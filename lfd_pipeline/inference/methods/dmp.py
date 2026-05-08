"""Generatore di traiettorie DMP (multi-demo, pesi medi o concat-LS).

Carica il modello salvato da ``learning/dmp/dmp_train.py`` ed effettua il
rollout dei 6 DMP (Ijspeert 2013) sullo stato y = [x, y, z, rx, ry, rz].

Adattamento ai task parameters:
  - posizioni e rotazioni di start/goal arrivano come [x, y, z, roll, pitch, yaw]
    (ZYX, radianti). Le rotazioni vengono convertite in rotvec (angle-axis)
    coerentemente con la rappresentazione interna del DMP;
  - se ``start_xyzrpy`` / ``goal_xyzrpy`` sono None viene usato l'endpoint
    medio sulle demo (``y0_mean`` / ``g_mean``);
  - ``duration_scale`` agisce come temporal scaling tau: T_out = tau * T_mean.

Il profilo gripper medio (campionato sulla fase s in [0,1] al training) viene
ricampionato sulla griglia di fase del rollout.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from .base import Trajectory


# ---------------------------------------------------------------------------
# rotation helpers (duplicati locali per non accoppiare i moduli)
# ---------------------------------------------------------------------------
def _rpy_zyx_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], float)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], float)
    return Rz @ Ry @ Rx


def _R_to_rotvec(R: np.ndarray) -> np.ndarray:
    cos_th = (np.trace(R) - 1.0) * 0.5
    cos_th = max(-1.0, min(1.0, cos_th))
    th = math.acos(cos_th)
    if th < 1e-9:
        return np.zeros(3, dtype=float)
    if abs(math.pi - th) < 1e-6:
        d = np.diag(R)
        i = int(np.argmax(d))
        col = (R[:, i] + np.eye(3)[:, i]) / math.sqrt(2.0 * (1.0 + d[i]))
        return col * th
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]], float) / (2.0 * math.sin(th))
    return axis * th


def _canonicalize_rotvec(r: np.ndarray,
                         r_ref: np.ndarray | None = None) -> np.ndarray:
    """Mette ``r`` sullo stesso ramo del log-map di un riferimento ``r_ref``.

    Per ogni rotvec esiste la rappresentazione equivalente
    ``r' = (1 - 2*pi/||r||) * r`` (stessa rotazione, asse opposto, angolo
    ``2*pi - ||r||``). Quando ``r_ref`` e' fornito scegliamo tra ``r`` e
    ``r'`` quella piu' vicina (in norma euclidea) a ``r_ref``: cosi' lo
    start/goal di inferenza vive sullo stesso ramo dei rotvec usati nel
    training (y0_mean / g_mean).

    Se ``r_ref`` e' None, fallback al canone "componente con modulo massimo
    positiva" (compatibilita' con modelli legacy che non hanno y0_mean/g_mean
    consistenti).
    """
    r = np.asarray(r, float).copy()
    norm = float(np.linalg.norm(r))
    if norm < 1e-9:
        return r
    r_alt = (1.0 - 2.0 * np.pi / norm) * r
    if r_ref is not None:
        r_ref = np.asarray(r_ref, float)
        if np.linalg.norm(r_alt - r_ref) < np.linalg.norm(r - r_ref):
            return r_alt
        return r
    k = int(np.argmax(np.abs(r)))
    if r[k] < 0.0:
        return r_alt
    return r


def _rpy_zyx_to_rotvec(rpy, r_ref: np.ndarray | None = None) -> np.ndarray:
    return _canonicalize_rotvec(
        _R_to_rotvec(_rpy_zyx_to_R(rpy[0], rpy[1], rpy[2])),
        r_ref=r_ref,
    )


def _rotvec_to_quat(r: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], float)
    axis = r / th
    s = math.sin(th * 0.5)
    q = np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(th * 0.5)], float)
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------
class DMPGenerator:
    """Generatore DMP per traiettorie [x, y, z, rotvec_x, rotvec_y, rotvec_z]
    + profilo gripper medio (interpolato sulla griglia di fase).
    """

    def __init__(self, model_npz_path: str | Path):
        d = np.load(str(model_npz_path), allow_pickle=True)
        self.W = np.asarray(d["W"], float)            # (6, n_bfs)
        self.C = np.asarray(d["C"], float)            # (n_bfs,)
        self.H = np.asarray(d["H"], float)            # (n_bfs,)
        self.alpha_z = float(d["alpha_z"])
        self.beta_z = float(d["beta_z"])
        self.alpha_s = float(d["alpha_s"])
        self.dt = float(d["dt"])
        self.n_bfs = int(d["n_bfs"])
        self.T_mean = float(d["T_mean"])
        self.y0_mean = np.asarray(d["y0_mean"], float)  # (6,)
        self.g_mean = np.asarray(d["g_mean"], float)    # (6,)
        self.s_ref = d["s_ref"] if "s_ref" in d.files else None
        self.grip_ref = d["grip_ref"] if "grip_ref" in d.files else np.array([])
        # Maschera per asse: True -> diagonal scaling Ijspeert
        # f = (g - y0) * f_norm; False -> forzante in unita' assolute
        # f = f_norm. Il default a True mantiene compatibilita' coi modelli
        # legacy senza la maschera salvata.
        if "use_scaling" in d.files:
            self.use_scaling = np.asarray(d["use_scaling"], dtype=bool)
        else:
            self.use_scaling = np.ones(6, dtype=bool)

    # ------------------------------------------------------------------ API
    def generate(
        self,
        start_xyzrpy: Optional[list],
        goal_xyzrpy: Optional[list],
        duration_scale: float = 1.0,
    ) -> Trajectory:
        # endpoint del rollout: parto dai default e sovrascrivo se richiesto.
        # Le rotazioni di start/goal vengono convertite in rotvec ASSOLUTO
        # (stesso spazio dei rotvec con cui il DMP e' stato allenato).
        y0 = self.y0_mean.copy()
        g = self.g_mean.copy()
        if start_xyzrpy is not None:
            y0[:3] = np.asarray(start_xyzrpy[:3], float)
            r_s = _rpy_zyx_to_rotvec(start_xyzrpy[3:], r_ref=self.y0_mean[3:])
            y0[3:] = r_s
        if goal_xyzrpy is not None:
            g[:3] = np.asarray(goal_xyzrpy[:3], float)
            r_g = _rpy_zyx_to_rotvec(goal_xyzrpy[3:], r_ref=self.g_mean[3:])
            g[3:] = r_g

        # rescaling temporale: tau scala sia la durata sia la fase
        tau = float(duration_scale)
        T_out = tau * self.T_mean
        steps = max(int(round(T_out / self.dt)) + 1, 2)
        t_grid = np.linspace(0.0, T_out, steps)

        # fase analitica s(t) = exp(-alpha_s * t / tau)
        s_grid = np.exp(-self.alpha_s * t_grid / max(tau, 1e-9))

        # rollout via Eulero forward sui 6 DMP indipendenti (vettoriale)
        # Per gli assi "attivi" (use_scaling=True) applica il diagonal scaling
        # Ijspeert: f = (g_new - y0_new) * f_norm.
        # Per gli assi "statici" nelle demo (use_scaling=False) la forzante
        # e' stata fittata in unita' assolute, quindi viene applicata
        # direttamente: f = f_norm. Cosi' un asse che le demo non hanno
        # mosso non genera moto spurio se in inferenza il target lo sposta
        # (la convergenza verso g e' garantita dal termine elastico
        # alpha_z * beta_z * (g - y) della transformation system).
        scale_eff = np.where(self.use_scaling, g - y0, 1.0)
        # eps-guard solo dove serve (assi attivi con scala numericamente nulla)
        eps = 1e-8
        active_tiny = self.use_scaling & (np.abs(scale_eff) < eps)
        if np.any(active_tiny):
            scale_eff = np.where(
                active_tiny,
                np.where(scale_eff == 0.0, eps, np.sign(scale_eff) * eps),
                scale_eff,
            )

        Y = np.zeros((steps, 6), float)
        dY = np.zeros((steps, 6), float)
        Y[0] = y0

        alpha_z = self.alpha_z
        beta_z = self.beta_z
        for i in range(1, steps):
            s = s_grid[i]
            psi = np.exp(-self.H * (s - self.C) ** 2)        # (n_bfs,)
            denom = psi.sum() + 1e-12
            # f_norm per ognuna delle 6 dim: (W @ psi) * s / denom
            f_norm = (self.W @ psi) * (s / denom)             # (6,)
            f = scale_eff * f_norm                             # (6,)
            ddY_i = (alpha_z * (beta_z * (g - Y[i - 1]) - tau * dY[i - 1]) + f) / (tau ** 2)
            dY[i] = dY[i - 1] + ddY_i * self.dt
            Y[i] = Y[i - 1] + dY[i] * self.dt

        xyz = Y[:, :3].copy()
        rotvec = Y[:, 3:6].copy()

        # rotvec assoluto -> quaternione con continuita' emisferica.
        quat = np.empty((steps, 4), float)
        prev = None
        for i in range(steps):
            q_abs = _rotvec_to_quat(rotvec[i])
            q_abs = q_abs / max(float(np.linalg.norm(q_abs)), 1e-12)
            if prev is not None and float(np.dot(q_abs, prev)) < 0.0:
                q_abs = -q_abs
            quat[i] = q_abs
            prev = q_abs

        # gripper: ricampiona il profilo medio sulla griglia di fase del rollout.
        # Usiamo il tempo normalizzato u = t/T_out in [0,1] (monotono crescente)
        # come parametro di ricampionamento, coerentemente con quanto fatto in GMR.
        if self.grip_ref.size > 0:
            u = t_grid / (t_grid[-1] + 1e-12)
            grip_out = np.interp(u,
                                 np.linspace(0.0, 1.0, len(self.grip_ref)),
                                 self.grip_ref)
        else:
            grip_out = None

        return Trajectory(t=t_grid, xyz_m=xyz, quat=quat, grip=grip_out)
