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
    training (y0_mean / g_mean), che a loro volta sono stati canonicalizzati
    in modo consistente inter-demo dal preprocessing (alignment
    quaternionico globale, vedi preprocess_demos.align_quat_to_ref).

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


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Inversa di ``_rotvec_to_quat`` (qx, qy, qz, qw -> rotvec)."""
    q = np.asarray(q, float)
    qw = float(max(-1.0, min(1.0, q[3])))
    theta = 2.0 * math.acos(qw)
    sin_h = math.sqrt(max(0.0, 1.0 - qw * qw))
    if sin_h < 1e-12:
        return np.zeros(3, float)
    axis = q[:3] / sin_h
    return axis * theta


def _quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverso di quaternione unitario (qx, qy, qz, qw)."""
    q = np.asarray(q, float)
    out = q.copy()
    out[..., :3] = -out[..., :3]
    return out


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Prodotto di Hamilton (qx, qy, qz, qw)."""
    q1 = np.asarray(q1, float)
    q2 = np.asarray(q2, float)
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    out = np.empty(np.broadcast_shapes(q1.shape, q2.shape), float)
    out[..., 0] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    out[..., 1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    out[..., 2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    out[..., 3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return out


def _rpy_zyx_to_quat(rpy) -> np.ndarray:
    return _rotvec_to_quat(_R_to_rotvec(_rpy_zyx_to_R(rpy[0], rpy[1], rpy[2])))


def _abs_rpy_to_rel_rotvec(rpy, q_ref: np.ndarray) -> np.ndarray:
    """rpy assoluto -> rotvec nel frame relativo a ``q_ref``.

        q_abs = rpy_to_quat(rpy)
        q_rel = q_ref^{-1} * q_abs
        r_rel = log(q_rel)
    """
    q_abs = _rpy_zyx_to_quat(rpy)
    q_rel = _quat_mul(_quat_inv(q_ref), q_abs)
    return _quat_to_rotvec(q_rel)


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
        # quaternione di riferimento per la ricentratura della rotazione.
        # Se assente (modello legacy) si assume identita': i rotvec sono
        # interpretati come rotazioni assolute (vecchia convenzione).
        self.q_ref = (np.asarray(d["q_ref"], float) if "q_ref" in d.files
                      else np.array([0.0, 0.0, 0.0, 1.0], float))

    # ------------------------------------------------------------------ API
    def generate(
        self,
        start_xyzrpy: Optional[list],
        goal_xyzrpy: Optional[list],
        duration_scale: float = 1.0,
    ) -> Trajectory:
        # endpoint del rollout: parto dai default e sovrascrivo se richiesto.
        # Per la rotazione lavoro nel frame RELATIVO a q_ref:
        #   r_rel = log(q_ref^{-1} * q_abs)
        # cosi' i rotvec di start/goal vivono nello stesso spazio dei rotvec
        # con cui il DMP e' stato allenato (preprocess_demos -> recenter).
        y0 = self.y0_mean.copy()
        g = self.g_mean.copy()
        if start_xyzrpy is not None:
            y0[:3] = np.asarray(start_xyzrpy[:3], float)
            r_rel_s = _abs_rpy_to_rel_rotvec(start_xyzrpy[3:], self.q_ref)
            y0[3:] = _canonicalize_rotvec(r_rel_s, r_ref=self.y0_mean[3:])
        if goal_xyzrpy is not None:
            g[:3] = np.asarray(goal_xyzrpy[:3], float)
            r_rel_g = _abs_rpy_to_rel_rotvec(goal_xyzrpy[3:], self.q_ref)
            g[3:] = _canonicalize_rotvec(r_rel_g, r_ref=self.g_mean[3:])

        # rescaling temporale: tau scala sia la durata sia la fase
        tau = float(duration_scale)
        T_out = tau * self.T_mean
        steps = max(int(round(T_out / self.dt)) + 1, 2)
        t_grid = np.linspace(0.0, T_out, steps)

        # fase analitica s(t) = exp(-alpha_s * t / tau)
        s_grid = np.exp(-self.alpha_s * t_grid / max(tau, 1e-9))

        # rollout via Eulero forward sui 6 DMP indipendenti (vettoriale)
        scale_new = g - y0
        # eps-guard per dimensioni "piatte"
        eps = 1e-8
        scale_new = np.where(np.abs(scale_new) < eps,
                             np.where(scale_new == 0.0, eps, np.sign(scale_new) * eps),
                             scale_new)

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
            f = scale_new * f_norm                             # (6,)
            ddY_i = (alpha_z * (beta_z * (g - Y[i - 1]) - tau * dY[i - 1]) + f) / (tau ** 2)
            dY[i] = dY[i - 1] + ddY_i * self.dt
            Y[i] = Y[i - 1] + dY[i] * self.dt

        xyz = Y[:, :3].copy()
        rotvec = Y[:, 3:6].copy()

        # rotvec relativo a q_ref -> quaternione ASSOLUTO con continuita'
        # emisferica:   q_abs = q_ref * exp(rotvec_rel)
        quat = np.empty((steps, 4), float)
        prev = None
        for i in range(steps):
            q_rel = _rotvec_to_quat(rotvec[i])
            q_abs = _quat_mul(self.q_ref, q_rel)
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
