"""Generatore di traiettorie Goal-Conditioned Behavioral Cloning.

Carica il modello salvato da ``learning/bc/bc_train.py`` (file ``.pt``) ed
effettua il rollout iterativo della policy MLP goal-conditioned:

    s_0  <- start (oppure s0_mean dalle demo)
    g    <- goal_xyzrpy (oppure sT_mean dalle demo)
    s~_t = [s_t, g - s_t]                   (stato esteso, R^12)
    a_t  = pi_theta(s~_t)                    (forward MLP, normalizz./denorm.)
    s_{t+1} = s_t + a_t

Il numero di step e' derivato dalla durata media delle demo (``T_mean``) e
da ``duration_scale``, coerentemente con DMP/GMM.

Anche con goal-conditioning, il BC accumula errore lungo il rollout
(covariate shift, Ross & Bagnell 2010) e in pratica non chiude esattamente
sul goal richiesto. Per riportare l'endpoint sul target del task viene
applicato sempre un goal-blending lineare in fase, identico a quello del
generatore GMM-GMR (Sez II-F: task-parameter retrieval).

Il profilo gripper medio (campionato su s in [0,1] al training) viene
ricampionato sulla griglia di fase del rollout, esattamente come in DMP/GMM.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parents[1]  # .../lfd_pipeline
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from learning.bc.bc_model import BCPolicy  # noqa: E402

from .base import Trajectory


# ---------------------------------------------------------------------------
# rotation helpers (duplicati per non accoppiare i moduli, come negli altri
# generatori)
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

    Se ``r_ref`` e' fornito si sceglie tra ``r`` e la rappresentazione
    equivalente ``(1 - 2*pi/||r||) * r`` quella piu' vicina (in norma
    euclidea) a ``r_ref``: cosi' lo start/goal di inferenza vive sullo
    stesso ramo dei rotvec del training (consistenti grazie all'alignment
    quaternionico globale fatto in preprocess_demos).

    Senza riferimento, fallback al canone "componente con modulo massimo
    positiva" (compatibilita' con modelli legacy).
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
    q = np.asarray(q, float)
    qw = float(max(-1.0, min(1.0, q[3])))
    theta = 2.0 * math.acos(qw)
    sin_h = math.sqrt(max(0.0, 1.0 - qw * qw))
    if sin_h < 1e-12:
        return np.zeros(3, float)
    return q[:3] / sin_h * theta


def _quat_inv(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    out = q.copy()
    out[..., :3] = -out[..., :3]
    return out


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
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


def _abs_rpy_to_rel_rotvec(rpy, q_ref: np.ndarray) -> np.ndarray:
    """rpy assoluto -> rotvec nel frame relativo a ``q_ref``."""
    R3 = _rpy_zyx_to_R(rpy[0], rpy[1], rpy[2])
    q_abs = _rotvec_to_quat(_R_to_rotvec(R3))
    q_rel = _quat_mul(_quat_inv(q_ref), q_abs)
    return _quat_to_rotvec(q_rel)


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------
class BCGenerator:
    """Generatore Goal-Conditioned BC per traiettorie
    [x, y, z, rotvec_x, rotvec_y, rotvec_z] + profilo gripper medio.

    Il goal-blending lineare in fase e' SEMPRE attivo: il GC-BC modula la
    traiettoria sul goal ma il rollout accumula errore (covariate shift),
    quindi serve la correzione end-of-trajectory per chiudere esattamente
    sul target del task (coerente con la formulazione di Sez II-F del paper).
    """

    def __init__(self, model_pt_path: str | Path, device: str = "cpu"):
        self.model, self.extra = BCPolicy.load_with_extra(
            Path(model_pt_path), map_location=device,
        )
        self.device = device

        ex = self.extra
        self.dt = float(ex["dt"])
        self.T_mean = float(ex["T_mean"])
        self.N_ref = int(ex.get("N_ref", round(self.T_mean / self.dt) + 1))
        self.s0_mean = np.asarray(ex["s0_mean"], float)   # (6,)
        self.sT_mean = np.asarray(ex["sT_mean"], float)   # (6,)
        self.grip_ref = np.asarray(ex.get("grip_ref", []), float)
        # quaternione di riferimento per la ricentratura della rotazione.
        self.q_ref = (np.asarray(ex["q_ref"], float) if "q_ref" in ex
                      else np.array([0.0, 0.0, 0.0, 1.0], float))
        # GC-BC ha state_dim = 12 = [s_t, g - s_t]; vanilla BC = 6.
        self.goal_conditioned = bool(ex.get("goal_conditioned",
                                            self.model.cfg.state_dim == 12))
        if self.model.cfg.state_dim not in (6, 12):
            raise ValueError(
                f"state_dim inatteso ({self.model.cfg.state_dim}); "
                "supportati 6 (vanilla BC) o 12 (GC-BC)."
            )

    # ------------------------------------------------------------------ API
    def generate(
        self,
        start_xyzrpy: Optional[list],
        goal_xyzrpy: Optional[list],
        duration_scale: float = 1.0,
    ) -> Trajectory:
        # 1) stato iniziale e goal in spazio [x,y,z,rotvec_REL] (relativo a q_ref)
        s0 = self.s0_mean.copy()
        if start_xyzrpy is not None:
            s0[:3] = np.asarray(start_xyzrpy[:3], float)
            r_rel_s = _abs_rpy_to_rel_rotvec(start_xyzrpy[3:], self.q_ref)
            s0[3:] = _canonicalize_rotvec(r_rel_s, r_ref=self.s0_mean[3:])

        g = self.sT_mean.copy()
        if goal_xyzrpy is not None:
            g[:3] = np.asarray(goal_xyzrpy[:3], float)
            r_rel_g = _abs_rpy_to_rel_rotvec(goal_xyzrpy[3:], self.q_ref)
            g[3:] = _canonicalize_rotvec(r_rel_g, r_ref=self.sT_mean[3:])

        # 2) numero di step coerente con DMP/GMM
        T_out = self.T_mean * float(duration_scale)
        steps = max(int(round(T_out / self.dt)) + 1, 2)
        t_grid = np.linspace(0.0, T_out, steps)
        s_grid = t_grid / (t_grid[-1] + 1e-12)   # fase normalizzata in [0, 1]

        # 3) rollout iterativo della policy
        Y = np.zeros((steps, 6), float)
        Y[0] = s0
        for i in range(1, steps):
            s_prev = Y[i - 1]
            if self.goal_conditioned:
                s_in = np.concatenate([s_prev, g - s_prev]).astype(np.float32)
            else:
                s_in = s_prev.astype(np.float32)
            a = self.model.predict_action(s_in)
            Y[i] = s_prev + a

        xyz = Y[:, :3].copy()
        rotvec = Y[:, 3:6].copy()

        # 4) goal-blending lineare in s, sempre attivo (vedi docstring).
        #    I rotvec del rollout sono RELATIVI a q_ref; per coerenza calcolo
        #    anche start/goal del blending in coordinate relative.
        d_pos0 = np.zeros(3); d_rv0 = np.zeros(3)
        d_pos1 = np.zeros(3); d_rv1 = np.zeros(3)
        if start_xyzrpy is not None:
            sxyz = np.asarray(start_xyzrpy[:3], float)
            srv = _abs_rpy_to_rel_rotvec(start_xyzrpy[3:], self.q_ref)
            srv = _canonicalize_rotvec(srv, r_ref=rotvec[0])
            d_pos0 = sxyz - xyz[0]
            d_rv0 = srv - rotvec[0]
        if goal_xyzrpy is not None:
            gxyz = np.asarray(goal_xyzrpy[:3], float)
            grv = _abs_rpy_to_rel_rotvec(goal_xyzrpy[3:], self.q_ref)
            grv = _canonicalize_rotvec(grv, r_ref=rotvec[-1])
            d_pos1 = gxyz - xyz[-1]
            d_rv1 = grv - rotvec[-1]
        if start_xyzrpy is not None or goal_xyzrpy is not None:
            for i, s in enumerate(s_grid):
                xyz[i]    = xyz[i]    + (1.0 - s) * d_pos0 + s * d_pos1
                rotvec[i] = rotvec[i] + (1.0 - s) * d_rv0  + s * d_rv1

        # 5) rotvec relativo a q_ref -> quaternioni ASSOLUTI con continuita'
        #    emisferica:   q_abs = q_ref * exp(rotvec_rel)
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

        # 6) gripper: replicato dal profilo medio delle demo (fuori dalla MLP).
        if self.grip_ref.size > 0:
            grip_out = np.interp(s_grid,
                                 np.linspace(0.0, 1.0, len(self.grip_ref)),
                                 self.grip_ref)
        else:
            grip_out = None

        return Trajectory(t=t_grid, xyz_m=xyz, quat=quat, grip=grip_out)
