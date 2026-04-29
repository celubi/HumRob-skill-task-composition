"""Generatore di traiettorie GMM-GMR.

Porta la logica di ``lfd_exec_real/gmm_infer.py`` dietro l'interfaccia
``TrajectoryGenerator``: niente I/O su CSV, solo un oggetto ``Trajectory``
in memoria. La rotazione di goal viene blendata in spazio tangente (rotvec),
quindi convertiamo internamente l'rpy ZYX -> rotvec.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from .base import Trajectory


# ---------------------------------------------------------------------------
# rotation helpers
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
    """Rotation matrix -> axis-angle (rotvec). Robusto su angoli piccoli."""
    cos_th = (np.trace(R) - 1.0) * 0.5
    cos_th = max(-1.0, min(1.0, cos_th))
    th = math.acos(cos_th)
    if th < 1e-9:
        return np.zeros(3, dtype=float)
    if abs(math.pi - th) < 1e-6:
        # caso prossimo a pi: estrai dall'auto-vettore di R
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


def _rpy_zyx_to_rotvec(rpy: list,
                       r_ref: np.ndarray | None = None) -> np.ndarray:
    return _canonicalize_rotvec(
        _R_to_rotvec(_rpy_zyx_to_R(rpy[0], rpy[1], rpy[2])),
        r_ref=r_ref,
    )


def _rotvec_to_quat(r: np.ndarray) -> np.ndarray:
    """rotvec (axis*angle) -> quaternion (qx, qy, qz, qw)."""
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
# GMR core (1D input, multi-dim output)
# ---------------------------------------------------------------------------
def _gauss_pdf_1d(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    x = x.reshape(-1, 1)
    mu = mu.reshape(1, -1)
    var = var.reshape(1, -1)
    coef = 1.0 / np.sqrt(2.0 * math.pi * var + 1e-12)
    return coef * np.exp(-0.5 * (x - mu) ** 2 / (var + 1e-12))


def _gmr_predict_mean(weights, means, covs, x_vals, in_idx, out_idx):
    K = means.shape[0]
    o = np.array(out_idx, int)
    i = np.array(in_idx, int)
    mu_x = means[:, i].reshape(K, -1)
    mu_y = means[:, o].reshape(K, -1)
    Sigma_xx = covs[:, i[:, None], i].reshape(K)
    Sigma_yx = covs[:, o[:, None], i].reshape(K, -1)

    var_x = Sigma_xx + 1e-9
    px = _gauss_pdf_1d(np.asarray(x_vals, float), mu_x.reshape(K), var_x)
    h = px * weights.reshape(1, K)
    h = h / (h.sum(axis=1, keepdims=True) + 1e-12)

    inv_xx = 1.0 / var_x
    Y = np.empty((len(x_vals), len(out_idx)), float)
    for n, x in enumerate(x_vals):
        mu_yx = mu_y + (Sigma_yx * inv_xx[:, None]) * (x - mu_x)
        Y[n] = (h[n][:, None] * mu_yx).sum(axis=0)
    return Y


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------
class GMMGMRGenerator:
    """Generatore GMM-GMR per traiettorie [x, y, z, rotvec_x, rotvec_y, rotvec_z]
    + profilo gripper medio (interpolato sulla griglia di fase).
    """

    def __init__(self, model_npz_path: str | Path):
        d = np.load(str(model_npz_path), allow_pickle=True)
        self.weights = d["weights"]
        self.means = d["means"]
        self.covariances = d["covariances"]
        self.in_idx = d["input_idx"]
        self.out_idx = d["output_idx"]
        self.dt = float(d["dt"])
        self.T_mean = float(d["T_mean"])
        self.s_ref = d["s_ref"]
        self.grip_ref = d["grip_ref"] if "grip_ref" in d.files else np.array([])
        # eventuale normalizzazione (z-score per-colonna su tutte le D)
        if "normalized" in d.files and bool(d["normalized"]):
            self.norm_mean = d["norm_mean"]
            self.norm_std = d["norm_std"]
            self.normalized = True
        else:
            self.norm_mean = np.zeros(self.means.shape[1])
            self.norm_std = np.ones(self.means.shape[1])
            self.normalized = False
        # quaternione di riferimento per la ricentratura della rotazione.
        self.q_ref = (np.asarray(d["q_ref"], float) if "q_ref" in d.files
                      else np.array([0.0, 0.0, 0.0, 1.0], float))

    # ------------------------------------------------------------------ API
    def generate(
        self,
        start_xyzrpy: Optional[list],
        goal_xyzrpy: Optional[list],
        duration_scale: float = 1.0,
    ) -> Trajectory:
        T_out = self.T_mean * float(duration_scale)
        steps = int(round(T_out / self.dt)) + 1
        t_grid = np.linspace(0.0, T_out, steps)
        s_grid = t_grid / (t_grid[-1] + 1e-12)

        # GMR sullo spazio normalizzato, poi de-normalizzazione delle uscite
        if self.normalized:
            s_in = (s_grid - self.norm_mean[0]) / self.norm_std[0]
        else:
            s_in = s_grid
        Y = _gmr_predict_mean(self.weights, self.means, self.covariances,
                              s_in, self.in_idx, self.out_idx)
        if self.normalized:
            Y = Y * self.norm_std[1:] + self.norm_mean[1:]

        xyz = Y[:, :3].copy()
        rotvec = Y[:, 3:6].copy()

        # ------------- adattamento start/goal -------------
        # Posizioni: blending lineare in s. Rotazioni: additivo in rotvec
        # (valido per piccoli delta, come nel codice di riferimento).
        d_pos0 = np.zeros(3); d_rv0 = np.zeros(3)
        d_pos1 = np.zeros(3); d_rv1 = np.zeros(3)

        if start_xyzrpy is not None:
            sxyz = np.asarray(start_xyzrpy[:3], float)
            # rotvec relativo a q_ref (stesso spazio del rotvec del GMR);
            # poi allineato al ramo log-map gia' usato dal GMR a t=0.
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

        # gripper: ricampiona il profilo medio sulla griglia di fase
        if self.grip_ref.size > 0:
            grip_out = np.interp(s_grid,
                                 np.linspace(0.0, 1.0, len(self.grip_ref)),
                                 self.grip_ref)
        else:
            grip_out = None

        return Trajectory(t=t_grid, xyz_m=xyz, quat=quat, grip=grip_out)
