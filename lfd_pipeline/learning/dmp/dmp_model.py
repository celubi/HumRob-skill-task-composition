"""DMP 1D - formulazione standard Ijspeert et al. 2013 [10].

Versione "pulita" rispetto al codice in src/.../lfd_learning/dmp:
  - tau fissato a 1.0 al fit; il rescaling temporale e' applicato all'inferenza.
  - Centri delle basis function: c_i = exp(-alpha_s * i/(N-1)),  i=0..N-1.
  - Larghezze: h_i = 1 / (c_{i+1} - c_i)^2  (ultimo padded col precedente).
  - Fase analitica: s(t) = exp(-alpha_s * t).
  - Diagonal scaling Ijspeert: f_norm = f_target / (g - y0), con eps-guard.
  - Le derivate (dy, ddy) arrivano dal preprocessing (no np.gradient interno).

Questo modulo espone solo il fit (closed-form weighted LS); il rollout vive
nel layer di inferenza.
"""

from __future__ import annotations

import numpy as np


def make_basis(n_bfs: int, alpha_s: float, T: float = 1.0):
    """Centri c_i e larghezze h_i delle Gaussian basis function.

    I centri sono distribuiti uniformemente in TEMPO sull'intervallo [0, T] e
    poi mappati in fase via s(t)=exp(-alpha_s*t). Cosi' coprono tutto l'arco
    di fase effettivamente percorso da una traiettoria di durata T (con tau=1
    al fit). Se T=1 si ricade nella convenzione 'normalizzata' classica.
    """
    i = np.arange(n_bfs, dtype=float)
    t_centers = i / max(n_bfs - 1, 1) * float(T)
    c = np.exp(-alpha_s * t_centers)
    diffs = np.diff(c)
    h_inner = 1.0 / (diffs ** 2)
    h = np.empty(n_bfs, dtype=float)
    h[:-1] = h_inner
    h[-1] = h_inner[-1] if len(h_inner) > 0 else 1.0
    return c, h


def psi_matrix(s: np.ndarray, c: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Matrice (T, N_bfs) delle basis function valutate sulla fase s."""
    s_vec = np.asarray(s, dtype=float).reshape(-1)
    diff = s_vec[:, None] - c[None, :]
    return np.exp(-h[None, :] * diff * diff)


class DMP1D:
    """DMP 1D Ijspeert con tau=1 al fit. Una istanza per dimensione."""

    def __init__(self, n_bfs: int, alpha_z: float, alpha_s: float, dt: float,
                 T: float = 1.0):
        self.n_bfs = int(n_bfs)
        self.alpha_z = float(alpha_z)
        self.beta_z = self.alpha_z / 4.0
        self.alpha_s = float(alpha_s)
        self.dt = float(dt)
        self.tau = 1.0
        self.T = float(T)

        self.c, self.h = make_basis(self.n_bfs, self.alpha_s, T=self.T)
        self.w = np.zeros(self.n_bfs, dtype=float)
        self.y0 = 0.0
        self.g = 0.0
        self._scale_demo = 1.0

    # --- internals --------------------------------------------------------
    def _phase(self, n: int) -> np.ndarray:
        """Fase analitica s(t) = exp(-alpha_s * t/tau) sulla griglia di N campioni."""
        t = np.arange(n, dtype=float) * self.dt
        return np.exp(-self.alpha_s * t / self.tau)

    def _design(self, s: np.ndarray) -> np.ndarray:
        """Matrice di design X = Psi * s / sum(Psi)  (T, N_bfs)."""
        Psi = psi_matrix(s, self.c, self.h)
        denom = Psi.sum(axis=1) + 1e-12
        return Psi * (s / denom)[:, None]

    @staticmethod
    def _safe_scale(scale: float, eps: float = 1e-8) -> float:
        if abs(scale) < eps:
            return np.sign(scale) * eps if scale != 0.0 else eps
        return float(scale)

    # --- fit --------------------------------------------------------------
    def fit(self, y: np.ndarray, dy: np.ndarray, ddy: np.ndarray) -> "DMP1D":
        """Locally-Weighted Regression per BF (Ijspeert 2013).

        Per ogni basis function i:
            w_i = sum_t psi_i(t) * s(t) * f_norm(t)
                / (sum_t psi_i(t) * s(t)^2 + eps)

        Disaccoppia i pesi e li mantiene limitati anche con BF strette
        (a differenza della LS globale).
        """
        y = np.asarray(y, dtype=float)
        dy = np.asarray(dy, dtype=float)
        ddy = np.asarray(ddy, dtype=float)
        if not (len(y) == len(dy) == len(ddy)):
            raise ValueError("y, dy, ddy devono avere la stessa lunghezza.")

        self.y0 = float(y[0])
        self.g = float(y[-1])

        s = self._phase(len(y))                  # (N,)
        Psi = psi_matrix(s, self.c, self.h)      # (N, n_bfs)

        # forcing target (Ijspeert 2013, tau=1)
        f_target = (self.tau ** 2) * ddy - self.alpha_z * (
            self.beta_z * (self.g - y) - self.tau * dy
        )
        self._scale_demo = self._safe_scale(self.g - self.y0)
        f_norm = f_target / self._scale_demo     # (N,)

        # LWR closed-form, per BF
        s_col = s[:, None]                       # (N, 1)
        num = (Psi * s_col * f_norm[:, None]).sum(axis=0)     # (n_bfs,)
        den = (Psi * (s_col ** 2)).sum(axis=0) + 1e-12        # (n_bfs,)
        self.w = num / den
        return self

    def fit_design(self):
        """Espone (X, target_normalizer) pronti per la regressione concatenata.

        Ritorna una funzione che, dati y, dy, ddy, restituisce la coppia
        (X, f_norm) per concat-LS multi-demo. Usata da dmp_train.py quando
        method='concat-ls'.
        """
        # alias semantico per chiarezza nel chiamante
        return self._design, self._safe_scale


def fit_demo_weights(y, dy, ddy, n_bfs, alpha_z, alpha_s, dt):
    """Helper: ritorna w (N_bfs,) per UNA dimensione di UNA demo."""
    dmp = DMP1D(n_bfs=n_bfs, alpha_z=alpha_z, alpha_s=alpha_s, dt=dt)
    dmp.fit(y, dy, ddy)
    return dmp.w, dmp._scale_demo
