"""Interfaccia comune per i generatori di traiettoria (GMM, BC, DMP, ...)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass
class Trajectory:
    """Traiettoria cartesiana in BASE.

    Convenzioni:
        t      : (N,)    secondi, t[0]=0
        xyz_m  : (N, 3)  metri
        quat   : (N, 4)  (qx, qy, qz, qw), continuita' emisferica garantita
        grip   : (N,) o None   posizione gripper (unita' SDK xArm, 0..850)
    """

    t: np.ndarray
    xyz_m: np.ndarray
    quat: np.ndarray
    grip: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return int(self.t.shape[0])


class TrajectoryGenerator(Protocol):
    """Tutti i metodi (GMM, BC, DMP) implementano questa interfaccia.

    `start_xyzrpy` / `goal_xyzrpy` sono in formato [x, y, z, roll, pitch, yaw]
    con xyz in metri e angoli in radianti, convenzione ZYX (xArm).
    Possono essere None se il metodo non li sfrutta.
    """

    def generate(
        self,
        start_xyzrpy: Optional[list],
        goal_xyzrpy: Optional[list],
        duration_scale: float = 1.0,
    ) -> Trajectory: ...
