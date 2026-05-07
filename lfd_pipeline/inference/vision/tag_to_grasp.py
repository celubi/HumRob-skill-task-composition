"""Da posa ArUco a posa di grasping del gripper.

Vincoli geometrici (frame BASE):
    - z_g = -z_t              (assi z opposti)
    - x_g = +x_t              (assi x allineati)
    - y_g = z_g x x_g         (= -y_t, garantisce destrorso e ortonormale)
La posizione del gripper coincide col centro del marker, eventualmente
spostata di ``offset_z_m`` lungo l'asse z del tag (per portare il centro del
tag tra le pinze tenendo conto dell'offset del TCP rispetto alla punta).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _R_to_rpy_zyx(R: np.ndarray) -> Tuple[float, float, float]:
    R = np.asarray(R, float)
    sp = -R[2, 0]
    sp = float(np.clip(sp, -1.0, 1.0))
    pitch = math.asin(sp)
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return float(roll), float(pitch), float(yaw)


def grasp_pose_from_tag(T_base_tag: np.ndarray,
                        offset_z_m: float = 0.0) -> list[float]:
    """Calcola la posa target del gripper in formato [x,y,z, r,p,y] (m, rad).

    ``offset_z_m`` e' espresso lungo l'asse z del *tag* (positivo = verso
    la camera, dato che dopo l'inversione z_g = -z_t).
    """
    R_t = np.asarray(T_base_tag, float)[:3, :3]
    p_t = np.asarray(T_base_tag, float)[:3, 3]

    x_t = R_t[:, 0]
    z_t = R_t[:, 2]

    x_g = x_t / np.linalg.norm(x_t)
    z_g = -z_t / np.linalg.norm(z_t)
    # ri-ortogonalizzazione di x_g rispetto a z_g
    x_g = x_g - np.dot(x_g, z_g) * z_g
    x_g = x_g / np.linalg.norm(x_g)
    y_g = np.cross(z_g, x_g)

    R_g = np.column_stack([x_g, y_g, z_g])
    p_g = p_t + offset_z_m * z_t  # spostamento lungo z del tag

    roll, pitch, yaw = _R_to_rpy_zyx(R_g)
    return [float(p_g[0]), float(p_g[1]), float(p_g[2]),
            roll, pitch, yaw]


def place_pose_from_tag(T_base_tag: np.ndarray,
                        offset_z_m: float = 0.20) -> list[float]:
    """Posa target del gripper per il PLACE in formato [x,y,z, r,p,y] (m, rad).

    Vincoli geometrici (frame BASE):
        - x_g = +Z_t   (asse +x del gripper allineato a +Z del tag)
        - z_g = +X_t   (asse +z del gripper allineato a +X del tag)
        - y_g = z_g x x_g  (destrorso, ortonormale)
    Posizione: centro del tag spostato di ``offset_z_m`` lungo +Z del tag
    (default 20 cm sopra il tag, lungo il suo asse Z uscente).
    """
    R_t = np.asarray(T_base_tag, float)[:3, :3]
    p_t = np.asarray(T_base_tag, float)[:3, 3]

    x_t = R_t[:, 0]
    z_t = R_t[:, 2]

    x_g = z_t / np.linalg.norm(z_t)
    z_g = x_t / np.linalg.norm(x_t)
    # ri-ortogonalizzazione di z_g rispetto a x_g
    z_g = z_g - np.dot(z_g, x_g) * x_g
    z_g = z_g / np.linalg.norm(z_g)
    y_g = np.cross(z_g, x_g)

    R_g = np.column_stack([x_g, y_g, z_g])
    p_g = p_t + offset_z_m * (z_t / np.linalg.norm(z_t))

    roll, pitch, yaw = _R_to_rpy_zyx(R_g)
    return [float(p_g[0]), float(p_g[1]), float(p_g[2]),
            roll, pitch, yaw]
