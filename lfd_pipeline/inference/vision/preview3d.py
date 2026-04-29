"""Anteprima 3D della traiettoria con il modello del robot in PyBullet.

La GUI mostra:
    - il modello URDF dello xArm 6 (mesh visive di ``xarm_description``,
      gripper standard incluso) posizionato nella configurazione HOME;
    - la polilinea della traiettoria del TCP;
    - terne (assi RGB) a intervalli regolari sui waypoint;
    - una terna piu' grande sul *goal pose* (posa di grasping richiesta);
    - una piccola terna animata sul TCP, opzionalmente trascinata via IK.

L'URDF viene generato al volo via ``xacro`` (ROS 2) la prima volta, con
i percorsi ``package://xarm_description/`` riscritti come path assoluti
in modo che PyBullet possa caricare le mesh STL senza un workspace
ROS installato.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# URDF generation / cache
# ---------------------------------------------------------------------------
_DESC_DIR = Path("/home/localadmin/xarm_ws-main/src/xarm_code/xarm_description").resolve()
_CACHE_DIR = Path.home() / ".cache" / "lfd_pipeline"
_URDF_PATH = _CACHE_DIR / "xarm6_with_gripper.urdf"
_AMENT_SHIM = _CACHE_DIR / "ament_shim"
_ROS_SETUP = "/opt/ros/jazzy/setup.bash"


def _build_ament_shim() -> Path:
    """Crea uno share/ minimale che simula l'install ament di
    ``xarm_description``, con symlink ai sotto-folder della sorgente.
    """
    share = _AMENT_SHIM / "share"
    pkg = share / "xarm_description"
    idx = share / "ament_index" / "resource_index" / "packages"
    idx.mkdir(parents=True, exist_ok=True)
    (idx / "xarm_description").touch()
    pkg.mkdir(parents=True, exist_ok=True)
    for sub in ("urdf", "meshes", "config", "rviz", "launch"):
        src = _DESC_DIR / sub
        dst = pkg / sub
        if src.is_dir() and not dst.exists():
            os.symlink(src, dst)
    return _AMENT_SHIM


def _generate_urdf(force: bool = False) -> Path:
    """Genera (o restituisce dalla cache) l'URDF di xarm6 + gripper standard,
    con i ``package://`` riscritti in path assoluti.
    """
    if _URDF_PATH.is_file() and not force:
        return _URDF_PATH

    if not Path(_ROS_SETUP).is_file():
        raise RuntimeError(
            f"ROS 2 setup non trovato in {_ROS_SETUP}; "
            "necessario per generare l'URDF via xacro."
        )

    shim = _build_ament_shim()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    xacro_in = _DESC_DIR / "urdf" / "xarm_device.urdf.xacro"
    cmd = (
        f"source {_ROS_SETUP} && "
        f"AMENT_PREFIX_PATH={shim}:$AMENT_PREFIX_PATH "
        f"xacro {xacro_in} dof:=6 robot_type:=xarm add_gripper:=true"
    )
    proc = subprocess.run(
        ["bash", "-lc", cmd], capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"xacro fallito (exit {proc.returncode}):\n{proc.stderr}"
        )

    urdf_text = proc.stdout
    abs_pkg = str(shim / "share" / "xarm_description")
    urdf_text = urdf_text.replace("package://xarm_description", f"file://{abs_pkg}")
    # PyBullet vuole path su filesystem, non file:// -> li teniamo entrambi gestiti
    urdf_text = urdf_text.replace("file://", "")

    _URDF_PATH.write_text(urdf_text)
    return _URDF_PATH


# ---------------------------------------------------------------------------
# RobotPreview3D
# ---------------------------------------------------------------------------
class RobotPreview3D:
    """Wrapper minimale attorno a PyBullet per visualizzare il robot e una
    traiettoria. Gestisce la propria simulazione in modalita' GUI.
    """

    AXES_LEN = 0.06    # m, terne sui waypoint
    GOAL_AXES_LEN = 0.10
    SAMPLE_EVERY = 10  # disegna una terna ogni N waypoint

    def __init__(self, home_joint_deg, animate: bool = False):
        # imports laziosi
        import pybullet as pb               # noqa: WPS433
        import pybullet_data                 # noqa: WPS433
        self._pb = pb
        self._pbdata = pybullet_data
        self.home_joint_deg = list(home_joint_deg)
        self.animate = bool(animate)

        urdf = _generate_urdf()
        self._client = pb.connect(pb.GUI)
        # tieni visibile il pannello laterale (hint sui controlli camera) e
        # disabilita il mouse-picking per evitare che il click selezioni i
        # rigid body invece di interagire con la camera.
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 1)
        pb.configureDebugVisualizer(pb.COV_ENABLE_MOUSE_PICKING, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 1)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, 0)
        # piano del tavolo per riferimento visivo
        try:
            pb.loadURDF("plane.urdf", basePosition=[0, 0, -0.0])
        except Exception:
            pass

        flags = pb.URDF_USE_INERTIA_FROM_FILE
        self.robot = pb.loadURDF(str(urdf), basePosition=[0, 0, 0],
                                 useFixedBase=True, flags=flags)

        # mappa joint name -> index e indici dei 6 giunti rivoluti del braccio
        # (joint1..joint6: gli altri "joint*" sono del gripper a 5 barre)
        self._joint_idx_by_name: dict[str, int] = {}
        for j in range(pb.getNumJoints(self.robot)):
            info = pb.getJointInfo(self.robot, j)
            self._joint_idx_by_name[info[1].decode("utf-8")] = j
        self._arm_joints = [
            self._joint_idx_by_name[f"joint{i}"] for i in range(1, 7)
            if f"joint{i}" in self._joint_idx_by_name
        ]

        # link "tool" / TCP: usiamo "link_tcp" se presente, altrimenti l'ultimo
        # link del braccio (link6).
        self.tcp_link = self._joint_idx_by_name.get(
            "joint_tcp", self._arm_joints[-1])

        # imposta HOME (deg -> rad)
        for j, deg in zip(self._arm_joints, self.home_joint_deg):
            pb.resetJointState(self.robot, j, math.radians(float(deg)))
        pb.stepSimulation()

        # viewpoint comodo sulla scena
        pb.resetDebugVisualizerCamera(
            cameraDistance=1.1,
            cameraYaw=45.0,
            cameraPitch=-30.0,
            cameraTargetPosition=[0.30, 0.0, 0.20],
        )
        # frame BASE
        self._draw_axes(np.eye(3), np.zeros(3), length=0.12, lifetime=0)

        self._traj_line_ids: list[int] = []
        self._tcp_marker_ids: list[int] = []

    # ----------------------------------------------------------------- utils
    def _quat_to_R(self, q):
        # PyBullet ordering: (qx, qy, qz, qw)
        x, y, z, w = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], float)

    def _rpy_zyx_to_quat(self, rpy):
        r, p, y = rpy
        # ZYX intrinsic = XYZ extrinsic
        return self._pb.getQuaternionFromEuler([r, p, y])

    def _draw_axes(self, R, t, length=0.06, lifetime=0):
        ids = []
        for i, color in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
            end = t + R[:, i] * length
            ids.append(self._pb.addUserDebugLine(
                t.tolist(), end.tolist(), color, lineWidth=2.0,
                lifeTime=lifetime,
            ))
        return ids

    # --------------------------------------------------------------- public
    def show_trajectory(self, traj, goal_xyzrpy: Optional[list] = None) -> None:
        pb = self._pb
        # Disabilita il rendering del visualizer durante il disegno, cosi'
        # PyBullet non flusha la GUI fra una addUserDebugLine e l'altra:
        # tutto compare in un singolo frame quando si riabilita.
        try:
            pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
        except Exception:
            pass
        try:
            # cancella eventuali precedenti
            for lid in self._traj_line_ids + self._tcp_marker_ids:
                try: pb.removeUserDebugItem(lid)
                except Exception: pass
            self._traj_line_ids.clear()
            self._tcp_marker_ids.clear()

            # polilinea TCP (verde)
            xyz = traj.xyz_m
            for i in range(1, traj.n):
                self._traj_line_ids.append(pb.addUserDebugLine(
                    xyz[i - 1].tolist(), xyz[i].tolist(),
                    lineColorRGB=[0.1, 0.8, 0.1], lineWidth=2.0, lifeTime=0,
                ))

            # terne RGB ogni N waypoint
            for i in range(0, traj.n, self.SAMPLE_EVERY):
                R = self._quat_to_R(traj.quat[i])
                self._traj_line_ids.extend(
                    self._draw_axes(R, xyz[i], length=self.AXES_LEN)
                )
            # terna sull'ultimo waypoint
            R_end = self._quat_to_R(traj.quat[-1])
            self._traj_line_ids.extend(
                self._draw_axes(R_end, xyz[-1], length=self.AXES_LEN * 1.2)
            )

            # terna sul goal richiesto (in giallo-magenta)
            if goal_xyzrpy is not None:
                t_goal = np.asarray(goal_xyzrpy[:3], float)
                q_goal = self._rpy_zyx_to_quat(goal_xyzrpy[3:])
                R_goal = self._quat_to_R(q_goal)
                self._traj_line_ids.extend(
                    self._draw_axes(R_goal, t_goal,
                                    length=self.GOAL_AXES_LEN)
                )
                self._traj_line_ids.append(pb.addUserDebugText(
                    "GOAL", textPosition=(t_goal + np.array([0, 0, 0.04])).tolist(),
                    textColorRGB=[1, 0.4, 0.0], textSize=1.2, lifeTime=0,
                ))
        finally:
            try:
                pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
            except Exception:
                pass

        # animazione opzionale: muovi il robot via IK lungo la traiettoria
        if self.animate:
            self._animate_ik(traj)

    def _animate_ik(self, traj) -> None:
        pb = self._pb
        # parametri IK
        n_iter = 100
        residual = 1e-4
        # pacing visibile (~ 30 Hz in finestra)
        period = 1.0 / 30.0
        last_t = time.perf_counter()
        for i in range(traj.n):
            tgt_pos = traj.xyz_m[i].tolist()
            tgt_orn = list(traj.quat[i])  # (qx,qy,qz,qw) - stessa convenzione PyBullet
            joints = pb.calculateInverseKinematics(
                self.robot, self.tcp_link,
                targetPosition=tgt_pos,
                targetOrientation=tgt_orn,
                maxNumIterations=n_iter,
                residualThreshold=residual,
            )
            for j_idx, j_val in zip(self._arm_joints, joints[:len(self._arm_joints)]):
                pb.resetJointState(self.robot, j_idx, float(j_val))
            pb.stepSimulation()
            # throttle
            now = time.perf_counter()
            sleep = period - (now - last_t)
            if sleep > 0:
                time.sleep(sleep)
            last_t = time.perf_counter()
        # alla fine: torna a HOME visivamente
        for j, deg in zip(self._arm_joints, self.home_joint_deg):
            pb.resetJointState(self.robot, j, math.radians(float(deg)))

    def wait_for_user(self, prompt: str = "[3D] INVIO per continuare ") -> None:
        """Tiene viva la GUI mentre l'utente ispeziona la scena."""
        import threading
        evt = threading.Event()

        def _ask():
            try:
                input(prompt)
            finally:
                evt.set()

        th = threading.Thread(target=_ask, daemon=True)
        th.start()
        while not evt.is_set():
            try:
                self._pb.stepSimulation()
            except Exception:
                break
            time.sleep(0.02)

    def close(self) -> None:
        try:
            self._pb.disconnect()
        except Exception:
            pass
