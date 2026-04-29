"""Driver minimale per xArm 6: connessione, HOME, esecuzione di una Trajectory.

L'esecuzione segue lo stile di ``replay_demo.py``: streaming cartesiano con
blending e controllo del buffer comandi, con il gripper guidato dalla
colonna ``grip`` della traiettoria.
"""

from __future__ import annotations

import math
import time

import numpy as np

from xarm.wrapper import XArmAPI

from inference.methods.base import Trajectory


# ---------------------------------------------------------------------------
# rotation helpers (locali per evitare dipendenze cicliche)
# ---------------------------------------------------------------------------
def _quat_to_rpy(qx: float, qy: float, qz: float, qw: float):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0.0:
        return 0.0, 0.0, 0.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)
    sinp = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
    pitch = math.asin(sinp)
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


def _unwrap(prev, cur):
    if prev is None:
        return cur
    out = []
    for a, b in zip(prev, cur):
        d = b - a
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        out.append(a + d)
    return tuple(out)


# ---------------------------------------------------------------------------
# robot lifecycle
# ---------------------------------------------------------------------------
def init_robot(ip: str, gripper_speed: int = 2000) -> XArmAPI:
    arm = XArmAPI(ip, is_radian=True)
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.2)
    try:
        arm.set_gripper_mode(0)
        arm.set_gripper_enable(True)
        arm.set_gripper_speed(gripper_speed)
    except Exception as e:
        print(f"[WARN] init gripper: {e}")
    return arm


def go_home(arm: XArmAPI, home_joint_deg, joint_speed_deg_s: float = 30.0) -> None:
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.1)
    arm.set_servo_angle(angle=list(home_joint_deg),
                        speed=joint_speed_deg_s, wait=True, is_radian=False)
    time.sleep(0.3)


def open_gripper(arm: XArmAPI, position: int = 850) -> None:
    try:
        arm.set_gripper_position(position, wait=True)
    except Exception as e:
        print(f"[WARN] open gripper: {e}")


# ---------------------------------------------------------------------------
# Trajectory -> waypoint list (mm + rad ZYX) per xArm
# ---------------------------------------------------------------------------
def trajectory_to_waypoints(traj: Trajectory):
    pts = []
    prev_rpy = None
    for i in range(traj.n):
        x_mm = float(traj.xyz_m[i, 0]) * 1000.0
        y_mm = float(traj.xyz_m[i, 1]) * 1000.0
        z_mm = float(traj.xyz_m[i, 2]) * 1000.0
        qx, qy, qz, qw = (float(v) for v in traj.quat[i])
        rpy = _quat_to_rpy(qx, qy, qz, qw)
        rpy = _unwrap(prev_rpy, rpy)
        prev_rpy = rpy
        grip = float(traj.grip[i]) if traj.grip is not None else None
        pts.append({"pose": [x_mm, y_mm, z_mm, *rpy], "gripper": grip})
    return pts


def downsample_waypoints(pts,
                         dist_min_mm: float = 15.0,
                         ang_min_rad: float = math.radians(1.0),
                         grip_min_delta: float = 20.0):
    if not pts:
        return pts
    out = [pts[0]]
    last = pts[0]
    for p in pts[1:]:
        moved = (
            math.dist(last["pose"][:3], p["pose"][:3]) >= dist_min_mm
            or max(abs(p["pose"][3] - last["pose"][3]),
                   abs(p["pose"][4] - last["pose"][4]),
                   abs(p["pose"][5] - last["pose"][5])) >= ang_min_rad
        )
        gripped = False
        if (grip_min_delta is not None
                and last["gripper"] is not None
                and p["gripper"] is not None):
            gripped = abs(p["gripper"] - last["gripper"]) >= grip_min_delta
        if moved or gripped:
            out.append(p)
            last = p
    if out[-1] is not pts[-1]:
        out.append(pts[-1])
    return out


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def _wait_buffer_ok(arm: XArmAPI, limit: int = 40) -> None:
    while True:
        ret = arm.get_cmdnum()
        num = (ret[1] if isinstance(ret, (list, tuple)) and len(ret) > 1
               else (ret if isinstance(ret, int) else 0))
        if num <= limit:
            return
        time.sleep(0.01)


def execute_trajectory(arm: XArmAPI, traj: Trajectory,
                       speed: float, acc: float,
                       blend_radius: float = 3.0,
                       grip_threshold: float = 20.0) -> None:
    pts = downsample_waypoints(trajectory_to_waypoints(traj))
    if not pts:
        raise RuntimeError("Traiettoria vuota.")
    print(f"[exec] waypoints dopo downsampling: {len(pts)}")

    arm.motion_enable(True)
    arm.clean_error()
    arm.set_mode(0)
    arm.set_state(0)

    first = pts[0]
    x, y, z, R, P, Y = first["pose"]
    print("[exec] muovo al primo waypoint (wait=True) ...")
    arm.set_position(x=x, y=y, z=z, roll=R, pitch=P, yaw=Y,
                     speed=speed, acc=acc, radius=0.0, wait=True)
    last_grip = None
    if first["gripper"] is not None:
        g = max(0.0, min(850.0, float(first["gripper"])))
        arm.set_gripper_position(g, wait=True)
        last_grip = first["gripper"]

    print(f"[exec] streaming {len(pts) - 1} waypoints ...")
    for p in pts[1:]:
        _wait_buffer_ok(arm)
        x, y, z, R, P, Y = p["pose"]
        code = arm.set_position(x=x, y=y, z=z, roll=R, pitch=P, yaw=Y,
                                speed=speed, acc=acc, radius=blend_radius,
                                wait=False)
        if code != 0:
            print(f"[exec] warn set_position code={code}")
        if (p["gripper"] is not None and last_grip is not None
                and abs(p["gripper"] - last_grip) >= grip_threshold):
            g = max(0.0, min(850.0, float(p["gripper"])))
            arm.set_gripper_position(g, wait=False)
            last_grip = p["gripper"]

    arm.set_pause_time(0.2)
    last = pts[-1]
    x, y, z, R, P, Y = last["pose"]
    arm.set_position(x=x, y=y, z=z, roll=R, pitch=P, yaw=Y,
                     speed=speed, acc=acc, radius=0.0, wait=True)
    if last["gripper"] is not None:
        g = max(0.0, min(850.0, float(last["gripper"])))
        arm.set_gripper_position(g, wait=True)
    print("[exec] done")


def shutdown(arm: XArmAPI) -> None:
    try:
        arm.set_mode(0)
        arm.set_state(0)
    finally:
        arm.disconnect()
