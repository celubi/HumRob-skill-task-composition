"""Replay di una traiettoria registrata su xArm 6.

Legge un CSV con header ``t,x,y,z,qx,qy,qz,qw,gripper`` (formato prodotto
da ``record_demo.py``) e lo riproduce sul robot in modalità Cartesiana
con buffering / blending, dopo essere passati per la HOME definita in
``robot_config``.

Uso:
    python3 replay_demo.py --csv ../demonstrations/pick/pick_01.csv
    python3 replay_demo.py --task pick --index 1
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config.robot_config import (  # noqa: E402
    DEMO_ROOT,
    GRIPPER_OPEN_POS,
    GRIPPER_SPEED,
    HOME_JOINT_DEG,
    ROBOT_IP,
)

try:
    from xarm.wrapper import XArmAPI
except ImportError as e:
    raise SystemExit("xArm Python SDK non trovato.") from e


# ---------------------------------------------------------------------------
# CSV / math
# ---------------------------------------------------------------------------
def quat_to_rpy(qx: float, qy: float, qz: float, qw: float):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0:
        return 0.0, 0.0, 0.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    sinr = 2 * (qw * qx + qy * qz)
    cosr = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)
    sinp = max(-1.0, min(1.0, 2 * (qw * qy - qz * qx)))
    pitch = math.asin(sinp)
    siny = 2 * (qw * qz + qx * qy)
    cosy = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


def unwrap(prev, cur):
    if prev is None:
        return cur
    out = []
    for a, b in zip(prev, cur):
        d = b - a
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        out.append(a + d)
    return tuple(out)


def load_csv(path: Path):
    """Ritorna lista di dict {pose: [x,y,z,r,p,y] in mm/rad, gripper, t}."""
    pts = []
    prev_rpy = None
    with open(path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            x = float(row["x"]) * 1000.0
            y = float(row["y"]) * 1000.0
            z = float(row["z"]) * 1000.0
            rpy = quat_to_rpy(float(row["qx"]), float(row["qy"]),
                              float(row["qz"]), float(row["qw"]))
            rpy = unwrap(prev_rpy, rpy)
            prev_rpy = rpy
            grip = float(row["gripper"]) if row.get("gripper") not in (None, "") else None
            t = float(row.get("t") or 0.0)
            pts.append({"pose": [x, y, z, *rpy], "gripper": grip, "t": t})
    return pts


def dist_mm(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def ang_max(a, b):
    return max(abs(a[3] - b[3]), abs(a[4] - b[4]), abs(a[5] - b[5]))


def downsample(pts, dist_min_mm=15.0, ang_min_rad=math.radians(1.0),
               grip_min_delta=20.0):
    if not pts:
        return pts
    out = [pts[0]]
    last = pts[0]
    for p in pts[1:]:
        moved = (dist_mm(last["pose"], p["pose"]) >= dist_min_mm
                 or ang_max(last["pose"], p["pose"]) >= ang_min_rad)
        gripped = False
        if grip_min_delta is not None and last["gripper"] is not None and p["gripper"] is not None:
            gripped = abs(p["gripper"] - last["gripper"]) >= grip_min_delta
        if moved or gripped:
            out.append(p)
            last = p
    if out[-1] is not pts[-1]:
        out.append(pts[-1])
    return out


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------
def init_robot(ip: str) -> XArmAPI:
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
        arm.set_gripper_speed(GRIPPER_SPEED)
    except Exception as e:
        print(f"[WARN] init gripper: {e}")
    return arm


def go_home(arm: XArmAPI) -> None:
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.1)
    arm.set_servo_angle(angle=list(HOME_JOINT_DEG), speed=30, wait=True, is_radian=False)
    time.sleep(0.3)


def wait_buffer_ok(arm: XArmAPI, limit: int = 40) -> None:
    while True:
        ret = arm.get_cmdnum()
        num = ret[1] if isinstance(ret, (list, tuple)) and len(ret) > 1 else (ret if isinstance(ret, int) else 0)
        if num <= limit:
            return
        time.sleep(0.01)


def replay(arm: XArmAPI, pts, speed: float, acc: float, radius: float,
           grip_threshold: float) -> None:
    arm.motion_enable(True)
    arm.clean_error()
    arm.set_mode(0)
    arm.set_state(0)

    first = pts[0]
    x, y, z, R_, P_, Y_ = first["pose"]
    print("[replay] vado al primo waypoint ...")
    arm.set_position(x=x, y=y, z=z, roll=R_, pitch=P_, yaw=Y_,
                     speed=speed, acc=acc, radius=0.0, wait=True)

    last_grip = None
    if first["gripper"] is not None:
        arm.set_gripper_position(max(0.0, min(850.0, float(first["gripper"]))), wait=True)
        last_grip = first["gripper"]

    print(f"[replay] streaming {len(pts) - 1} waypoints ...")
    for p in pts[1:]:
        wait_buffer_ok(arm)
        x, y, z, R_, P_, Y_ = p["pose"]
        code = arm.set_position(x=x, y=y, z=z, roll=R_, pitch=P_, yaw=Y_,
                                speed=speed, acc=acc, radius=radius, wait=False)
        if code != 0:
            print(f"[replay] warn set_position code={code}")
        if p["gripper"] is not None and last_grip is not None:
            if abs(p["gripper"] - last_grip) >= grip_threshold:
                arm.set_gripper_position(max(0.0, min(850.0, float(p["gripper"]))), wait=False)
                last_grip = p["gripper"]

    arm.set_pause_time(0.2)
    last = pts[-1]
    x, y, z, R_, P_, Y_ = last["pose"]
    arm.set_position(x=x, y=y, z=z, roll=R_, pitch=P_, yaw=Y_,
                     speed=speed, acc=acc, radius=0.0, wait=True)
    if last["gripper"] is not None:
        arm.set_gripper_position(max(0.0, min(850.0, float(last["gripper"]))), wait=True)
    print("[replay] done")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def resolve_csv(args) -> Path:
    if args.csv:
        p = Path(args.csv).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"CSV non trovato: {p}")
        return p
    if not args.task:
        raise SystemExit("Specifica --csv oppure --task (+ --index).")
    task_dir = (args.demo_root or DEMO_ROOT) / args.task
    idx = args.index
    if idx is None:
        # ultimo file disponibile
        files = sorted(task_dir.glob(f"{args.task}_*.csv"))
        if not files:
            raise SystemExit(f"Nessuna demo trovata in {task_dir}")
        return files[-1]
    fname = f"{args.task}_{str(idx).zfill(args.zfill)}.csv"
    p = task_dir / fname
    if not p.is_file():
        raise SystemExit(f"Demo non trovata: {p}")
    return p


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Replay di una demo CSV su xArm 6.")
    ap.add_argument("--csv", help="Path al file CSV.")
    ap.add_argument("--task", help="Nome task (sottocartella di demonstrations).")
    ap.add_argument("--index", type=int, help="Indice della demo (es. 1, 2, ...).")
    ap.add_argument("--zfill", type=int, default=2, help="Zero-padding indice (default: 2).")
    ap.add_argument("--demo-root", type=Path, default=None,
                    help=f"Root demo (default: {DEMO_ROOT}).")
    ap.add_argument("--robot-ip", default=ROBOT_IP, help=f"IP xArm (default: {ROBOT_IP}).")
    ap.add_argument("--speed", type=float, default=150.0, help="Velocità Cartesiana mm/s.")
    ap.add_argument("--acc", type=float, default=1000.0, help="Accelerazione mm/s^2.")
    ap.add_argument("--radius", type=float, default=3.0, help="Raggio di blending mm.")
    ap.add_argument("--dist-min-mm", type=float, default=15.0,
                    help="Downsampling: min spostamento [mm] tra waypoint.")
    ap.add_argument("--ang-min-deg", type=float, default=1.0,
                    help="Downsampling: min variazione angolare [deg].")
    ap.add_argument("--no-downsample", action="store_true",
                    help="Disabilita il downsampling (invia ogni riga del CSV).")
    ap.add_argument("--no-gripper", action="store_true", help="Ignora la colonna gripper.")
    ap.add_argument("--grip-threshold", type=float, default=20.0,
                    help="Min variazione gripper (unità SDK) per emettere un comando.")
    ap.add_argument("--no-home", action="store_true",
                    help="Salta il movimento iniziale a HOME.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = resolve_csv(args)
    print(f"[replay] CSV: {csv_path}")

    pts = load_csv(csv_path)
    if not pts:
        raise SystemExit("CSV vuoto o non valido.")
    if args.no_gripper:
        for p in pts:
            p["gripper"] = None
    if not args.no_downsample:
        pts = downsample(pts,
                         dist_min_mm=args.dist_min_mm,
                         ang_min_rad=math.radians(args.ang_min_deg),
                         grip_min_delta=None if args.no_gripper else args.grip_threshold)
    print(f"[replay] waypoints dopo preprocessing: {len(pts)}")

    arm = init_robot(args.robot_ip)
    try:
        if not args.no_home:
            print(f"[replay] HOME (deg): {HOME_JOINT_DEG}")
            go_home(arm)
            # Apri il gripper per partire da uno stato noto.
            try:
                arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)
            except Exception:
                pass
        replay(arm, pts,
               speed=args.speed, acc=args.acc, radius=args.radius,
               grip_threshold=args.grip_threshold)
    finally:
        try:
            arm.set_mode(0)
            arm.set_state(0)
        finally:
            arm.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
