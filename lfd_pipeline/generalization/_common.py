"""Helper condivisi per i 3 script di generalization (pick / place / pour).

Espone:
    - factory dei modelli (BC / DMP / GMM-GMR)
    - acquisizione di start manuali in gravity-comp
    - acquisizione di N pose ArUco con LiveAruco
    - salvataggio CSV delle traiettorie generate
    - utility I/O (inputs.json, metadata.json)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parent  # .../lfd_pipeline
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config.robot_config import (  # noqa: E402
    ARUCO_DICT_NAME,
    ARUCO_MARKER_LEN_M,
    EXEC_DEFAULT_ACC,
    EXEC_DEFAULT_BLEND_RADIUS,
    EXEC_DEFAULT_SPEED,
    GRIPPER_OPEN_POS,
    HOME_JOINT_DEG,
    REALSENSE_FPS,
    REALSENSE_HEIGHT,
    REALSENSE_WIDTH,
    ROBOT_IP,
    T_EE_CAM,
    TRAINED_MODELS_ROOT,
)
from inference.robot.arm_io import (  # noqa: E402, F401
    go_home,
    init_robot,
    open_gripper,
    shutdown,
)
from inference.vision.live_aruco import LiveAruco  # noqa: E402

METHODS_ALL = ("bc", "dmp", "gmm")
GENERALIZATION_ROOT = _PKG_ROOT / "generalization_results"


# ---------------------------------------------------------------------------
# Generator factory
# ---------------------------------------------------------------------------
def default_model_path(task: str, method: str) -> Path:
    ext = "pt" if method == "bc" else "npz"
    return TRAINED_MODELS_ROOT / f"{task}_{method}.{ext}"


def make_generator(method: str, model_path: Path, device: str):
    if method == "bc":
        from inference.methods.bc import BCGenerator
        return BCGenerator(model_path, device=device)
    if method == "dmp":
        from inference.methods.dmp import DMPGenerator
        return DMPGenerator(model_path)
    if method == "gmm":
        from inference.methods.gmm_gmr import GMMGMRGenerator
        return GMMGMRGenerator(model_path)
    raise ValueError(f"Metodo non supportato: {method!r} "
                     f"(validi: {METHODS_ALL}).")


def resolve_model_path(args: argparse.Namespace, method: str, task: str) -> Path:
    override = getattr(args, f"{method}_model", None)
    return override if override is not None else default_model_path(task, method)


def load_generators(args: argparse.Namespace, task: str) -> dict:
    """Carica i modelli richiesti; salta con warning quelli mancanti."""
    generators: dict[str, object] = {}
    for method in args.methods:
        path = resolve_model_path(args, method, task)
        if not Path(path).is_file():
            print(f"[gen] [warn] modello {method.upper()} non trovato: "
                  f"{path} -- skip metodo.")
            continue
        try:
            print(f"[gen] carico {method.upper()} : {path}")
            generators[method] = make_generator(method, path, args.device)
        except Exception as e:
            print(f"[gen] [warn] {method.upper()} non caricato: {e} -- skip.")
            continue
    if not generators:
        raise SystemExit("[gen] nessun modello disponibile; uscita.")
    return generators


# ---------------------------------------------------------------------------
# Robot helpers
# ---------------------------------------------------------------------------
def get_current_xyzrpy(arm) -> list[float]:
    code, pose = arm.get_position(is_radian=True)
    if code != 0 or pose is None:
        raise SystemExit(f"[gen] impossibile leggere la posa del TCP (code={code}).")
    x_mm, y_mm, z_mm, roll, pitch, yaw = pose[:6]
    return [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0,
            float(roll), float(pitch), float(yaw)]


def acquire_starts_manual(arm, n_starts: int, task: str) -> list[list[float]]:
    """Acquisisce N pose di partenza in gravity-comp. Ritorna una lista
    di [x,y,z,roll,pitch,yaw] (m, rad)."""
    print(f"\n[{task}] acquisizione di {n_starts} pose di partenza in modalita' "
          "manuale (gravity-compensation).")
    try:
        arm.clean_warn()
    except Exception:
        pass
    arm.set_mode(2)
    arm.set_state(0)
    time.sleep(0.5)
    starts = []
    try:
        for i in range(n_starts):
            input(f"[{task}] muovi MANUALMENTE il braccio al "
                  f"start #{i + 1}/{n_starts}, INVIO per acquisire... ")
            s = get_current_xyzrpy(arm)
            print(f"  start#{i + 1}: xyz={[round(v, 4) for v in s[:3]]} "
                  f"rpy={[round(v, 4) for v in s[3:]]}")
            starts.append(s)
    finally:
        print(f"[{task}] ritorno in modalita' position.")
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.5)
    return starts


def acquire_tags(arm, marker_id: int, n_tags: int, marker_len: float,
                 marker_timeout: float, n_stable: int,
                 task: str) -> list[np.ndarray]:
    """Acquisisce N pose del marker ArUco con id ``marker_id``. Ritorna una
    lista di matrici 4x4 (T_base_tag)."""
    print(f"\n[{task}] acquisizione di {n_tags} pose ArUco (id={marker_id}).")
    live = LiveAruco(
        arm,
        T_ee_cam=T_EE_CAM,
        marker_len_m=marker_len,
        dict_name=ARUCO_DICT_NAME,
        width=REALSENSE_WIDTH,
        height=REALSENSE_HEIGHT,
        fps=REALSENSE_FPS,
    )
    live.start()
    tags: list[np.ndarray] = []
    try:
        for j in range(n_tags):
            input(f"[{task}] inquadra il marker (posa #{j + 1}/{n_tags}), "
                  "INVIO per acquisire... ")
            T = live.wait_for_marker(
                marker_id,
                timeout_s=marker_timeout,
                n_stable=n_stable,
            )
            if T is None:
                raise SystemExit(
                    f"[{task}] marker id={marker_id} non rilevato (posa #{j + 1}).")
            print(f"  tag#{j + 1} center (m): {T[:3, 3].round(4).tolist()}")
            tags.append(T.copy())
    finally:
        live.stop()
    return tags


# ---------------------------------------------------------------------------
# I/O traiettorie generate + metadati
# ---------------------------------------------------------------------------
def save_generated_csv(path: Path, traj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "z", "qx", "qy", "qz", "qw", "gripper"])
        for i in range(traj.n):
            w.writerow([
                f"{traj.t[i]:.6f}",
                f"{traj.xyz_m[i, 0]:.6f}",
                f"{traj.xyz_m[i, 1]:.6f}",
                f"{traj.xyz_m[i, 2]:.6f}",
                f"{traj.quat[i, 0]:.8f}",
                f"{traj.quat[i, 1]:.8f}",
                f"{traj.quat[i, 2]:.8f}",
                f"{traj.quat[i, 3]:.8f}",
                f"{(traj.grip[i] if traj.grip is not None else float('nan')):.6f}",
            ])


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def add_common_cli(p: argparse.ArgumentParser, *, default_marker_id: int) -> None:
    p.add_argument("--robot-ip", default=ROBOT_IP,
                   help=f"IP xArm (default: {ROBOT_IP}).")
    p.add_argument("--device", default="cpu",
                   help="Device torch per BC (default: cpu).")
    p.add_argument("--methods", nargs="+", choices=METHODS_ALL,
                   default=list(METHODS_ALL),
                   help=f"Metodi da valutare (default: {' '.join(METHODS_ALL)}).")
    p.add_argument("--bc-model", type=Path, default=None,
                   help="Override path modello BC.")
    p.add_argument("--dmp-model", type=Path, default=None,
                   help="Override path modello DMP.")
    p.add_argument("--gmm-model", type=Path, default=None,
                   help="Override path modello GMM.")
    p.add_argument("--marker-id", type=int, default=default_marker_id,
                   help=f"ID ArUco (default: {default_marker_id}).")
    p.add_argument("--marker-len", type=float, default=ARUCO_MARKER_LEN_M,
                   help=f"Lato fisico del marker [m] (default: {ARUCO_MARKER_LEN_M}).")
    p.add_argument("--marker-timeout", type=float, default=30.0,
                   help="Timeout di rilevamento marker [s] (default: 30).")
    p.add_argument("--n-stable", type=int, default=5,
                   help="N. di frame consecutivi su cui mediare la posa del marker.")
    p.add_argument("--duration-scale", type=float, default=1.0,
                   help="Riscala la durata della traiettoria (1.0 = T_mean del modello).")
    p.add_argument("--out-root", type=Path, default=GENERALIZATION_ROOT,
                   help=f"Cartella output (default: {GENERALIZATION_ROOT}).")
