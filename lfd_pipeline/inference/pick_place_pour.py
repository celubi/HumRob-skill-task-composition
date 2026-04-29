"""Pipeline combinata PICK -> LIFT -> PLACE -> POUR su xArm 6.

Flusso:
    1. HOME (posa di scansione) e gripper aperto.
    2. Acquisizione di DUE marker:
       - pick (default id=2) -> posa di pick (grasp_pose_from_tag).
       - place (default id=3) -> posa di place (offset come pour scripts).
    3. Calcolo pour goal = place + offset_y / rotazione attorno a +Z del tag.
    4. Esecuzione sequenziale, con generazione "on the fly" delle traiettorie:
       posizioni di partenza di place e pour sono lette dal robot DOPO
       l'esecuzione della fase precedente (cosi' si vede l'effetto di un
       modello sull'altro).
    5. Lift = traslazione cartesiana lineare di +5 cm lungo +Z del WORLD,
       eseguita dopo il pick per staccare l'oggetto da terra.
    6. Gripper: aperto in HOME, chiuso post-pick, chiuso fino a fine pour.

Per ogni fase si puo' scegliere il metodo (BC / DMP / GMM-GMR) e il path al
modello via flag CLI. Default: ``dmp`` per tutte e tre, modello in
``trained_models/<primitive>_dmp.npz``.

Uso:
    python3 pick_place_pour.py
    python3 pick_place_pour.py --pick-method dmp --place-method gmm --pour-method bc
    python3 pick_place_pour.py --no-execute        # solo generazione + preview
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
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
    DEFAULT_RECORD_RATE_HZ,
    EXEC_DEFAULT_ACC,
    EXEC_DEFAULT_BLEND_RADIUS,
    EXEC_DEFAULT_SPEED,
    GRIPPER_CLOSED_POS,
    GRIPPER_OPEN_POS,
    HOME_JOINT_DEG,
    REALSENSE_FPS,
    REALSENSE_HEIGHT,
    REALSENSE_WIDTH,
    ROBOT_IP,
    T_EE_CAM,
    TRAINED_MODELS_ROOT,
)
from inference.robot.arm_io import (  # noqa: E402
    execute_trajectory,
    go_home,
    init_robot,
    open_gripper,
    shutdown,
)
from inference.vision.live_aruco import LiveAruco  # noqa: E402
from inference.vision.tag_to_grasp import grasp_pose_from_tag  # noqa: E402


# ---------------------------------------------------------------------------
# Costanti del task combinato
# ---------------------------------------------------------------------------
PICK_MARKER_ID_DEFAULT = 3
PLACE_MARKER_ID_DEFAULT = 2

# Offset z dell'azione di PICK (lungo z del pick tag, metri).
PICK_OFFSET_Z_M = -0.02

# Offset del gripper rispetto al place tag (assi del tag, in metri).
# x_t / y_t / z_t definiscono il punto sopra il bicchiere.
PLACE_OFFSET_X_M = 0.20
PLACE_OFFSET_Y_M = 0.08
PLACE_OFFSET_Z_M = -0.06

# Pour goal vs place: offset y_t e rotazione attorno a z_t del place tag.
POUR_GOAL_OFFSET_Y_M = +0.02
POUR_GOAL_ROT_Z_DEG = -120.0
# Offset z dell'azione di POUR (lungo z del place tag, metri).
POUR_OFFSET_Z_M = -0.06

# Sollevamento post-pick lungo +Z del WORLD.
LIFT_Z_M = 0.05

# Dove salvare le registrazioni di esecuzione (planned + actual + metadata).
RECORDED_EXECUTIONS_ROOT = _PKG_ROOT / "recorded_executions"


# ---------------------------------------------------------------------------
# Helper geometria
# ---------------------------------------------------------------------------
def pour_pose_from_tag(T_base_tag: np.ndarray,
                       offset_x_m: float,
                       offset_y_m: float,
                       offset_z_m: float,
                       rot_z_deg: float = 0.0) -> list[float]:
    """Posa di pour. Orientamento base come grasp_pose_from_tag
    (z_g = -z_t, x_g = +x_t), eventualmente ruotato di ``rot_z_deg``
    attorno a +Z del tag (in frame world: R_axis · R_g). Posizione =
    centro del tag traslato lungo x_t, y_t, z_t."""
    R_t = np.asarray(T_base_tag, float)[:3, :3]
    p_t = np.asarray(T_base_tag, float)[:3, 3]
    x_t = R_t[:, 0]
    y_t = R_t[:, 1]
    z_t = R_t[:, 2]

    # orientamento base (stesso di grasp_pose_from_tag)
    x_g = x_t / np.linalg.norm(x_t)
    z_g = -z_t / np.linalg.norm(z_t)
    x_g = x_g - np.dot(x_g, z_g) * z_g
    x_g = x_g / np.linalg.norm(x_g)
    y_g = np.cross(z_g, x_g)
    R_g = np.column_stack([x_g, y_g, z_g])

    # rotazione attorno a +Z del tag (Rodrigues, world frame)
    if abs(rot_z_deg) > 1e-9:
        theta = float(np.deg2rad(rot_z_deg))
        k = z_t / np.linalg.norm(z_t)
        K = np.array([[0.0, -k[2], k[1]],
                      [k[2], 0.0, -k[0]],
                      [-k[1], k[0], 0.0]], float)
        R_axis = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
        R_g = R_axis @ R_g

    p_g = p_t + offset_x_m * x_t + offset_y_m * y_t + offset_z_m * z_t

    # R -> rpy ZYX (stessa convenzione di tag_to_grasp._R_to_rpy_zyx)
    sp = float(np.clip(-R_g[2, 0], -1.0, 1.0))
    pitch = float(np.arcsin(sp))
    if abs(np.cos(pitch)) > 1e-8:
        roll = float(np.arctan2(R_g[2, 1], R_g[2, 2]))
        yaw = float(np.arctan2(R_g[1, 0], R_g[0, 0]))
    else:
        roll = 0.0
        yaw = float(np.arctan2(-R_g[0, 1], R_g[1, 1]))
    return [float(p_g[0]), float(p_g[1]), float(p_g[2]), roll, pitch, yaw]


def get_current_xyzrpy(arm) -> list[float]:
    code, pose = arm.get_position(is_radian=True)
    if code != 0 or pose is None:
        raise SystemExit(f"[combined] impossibile leggere la posa del TCP (code={code}).")
    x_mm, y_mm, z_mm, roll, pitch, yaw = pose[:6]
    return [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0,
            float(roll), float(pitch), float(yaw)]


# ---------------------------------------------------------------------------
# Generator factory
# ---------------------------------------------------------------------------
METHODS = ("bc", "dmp", "gmm")


def default_model_path(primitive: str, method: str) -> Path:
    ext = "pt" if method == "bc" else "npz"
    return TRAINED_MODELS_ROOT / f"{primitive}_{method}.{ext}"


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
    raise ValueError(f"Metodo non supportato: {method!r} (validi: {METHODS}).")


# ---------------------------------------------------------------------------
# Preview helper (per-fase, con istanza PyBullet usa-e-getta)
# ---------------------------------------------------------------------------
def show_preview(arm, traj, goal_xyzrpy, animate: bool, label: str) -> None:
    from inference.vision.preview3d import RobotPreview3D
    print(f"[{label}] apro la preview 3D (PyBullet)...")
    code, q_now_deg = arm.get_servo_angle(is_radian=False)
    q_init = (list(q_now_deg[:6]) if code == 0 and q_now_deg is not None
              else list(HOME_JOINT_DEG))
    preview = RobotPreview3D(q_init, animate=animate)
    try:
        preview.show_trajectory(traj, goal_xyzrpy=goal_xyzrpy)
        preview.wait_for_user(f"[{label}] INVIO per procedere... ")
    finally:
        preview.close()


def fmt_pose(p: list[float]) -> str:
    return (f"xyz={[round(v, 4) for v in p[:3]]}  "
            f"rpy={[round(v, 4) for v in p[3:]]}")


# ---------------------------------------------------------------------------
# Registrazione esecuzioni (planned + actual)
# ---------------------------------------------------------------------------
def _rpy_zyx_to_quat(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return qx, qy, qz, qw


def save_trajectory_csv(path: Path, traj) -> None:
    """Salva la traiettoria PIANIFICATA dal modello (input dell'execute)."""
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


class ExecutionRecorder:
    """Registratore in background della posa TCP + gripper a frequenza fissa.

    Stesso schema di ``recording.record_demo.DemoRecorder``, ridotto a
    begin/end_and_save (niente toggle/discard). Lo stesso thread serve piu'
    sessioni in sequenza (pick, place, pour).
    """

    def __init__(self, arm, rate_hz: float):
        self.arm = arm
        self.dt = 1.0 / float(rate_hz)
        self._recording = threading.Event()
        self._stop = threading.Event()
        self._buffer: list[tuple] = []
        self._t0: float | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def begin(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._t0 = None
        self._recording.set()

    def end_and_save(self, csv_path: Path) -> int:
        self._recording.clear()
        with self._lock:
            samples = list(self._buffer)
            self._buffer.clear()
            self._t0 = None
        if not samples:
            return 0
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["t", "x", "y", "z", "qx", "qy", "qz", "qw", "gripper"])
            for row in samples:
                w.writerow(row)
        return len(samples)

    # --- internal ---
    def _read_pose(self):
        code, pose = self.arm.get_position(is_radian=True)
        if code != 0 or pose is None:
            return None
        x_mm, y_mm, z_mm, roll, pitch, yaw = pose[:6]
        qx, qy, qz, qw = _rpy_zyx_to_quat(roll, pitch, yaw)
        return (x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0, qx, qy, qz, qw)

    def _read_gripper(self) -> float:
        try:
            code, pos = self.arm.get_gripper_position()
            if code == 0 and pos is not None:
                return float(pos)
        except Exception:
            pass
        return float("nan")

    def _loop(self) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(min(self.dt, next_t - now))
                continue
            next_t += self.dt
            if not self._recording.is_set():
                if now - next_t > 1.0:
                    next_t = now + self.dt
                continue
            pose = self._read_pose()
            if pose is None:
                continue
            grip = self._read_gripper()
            with self._lock:
                if self._t0 is None:
                    self._t0 = now
                t_rel = now - self._t0
                self._buffer.append((
                    f"{t_rel:.9f}",
                    f"{pose[0]:.6f}", f"{pose[1]:.6f}", f"{pose[2]:.6f}",
                    f"{pose[3]:.8f}", f"{pose[4]:.8f}",
                    f"{pose[5]:.8f}", f"{pose[6]:.8f}",
                    f"{grip:.6f}",
                ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline combinata PICK->LIFT->PLACE->POUR.")
    p.add_argument("--robot-ip", default=ROBOT_IP, help=f"IP xArm (default: {ROBOT_IP}).")
    p.add_argument("--device", default="cpu",
                   help="Device torch (BC). Default: cpu.")

    # Modelli per fase
    p.add_argument("--pick-method", choices=METHODS, default="gmm",
                   help="Metodo per pick (default: gmm).")
    p.add_argument("--place-method", choices=METHODS, default="gmm",
                   help="Metodo per place (default: gmm).")
    p.add_argument("--pour-method", choices=METHODS, default="gmm",
                   help="Metodo per pour (default: gmm).")
    p.add_argument("--pick-model", type=Path, default=None,
                   help="Override path modello pick. "
                        "Default: trained_models/pick_<method>.{pt|npz}.")
    p.add_argument("--place-model", type=Path, default=None,
                   help="Override path modello place.")
    p.add_argument("--pour-model", type=Path, default=None,
                   help="Override path modello pour.")

    # Vision / marker
    p.add_argument("--marker-len", type=float, default=ARUCO_MARKER_LEN_M,
                   help=f"Lato fisico del marker [m] (default: {ARUCO_MARKER_LEN_M}).")
    p.add_argument("--pick-marker-id", type=int, default=PICK_MARKER_ID_DEFAULT,
                   help=f"ID ArUco del pick (default: {PICK_MARKER_ID_DEFAULT}).")
    p.add_argument("--place-marker-id", type=int, default=PLACE_MARKER_ID_DEFAULT,
                   help=f"ID ArUco del place (default: {PLACE_MARKER_ID_DEFAULT}).")
    p.add_argument("--marker-timeout", type=float, default=30.0,
                   help="Timeout di rilevamento marker [s] (default: 30).")
    p.add_argument("--n-stable", type=int, default=5,
                   help="N. di frame consecutivi su cui mediare la posa del marker.")

    # Offset z per fase (separati per chiarezza: pick / place / pour)
    p.add_argument("--pick-offset-z", type=float, default=PICK_OFFSET_Z_M,
                   help=f"Offset PICK lungo z del pick tag [m] "
                        f"(default: {PICK_OFFSET_Z_M}).")
    p.add_argument("--place-offset-z", type=float, default=PLACE_OFFSET_Z_M,
                   help=f"Offset PLACE lungo z del place tag [m] "
                        f"(default: {PLACE_OFFSET_Z_M}).")
    p.add_argument("--pour-offset-z", type=float, default=POUR_OFFSET_Z_M,
                   help=f"Offset POUR lungo z del place tag [m] "
                        f"(default: {POUR_OFFSET_Z_M}).")

    # Geometria place / pour goal (assi del place tag)
    p.add_argument("--place-offset-x", type=float, default=PLACE_OFFSET_X_M,
                   help=f"Offset gripper-place lungo x_t [m] (default: {PLACE_OFFSET_X_M}).")
    p.add_argument("--place-offset-y", type=float, default=PLACE_OFFSET_Y_M,
                   help=f"Offset gripper-place lungo y_t [m] (default: {PLACE_OFFSET_Y_M}).")
    p.add_argument("--pour-goal-offset-y", type=float, default=POUR_GOAL_OFFSET_Y_M,
                   help=f"Offset pour-goal vs place lungo y_t [m] "
                        f"(default: {POUR_GOAL_OFFSET_Y_M}).")
    p.add_argument("--pour-goal-rot-z-deg", type=float, default=POUR_GOAL_ROT_Z_DEG,
                   help=f"Rotazione pour-goal vs place attorno a +Z del tag "
                        f"[deg] (default: {POUR_GOAL_ROT_Z_DEG}).")
    p.add_argument("--lift-z-m", type=float, default=LIFT_Z_M,
                   help=f"Sollevamento post-pick lungo +Z world [m] "
                        f"(default: {LIFT_Z_M}).")

    # Esecuzione
    p.add_argument("--duration-scale", type=float, default=1.0,
                   help="Riscala la durata di TUTTE le traiettorie (1.0 = come demo).")
    p.add_argument("--exec-speed", type=float, default=EXEC_DEFAULT_SPEED,
                   help=f"Velocita' cartesiana mm/s (default: {EXEC_DEFAULT_SPEED}).")
    p.add_argument("--exec-acc", type=float, default=EXEC_DEFAULT_ACC,
                   help=f"Accelerazione mm/s^2 (default: {EXEC_DEFAULT_ACC}).")
    p.add_argument("--blend-radius", type=float, default=EXEC_DEFAULT_BLEND_RADIUS,
                   help=f"Raggio blending mm (default: {EXEC_DEFAULT_BLEND_RADIUS}).")
    p.add_argument("--no-execute", action="store_true",
                   help="Genera le traiettorie e mostra la preview, ma NON esegue.")
    p.add_argument("--preview3d", type=bool, default=True,
                   help="Apri una preview 3D PyBullet prima di ogni fase.")
    p.add_argument("--preview3d-animate", type=bool, default=False,
                   help="Anima il robot lungo la traiettoria nella preview 3D.")

    # Registrazione esecuzioni (default: ON)
    p.add_argument("--no-record", action="store_true",
                   help="Disattiva la registrazione delle esecuzioni "
                        "(default: registrazione attiva).")
    p.add_argument("--record-dir", type=Path, default=RECORDED_EXECUTIONS_ROOT,
                   help=f"Cartella radice dove salvare le registrazioni "
                        f"(default: {RECORDED_EXECUTIONS_ROOT}).")
    p.add_argument("--record-rate", type=float, default=DEFAULT_RECORD_RATE_HZ,
                   help=f"Sampling rate per la registrazione actual [Hz] "
                        f"(default: {DEFAULT_RECORD_RATE_HZ}).")

    return p.parse_args()


def resolve_model_paths(args: argparse.Namespace) -> None:
    if args.pick_model is None:
        args.pick_model = default_model_path("pick", args.pick_method)
    if args.place_model is None:
        args.place_model = default_model_path("place", args.place_method)
    if args.pour_model is None:
        args.pour_model = default_model_path("pour", args.pour_method)
    for prim, path in (("pick", args.pick_model),
                       ("place", args.place_model),
                       ("pour", args.pour_model)):
        if not Path(path).is_file():
            raise SystemExit(f"Modello {prim} non trovato: {path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    resolve_model_paths(args)

    print("[combined] METODI / MODELLI")
    print(f"  pick  : {args.pick_method:>3s}  ->  {args.pick_model}")
    print(f"  place : {args.place_method:>3s}  ->  {args.place_model}")
    print(f"  pour  : {args.pour_method:>3s}  ->  {args.pour_model}")
    print(f"[combined] marker pick={args.pick_marker_id}, "
          f"place={args.place_marker_id}, lato {args.marker_len*1000:.0f} mm")
    print(f"[combined] pick  offset z (pick tag z) : {args.pick_offset_z:+.3f} m")
    print(f"[combined] place offset (place tag)    : "
          f"dx={args.place_offset_x:+.3f} dy={args.place_offset_y:+.3f} "
          f"dz={args.place_offset_z:+.3f} m")
    print(f"[combined] pour goal vs place          : "
          f"dy={args.pour_goal_offset_y:+.3f} m  "
          f"rot_z={args.pour_goal_rot_z_deg:+.1f} deg  "
          f"dz={args.pour_offset_z:+.3f} m (tag axes)")
    print(f"[combined] lift world +Z               : {args.lift_z_m*1000:+.0f} mm")

    # Setup registrazione (default ON, disattivabile con --no-record)
    record = (not args.no_record) and (not args.no_execute)
    run_dir: Path | None = None
    if record:
        ts = time.strftime("%Y%m%d-%H%M%S")
        run_dir = Path(args.record_dir) / f"run_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[combined] registrazione attiva -> {run_dir}")
    elif args.no_record:
        print("[combined] registrazione DISATTIVATA (--no-record).")
    elif args.no_execute:
        print("[combined] --no-execute: registrazione actual non disponibile "
              "(salvero' solo le traiettorie planned se la run dir e' fornita).")

    arm = init_robot(args.robot_ip)
    recorder: ExecutionRecorder | None = None
    try:
        # 1) HOME + gripper aperto
        print(f"\n[combined] HOME (deg): {HOME_JOINT_DEG}")
        go_home(arm, HOME_JOINT_DEG)
        open_gripper(arm, GRIPPER_OPEN_POS)

        # 2) acquisizione marker pick e place
        live = LiveAruco(
            arm,
            T_ee_cam=T_EE_CAM,
            marker_len_m=args.marker_len,
            dict_name=ARUCO_DICT_NAME,
            width=REALSENSE_WIDTH,
            height=REALSENSE_HEIGHT,
            fps=REALSENSE_FPS,
        )
        live.start()
        try:
            input(f"[combined] inquadra il marker di PICK (id={args.pick_marker_id}), "
                  "INVIO per acquisire...")
            print(f"[combined] cerco marker pick id={args.pick_marker_id} "
                  f"(timeout {args.marker_timeout:.0f}s)...")
            T_pick_tag = live.wait_for_marker(
                args.pick_marker_id,
                timeout_s=args.marker_timeout,
                n_stable=args.n_stable,
            )
            if T_pick_tag is None:
                raise SystemExit(f"[combined] marker pick id={args.pick_marker_id} non rilevato.")
            print(f"[combined] pick tag (m): {T_pick_tag[:3, 3].round(4).tolist()}")

            input(f"[combined] inquadra il marker di PLACE (id={args.place_marker_id}), "
                  "INVIO per acquisire...")
            print(f"[combined] cerco marker place id={args.place_marker_id} "
                  f"(timeout {args.marker_timeout:.0f}s)...")
            T_place_tag = live.wait_for_marker(
                args.place_marker_id,
                timeout_s=args.marker_timeout,
                n_stable=args.n_stable,
            )
            if T_place_tag is None:
                raise SystemExit(f"[combined] marker place id={args.place_marker_id} non rilevato.")
            print(f"[combined] place tag (m): {T_place_tag[:3, 3].round(4).tolist()}")
        finally:
            live.stop()

        # 3) target poses (place_goal e pour_goal sono fissi; le START arrivano "live")
        pick_goal = grasp_pose_from_tag(T_pick_tag, offset_z_m=args.pick_offset_z)
        place_goal = pour_pose_from_tag(
            T_place_tag,
            offset_x_m=args.place_offset_x,
            offset_y_m=args.place_offset_y,
            offset_z_m=args.place_offset_z,
            rot_z_deg=0.0,
        )
        pour_goal = pour_pose_from_tag(
            T_place_tag,
            offset_x_m=args.place_offset_x,
            offset_y_m=args.place_offset_y + args.pour_goal_offset_y,
            offset_z_m=args.pour_offset_z,
            rot_z_deg=args.pour_goal_rot_z_deg,
        )
        print(f"\n[combined] pick  goal : {fmt_pose(pick_goal)}")
        print(f"[combined] place goal : {fmt_pose(place_goal)}")
        print(f"[combined] pour  goal : {fmt_pose(pour_goal)}")

        # 4) generatori
        pick_gen = make_generator(args.pick_method, args.pick_model, args.device)
        place_gen = make_generator(args.place_method, args.place_model, args.device)
        pour_gen = make_generator(args.pour_method, args.pour_model, args.device)

        # 4b) recorder (se attivo)
        if record:
            recorder = ExecutionRecorder(arm, rate_hz=args.record_rate)
            recorder.start()

        # =================================================================
        # PICK
        # =================================================================
        print("\n=== PICK ===")
        traj_pick = pick_gen.generate(
            start_xyzrpy=None,           # parto da HOME (mean/y0 del modello)
            goal_xyzrpy=pick_goal,
            duration_scale=args.duration_scale,
        )
        print(f"[pick] traj N={traj_pick.n}  T={traj_pick.t[-1]:.2f}s  "
              f"endpoint xyz={traj_pick.xyz_m[-1].round(4).tolist()}")
        if args.preview3d:
            show_preview(arm, traj_pick, pick_goal, args.preview3d_animate, "pick")

        if args.no_execute:
            print("[combined] --no-execute: salto pick.")
        else:
            print("[combined] esecuzione pick tra 2s...")
            time.sleep(2.0)
            traj_pick.grip = None
            if recorder is not None:
                recorder.begin()
            execute_trajectory(arm, traj_pick,
                               speed=args.exec_speed, acc=args.exec_acc,
                               blend_radius=args.blend_radius)
            print(f"[pick] chiudo il gripper a {GRIPPER_CLOSED_POS}.")
            arm.set_gripper_position(GRIPPER_CLOSED_POS, wait=True)
            if recorder is not None and run_dir is not None:
                save_trajectory_csv(run_dir / "pick_planned.csv", traj_pick)
                n = recorder.end_and_save(run_dir / "pick_actual.csv")
                print(f"[pick] registrazione: {n} campioni -> "
                      f"{run_dir / 'pick_actual.csv'}")

        # =================================================================
        # LIFT (+lift_z_m lungo +Z del WORLD, cartesiano lineare)
        # =================================================================
        print(f"\n=== LIFT (+{args.lift_z_m*1000:.0f} mm world Z) ===")
        if args.no_execute:
            print("[combined] --no-execute: salto lift.")
        else:
            cur = get_current_xyzrpy(arm)
            x_mm = cur[0] * 1000.0
            y_mm = cur[1] * 1000.0
            z_mm_target = (cur[2] + args.lift_z_m) * 1000.0
            print(f"[lift] z corrente {cur[2]*1000:.1f} mm -> target {z_mm_target:.1f} mm")
            arm.set_position(
                x=x_mm, y=y_mm, z=z_mm_target,
                roll=cur[3], pitch=cur[4], yaw=cur[5],
                speed=args.exec_speed, acc=args.exec_acc, radius=0.0,
                wait=True,
            )

        # =================================================================
        # PLACE (start = posa CORRENTE letta dopo lift)
        # =================================================================
        print("\n=== PLACE ===")
        if args.no_execute:
            # senza esecuzione la posa "corrente" e' HOME; come stima usiamo
            # il pick goal sollevato (solo per la preview).
            place_start = list(pick_goal)
            place_start[2] += args.lift_z_m
        else:
            place_start = get_current_xyzrpy(arm)
        print(f"[place] start (live) : {fmt_pose(place_start)}")
        print(f"[place] goal         : {fmt_pose(place_goal)}")
        traj_place = place_gen.generate(
            start_xyzrpy=place_start,
            goal_xyzrpy=place_goal,
            duration_scale=args.duration_scale,
        )
        print(f"[place] traj N={traj_place.n}  T={traj_place.t[-1]:.2f}s  "
              f"endpoint xyz={traj_place.xyz_m[-1].round(4).tolist()}")
        if args.preview3d:
            show_preview(arm, traj_place, place_goal, args.preview3d_animate, "place")

        if args.no_execute:
            print("[combined] --no-execute: salto place.")
        else:
            print("[combined] esecuzione place tra 2s...")
            time.sleep(2.0)
            traj_place.grip = None
            if recorder is not None:
                recorder.begin()
            execute_trajectory(arm, traj_place,
                               speed=args.exec_speed, acc=args.exec_acc,
                               blend_radius=args.blend_radius)
            if recorder is not None and run_dir is not None:
                save_trajectory_csv(run_dir / "place_planned.csv", traj_place)
                n = recorder.end_and_save(run_dir / "place_actual.csv")
                print(f"[place] registrazione: {n} campioni -> "
                      f"{run_dir / 'place_actual.csv'}")

        # =================================================================
        # POUR (start = posa CORRENTE letta dopo place)
        # =================================================================
        print("\n=== POUR ===")
        if args.no_execute:
            pour_start = list(place_goal)
        else:
            pour_start = get_current_xyzrpy(arm)
        print(f"[pour] start (live) : {fmt_pose(pour_start)}")
        print(f"[pour] goal         : {fmt_pose(pour_goal)}")
        traj_pour = pour_gen.generate(
            start_xyzrpy=pour_start,
            goal_xyzrpy=pour_goal,
            duration_scale=args.duration_scale,
        )
        print(f"[pour] traj N={traj_pour.n}  T={traj_pour.t[-1]:.2f}s  "
              f"endpoint xyz={traj_pour.xyz_m[-1].round(4).tolist()}")
        if args.preview3d:
            show_preview(arm, traj_pour, pour_goal, args.preview3d_animate, "pour")

        if args.no_execute:
            print("[combined] --no-execute: salto pour.")
        else:
            print("[combined] esecuzione pour tra 2s...")
            time.sleep(2.0)
            traj_pour.grip = None
            if recorder is not None:
                recorder.begin()
            execute_trajectory(arm, traj_pour,
                               speed=args.exec_speed, acc=args.exec_acc,
                               blend_radius=args.blend_radius)
            if recorder is not None and run_dir is not None:
                save_trajectory_csv(run_dir / "pour_planned.csv", traj_pour)
                n = recorder.end_and_save(run_dir / "pour_actual.csv")
                print(f"[pour] registrazione: {n} campioni -> "
                      f"{run_dir / 'pour_actual.csv'}")

        # metadata della run
        if run_dir is not None:
            metadata = {
                "timestamp": run_dir.name.removeprefix("run_"),
                "methods": {
                    "pick": args.pick_method,
                    "place": args.place_method,
                    "pour": args.pour_method,
                },
                "models": {
                    "pick": str(args.pick_model),
                    "place": str(args.place_model),
                    "pour": str(args.pour_model),
                },
                "marker_ids": {
                    "pick": args.pick_marker_id,
                    "place": args.place_marker_id,
                },
                "offsets_m": {
                    "pick_z": args.pick_offset_z,
                    "place": {"x": args.place_offset_x,
                              "y": args.place_offset_y,
                              "z": args.place_offset_z},
                    "pour_goal_vs_place": {
                        "y": args.pour_goal_offset_y,
                        "z": args.pour_offset_z,
                        "rot_z_deg": args.pour_goal_rot_z_deg,
                    },
                    "lift_z_world": args.lift_z_m,
                },
                "exec": {
                    "speed_mm_s": args.exec_speed,
                    "acc_mm_s2": args.exec_acc,
                    "blend_radius_mm": args.blend_radius,
                    "duration_scale": args.duration_scale,
                    "record_rate_hz": args.record_rate,
                },
                "goals": {
                    "pick": pick_goal,
                    "place": place_goal,
                    "pour": pour_goal,
                },
            }
            with open(run_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"[combined] metadata -> {run_dir / 'metadata.json'}")

        print("\n[combined] sequenza completata. Gripper chiuso.")
    finally:
        if recorder is not None:
            recorder.shutdown()
        shutdown(arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
