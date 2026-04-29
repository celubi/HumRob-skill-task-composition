"""Pipeline di inferenza GMM-GMR per la primitiva POUR.

Flusso:
    1. connessione xArm e ritorno a HOME (gia' adatta a osservare la scena);
    2. apertura del gripper;
    3. avvio dello stream live RealSense + detect ArUco (finestra OpenCV);
    4. attesa rilevamento del marker e calcolo della posa di pour
       (z gripper opposto a z tag, x allineati, posizione spostata di
       +20 cm lungo x_t e +15 cm lungo y_t);
    5. generazione della traiettoria con il modello GMM-GMR addestrato;
    6. esecuzione cartesiana della traiettoria sul robot.

Uso:
    python3 pour_gmm.py
    python3 pour_gmm.py --marker-id 46
    python3 pour_gmm.py --no-execute     # solo generazione + preview
"""

from __future__ import annotations

import argparse
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
    ARUCO_OBJECT_ID,
    EXEC_DEFAULT_ACC,
    EXEC_DEFAULT_BLEND_RADIUS,
    EXEC_DEFAULT_SPEED,
    GRASP_OFFSET_Z_M,
    GRIPPER_OPEN_POS,
    HOME_JOINT_DEG,
    REALSENSE_FPS,
    REALSENSE_HEIGHT,
    REALSENSE_WIDTH,
    ROBOT_IP,
    T_EE_CAM,
    TRAINED_MODELS_ROOT,
)
from inference.methods.gmm_gmr import GMMGMRGenerator  # noqa: E402
from inference.robot.arm_io import (  # noqa: E402
    execute_trajectory,
    go_home,
    init_robot,
    open_gripper,
    shutdown,
)
from inference.vision.live_aruco import LiveAruco  # noqa: E402


# Offset specifici della primitiva POUR (lungo gli assi del tag, in metri).
POUR_OFFSET_X_M = 0.20
POUR_OFFSET_Y_M = 0.15
# Offset del goal rispetto allo start lungo y_t: rompe la degenerazione
# start==goal che fa collassare il forcing DMP e mediare a zero le azioni BC.
POUR_GOAL_OFFSET_Y_M = -0.02
# Rotazione del goal attorno a +Z del tag rispetto allo start (gradi).
# Coerente con la demo: il polso parte verticale e finisce inclinato.
POUR_GOAL_ROT_Z_DEG = -120.0


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inferenza GMM-GMR per POUR su xArm 6.")
    p.add_argument("--robot-ip", default=ROBOT_IP, help=f"IP xArm (default: {ROBOT_IP}).")
    p.add_argument("--model", type=Path,
                   default=TRAINED_MODELS_ROOT / "pour_gmm.npz",
                   help="Path al modello GMM (.npz).")
    p.add_argument("--marker-id", type=int, default=ARUCO_OBJECT_ID,
                   help=f"ID ArUco del target di pour (default: {ARUCO_OBJECT_ID}).")
    p.add_argument("--marker-len", type=float, default=ARUCO_MARKER_LEN_M,
                   help=f"Lato fisico del marker [m] (default: {ARUCO_MARKER_LEN_M}).")
    p.add_argument("--pour-offset-x", type=float, default=POUR_OFFSET_X_M,
                   help=f"Offset lungo x_t [m] (default: {POUR_OFFSET_X_M}).")
    p.add_argument("--pour-offset-y", type=float, default=POUR_OFFSET_Y_M,
                   help=f"Offset lungo y_t [m] (default: {POUR_OFFSET_Y_M}).")
    p.add_argument("--grasp-offset-z", type=float, default=GRASP_OFFSET_Z_M,
                   help=f"Offset lungo z_t [m] (default: {GRASP_OFFSET_Z_M}).")
    p.add_argument("--goal-offset-y", type=float, default=POUR_GOAL_OFFSET_Y_M,
                   help=f"Offset goal-vs-start lungo y_t [m] "
                        f"(default: {POUR_GOAL_OFFSET_Y_M}).")
    p.add_argument("--goal-rot-z-deg", type=float, default=POUR_GOAL_ROT_Z_DEG,
                   help=f"Rotazione del goal attorno a +Z del tag rispetto "
                        f"allo start [deg] (default: {POUR_GOAL_ROT_Z_DEG}).")
    p.add_argument("--duration-scale", type=float, default=1.0,
                   help="Riscala la durata della traiettoria (1.0 = come demo).")
    p.add_argument("--exec-speed", type=float, default=EXEC_DEFAULT_SPEED,
                   help=f"Velocita' cartesiana mm/s (default: {EXEC_DEFAULT_SPEED}).")
    p.add_argument("--exec-acc", type=float, default=EXEC_DEFAULT_ACC,
                   help=f"Accelerazione mm/s^2 (default: {EXEC_DEFAULT_ACC}).")
    p.add_argument("--blend-radius", type=float, default=EXEC_DEFAULT_BLEND_RADIUS,
                   help=f"Raggio blending mm (default: {EXEC_DEFAULT_BLEND_RADIUS}).")
    p.add_argument("--marker-timeout", type=float, default=30.0,
                   help="Timeout di rilevamento marker [s] (default: 30).")
    p.add_argument("--n-stable", type=int, default=5,
                   help="N. di frame consecutivi su cui mediare la posa del marker.")
    p.add_argument("--no-execute", action="store_true",
                   help="Genera la traiettoria e mostra la preview, ma NON la esegue.")
    p.add_argument("--save-csv", type=Path, default=None,
                   help="Salva la traiettoria generata in un CSV (formato "
                        "t,x,y,z,qx,qy,qz,qw,gripper).")
    p.add_argument("--preview3d", type=bool, default=True,
                   help="Apri una finestra PyBullet con il modello del robot "
                        "e la traiettoria generata, prima dell'esecuzione.")
    p.add_argument("--preview3d-animate", type=bool, default=True,
                   help="Anima il robot lungo la traiettoria nella preview "
                        "3D (IK risolto da PyBullet).")
    return p.parse_args()


def save_trajectory_csv(path: Path, traj) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "z", "qx", "qy", "qz", "qw", "gripper"])
        for i in range(traj.n):
            row = [
                f"{traj.t[i]:.6f}",
                f"{traj.xyz_m[i, 0]:.6f}",
                f"{traj.xyz_m[i, 1]:.6f}",
                f"{traj.xyz_m[i, 2]:.6f}",
                f"{traj.quat[i, 0]:.8f}",
                f"{traj.quat[i, 1]:.8f}",
                f"{traj.quat[i, 2]:.8f}",
                f"{traj.quat[i, 3]:.8f}",
                f"{(traj.grip[i] if traj.grip is not None else float('nan')):.6f}",
            ]
            w.writerow(row)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"Modello non trovato: {args.model}")

    print(f"[gmm-pour] model     : {args.model}")
    print(f"[gmm-pour] marker id : {args.marker_id}  (lato {args.marker_len*1000:.0f} mm,"
          f" dict={ARUCO_DICT_NAME})")
    print(f"[gmm-pour] offset    : dx={args.pour_offset_x:+.3f} m  "
          f"dy={args.pour_offset_y:+.3f} m  dz={args.grasp_offset_z:+.3f} m "
          "(assi del tag)")
    print(f"[gmm-pour] goal delta: dy={args.goal_offset_y:+.3f} m  "
          f"rot_z={args.goal_rot_z_deg:+.1f} deg (rispetto allo start)")

    # 1) robot up + HOME (la HOME e' gia' la posa di scansione)
    arm = init_robot(args.robot_ip)
    try:
        print(f"[gmm-pour] HOME (deg) : {HOME_JOINT_DEG}")
        go_home(arm, HOME_JOINT_DEG)

        # 2) gripper aperto come stato di partenza noto
        open_gripper(arm, GRIPPER_OPEN_POS)

        # 3) live preview
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
            input("[gmm-pour] inquadra il marker del target di pour, "
                  "poi premi INVIO per acquisire la posa...")

            # 4) acquisizione posa marker -> pour pose target
            print(f"[gmm-pour] cerco marker id={args.marker_id} "
                  f"(timeout {args.marker_timeout:.0f}s, mediato su "
                  f"{args.n_stable} frame)...")
            T_base_tag = live.wait_for_marker(
                args.marker_id,
                timeout_s=args.marker_timeout,
                n_stable=args.n_stable,
            )
            if T_base_tag is None:
                raise SystemExit(f"[gmm-pour] marker id={args.marker_id} non rilevato.")

            start_xyzrpy = pour_pose_from_tag(
                T_base_tag,
                offset_x_m=args.pour_offset_x,
                offset_y_m=args.pour_offset_y,
                offset_z_m=args.grasp_offset_z,
                rot_z_deg=0.0,
            )
            goal_xyzrpy = pour_pose_from_tag(
                T_base_tag,
                offset_x_m=args.pour_offset_x,
                offset_y_m=args.pour_offset_y + args.goal_offset_y,
                offset_z_m=args.grasp_offset_z,
                rot_z_deg=args.goal_rot_z_deg,
            )
            print(f"[gmm-pour] tag center (m): {T_base_tag[:3, 3].round(4).tolist()}")
            print(f"[gmm-pour] pour start    : "
                  f"xyz={[round(v, 4) for v in start_xyzrpy[:3]]}  "
                  f"rpy={[round(v, 4) for v in start_xyzrpy[3:]]} (rad)")
            print(f"[gmm-pour] pour goal     : "
                  f"xyz={[round(v, 4) for v in goal_xyzrpy[:3]]}  "
                  f"rpy={[round(v, 4) for v in goal_xyzrpy[3:]]} (rad)")

            # 5) generazione traiettoria GMM-GMR
            # start sopra il bicchiere, goal con piccolo offset in y_t e
            # rotazione di --goal-rot-z-deg attorno a +Z del tag (coerente
            # con la demo: il polso parte dritto e finisce inclinato).
            gen = GMMGMRGenerator(args.model)
            traj = gen.generate(
                start_xyzrpy=start_xyzrpy,
                goal_xyzrpy=goal_xyzrpy,
                duration_scale=args.duration_scale,
            )
            print(f"[gmm-pour] traj N={traj.n}  T={traj.t[-1]:.2f}s  "
                  f"dt={(traj.t[1]-traj.t[0]):.3f}s  "
                  f"grip={'si' if traj.grip is not None else 'no'}")
            print(f"[gmm-pour] endpoint xyz   : {traj.xyz_m[-1].round(4).tolist()}")

            if args.save_csv is not None:
                save_trajectory_csv(args.save_csv, traj)
                print(f"[gmm-pour] traiettoria salvata in {args.save_csv}")

            # 5b) preview 3D opzionale (PyBullet + URDF xArm6)
            preview = None
            if args.preview3d:
                from inference.vision.preview3d import RobotPreview3D
                print("[gmm-pour] apro la preview 3D (PyBullet)...")
                print("[gmm-pour] controlli camera: Ctrl+sx=ruota, "
                      "Ctrl+centrale=pan, rotella=zoom")
                preview = RobotPreview3D(HOME_JOINT_DEG,
                                         animate=args.preview3d_animate)
                preview.show_trajectory(traj, goal_xyzrpy=goal_xyzrpy)
                preview.wait_for_user(
                    "[3D] ispeziona la traiettoria. INVIO per procedere... ")

            # 6) esecuzione
            try:
                if args.no_execute:
                    print("[gmm-pour] --no-execute attivo: salto l'esecuzione.")
                    if preview is None:
                        input("[gmm-pour] INVIO per chiudere la preview...")
                else:
                    print("[gmm-pour] esecuzione tra 2s ...")
                    time.sleep(2.0)
                    # La primitiva pour non afferra: ignoriamo il segnale di
                    # gripper della policy e lasciamo lo stato corrente.
                    print("[gmm-pour] gripper: ignoro il segnale di traiettoria, "
                          "nessuna azione post-esecuzione.")
                    traj.grip = None
                    execute_trajectory(arm, traj,
                                       speed=args.exec_speed,
                                       acc=args.exec_acc,
                                       blend_radius=args.blend_radius)
            finally:
                if preview is not None:
                    preview.close()
        finally:
            live.stop()
    finally:
        shutdown(arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
