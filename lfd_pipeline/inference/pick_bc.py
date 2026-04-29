"""Pipeline di inferenza Goal-Conditioned BC per la primitiva PICK.

Stessa interfaccia di ``pick_dmp.py`` / ``pick_gmm.py``, ma usa
``BCGenerator`` (modello salvato da ``learning/bc/bc_train.py`` come ``.pt``).
Il rollout BC accumula errore lungo la traiettoria: il goal-blending lineare
in fase e' SEMPRE attivo per chiudere esattamente sul target del task,
coerentemente con DMP (attractor) e GMM-GMR (regression at s=1).

Uso:
    python3 pick_bc.py
    python3 pick_bc.py --marker-id 46 --grasp-offset-z 0.0
    python3 pick_bc.py --no-execute        # solo generazione + preview
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
from inference.methods.bc import BCGenerator  # noqa: E402
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
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inferenza BC per PICK su xArm 6.")
    p.add_argument("--robot-ip", default=ROBOT_IP, help=f"IP xArm (default: {ROBOT_IP}).")
    p.add_argument("--model", type=Path,
                   default=TRAINED_MODELS_ROOT / "pick_bc.pt",
                   help="Path al modello BC (.pt).")
    p.add_argument("--device", default="cpu",
                   help="Device torch per l'inferenza (cpu, cuda, ...).")
    p.add_argument("--marker-id", type=int, default=ARUCO_OBJECT_ID,
                   help=f"ID ArUco dell'oggetto (default: {ARUCO_OBJECT_ID}).")
    p.add_argument("--marker-len", type=float, default=ARUCO_MARKER_LEN_M,
                   help=f"Lato fisico del marker [m] (default: {ARUCO_MARKER_LEN_M}).")
    p.add_argument("--grasp-offset-z", type=float, default=GRASP_OFFSET_Z_M,
                   help="Offset lungo l'asse z del tag [m] (default: "
                        f"{GRASP_OFFSET_Z_M}).")
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
    p.add_argument("--preview3d-animate", action="store_true",
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

    print(f"[bc] model      : {args.model}")
    print(f"[bc] marker id  : {args.marker_id}  (lato {args.marker_len*1000:.0f} mm,"
          f" dict={ARUCO_DICT_NAME})")

    # 1) robot up + HOME (la HOME e' gia' la posa di scansione)
    arm = init_robot(args.robot_ip)
    try:
        print(f"[bc] HOME (deg) : {HOME_JOINT_DEG}")
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
            input("[bc] posiziona l'oggetto sotto la camera, "
                  "poi premi INVIO per acquisire la posa...")

            # 4) acquisizione posa marker -> grasp pose target
            print(f"[bc] cerco marker id={args.marker_id} "
                  f"(timeout {args.marker_timeout:.0f}s, mediato su "
                  f"{args.n_stable} frame)...")
            T_base_tag = live.wait_for_marker(
                args.marker_id,
                timeout_s=args.marker_timeout,
                n_stable=args.n_stable,
            )
            if T_base_tag is None:
                raise SystemExit(f"[bc] marker id={args.marker_id} non rilevato.")

            goal_xyzrpy = grasp_pose_from_tag(T_base_tag,
                                              offset_z_m=args.grasp_offset_z)
            print(f"[bc] tag center (m): {T_base_tag[:3, 3].round(4).tolist()}")
            print(f"[bc] grasp goal     : "
                  f"xyz={[round(v, 4) for v in goal_xyzrpy[:3]]}  "
                  f"rpy={[round(v, 4) for v in goal_xyzrpy[3:]]} (rad)")

            # 5) generazione traiettoria BC
            gen = BCGenerator(args.model, device=args.device)
            traj = gen.generate(
                start_xyzrpy=None,             # parto dalla HOME (stessa della demo)
                goal_xyzrpy=goal_xyzrpy,
                duration_scale=args.duration_scale,
            )
            print(f"[bc] traj N={traj.n}  T={traj.t[-1]:.2f}s  "
                  f"dt={(traj.t[1]-traj.t[0]):.3f}s  "
                  f"grip={'si' if traj.grip is not None else 'no'}")
            print(f"[bc] endpoint xyz   : {traj.xyz_m[-1].round(4).tolist()}")

            if args.save_csv is not None:
                save_trajectory_csv(args.save_csv, traj)
                print(f"[bc] traiettoria salvata in {args.save_csv}")

            # 5b) preview 3D opzionale (PyBullet + URDF xArm6)
            preview = None
            if args.preview3d:
                from inference.vision.preview3d import RobotPreview3D
                print("[bc] apro la preview 3D (PyBullet)...")
                print("[bc] controlli camera: Ctrl+sx=ruota, "
                      "Ctrl+centrale=pan, rotella=zoom")
                preview = RobotPreview3D(HOME_JOINT_DEG,
                                         animate=args.preview3d_animate)
                preview.show_trajectory(traj, goal_xyzrpy=goal_xyzrpy)
                preview.wait_for_user(
                    "[3D] ispeziona la traiettoria. INVIO per procedere... ")

            # 6) esecuzione
            try:
                if args.no_execute:
                    print("[bc] --no-execute attivo: salto l'esecuzione.")
                    if preview is None:
                        input("[bc] INVIO per chiudere la preview...")
                else:
                    print("[bc] esecuzione tra 2s ...")
                    time.sleep(2.0)
                    # Gripper: ignoriamo il segnale generato dalla policy MLP
                    # (ricostruito grossolanamente dal profilo medio delle
                    # demo) e chiudiamo SOLO al termine della traiettoria.
                    # Coerente con i flag --close-after-traj di pick_dmp/gmm.
                    print("[bc] gripper: ignoro il segnale di traiettoria, "
                          "chiudo dopo l'esecuzione.")
                    traj.grip = None
                    execute_trajectory(arm, traj,
                                       speed=args.exec_speed,
                                       acc=args.exec_acc,
                                       blend_radius=args.blend_radius)
                    print(f"[bc] chiudo il gripper a {GRIPPER_CLOSED_POS} "
                          "(post-traiettoria).")
                    arm.set_gripper_position(GRIPPER_CLOSED_POS, wait=True)
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
