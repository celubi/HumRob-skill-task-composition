"""Pipeline di inferenza GMM-GMR per la primitiva PLACE.

Differenze rispetto a ``pick_gmm.py``:
    - NON si torna a HOME: l'utente posiziona manualmente il braccio nella
      posa iniziale del place (es. punto sopra l'oggetto gia' afferrato).
      La posa corrente del TCP viene letta dal controller e usata come
      ``start_xyzrpy`` del generatore.
    - Il gripper NON viene aperto all'inizio (sta gia' tenendo l'oggetto).
    - Al termine della traiettoria si APRE il gripper per rilasciare
      l'oggetto (default: --open-after-traj True).
    - Il marker ArUco indica il punto in cui depositare l'oggetto; la
      posa di goal e' calcolata con la stessa convenzione di pick
      (gripper z opposto al z del tag, x allineati).

Uso:
    python3 place_gmm.py
    python3 place_gmm.py --marker-id 8 --grasp-offset-z 0.0
    python3 place_gmm.py --no-execute     # solo generazione + preview
"""

from __future__ import annotations

import argparse
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
    shutdown,
)
from inference.vision.live_aruco import LiveAruco  # noqa: E402
from inference.vision.tag_to_grasp import grasp_pose_from_tag  # noqa: E402


# ---------------------------------------------------------------------------
# helper: legge la posa corrente del TCP dal controller xArm.
# ---------------------------------------------------------------------------
def get_current_xyzrpy(arm) -> list[float]:
    """Ritorna [x,y,z, roll,pitch,yaw] in metri/radianti (ZYX)."""
    code, pose = arm.get_position(is_radian=True)
    if code != 0 or pose is None:
        raise SystemExit(f"[place] impossibile leggere la posa del TCP (code={code}).")
    x_mm, y_mm, z_mm, roll, pitch, yaw = pose[:6]
    return [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0,
            float(roll), float(pitch), float(yaw)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inferenza GMM-GMR per PLACE su xArm 6.")
    p.add_argument("--robot-ip", default=ROBOT_IP, help=f"IP xArm (default: {ROBOT_IP}).")
    p.add_argument("--model", type=Path,
                   default=TRAINED_MODELS_ROOT / "place_gmm.npz",
                   help="Path al modello GMM (.npz).")
    p.add_argument("--marker-id", type=int, default=ARUCO_OBJECT_ID,
                   help=f"ID ArUco del punto di place (default: {ARUCO_OBJECT_ID}).")
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
    p.add_argument("--open-after-traj", type=bool, default=True,
                   help="Ignora il comando di gripper generato dal motion encoder "
                        "e apre completamente il gripper SOLO dopo che la "
                        "traiettoria GMM e' stata interamente eseguita.")
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

    print(f"[gmm-place] model     : {args.model}")
    print(f"[gmm-place] marker id : {args.marker_id}  (lato {args.marker_len*1000:.0f} mm,"
          f" dict={ARUCO_DICT_NAME})")

    # 1) connessione xArm + HOME (posa di scansione del marker).
    arm = init_robot(args.robot_ip)
    try:
        print(f"[place] HOME (deg) : {HOME_JOINT_DEG}")
        go_home(arm, HOME_JOINT_DEG)

        # 2) live preview ArUco (eye-in-hand) per acquisire il punto di place.
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
            input("[place] inquadra il marker del punto di place, "
                  "poi premi INVIO per acquisire la posa...")

            # 3) acquisizione posa marker -> place pose target (goal).
            print(f"[place] cerco marker id={args.marker_id} "
                  f"(timeout {args.marker_timeout:.0f}s, mediato su "
                  f"{args.n_stable} frame)...")
            T_base_tag = live.wait_for_marker(
                args.marker_id,
                timeout_s=args.marker_timeout,
                n_stable=args.n_stable,
            )
            if T_base_tag is None:
                raise SystemExit(f"[place] marker id={args.marker_id} non rilevato.")

            goal_xyzrpy = grasp_pose_from_tag(T_base_tag,
                                              offset_z_m=args.grasp_offset_z)
            print(f"[place] tag center (m): {T_base_tag[:3, 3].round(4).tolist()}")
            print(f"[place] place goal     : "
                  f"xyz={[round(v, 4) for v in goal_xyzrpy[:3]]}  "
                  f"rpy={[round(v, 4) for v in goal_xyzrpy[3:]]} (rad)")

            # 4) modalita' manuale (teach / gravity compensation): il
            #    braccio diventa cedevole e l'utente lo posiziona a mano
            #    nella posa iniziale del place (oggetto gia' afferrato).
            print("[place] passaggio in modalita' manuale (gravity compensation).")
            try:
                arm.clean_warn()
            except Exception:
                pass
            arm.set_mode(2)
            arm.set_state(0)
            time.sleep(0.5)

            input("[place] posiziona MANUALMENTE il braccio nella posa di partenza "
                  "del place, poi premi INVIO...")

            # 5) ritorno in modalita' position e leggo la posa CORRENTE come start.
            print("[place] ritorno in modalita' position.")
            arm.set_mode(0)
            arm.set_state(0)
            time.sleep(0.5)

            start_xyzrpy = get_current_xyzrpy(arm)
            print(f"[place] start xyz   : {[round(v, 4) for v in start_xyzrpy[:3]]}  "
                  f"rpy={[round(v, 4) for v in start_xyzrpy[3:]]} (rad)")

            # 6) generazione traiettoria GMM-GMR
            gen = GMMGMRGenerator(args.model)
            traj = gen.generate(
                start_xyzrpy=start_xyzrpy,     # parto dalla posa CORRENTE
                goal_xyzrpy=goal_xyzrpy,
                duration_scale=args.duration_scale,
            )
            print(f"[place] traj N={traj.n}  T={traj.t[-1]:.2f}s  "
                  f"dt={(traj.t[1]-traj.t[0]):.3f}s  "
                  f"grip={'si' if traj.grip is not None else 'no'}")
            print(f"[place] endpoint xyz   : {traj.xyz_m[-1].round(4).tolist()}")

            if args.save_csv is not None:
                save_trajectory_csv(args.save_csv, traj)
                print(f"[place] traiettoria salvata in {args.save_csv}")

            # 5b) preview 3D opzionale (PyBullet + URDF xArm6)
            preview = None
            if args.preview3d:
                from inference.vision.preview3d import RobotPreview3D
                print("[place] apro la preview 3D (PyBullet)...")
                print("[place] controlli camera: Ctrl+sx=ruota, "
                      "Ctrl+centrale=pan, rotella=zoom")
                # uso la posa corrente ai giunti come stato iniziale del viewer
                code, q_now_deg = arm.get_servo_angle(is_radian=False)
                q_init = (list(q_now_deg[:6]) if code == 0 and q_now_deg is not None
                          else list(HOME_JOINT_DEG))
                preview = RobotPreview3D(q_init,
                                         animate=args.preview3d_animate)
                preview.show_trajectory(traj, goal_xyzrpy=goal_xyzrpy)
                preview.wait_for_user(
                    "[3D] ispeziona la traiettoria. INVIO per procedere... ")

            # 6) esecuzione
            try:
                if args.no_execute:
                    print("[place] --no-execute attivo: salto l'esecuzione.")
                    if preview is None:
                        input("[place] INVIO per chiudere la preview...")
                else:
                    print("[place] esecuzione tra 2s ...")
                    time.sleep(2.0)
                    if args.open_after_traj:
                        print("[place] --open-after-traj: ignoro il gripper "
                              "decodificato dalla traiettoria.")
                        traj.grip = None
                    execute_trajectory(arm, traj,
                                       speed=args.exec_speed,
                                       acc=args.exec_acc,
                                       blend_radius=args.blend_radius)
                    if args.open_after_traj:
                        print(f"[place] apro il gripper a {GRIPPER_OPEN_POS} "
                              "(post-traiettoria) per rilasciare l'oggetto.")
                        arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)
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
