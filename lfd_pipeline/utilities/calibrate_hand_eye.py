"""Hand-eye calibration eye-in-hand per xArm 6 + RealSense + ChArUco.

Due modalita':

  record:  pone il braccio in TEACH MODE (free-drive). Sposti tu il
           robot a mano, premi INVIO ad ogni posa per registrarne i
           giunti, ESC/'q' per finire. Salva un JSON con la lista di
           configurazioni di giunti.

  run:     riproduce la lista di pose (move-to-joint con wait), ad ogni
           posa cattura un frame RGB dalla RealSense, rileva un target
           ChArUco e accumula (R_target2cam, t_target2cam) e
           (R_gripper2base, t_gripper2base). Alla fine calcola la
           trasformazione T_EE_CAM (gripper -> camera color-optical) con
           cv2.calibrateHandEye e la salva in NPZ + YAML.

Convenzioni:
  - "EE" = posa restituita da xArm.position (rispetta il TCP offset
    attivo sul controller). Quindi T_EE_CAM uscira' coerente con la
    stessa convenzione usata in inference/.
  - Il target deve restare FERMO durante l'intera acquisizione.

Esempi:

  # 1) acquisisci 20 pose in free-drive
  python3 utilities/calibrate_hand_eye.py record \\
      --out utilities/calib_poses.json --n-poses 20

  # 2) esegui la calibrazione (board: 5x7 squares, 35 mm square, 26 mm marker)
  python3 utilities/calibrate_hand_eye.py run \\
      --poses utilities/calib_poses.json \\
      --squares-x 5 --squares-y 7 \\
      --square-len 0.035 --marker-len 0.026 \\
      --aruco-dict DICT_5X5_50 \\
      --out utilities/T_ee_cam.npz
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Permette gli import dalla root della pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.robot_config import (  # noqa: E402
    ROBOT_IP,
    REALSENSE_WIDTH,
    REALSENSE_HEIGHT,
    REALSENSE_FPS,
)

from xarm.wrapper import XArmAPI  # noqa: E402


# ---------------------------------------------------------------------------
# rotation helpers (RPY ZYX gradi <-> matrice)
# ---------------------------------------------------------------------------
def _rpy_deg_to_R(roll_d, pitch_d, yaw_d):
    r, p, y = (math.radians(a) for a in (roll_d, pitch_d, yaw_d))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _R_to_rpy_deg(R):
    sp = -R[2, 0]
    sp = float(np.clip(sp, -1.0, 1.0))
    pitch = math.asin(sp)
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


# ---------------------------------------------------------------------------
# robot helpers
# ---------------------------------------------------------------------------
def _connect_arm(ip: str, mode: int = 0) -> XArmAPI:
    arm = XArmAPI(ip, is_radian=False)
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(mode)
    arm.set_state(0)
    time.sleep(0.2)
    return arm


def _read_joints(arm: XArmAPI):
    j = arm.angles
    if j is None:
        return None
    return [float(v) for v in j[:6]]


def _read_pose_ee(arm: XArmAPI):
    """Posa EE (TCP) come (R_ee_base, t_ee_base) in metri."""
    p = arm.position
    if p is None:
        return None, None
    x_mm, y_mm, z_mm, roll, pitch, yaw = (float(v) for v in p[:6])
    R = _rpy_deg_to_R(roll, pitch, yaw)
    t = np.array([x_mm, y_mm, z_mm], float) / 1000.0
    return R, t


# ---------------------------------------------------------------------------
# RECORD: free-drive teach con preview RGB+depth
# ---------------------------------------------------------------------------
def cmd_record(args):
    import cv2
    import pyrealsense2 as rs

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- ChArUco opzionale per anteprima detection ---
    adict = board = None
    if args.preview_charuco:
        try:
            adict, board = _build_charuco(args)
        except SystemExit as e:
            print(f"[record] preview ChArUco disabilitata ({e})")

    # --- start RealSense (color + depth allineati) ---
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, REALSENSE_WIDTH, REALSENSE_HEIGHT,
                      rs.format.bgr8, REALSENSE_FPS)
    cfg.enable_stream(rs.stream.depth, REALSENSE_WIDTH, REALSENSE_HEIGHT,
                      rs.format.z16, REALSENSE_FPS)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    vprof = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = vprof.get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], float)
    D = np.array(intr.coeffs if intr.coeffs else [0, 0, 0, 0, 0], float)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"[record] RealSense ok  fx={intr.fx:.1f} cx={intr.ppx:.1f} "
          f"depth_scale={depth_scale}")

    # warm-up
    for _ in range(15):
        try:
            pipe.wait_for_frames(timeout_ms=1000)
        except Exception:
            pass

    arm = _connect_arm(args.ip, mode=0)
    print("[record] attivo TEACH MODE (free-drive). Muovi il braccio a mano.")
    print("         SPAZIO = registra posa | 'd' = scarta ultima | 'q'/ESC = fine")
    arm.set_mode(2)
    arm.set_state(0)
    time.sleep(0.3)

    win = "calib record (SPACE=save, d=drop, q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    poses = []
    try:
        while True:
            try:
                frames = pipe.wait_for_frames(timeout_ms=2000)
            except Exception:
                continue
            frames = align.process(frames)
            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            if not cf:
                continue
            img = np.asanyarray(cf.get_data())
            h, w = img.shape[:2]
            cx_px, cy_px = w // 2, h // 2

            annot = img.copy()

            # --- ChArUco preview (riusa _detect_charuco_pose) ---
            n_corners = 0
            if board is not None:
                try:
                    pose_t, n_corners = _detect_charuco_pose(img, K, D, adict, board)
                except Exception:
                    pose_t, n_corners = None, 0
                if pose_t is not None:
                    rvec, tvec = pose_t
                    cv2.drawFrameAxes(annot, K, D, rvec, tvec, 0.05)

            # --- depth nel centro (media 5x5 px per stabilita') ---
            dist_m = None
            if df is not None:
                depth = np.asanyarray(df.get_data())
                r = 2
                patch = depth[max(0, cy_px-r):cy_px+r+1,
                              max(0, cx_px-r):cx_px+r+1]
                vals = patch[patch > 0]
                if vals.size:
                    dist_m = float(np.median(vals)) * depth_scale

            # --- HUD ---
            cv2.drawMarker(annot, (cx_px, cy_px), (0, 255, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)
            cv2.circle(annot, (cx_px, cy_px), 30, (0, 255, 255), 1)

            # range consigliato: 0.20-0.50 m
            if dist_m is None:
                dist_str = "depth: --"
                color = (60, 60, 60)
            else:
                dist_str = f"depth center: {dist_m*100:5.1f} cm"
                if 0.20 <= dist_m <= 0.50:
                    color = (0, 200, 0)
                elif 0.15 <= dist_m <= 0.60:
                    color = (0, 165, 255)
                else:
                    color = (0, 0, 255)

            cv2.rectangle(annot, (0, 0), (w, 56), (0, 0, 0), -1)
            cv2.putText(annot, dist_str, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(annot,
                        f"poses: {len(poses)}/{args.n_poses}   "
                        f"charuco corners: {n_corners}",
                        (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (220, 220, 220), 1)

            cv2.imshow(win, annot)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == ord('d'):
                if poses:
                    poses.pop()
                    print(f"[record] rimossa: ora {len(poses)} pose")
            elif key == ord(' '):
                j = _read_joints(arm)
                if j is None:
                    print("[record] errore lettura giunti, riprovo")
                    continue
                poses.append(j)
                msg = (f"[record] [{len(poses)}/{args.n_poses}] joints (deg) = "
                       + "[" + ", ".join(f"{v:+.3f}" for v in j) + "]")
                if dist_m is not None:
                    msg += f"  | center depth = {dist_m*100:.1f} cm"
                if board is not None:
                    msg += f"  | charuco corners = {n_corners}"
                print(msg)
                if len(poses) >= args.n_poses:
                    print("[record] raggiunto numero target di pose.")
                    break
    finally:
        try:
            cv2.destroyWindow(win)
        except Exception:
            pass
        try:
            pipe.stop()
        except Exception:
            pass
        try:
            arm.set_mode(0)
            arm.set_state(0)
        except Exception:
            pass
        try:
            arm.disconnect()
        except Exception:
            pass

    if not poses:
        print("[record] nessuna posa registrata.")
        return

    out_path.write_text(json.dumps({"poses_joint_deg": poses}, indent=2))
    print(f"[record] salvate {len(poses)} pose in {out_path}")


# ---------------------------------------------------------------------------
# RUN: muovi -> capture -> detect ChArUco -> hand-eye
# ---------------------------------------------------------------------------
def _build_charuco(args):
    import cv2
    import cv2.aruco as aruco

    if not hasattr(aruco, args.aruco_dict):
        raise SystemExit(f"Dizionario ArUco sconosciuto: {args.aruco_dict}")
    adict = aruco.getPredefinedDictionary(getattr(aruco, args.aruco_dict))

    # API nuova (opencv-contrib >= 4.7) vs vecchia
    if hasattr(aruco, "CharucoBoard"):
        try:
            board = aruco.CharucoBoard(
                (args.squares_x, args.squares_y),
                args.square_len, args.marker_len, adict,
            )
        except TypeError:
            board = aruco.CharucoBoard_create(
                args.squares_x, args.squares_y,
                args.square_len, args.marker_len, adict,
            )
    else:
        board = aruco.CharucoBoard_create(
            args.squares_x, args.squares_y,
            args.square_len, args.marker_len, adict,
        )
    return adict, board


def _detect_charuco_pose(img_bgr, K, D, adict, board):
    """Restituisce (rvec, tvec, n_corners) target->camera, oppure None."""
    import cv2
    import cv2.aruco as aruco

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if hasattr(aruco, "CharucoDetector"):
        detector = aruco.CharucoDetector(board)
        ch_corners, ch_ids, mk_corners, mk_ids = detector.detectBoard(gray)
    else:
        params = aruco.DetectorParameters_create()
        mk_corners, mk_ids, _ = aruco.detectMarkers(gray, adict, parameters=params)
        if mk_ids is None or len(mk_ids) == 0:
            return None, None
        _, ch_corners, ch_ids = aruco.interpolateCornersCharuco(
            mk_corners, mk_ids, gray, board
        )

    if ch_ids is None or len(ch_ids) < 6:
        return None, (0 if ch_ids is None else len(ch_ids))

    # Pose estimation
    rvec = np.zeros((3, 1), float)
    tvec = np.zeros((3, 1), float)
    if hasattr(aruco, "estimatePoseCharucoBoard"):
        ok, rvec, tvec = aruco.estimatePoseCharucoBoard(
            ch_corners, ch_ids, board, K, D, rvec, tvec, False
        )
        if not ok:
            return None, len(ch_ids)
    else:
        # fallback: solvePnP sui corner ChArUco con punti 3D dal board
        obj_pts = board.getChessboardCorners()[ch_ids.flatten()]
        ok, rvec, tvec = cv2.solvePnP(obj_pts, ch_corners, K, D)
        if not ok:
            return None, len(ch_ids)
    return (rvec, tvec), len(ch_ids)


def _save_yaml(path: Path, T: np.ndarray):
    """Scrive un YAML in stile OpenCV (compatibile con load_T_yaml gia' esistente)."""
    flat = T.flatten().tolist()
    body = (
        "%YAML:1.0\n---\n"
        "T_ee_cam: !!opencv-matrix\n"
        "   rows: 4\n"
        "   cols: 4\n"
        "   dt: d\n"
        "   data: [ "
        + ", ".join(f"{v:.16e}" for v in flat)
        + " ]\n"
    )
    path.write_text(body)


def cmd_run(args):
    import cv2
    import pyrealsense2 as rs

    poses_path = Path(args.poses)
    data = json.loads(poses_path.read_text())
    poses = data["poses_joint_deg"]
    print(f"[run] caricate {len(poses)} pose da {poses_path}")

    adict, board = _build_charuco(args)

    # --- start RealSense ---
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, REALSENSE_WIDTH, REALSENSE_HEIGHT,
                      rs.format.bgr8, REALSENSE_FPS)
    profile = pipe.start(cfg)
    try:
        vprof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = vprof.get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx],
                      [0, intr.fy, intr.ppy],
                      [0, 0, 1]], float)
        D = np.array(intr.coeffs if intr.coeffs else [0, 0, 0, 0, 0], float)
        print(f"[run] RealSense intrinsics: fx={intr.fx:.2f} fy={intr.fy:.2f} "
              f"cx={intr.ppx:.2f} cy={intr.ppy:.2f}  D={D.tolist()}")
        # scalda la camera
        for _ in range(15):
            pipe.wait_for_frames(timeout_ms=1000)

        # --- arm ---
        arm = _connect_arm(args.ip, mode=0)

        R_g2b_list, t_g2b_list = [], []
        R_t2c_list, t_t2c_list = [], []
        debug_dir = Path(args.debug_dir) if args.debug_dir else None
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)

        try:
            for i, joints in enumerate(poses):
                print(f"\n[run] === posa {i+1}/{len(poses)} ===")
                arm.set_mode(0)
                arm.set_state(0)
                code = arm.set_servo_angle(angle=list(joints),
                                           speed=args.joint_speed,
                                           wait=True, is_radian=False)
                if code != 0:
                    print(f"[run] WARN set_servo_angle code={code}, salto")
                    continue
                time.sleep(args.settle_s)

                # media di alcuni frame per stabilita'
                imgs = []
                for _ in range(args.frame_avg):
                    frames = pipe.wait_for_frames(timeout_ms=2000)
                    cf = frames.get_color_frame()
                    if cf:
                        imgs.append(np.asanyarray(cf.get_data()))
                if not imgs:
                    print("[run] nessun frame, salto")
                    continue
                img = imgs[-1]  # ultimo (gli altri sono solo per stabilizzare AE)

                R_e, t_e = _read_pose_ee(arm)
                if R_e is None:
                    print("[run] lettura posa fallita, salto")
                    continue

                pose_target, n_corners = _detect_charuco_pose(img, K, D, adict, board)
                if pose_target is None:
                    print(f"[run] ChArUco non rilevato (corners={n_corners}), salto")
                    if debug_dir:
                        cv2.imwrite(str(debug_dir / f"miss_{i:02d}.png"), img)
                    continue
                rvec, tvec = pose_target
                R_tc, _ = cv2.Rodrigues(rvec)
                t_tc = tvec.reshape(3)

                R_g2b_list.append(R_e)
                t_g2b_list.append(t_e)
                R_t2c_list.append(R_tc)
                t_t2c_list.append(t_tc)

                print(f"[run] ok corners={n_corners}  "
                      f"|t_target_cam|={np.linalg.norm(t_tc)*100:.1f} cm")
                if debug_dir:
                    cv2.imwrite(str(debug_dir / f"ok_{i:02d}.png"), img)
        finally:
            try:
                arm.disconnect()
            except Exception:
                pass

    finally:
        try:
            pipe.stop()
        except Exception:
            pass

    n = len(R_g2b_list)
    print(f"\n[run] pose valide raccolte: {n}/{len(poses)}")
    if n < 5:
        raise SystemExit("Servono almeno ~5 pose valide per la calibrazione.")

    # --- hand-eye ---
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    print("\n=== Risultati cv2.calibrateHandEye (T_EE_CAM = gripper -> camera) ===")
    results = {}
    for name, m in methods.items():
        R_c2g, t_c2g = cv2.calibrateHandEye(
            R_g2b_list, t_g2b_list,
            R_t2c_list, t_t2c_list,
            method=m,
        )
        T = np.eye(4)
        T[:3, :3] = R_c2g
        T[:3, 3] = t_c2g.reshape(3)
        rpy = _R_to_rpy_deg(R_c2g)
        results[name] = T
        print(f"  [{name:10s}] t (mm) = "
              f"[{t_c2g[0,0]*1000:+8.2f}, {t_c2g[1,0]*1000:+8.2f}, {t_c2g[2,0]*1000:+8.2f}]"
              f"  rpy (deg) = [{rpy[0]:+7.2f}, {rpy[1]:+7.2f}, {rpy[2]:+7.2f}]")

    # --- consistenza: ricostruisci T_base_target da ogni coppia, verifica varianza ---
    chosen = args.method.upper()
    if chosen not in results:
        raise SystemExit(f"Metodo {chosen} non riconosciuto.")
    T_ee_cam = results[chosen]

    print(f"\n[run] uso metodo: {chosen}")
    bts = []
    for R_eb, t_eb, R_tc, t_tc in zip(R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list):
        T_base_ee = np.eye(4); T_base_ee[:3, :3] = R_eb; T_base_ee[:3, 3] = t_eb
        T_cam_target = np.eye(4); T_cam_target[:3, :3] = R_tc; T_cam_target[:3, 3] = t_tc
        T_base_target = T_base_ee @ T_ee_cam @ T_cam_target
        bts.append(T_base_target[:3, 3])
    bts = np.array(bts)
    mean = bts.mean(axis=0)
    std_mm = bts.std(axis=0) * 1000.0
    print(f"[run] target nel base frame: mean(m)={mean.tolist()}  "
          f"std(mm)={std_mm.tolist()}")
    print(f"[run] residuo posizionale RMS = {np.linalg.norm(std_mm):.2f} mm "
          "(target dovrebbe essere fisso: minore = meglio)")

    # --- salvataggi ---
    out_npz = Path(args.out)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz,
             T_ee_cam=T_ee_cam,
             method=chosen,
             results={k: v for k, v in results.items()},
             rms_mm=float(np.linalg.norm(std_mm)),
             K=K, D=D)
    print(f"[run] salvato NPZ -> {out_npz}")

    out_yaml = out_npz.with_suffix(".yaml")
    _save_yaml(out_yaml, T_ee_cam)
    print(f"[run] salvato YAML -> {out_yaml}")

    print("\n--- T_EE_CAM ---")
    for row in T_ee_cam:
        print("  [" + ", ".join(f"{v:+.6f}" for v in row) + "]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("record", help="Free-drive teach: registra pose ai giunti")
    pr.add_argument("--ip", default=ROBOT_IP)
    pr.add_argument("--out", default="utilities/calib_poses.json")
    pr.add_argument("--n-poses", type=int, default=20)
    # parametri opzionali per anteprima ChArUco durante il record
    pr.add_argument("--preview-charuco", action="store_true",
                    help="rileva e disegna il ChArUco nella preview")
    pr.add_argument("--squares-x", type=int, default=5)
    pr.add_argument("--squares-y", type=int, default=7)
    pr.add_argument("--square-len", type=float, default=0.035)
    pr.add_argument("--marker-len", type=float, default=0.026)
    pr.add_argument("--aruco-dict", default="DICT_5X5_50")
    pr.set_defaults(func=cmd_record)

    pn = sub.add_parser("run", help="Esegui calibrazione hand-eye")
    pn.add_argument("--ip", default=ROBOT_IP)
    pn.add_argument("--poses", required=True, help="JSON con 'poses_joint_deg'")
    pn.add_argument("--out", default="utilities/T_ee_cam.npz")
    pn.add_argument("--squares-x", type=int, default=5,
                    help="numero di QUADRATI lungo X (colonne)")
    pn.add_argument("--squares-y", type=int, default=7,
                    help="numero di QUADRATI lungo Y (righe)")
    pn.add_argument("--square-len", type=float, default=0.035,
                    help="lato quadrato (m)")
    pn.add_argument("--marker-len", type=float, default=0.026,
                    help="lato marker ArUco interno (m)")
    pn.add_argument("--aruco-dict", default="DICT_5X5_50")
    pn.add_argument("--joint-speed", type=float, default=20.0,
                    help="velocita' di set_servo_angle (deg/s)")
    pn.add_argument("--settle-s", type=float, default=0.6,
                    help="pausa dopo il movimento prima di catturare (s)")
    pn.add_argument("--frame-avg", type=int, default=5,
                    help="numero di frame da scartare/aspettare prima del capture")
    pn.add_argument("--method", default="PARK",
                    choices=["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"])
    pn.add_argument("--debug-dir", default=None,
                    help="se settato, salva le immagini catturate qui")
    pn.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
