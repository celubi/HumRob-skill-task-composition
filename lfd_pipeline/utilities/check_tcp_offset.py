"""Verifica diagnostica del TCP offset e di T_EE_CAM sull'xArm 6.

Stampa lo stato del controller (TCP offset, world offset, posa corrente,
FK alla flangia) e, confrontando la posa restituita da ``get_position``
con la FK della flangia, deduce se nel codice "EE" coincide con la
flangia o con la punta del gripper. In base a questo, indica quale forma
di ``T_EE_CAM`` e' coerente con i valori di calibrazione ufficiali
UFactory per la "Camera Stand for Intel RealSense D435":

    EULER_EEF_TO_COLOR_OPT = [0.067052239, -0.0311387575, 0.021611456,
                              -0.004202176, -0.00848499, 1.5898775]
                             (xyz [m] + rpy [rad], flangia -> ottico RGB)

Uso:
    python3 utilities/check_tcp_offset.py [--ip 192.168.1.221]

Lo script e' di sola lettura: non muove il robot, non modifica offset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Import config condivisa (ROBOT_IP, T_EE_CAM)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.robot_config import ROBOT_IP, T_EE_CAM  # noqa: E402

from xarm.wrapper import XArmAPI  # noqa: E402


# Calibrazione ufficiale UFactory (flangia -> frame ottico RGB D435)
# Fonte: ufactory_vision/ggcnn_grasping_demo/example/realsense_d435/run_rs_d435_grasp.py
UFACTORY_FLANGE_TO_CAM_XYZ_M = np.array(
    [0.067052239, -0.0311387575, 0.021611456], dtype=float
)
UFACTORY_FLANGE_TO_CAM_RPY_RAD = np.array(
    [-0.004202176, -0.00848499, 1.5898775], dtype=float
)


def _safe_call(label, fn, *args, **kwargs):
    try:
        ret = fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - dipende da SDK
        print(f"  [WARN] {label}: eccezione {exc!r}")
        return None
    return ret


def _fmt_vec(v, fmt="{: .4f}"):
    return "[" + ", ".join(fmt.format(float(x)) for x in v) + "]"


def _unpack(ret):
    """Normalizza il classico ritorno (code, payload) dell'SDK xArm."""
    if isinstance(ret, (list, tuple)) and len(ret) == 2 and isinstance(ret[0], int):
        return ret[0], ret[1]
    return 0, ret


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default=ROBOT_IP, help="IP del controller xArm")
    args = parser.parse_args()

    print(f"[info] connessione a {args.ip} ...")
    arm = XArmAPI(args.ip, is_radian=False)
    try:
        arm.clean_warn()
        arm.clean_error()
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)

        print("\n=== STATO CONTROLLER ===")
        tcp_off = _safe_call("tcp_offset", lambda: arm.tcp_offset)
        print(f"  TCP offset (mm/deg)    : {_fmt_vec(tcp_off) if tcp_off is not None else 'N/A'}")

        world_off = _safe_call("world_offset", lambda: arm.world_offset)
        print(f"  World offset (mm/deg)  : {_fmt_vec(world_off) if world_off is not None else 'N/A'}")

        tcp_load = _safe_call("tcp_load", lambda: arm.tcp_load)
        if tcp_load is not None:
            print(f"  TCP load (kg, x/y/z mm): {tcp_load}")

        joints = _safe_call("angles", lambda: arm.angles)
        print(f"  Joints (deg)           : {_fmt_vec(joints[:6]) if joints is not None else 'N/A'}")

        pose_tcp = _safe_call("position", lambda: arm.position)
        print(f"  position (mm/deg)      : {_fmt_vec(pose_tcp[:6]) if pose_tcp is not None else 'N/A'}")

        # FK alla flangia (ignora il TCP offset)
        pose_flange = None
        if joints is not None:
            code, pose_flange = _unpack(
                _safe_call("get_forward_kinematics", arm.get_forward_kinematics, list(joints[:6]))
            )
            print(f"  FK flangia   (mm/deg)  [code={code}]: {_fmt_vec(pose_flange[:6]) if pose_flange is not None else 'N/A'}")

        # ---------------------------------------------------------------
        # diagnosi: get_position == flangia o == punta TCP?
        # ---------------------------------------------------------------
        print("\n=== DIAGNOSI 'EE' nel codice ===")
        delta_xyz_mm = None
        if pose_tcp is not None and pose_flange is not None:
            delta_xyz_mm = np.array(pose_tcp[:3], float) - np.array(pose_flange[:3], float)
            dist = float(np.linalg.norm(delta_xyz_mm))
            print(f"  delta TCP-flangia (mm): {_fmt_vec(delta_xyz_mm, '{: .2f}')}  |  norma = {dist:.2f} mm")

            tcp_offset_active = tcp_off is not None and any(abs(float(v)) > 0.5 for v in tcp_off[:3])
            if not tcp_offset_active and dist < 1.0:
                ee_kind = "flangia"
            elif tcp_offset_active:
                ee_kind = "punta TCP del gripper"
            else:
                ee_kind = "?? (controllare manualmente)"
            print(f"  --> 'EE' usato da arm.get_position() = {ee_kind}")
        else:
            ee_kind = None
            print("  Impossibile dedurre EE: posizioni non disponibili.")

        # ---------------------------------------------------------------
        # T_EE_CAM atteso vs configurato
        # ---------------------------------------------------------------
        print("\n=== T_EE_CAM ===")
        print("  T_EE_CAM attualmente in config/robot_config.py:")
        for row in T_EE_CAM:
            print("    " + _fmt_vec(row))

        tx_cfg, ty_cfg, tz_cfg = (float(T_EE_CAM[i, 3]) for i in range(3))
        print(f"  traslazione configurata (m): tx={tx_cfg:+.4f}  ty={ty_cfg:+.4f}  tz={tz_cfg:+.4f}")

        ux, uy, uz = UFACTORY_FLANGE_TO_CAM_XYZ_M
        print("\n  Calibrazione ufficiale UFactory (flangia -> camera color-optical):")
        print(f"    xyz (m): tx={ux:+.4f}  ty={uy:+.4f}  tz={uz:+.4f}")
        print(f"    rpy (rad): {_fmt_vec(UFACTORY_FLANGE_TO_CAM_RPY_RAD)}")

        # tz atteso in funzione del tipo di EE
        if ee_kind == "flangia":
            tz_expected = uz  # +0.0216
            note = "EE = flangia  =>  tz_T_EE_CAM atteso ≈ +0.0216 m"
        elif ee_kind == "punta TCP del gripper" and tcp_off is not None:
            tcp_z_m = float(tcp_off[2]) / 1000.0
            tz_expected = uz - tcp_z_m
            note = (f"EE = TCP (offset_z = {tcp_z_m*1000:.1f} mm)  =>  "
                    f"tz_T_EE_CAM atteso ≈ {tz_expected:+.4f} m")
        else:
            tz_expected = None
            note = "Atteso non calcolabile (EE non determinato)."

        print(f"\n  {note}")
        if tz_expected is not None:
            err_mm = (tz_cfg - tz_expected) * 1000.0
            print(f"  Errore tz attuale: {err_mm:+.1f} mm")
            if abs(err_mm) > 5.0:
                print("  [!] Scostamento > 5 mm: aggiornare T_EE_CAM (vedi suggerimento sotto).")
            else:
                print("  [ok] tz coerente con la calibrazione ufficiale.")

        # Suggerimento di matrice
        print("\n=== SUGGERIMENTO ===")
        if ee_kind == "flangia":
            print("  Sostituire T_EE_CAM in config/robot_config.py con:")
            print("    T_EE_CAM = _np.array([")
            print("        [ 0.0, -1.0,  0.0,  0.0671],")
            print("        [ 1.0,  0.0,  0.0, -0.0311],")
            print("        [ 0.0,  0.0,  1.0,  0.0216],")
            print("        [ 0.0,  0.0,  0.0,  1.0   ],")
            print("    ], dtype=float)")
            print("  (rotazione: solo Rz=+90°; i tilt UFactory ~0.3° sono trascurabili)")
        elif ee_kind == "punta TCP del gripper" and tcp_off is not None:
            tcp_z_m = float(tcp_off[2]) / 1000.0
            print("  La forma attuale (Rz=+90°, tx=+0.067, ty=-0.031) e' OK.")
            print(f"  tz dovrebbe essere: {uz - tcp_z_m:+.4f} m   "
                  f"(= 0.0216 - TCP_offset_z={tcp_z_m*1000:.1f} mm)")
        else:
            print("  Determinare prima il tipo di EE (vedi diagnosi sopra), poi rieseguire.")

    finally:
        try:
            arm.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
