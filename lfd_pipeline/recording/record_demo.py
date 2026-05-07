"""Registrazione di dimostrazioni cinestetiche per xArm 6.

Uso tipico:
    python3 record_demo.py --task pick
    python3 record_demo.py --task pour --rate 100

Comandi da tastiera (pynput, finestra terminale in foreground):
    SPACE  start / stop registrazione (toggle).
    n      scarta la registrazione in corso (se attiva) e azzera il buffer.
    h      torna a modalità posizione, vai a HOME, rientra in manual mode.
    q      esci (chiude in sicurezza il robot).

Output:
    <DEMO_ROOT>/<task>/<task>_<NN>.csv
con header: t,x,y,z,qx,qy,qz,qw,gripper
- x,y,z in metri
- qx,qy,qz,qw quaternione unitario (orientazione TCP)
- gripper: apertura letta da arm.get_gripper_position() (unità SDK)
- t: tempo relativo dal primo campione della demo
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from xarm.wrapper import XArmAPI
from keyboard_controller import KeyboardController
from demo_recorder import DemoRecorder

# Permetti l'import di config.* eseguendo lo script da qualunque cwd.
_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config.robot_config import (  # noqa: E402
    DEFAULT_RECORD_RATE_HZ,
    DEMO_ROOT,
    GRIPPER_CLOSE_SPEED,
    GRIPPER_CLOSED_POS,
    GRIPPER_OPEN_POS,
    GRIPPER_SPEED,
    HOME_JOINT_DEG,
    ROBOT_IP,
)

# ---------------------------------------------------------------------------
# Robot helpers
# ---------------------------------------------------------------------------
def init_robot(ip: str) -> XArmAPI:
    arm = XArmAPI(ip, is_radian=True)
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.2)
    # Gripper
    try:
        arm.set_gripper_mode(0)
        arm.set_gripper_enable(True)
        arm.set_gripper_speed(GRIPPER_SPEED)
    except Exception as e:
        print(f"[WARN] init gripper: {e}")
    return arm


def go_home(arm: XArmAPI, joints_deg) -> None:
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.1)
    # joints_deg in gradi; speed in deg/s quando is_radian=False.
    arm.set_servo_angle(angle=list(joints_deg), speed=30, wait=True, is_radian=False)
    time.sleep(0.5)


def enter_manual_mode(arm: XArmAPI) -> None:
    """Mette il robot in modalità teach (gravity compensation)."""
    # Pulisci eventuali warning latenti prima di cambiare modalità.
    try:
        arm.clean_warn()
    except Exception:
        pass
    arm.set_mode(2)
    arm.set_state(0)
    # Attendi che il controller si stabilizzi nella nuova modalità prima
    # di consentire il movimento manuale.
    time.sleep(0.5)


def recover_from_error(arm: XArmAPI) -> bool:
    """Pulisce eventuali errori/warning del controller e ripristina lo stato.

    Tipico per ControllerError code 37 (collision / joint speed), che lascia
    il robot in stato di errore finché non viene esplicitamente cleanato.
    Ritorna True se il controller risulta privo di errori dopo il recovery.
    """
    try:
        err_code, warn_code = arm.get_err_warn_code()
    except Exception:
        err_code, warn_code = -1, -1

    if err_code == 0 and warn_code == 0:
        return True
    print(f"\n[GUARD] ControllerError code={err_code} warn={warn_code} -> recovery")

    try:
        arm.clean_warn()
    except Exception:
        pass

    try:
        arm.clean_error()
    except Exception:
        pass

    try:
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.2)
    except Exception as e:
        print(f"[GUARD] errore durante il reset: {e}")

    try:
        err_code, warn_code = arm.get_err_warn_code()
    except Exception:
        return False
    
    return err_code == 0

def shutdown_robot(arm: XArmAPI) -> None:
    try:
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.1)
    finally:
        arm.disconnect()


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------
HELP = """
Comandi:
  SPACE  start / stop registrazione (toggle)
  n      scarta il buffer corrente (se in registrazione)
  g      apri / chiudi il gripper (toggle)
  h      torna a HOME (esce e rientra in manual mode)
  q      esci
"""


def toggle_gripper(arm: XArmAPI, currently_open: bool) -> bool:
    """Apre o chiude il gripper. Ritorna il nuovo stato (True = aperto).

    La chiusura usa una velocità ridotta (GRIPPER_CLOSE_SPEED) per consentire
    un campionamento adeguato a 50 Hz; l'apertura usa la velocità di default.
    """
    target = GRIPPER_CLOSED_POS if currently_open else GRIPPER_OPEN_POS
    try:
        if currently_open:
            # stiamo per chiudere: rallenta
            arm.set_gripper_speed(GRIPPER_CLOSE_SPEED)
        else:
            # stiamo per aprire: ripristina velocità default
            arm.set_gripper_speed(GRIPPER_SPEED)
        arm.set_gripper_position(target, wait=False)
    except Exception as e:
        print(f"[WARN] gripper: {e}")
        return currently_open
    return not currently_open


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Registrazione demo cinestetiche per xArm 6.")
    p.add_argument("--task", required=True, help="Nome operazione (es. pick, place, pour).")
    p.add_argument("--robot-ip", default=ROBOT_IP, help=f"IP del controller xArm (default: {ROBOT_IP}).")
    p.add_argument("--rate", type=float, default=DEFAULT_RECORD_RATE_HZ,
                   help=f"Frequenza di campionamento [Hz] (default: {DEFAULT_RECORD_RATE_HZ}).")
    p.add_argument("--out-dir", type=Path, default=DEMO_ROOT,
                   help=f"Cartella radice di output (default: {DEMO_ROOT}).")
    p.add_argument("--zfill", type=int, default=2, help="Zero-padding dell'indice nel nome file (default: 2).")
    return p.parse_args()


def print_status(recorder: DemoRecorder) -> None:
    state = "REC ●" if recorder.is_recording() else "IDLE "
    n = recorder.sample_count()
    sys.stdout.write(f"\r[{state}]  campioni: {n:6d}   ")
    sys.stdout.flush()


def main() -> int:
    args = parse_args()

    print(f"Connessione a xArm @ {args.robot_ip} ...")
    arm = init_robot(args.robot_ip)

    print(f"Movimento a HOME (deg): {HOME_JOINT_DEG}")
    go_home(arm, HOME_JOINT_DEG)

    print("Passaggio in modalità manuale (gravity compensation).")
    enter_manual_mode(arm)

    out_dir = args.out_dir
    print(f"Output: {out_dir / args.task}")
    print(HELP)

    recorder = DemoRecorder(arm, args.task, out_dir, args.rate, args.zfill)
    recorder.start_thread()

    kb = KeyboardController()
    kb.start()

    # SIGINT -> uscita pulita
    interrupted = threading.Event()

    def _sigint(_sig, _frm):
        interrupted.set()

    signal.signal(signal.SIGINT, _sigint)

    # Guard sui ControllerError (es. code 37 = collision/joint speed):
    # registra una callback che, alla prima notifica di errore, scarta la
    # registrazione in corso, pulisce l'errore e riporta il robot in manual.
    error_evt = threading.Event()

    def _on_err_warn(item):
        err = item.get('error_code', 0) if isinstance(item, dict) else 0
        if err and err != 0:
            error_evt.set()

    try:
        arm.register_error_warn_changed_callback(_on_err_warn)
    except Exception as e:
        print(f"[WARN] impossibile registrare callback errori: {e}")

    # Stato gripper iniziale: apriamolo per partire da uno stato noto.
    gripper_open = True
    try:
        arm.set_gripper_position(GRIPPER_OPEN_POS, wait=False)
    except Exception:
        pass

    try:
        while not interrupted.is_set():
            if error_evt.is_set():
                error_evt.clear()
                if recorder.is_recording():
                    print()
                    print(recorder.discard())
                if recover_from_error(arm):
                    enter_manual_mode(arm)
                    print("[GUARD] robot ripristinato in modalità manuale.")
                else:
                    print("[GUARD] recovery non riuscito; controlla il robot.")

            cmd = kb.pop()
            if cmd is None:
                print_status(recorder)
                time.sleep(0.05)
                continue
            print()  # newline dopo lo status inline
            if cmd == "space":
                print(recorder.toggle())
            elif cmd == "n":
                print(recorder.discard())
            elif cmd == "g":
                gripper_open = toggle_gripper(arm, gripper_open)
                print(f"Gripper -> {'APERTO' if gripper_open else 'CHIUSO'}")
            elif cmd == "h":
                if recorder.is_recording():
                    print(recorder.toggle())  # ferma e salva prima di muovere
                print("Ritorno a HOME ...")
                go_home(arm, HOME_JOINT_DEG)
                enter_manual_mode(arm)
                print("Pronto.")
            elif cmd == "q":
                break
    finally:
        print("\nChiusura ...")
        if recorder.is_recording():
            print(recorder.toggle())
        recorder.stop()
        kb.stop()
        shutdown_robot(arm)

    return 0


if __name__ == "__main__":
    sys.exit(main())
