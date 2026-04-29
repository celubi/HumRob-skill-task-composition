"""Replay di una traiettoria CSV in SIMULAZIONE (PyBullet).

Stessa interfaccia di selezione del CSV di ``replay_demo.py`` ma invece di
muovere il robot reale apre una preview 3D PyBullet che anima il modello
URDF dello xArm 6 lungo la traiettoria. Coerente con il flag
``--preview3d`` degli script di inferenza.

Il CSV deve contenere almeno le colonne ``t,x,y,z,qx,qy,qz,qw``; la colonna
``gripper`` e' opzionale (NaN o assente vengono accettati). Eventuali
colonne extra (es. ``rx,ry,rz`` dei CSV preprocessati) sono ignorate.

Uso:
    python3 sim_demo.py --csv ../demonstrations/pick/pick_01.csv
    python3 sim_demo.py --task pick --index 1
    python3 sim_demo.py --csv ../evaluation_results/pour/dmp/generated_pour_03.csv
    python3 sim_demo.py --csv ../recorded_executions/run_*/pour_actual.csv --animate
    python3 sim_demo.py --dir ../evaluation_results/pour/dmp
    python3 sim_demo.py --dir ../evaluation_results/pour --recursive
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config.robot_config import (  # noqa: E402
    DEMO_ROOT,
    HOME_JOINT_DEG,
)
from inference.methods.base import Trajectory  # noqa: E402


REQUIRED_COLS = ["t", "x", "y", "z", "qx", "qy", "qz", "qw"]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_trajectory(path: Path) -> Trajectory:
    """Legge un CSV e ritorna un oggetto Trajectory (compatibile con
    RobotPreview3D.show_trajectory). Gripper opzionale; valori NaN o
    colonna mancante producono ``traj.grip = None``."""
    rows = []
    with open(path) as f:
        rdr = csv.DictReader(f)
        fieldnames = rdr.fieldnames or []
        missing = [c for c in REQUIRED_COLS if c not in fieldnames]
        if missing:
            raise SystemExit(f"{path}: colonne mancanti {missing}")
        has_grip = "gripper" in fieldnames
        for row in rdr:
            rec = {k: float(row[k]) for k in REQUIRED_COLS}
            if has_grip:
                v = row.get("gripper", "")
                try:
                    rec["gripper"] = float(v) if v not in ("", None) else float("nan")
                except ValueError:
                    rec["gripper"] = float("nan")
            else:
                rec["gripper"] = float("nan")
            rows.append(rec)
    if not rows:
        raise SystemExit(f"CSV vuoto: {path}")

    t = np.array([r["t"] for r in rows], float)
    xyz = np.array([[r["x"], r["y"], r["z"]] for r in rows], float)
    quat = np.array([[r["qx"], r["qy"], r["qz"], r["qw"]] for r in rows], float)
    grip_arr = np.array([r["gripper"] for r in rows], float)
    grip = grip_arr if not np.all(np.isnan(grip_arr)) else None

    # Riallinea il tempo a t[0]=0 per coerenza con i Trajectory generati.
    if t[0] != 0.0:
        t = t - t[0]

    return Trajectory(t=t, xyz_m=xyz, quat=quat, grip=grip)


# ---------------------------------------------------------------------------
# CLI / file resolution
# ---------------------------------------------------------------------------
def resolve_csv(args: argparse.Namespace) -> Path:
    """Stessa logica di ``replay_demo.resolve_csv``."""
    if args.csv:
        p = Path(args.csv).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"CSV non trovato: {p}")
        return p
    if not args.task:
        raise SystemExit("Specifica --csv, --task (+ --index), oppure --dir.")
    task_dir = (args.demo_root or DEMO_ROOT) / args.task
    idx = args.index
    if idx is None:
        files = sorted(task_dir.glob(f"{args.task}_*.csv"))
        if not files:
            raise SystemExit(f"Nessuna demo trovata in {task_dir}")
        return files[-1]
    fname = f"{args.task}_{str(idx).zfill(args.zfill)}.csv"
    p = task_dir / fname
    if not p.is_file():
        raise SystemExit(f"Demo non trovata: {p}")
    return p


def collect_csvs(dir_path: Path, recursive: bool) -> list[Path]:
    """Lista ordinata dei CSV in ``dir_path`` (eventualmente ricorsiva)."""
    dir_path = dir_path.expanduser().resolve()
    if not dir_path.is_dir():
        raise SystemExit(f"Cartella non trovata: {dir_path}")
    files = sorted(dir_path.rglob("*.csv") if recursive
                   else dir_path.glob("*.csv"))
    if not files:
        flag = " (recursive)" if recursive else ""
        raise SystemExit(f"Nessun CSV in {dir_path}{flag}.")
    return files


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Replay in SIMULAZIONE (PyBullet) di una traiettoria CSV.")
    ap.add_argument("--csv", help="Path al file CSV (singolo).")
    ap.add_argument("--task", help="Nome task (sottocartella di demo-root).")
    ap.add_argument("--index", type=int, help="Indice della demo (es. 1, 2, ...).")
    ap.add_argument("--dir", type=Path, default=None,
                    help="Cartella di CSV da riprodurre in sequenza "
                         "(uno alla volta, INVIO per il successivo).")
    ap.add_argument("--recursive", action="store_true",
                    help="Con --dir: cerca i CSV anche nelle sottocartelle.")
    ap.add_argument("--zfill", type=int, default=2,
                    help="Zero-padding indice (default: 2).")
    ap.add_argument("--demo-root", type=Path, default=None,
                    help=f"Root delle demo per --task (default: {DEMO_ROOT}).")
    ap.add_argument("--animate", action="store_true",
                    help="Anima il robot lungo la traiettoria via IK (~30 Hz). "
                         "Default: mostra solo il path statico, plottato subito.")
    ap.add_argument("--initial-joints", type=float, nargs=6, default=None,
                    metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
                    help="Configurazione iniziale dei giunti [deg]. "
                         f"Default: HOME ({HOME_JOINT_DEG}).")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def show_one(csv_path: Path, q_init: list[float], animate: bool,
             label: str, prompt: str) -> None:
    """Carica un CSV e ne mostra la preview 3D PyBullet (open + close)."""
    print(f"\n[sim] {label}: {csv_path}")
    traj = load_trajectory(csv_path)
    dt = (traj.t[1] - traj.t[0]) if traj.n > 1 else 0.0
    print(f"[sim] traj N={traj.n}  T={traj.t[-1]:.2f}s  dt={dt:.3f}s  "
          f"grip={'si' if traj.grip is not None else 'no'}")
    print(f"[sim] start xyz={traj.xyz_m[0].round(4).tolist()}")
    print(f"[sim] end   xyz={traj.xyz_m[-1].round(4).tolist()}")

    from inference.vision.preview3d import RobotPreview3D
    print("[sim] apro la preview 3D (PyBullet)...")
    preview = RobotPreview3D(q_init, animate=animate)
    try:
        preview.show_trajectory(traj)
        preview.wait_for_user(prompt)
    finally:
        preview.close()


def main() -> int:
    args = parse_args()
    q_init = list(args.initial_joints) if args.initial_joints else list(HOME_JOINT_DEG)
    animate = args.animate

    if args.dir is not None:
        csvs = collect_csvs(args.dir, args.recursive)
        print(f"[sim] {len(csvs)} CSV trovati in {args.dir} "
              f"(recursive={args.recursive}).")
        print("[sim] controlli camera: Ctrl+sx=ruota, Ctrl+centrale=pan, "
              "rotella=zoom")
        try:
            for i, p in enumerate(csvs, start=1):
                rel = p.relative_to(args.dir.expanduser().resolve())
                label = f"({i}/{len(csvs)}) {rel}"
                prompt = (f"[sim] INVIO per il prossimo... "
                          if i < len(csvs) else "[sim] INVIO per chiudere... ")
                show_one(p, q_init, animate, label, prompt)
        except KeyboardInterrupt:
            print("\n[sim] interrotto dall'utente.")
        return 0

    # singolo CSV (--csv o --task --index)
    csv_path = resolve_csv(args)
    print("[sim] controlli camera: Ctrl+sx=ruota, Ctrl+centrale=pan, "
          "rotella=zoom")
    show_one(csv_path, q_init, animate, csv_path.name,
             "[sim] INVIO per chiudere... ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
