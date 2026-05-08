"""Riproduzione delle dimostrazioni con BC / DMP / GMM-GMR.

Per ogni demo pre-processata del task specificato, ciascun modello selezionato
genera una traiettoria con:
    start = primo sample della demo
    goal  = ultimo sample della demo
(start/goal in formato [x, y, z, roll, pitch, yaw] in BASE; xyz e quaternione
sono presi dal CSV ``processed_<task>_NN.csv`` e il quaternione e' convertito
in rpy ZYX coerente con la convenzione xArm.)

Le traiettorie generate vengono salvate in CSV per analisi a posteriori
(estrazione di metriche di tracking/replay-error vs. demo).

Output:
    evaluation_results/<task>/<method>/generated_<task>_<NN>.csv
    (header: t,x,y,z,qx,qy,qz,qw,gripper)

Uso:
    python3 reproduce_demos.py --task pick
    python3 reproduce_demos.py --task pour --methods bc dmp gmm
    python3 reproduce_demos.py --task place --indices 1 2 3
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parent  # .../lfd_pipeline
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config.robot_config import (  # noqa: E402
    PREPROCESSED_ROOT,
    TRAINED_MODELS_ROOT,
)


METHODS_ALL = ("bc", "dmp", "gmm")
DEMO_HEADER = ["t", "x", "y", "z", "qx", "qy", "qz", "qw",
               "rx", "ry", "rz", "gripper"]
OUT_HEADER = ["t", "x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]
EVALUATION_ROOT = _PKG_ROOT / "reproduction_results"


# ---------------------------------------------------------------------------
# I/O demo + estrazione endpoint
# ---------------------------------------------------------------------------
def load_preprocessed_demo(path: Path) -> dict:
    """Carica un CSV ``processed_<task>_NN.csv`` come dict di array."""
    rows = []
    with open(path) as f:
        rdr = csv.DictReader(f)
        missing = [c for c in DEMO_HEADER if c not in (rdr.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path}: colonne mancanti {missing}")
        for row in rdr:
            rows.append([float(row[k]) for k in DEMO_HEADER])
    if not rows:
        raise SystemExit(f"{path}: CSV vuoto.")
    arr = np.asarray(rows, float)
    return {
        "name": path.name,
        "t": arr[:, 0],
        "xyz": arr[:, 1:4],
        "quat": arr[:, 4:8],     # qx, qy, qz, qw (assoluto)
        "rotvec": arr[:, 8:11],  # rotvec assoluto (log-map del quaternione)
        "grip": arr[:, 11],
    }


def quat_to_rpy_zyx(qx: float, qy: float, qz: float, qw: float):
    """Quaternione (xyzw) -> rpy ZYX (Tait-Bryan, Rz*Ry*Rx). Conv. xArm."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0.0:
        return 0.0, 0.0, 0.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)
    sinp = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
    pitch = math.asin(sinp)
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny, cosy)
    return float(roll), float(pitch), float(yaw)


def demo_endpoints(demo: dict) -> tuple[list[float], list[float]]:
    """Restituisce (start, goal) come [x,y,z,r,p,y] dal primo e ultimo sample."""
    def to_xyzrpy(i: int) -> list[float]:
        x, y, z = demo["xyz"][i]
        qx, qy, qz, qw = demo["quat"][i]
        roll, pitch, yaw = quat_to_rpy_zyx(qx, qy, qz, qw)
        return [float(x), float(y), float(z), roll, pitch, yaw]
    return to_xyzrpy(0), to_xyzrpy(-1)


def save_generated_csv(path: Path, traj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(OUT_HEADER)
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


# ---------------------------------------------------------------------------
# Generator factory (riusa i moduli inference.methods)
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


# ---------------------------------------------------------------------------
# Selezione demo
# ---------------------------------------------------------------------------
def collect_demo_files(in_root: Path, task: str,
                       indices: list[int] | None) -> list[Path]:
    task_dir = in_root / task
    if not task_dir.is_dir():
        raise SystemExit(f"Cartella demo non trovata: {task_dir}")
    pattern = re.compile(rf"^processed_{re.escape(task)}_(\d+)\.csv$")
    files = sorted(p for p in task_dir.iterdir() if pattern.match(p.name))
    if not files:
        raise SystemExit(f"Nessuna demo in {task_dir}")
    if indices:
        wanted = set(indices)
        files = [f for f in files
                 if int(pattern.match(f.name).group(1)) in wanted]
        if not files:
            raise SystemExit(f"Nessuna demo trovata per indici {indices}.")
    return files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Riproduzione delle demo con BC/DMP/GMM-GMR per valutazione.")
    p.add_argument("--task", required=True,
                   help="Nome primitiva (es. pick, place, pour).")
    p.add_argument("--methods", nargs="+", choices=METHODS_ALL,
                   default=list(METHODS_ALL),
                   help=f"Metodi da valutare (default: {' '.join(METHODS_ALL)}).")
    p.add_argument("--bc-model", type=Path, default=None,
                   help="Override path modello BC. "
                        "Default: trained_models/<task>_bc.pt.")
    p.add_argument("--dmp-model", type=Path, default=None,
                   help="Override path modello DMP. "
                        "Default: trained_models/<task>_dmp.npz.")
    p.add_argument("--gmm-model", type=Path, default=None,
                   help="Override path modello GMM. "
                        "Default: trained_models/<task>_gmm.npz.")
    p.add_argument("--device", default="cpu",
                   help="Device torch per BC (default: cpu).")
    p.add_argument("--indices", type=int, nargs="+", default=None,
                   help="Indici demo da processare (default: tutti).")
    p.add_argument("--in-root", type=Path, default=PREPROCESSED_ROOT,
                   help=f"Cartella demo pre-processate (default: {PREPROCESSED_ROOT}).")
    p.add_argument("--out-root", type=Path, default=EVALUATION_ROOT,
                   help=f"Cartella output (default: {EVALUATION_ROOT}).")
    p.add_argument("--duration-scale", type=float, default=1.0,
                   help="Riscala la durata della traiettoria generata "
                        "(1.0 = T_mean del modello; valori <1 accorciano).")
    return p.parse_args()


def resolve_model_path(args: argparse.Namespace, method: str, task: str) -> Path:
    override = getattr(args, f"{method}_model")
    return override if override is not None else default_model_path(task, method)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    files = collect_demo_files(args.in_root, args.task, args.indices)
    pattern = re.compile(rf"^processed_{re.escape(args.task)}_(\d+)\.csv$")

    print(f"[eval] task     : {args.task}")
    print(f"[eval] in_root  : {args.in_root}")
    print(f"[eval] out_root : {args.out_root}")
    print(f"[eval] metodi   : {' '.join(args.methods)}")
    print(f"[eval] {len(files)} demo da processare:")
    for f in files:
        print(f"  - {f.name}")

    # Carica i generatori (uno per metodo, riutilizzati su tutte le demo).
    # Se un modello manca o non si carica, il metodo viene saltato con warning.
    generators: dict[str, object] = {}
    for method in args.methods:
        path = resolve_model_path(args, method, args.task)
        if not path.is_file():
            print(f"[eval] [warn] modello {method.upper()} non trovato: "
                  f"{path} -- skip metodo.")
            continue
        print(f"[eval] carico {method.upper()} : {path}")
        try:
            generators[method] = make_generator(method, path, args.device)
        except Exception as e:
            print(f"[eval] [warn] {method.upper()} non caricato: {e} -- skip metodo.")
            continue

    if not generators:
        raise SystemExit("[eval] nessun modello disponibile; uscita.")

    out_task_dir = args.out_root / args.task
    n_total = 0
    for f in files:
        m = pattern.match(f.name)
        idx_str = m.group(1)

        demo = load_preprocessed_demo(f)
        start_xyzrpy, goal_xyzrpy = demo_endpoints(demo)
        T_demo = float(demo["t"][-1] - demo["t"][0])
        N_demo = int(len(demo["t"]))

        print(f"\n[eval] {f.name}: N={N_demo}  T={T_demo:.3f}s")
        print(f"  start xyz={[round(v, 4) for v in start_xyzrpy[:3]]} "
              f"rpy={[round(v, 4) for v in start_xyzrpy[3:]]}")
        print(f"  goal  xyz={[round(v, 4) for v in goal_xyzrpy[:3]]} "
              f"rpy={[round(v, 4) for v in goal_xyzrpy[3:]]}")

        for method, gen in generators.items():
            try:
                traj = gen.generate(
                    start_xyzrpy=start_xyzrpy,
                    goal_xyzrpy=goal_xyzrpy,
                    duration_scale=args.duration_scale,
                )
            except Exception as e:
                print(f"  [{method}] [error] generate fallito: {e}")
                continue
            out_path = out_task_dir / method / f"generated_{args.task}_{idx_str}.csv"
            save_generated_csv(out_path, traj)
            print(f"  [{method}] N={traj.n}  T={traj.t[-1]:.3f}s  -> {out_path}")
            n_total += 1

    print(f"\n[eval] DONE. Salvate {n_total} traiettorie in {out_task_dir}/<method>/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
