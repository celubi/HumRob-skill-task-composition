"""Pre-processing model-specific per GMM-GMR (Section II-E del paper).

Per ogni demo gi\u00e0 passata dalla common-preprocessing produce:
  - una variabile di fase normalizzata  s = (t - t0) / (t_end - t0) \u2208 [0, 1]
  - la matrice di osservazione  Z = [s | x, y, z, rx, ry, rz]   (N \u00d7 7)

Tutte le demo del task vengono concatenate in un unico dataset salvato in
``gmm_preprocessed_demonstrations/<task>/<task>_gmm_dataset.npz`` con anche:
  - ``s_ref``, ``grip_ref`` (gripper medio interpolato sulla griglia s_ref)
  - ``T_mean`` (durata media delle demo)
  - ``demo_files``, ``demo_lengths`` (audit)

Per ispezione visiva, viene salvato anche un CSV per ciascuna demo con la
colonna ``s`` aggiunta in testa (header: ``s,x,y,z,rx,ry,rz,gripper``).

Uso:
    python3 gmm_preprocess.py --task pick
    python3 gmm_preprocess.py --task pour --index 3
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parents[1]  # .../lfd_pipeline
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config.robot_config import (  # noqa: E402
    GMM_PREPROCESSED_ROOT,
    PREPROCESSED_ROOT,
)

# colonne attese nel CSV di common-preprocessing
COMMON_HEADER = ["t", "x", "y", "z", "qx", "qy", "qz", "qw",
                 "rx", "ry", "rz", "gripper"]

# colonne usate da GMM-GMR (escludiamo i quaternioni: ridondanti rispetto a rx,ry,rz)
Y_COLS = ["x", "y", "z", "rx", "ry", "rz"]

# header del CSV per ispezione visiva
SCSV_HEADER = ["s", "x", "y", "z", "rx", "ry", "rz", "gripper"]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_processed_csv(path: Path):
    """Ritorna (t, y, gripper) con y di shape (N, 6) = [x,y,z,rx,ry,rz]."""
    rows = []
    with open(path) as f:
        rdr = csv.DictReader(f)
        missing = [c for c in COMMON_HEADER if c not in (rdr.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV {path} - colonne mancanti: {missing}")
        for row in rdr:
            rows.append([float(row[k]) for k in COMMON_HEADER])
    if not rows:
        raise ValueError(f"CSV vuoto: {path}")
    arr = np.asarray(rows, dtype=float)
    t = arr[:, 0]
    # x,y,z = colonne 1..3; rx,ry,rz = colonne 8..10
    y = np.column_stack([arr[:, 1:4], arr[:, 8:11]])
    grip = arr[:, 11]
    return t, y, grip


def collect_inputs(in_root: Path, task: str, index: int | None, zfill: int,
                   first_k: int | None = None):
    task_dir = in_root / task
    if not task_dir.is_dir():
        raise SystemExit(f"Cartella demo processate non trovata: {task_dir}")
    if index is not None and first_k is not None:
        raise SystemExit("Usa --index oppure --first-k, non entrambi.")
    if index is not None:
        fname = f"processed_{task}_{str(index).zfill(zfill)}.csv"
        f = task_dir / fname
        if not f.is_file():
            raise SystemExit(f"Demo non trovata: {f}")
        return [f]
    pattern = re.compile(rf"^processed_{re.escape(task)}_(\d+)\.csv$")
    files = sorted(p for p in task_dir.iterdir() if pattern.match(p.name))
    if not files:
        raise SystemExit(f"Nessuna demo processata trovata in {task_dir}")
    if first_k is not None:
        if first_k <= 0:
            raise SystemExit("--first-k deve essere > 0.")
        if first_k > len(files):
            raise SystemExit(
                f"--first-k={first_k} ma sono disponibili solo {len(files)} demo in {task_dir}."
            )
        files = files[:first_k]
    return files


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def phase_variable(t: np.ndarray) -> np.ndarray:
    t0, tN = float(t[0]), float(t[-1])
    span = max(tN - t0, 1e-12)
    return (t - t0) / span


def save_inspection_csv(out_path: Path, s: np.ndarray, y: np.ndarray, grip: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SCSV_HEADER)
        for i in range(len(s)):
            w.writerow([f"{s[i]:.9f}", *(f"{v:.9f}" for v in y[i]),
                        f"{grip[i]:.9f}"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-processing GMM-specifico (fase s).")
    p.add_argument("--task", required=True, help="Nome primitiva (es. pick, place, pour).")
    p.add_argument("--index", type=int, default=None,
                   help="Processa solo la demo con questo indice. Se omesso, le processa tutte.")
    p.add_argument("--first-k", type=int, default=None,
                   help="Usa solo le prime K demo (in ordine alfabetico). Mutualmente esclusivo con --index.")
    p.add_argument("--in-root", type=Path, default=PREPROCESSED_ROOT,
                   help=f"Root delle demo gi\u00e0 pre-processate (default: {PREPROCESSED_ROOT}).")
    p.add_argument("--out-root", type=Path, default=GMM_PREPROCESSED_ROOT,
                   help=f"Root di output (default: {GMM_PREPROCESSED_ROOT}).")
    p.add_argument("--zfill", type=int, default=2, help="Zero-padding indice (default: 2).")
    p.add_argument("--no-inspection-csv", action="store_true",
                   help="Non salvare i CSV per-demo per ispezione visiva.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inputs = collect_inputs(args.in_root, args.task, args.index, args.zfill,
                            first_k=args.first_k)
    out_dir = args.out_root / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"GMM preprocessing su {len(inputs)} demo per '{args.task}'")
    print(f"  in : {args.in_root / args.task}")
    print(f"  out: {out_dir}")

    Z_list = []
    grip_list = []
    s_list = []
    durations = []
    n_per_demo = []
    demo_names = []

    for f in inputs:
        t, y, grip = load_processed_csv(f)
        s = phase_variable(t)
        Z = np.column_stack([s, y])  # (N, 7)
        Z_list.append(Z)
        grip_list.append(grip)
        s_list.append(s)
        durations.append(float(t[-1] - t[0]))
        n_per_demo.append(len(s))
        demo_names.append(f.name)

        if not args.no_inspection_csv:
            ins_path = out_dir / f.name.replace("processed_", "gmm_processed_")
            save_inspection_csv(ins_path, s, y, grip)
            print(f"  {f.name} -> {ins_path.name}  (N={len(s)}, T={durations[-1]:.2f}s)")
        else:
            print(f"  {f.name}  (N={len(s)}, T={durations[-1]:.2f}s)")

    Z_concat = np.vstack(Z_list)

    # griglia di riferimento s_ref con N = media degli N_k.
    N_ref = int(round(float(np.mean(n_per_demo))))
    s_ref = np.linspace(0.0, 1.0, N_ref)
    # gripper medio: ricampiono ogni grip su s_ref e medio.
    grip_on_ref = np.stack([np.interp(s_ref, s_k, g_k)
                            for s_k, g_k in zip(s_list, grip_list)], axis=0)
    grip_ref = grip_on_ref.mean(axis=0)

    out_npz = out_dir / f"{args.task}_gmm_dataset.npz"
    np.savez(
        out_npz,
        Z_concat=Z_concat,
        s_ref=s_ref,
        grip_ref=grip_ref,
        T_mean=float(np.mean(durations)),
        demo_lengths=np.asarray(n_per_demo, dtype=int),
        demo_files=np.asarray(demo_names),
        y_columns=np.asarray(Y_COLS),
    )
    print(f"\nDataset GMM salvato: {out_npz}")
    print(f"  Z_concat: {Z_concat.shape}  |  s_ref: {s_ref.shape}  |  T_mean={np.mean(durations):.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
