"""Pre-processing model-specific per DMP (Section II-D del paper).

Le demo arrivano dalla common-preprocessing su griglia temporale uniforme
(t, x, y, z, qx, qy, qz, qw, rx, ry, rz, gripper). Per il DMP serve, in
aggiunta, la velocita' e l'accelerazione di ciascuna componente dello stato
6D ``y = [x, y, z, rx, ry, rz]``, ottenute via differenze finite (np.gradient,
edge_order=2).

Per ogni demo viene salvato:
  - un CSV di ispezione con t, y, dy, ddy, gripper
  - un'entry nel dataset consolidato ``<task>_dmp_dataset.npz``

Il dataset consolidato contiene object-arrays (una entry per demo) cosi' da
mantenere lunghezze potenzialmente diverse senza padding artificiali, e gli
endpoint di ogni demo (utili al fit DMP).

Uso:
    python3 dmp_preprocess.py --task pick
    python3 dmp_preprocess.py --task pour --index 3
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
    DMP_PREPROCESSED_ROOT,
    PREPROCESSED_ROOT,
)
from preprocessing.preprocess_demos import load_qref_sidecar  # noqa: E402

# colonne attese nel CSV di common-preprocessing
COMMON_HEADER = ["t", "x", "y", "z", "qx", "qy", "qz", "qw",
                 "rx", "ry", "rz", "gripper"]

# componenti dello stato 6D usato dal DMP
Y_COLS = ["x", "y", "z", "rx", "ry", "rz"]

# header del CSV di ispezione visiva
INSPECTION_HEADER = (
    ["t"]
    + Y_COLS
    + [f"d{c}" for c in Y_COLS]
    + [f"dd{c}" for c in Y_COLS]
    + ["gripper"]
)

# tolleranza relativa per verificare l'uniformita' della time-grid
DT_RTOL = 1e-3


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
def estimate_dt(t: np.ndarray) -> float:
    """Stima dt da una time-grid uniforme; verifica l'uniformita'."""
    if len(t) < 2:
        raise ValueError("Demo troppo corta per stimare dt (N<2).")
    diffs = np.diff(t)
    dt = float(np.mean(diffs))
    if dt <= 0.0:
        raise ValueError(f"dt non positivo ({dt}); time-grid non valida.")
    if not np.allclose(diffs, dt, rtol=DT_RTOL, atol=1e-6):
        spread = float(np.max(diffs) - np.min(diffs))
        raise ValueError(
            f"Time-grid non uniforme: dt_mean={dt:.6f}, spread={spread:.2e}. "
            "Atteso input dalla common-preprocessing su griglia uniforme."
        )
    return dt


def finite_diff_derivatives(y: np.ndarray, dt: float):
    """ẏ e ÿ via differenze finite centrate (np.gradient, edge_order=2)."""
    dy = np.gradient(y, dt, axis=0, edge_order=2)
    ddy = np.gradient(dy, dt, axis=0, edge_order=2)
    return dy, ddy


def save_inspection_csv(out_path: Path, t: np.ndarray, y: np.ndarray,
                        dy: np.ndarray, ddy: np.ndarray,
                        grip: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(INSPECTION_HEADER)
        for i in range(len(t)):
            w.writerow([
                f"{t[i]:.9f}",
                *(f"{v:.9f}" for v in y[i]),
                *(f"{v:.9f}" for v in dy[i]),
                *(f"{v:.9f}" for v in ddy[i]),
                f"{grip[i]:.9f}",
            ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-processing DMP-specifico (derivate).")
    p.add_argument("--task", required=True, help="Nome primitiva (es. pick, place, pour).")
    p.add_argument("--index", type=int, default=None,
                   help="Processa solo la demo con questo indice. Se omesso, le processa tutte.")
    p.add_argument("--first-k", type=int, default=None,
                   help="Usa solo le prime K demo (in ordine alfabetico). Mutualmente esclusivo con --index.")
    p.add_argument("--in-root", type=Path, default=PREPROCESSED_ROOT,
                   help=f"Root delle demo gia' pre-processate (default: {PREPROCESSED_ROOT}).")
    p.add_argument("--out-root", type=Path, default=DMP_PREPROCESSED_ROOT,
                   help=f"Root di output (default: {DMP_PREPROCESSED_ROOT}).")
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

    print(f"DMP preprocessing su {len(inputs)} demo per '{args.task}'")
    print(f"  in : {args.in_root / args.task}")
    print(f"  out: {out_dir}")

    # q_ref dal sidecar prodotto da preprocess_demos: serve per ricomporre la
    # rotazione assoluta a inferenza (q_abs = q_ref * exp(r_rel)).
    try:
        q_ref = load_qref_sidecar(args.in_root / args.task)
        print(f"  q_ref : {q_ref.round(4).tolist()}")
    except FileNotFoundError:
        q_ref = np.array([0.0, 0.0, 0.0, 1.0], float)
        print("  [warn] qref.json non trovato: assumo q_ref=identita' "
              "(rotazioni assolute). Riproc. con preprocess_demos per "
              "abilitare la ricentratura.")

    Y_list, dY_list, ddY_list = [], [], []
    t_list, grip_list = [], []
    durations, n_per_demo = [], []
    y0_list, g_list = [], []
    demo_names = []
    dt_values = []

    for f in inputs:
        t, y, grip = load_processed_csv(f)
        dt = estimate_dt(t)
        dy, ddy = finite_diff_derivatives(y, dt)

        Y_list.append(y)
        dY_list.append(dy)
        ddY_list.append(ddy)
        t_list.append(t)
        grip_list.append(grip)
        durations.append(float(t[-1] - t[0]))
        n_per_demo.append(len(t))
        y0_list.append(y[0].copy())
        g_list.append(y[-1].copy())
        demo_names.append(f.name)
        dt_values.append(dt)

        if not args.no_inspection_csv:
            ins_path = out_dir / f.name.replace("processed_", "dmp_processed_")
            save_inspection_csv(ins_path, t, y, dy, ddy, grip)
            print(f"  {f.name} -> {ins_path.name}  "
                  f"(N={len(t)}, T={durations[-1]:.2f}s, dt={dt:.4f}s)")
        else:
            print(f"  {f.name}  (N={len(t)}, T={durations[-1]:.2f}s, dt={dt:.4f}s)")

    # dt unico (tutte le demo passano da common-preprocessing con stesso dt).
    dt_arr = np.asarray(dt_values, float)
    if not np.allclose(dt_arr, dt_arr[0], rtol=DT_RTOL, atol=1e-6):
        print(
            f"  [warn] dt non costante tra le demo: min={dt_arr.min():.6f}, "
            f"max={dt_arr.max():.6f}. Salvo dt_mean."
        )
    dt_out = float(dt_arr.mean())

    out_npz = out_dir / f"{args.task}_dmp_dataset.npz"
    np.savez(
        out_npz,
        # liste per-demo (object-arrays: lunghezze potenzialmente diverse)
        Y_list=np.array(Y_list, dtype=object),
        dY_list=np.array(dY_list, dtype=object),
        ddY_list=np.array(ddY_list, dtype=object),
        t_list=np.array(t_list, dtype=object),
        grip_list=np.array(grip_list, dtype=object),
        # endpoints per-demo (matrici regolari)
        y0_list=np.asarray(y0_list, dtype=float),     # (K, 6)
        g_list=np.asarray(g_list, dtype=float),       # (K, 6)
        # metadati
        dt=dt_out,
        T_list=np.asarray(durations, dtype=float),
        T_mean=float(np.mean(durations)),
        demo_lengths=np.asarray(n_per_demo, dtype=int),
        demo_files=np.asarray(demo_names),
        y_columns=np.asarray(Y_COLS),
        # quaternione di riferimento per la ricentratura della rotazione:
        # rx,ry,rz nelle Y_list sono log(q_ref^{-1} * q_abs).
        q_ref=np.asarray(q_ref, float),
    )
    print(f"\nDataset DMP salvato: {out_npz}")
    print(f"  K_demos={len(inputs)}  |  dt={dt_out:.4f}s  |  T_mean={np.mean(durations):.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
