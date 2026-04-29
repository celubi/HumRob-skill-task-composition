"""Pre-processing model-specific per Goal-Conditioned Behavioral Cloning.

Le demo arrivano dalla common-preprocessing su griglia temporale uniforme
(t, x, y, z, qx, qy, qz, qw, rx, ry, rz, gripper). Per il GC-BC formuliamo
lo skill learning come regressione supervisionata stato-esteso -> azione,
dove lo stato include esplicitamente l'informazione sul goal:

    s_t  = [p_t, r_t]                       in R^6   (TCP + angle-axis)
    g    = s_T   (endpoint della demo)        in R^6
    s~_t = [s_t, g - s_t]                   in R^12  (stato goal-conditioned)
    a_t  = s_{t+1} - s_t                     in R^6  (incremento one-step)

Rispetto al BC "vanilla" (Section II-C del paper), il goal-conditioning
elimina la convergenza al sT_mean delle demo: la policy puo' modulare il
comportamento in funzione del goal richiesto a inference dalla vision
layer (Sez II-F, task-parameter retrieval).

La codifica come delta-to-go (g - s_t) anziche' goal assoluto e' invariante
per traslazione del goal e generalizza meglio con poche demo.

Coerentemente con DMP e GMM-GMR, il gripper non entra nello stato/azione: e'
una grandezza discreta (open/close) che viene replicata a inference dal
segnale demo medio.

Per ogni demo viene salvato:
  - un CSV di ispezione con t, s~_t, a_t (un sample in meno della demo)
  - un'entry nel dataset consolidato ``<task>_bc_dataset.npz``

Uso:
    python3 bc_preprocess.py --task pick
    python3 bc_preprocess.py --task pour --index 3
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
    BC_PREPROCESSED_ROOT,
    PREPROCESSED_ROOT,
)
from preprocessing.preprocess_demos import load_qref_sidecar  # noqa: E402

# colonne attese nel CSV di common-preprocessing
COMMON_HEADER = ["t", "x", "y", "z", "qx", "qy", "qz", "qw",
                 "rx", "ry", "rz", "gripper"]

# componenti dello stato 6D di base e dello stato goal-conditioned 12D
S_COLS = ["x", "y", "z", "rx", "ry", "rz"]
DG_COLS = [f"dg_{c}" for c in S_COLS]
S_TILDE_COLS = S_COLS + DG_COLS
A_COLS = [f"d{c}" for c in S_COLS]

INSPECTION_HEADER = ["t"] + S_TILDE_COLS + A_COLS

# tolleranza relativa per verificare l'uniformita' della time-grid
DT_RTOL = 1e-3


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_processed_csv(path: Path):
    """Ritorna (t, s, gripper) con s di shape (N, 6) = [x,y,z,rx,ry,rz]."""
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
    s = np.column_stack([arr[:, 1:4], arr[:, 8:11]])
    grip = arr[:, 11]
    return t, s, grip


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


def build_state_action(s: np.ndarray):
    """Costruisce (S_tilde, A_out) per una singola demo (goal-conditioned).

    s : (N, 6) traiettoria demo gia' processata.

    g  = s[-1]                              (goal = endpoint della demo)
    S_tilde[t] = [s[t], g - s[t]]   per t = 0..N-2   -> shape (N-1, 12)
    A_out[t]   = s[t+1] - s[t]      per t = 0..N-2   -> shape (N-1, 6)
    """
    g = s[-1]
    S_in = s[:-1]
    Dg = g[None, :] - S_in
    S_tilde = np.concatenate([S_in, Dg], axis=1).astype(float)
    A_out = np.diff(s, axis=0).astype(float)
    return S_tilde, A_out


def save_inspection_csv(out_path: Path, t: np.ndarray, S_in: np.ndarray,
                        A_out: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(INSPECTION_HEADER)
        # t va da 0 a N-2 (un sample in meno della demo originale)
        for i in range(len(S_in)):
            w.writerow([
                f"{t[i]:.9f}",
                *(f"{v:.9f}" for v in S_in[i]),
                *(f"{v:.9f}" for v in A_out[i]),
            ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-processing BC-specifico (state-action pairs).")
    p.add_argument("--task", required=True, help="Nome primitiva (es. pick, place, pour).")
    p.add_argument("--index", type=int, default=None,
                   help="Processa solo la demo con questo indice. Se omesso, le processa tutte.")
    p.add_argument("--first-k", type=int, default=None,
                   help="Usa solo le prime K demo. Mutualmente esclusivo con --index.")
    p.add_argument("--in-root", type=Path, default=PREPROCESSED_ROOT,
                   help=f"Root delle demo gia' pre-processate (default: {PREPROCESSED_ROOT}).")
    p.add_argument("--out-root", type=Path, default=BC_PREPROCESSED_ROOT,
                   help=f"Root di output (default: {BC_PREPROCESSED_ROOT}).")
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

    print(f"BC preprocessing su {len(inputs)} demo per '{args.task}'")
    print(f"  in : {args.in_root / args.task}")
    print(f"  out: {out_dir}")

    try:
        q_ref = load_qref_sidecar(args.in_root / args.task)
        print(f"  q_ref : {q_ref.round(4).tolist()}")
    except FileNotFoundError:
        q_ref = np.array([0.0, 0.0, 0.0, 1.0], float)
        print("  [warn] qref.json non trovato: assumo q_ref=identita'.")

    S_list, A_list = [], []
    grip_list, t_list = [], []
    s0_list, sT_list = [], []
    durations, n_per_demo = [], []
    demo_names, dt_values = [], []

    for f in inputs:
        t, s, grip = load_processed_csv(f)
        dt = estimate_dt(t)
        S_in, A_out = build_state_action(s)

        S_list.append(S_in)
        A_list.append(A_out)
        grip_list.append(grip)
        t_list.append(t)
        s0_list.append(s[0].copy())
        sT_list.append(s[-1].copy())
        durations.append(float(t[-1] - t[0]))
        n_per_demo.append(len(t))
        demo_names.append(f.name)
        dt_values.append(dt)

        if not args.no_inspection_csv:
            ins_path = out_dir / f.name.replace("processed_", "bc_processed_")
            save_inspection_csv(ins_path, t, S_in, A_out)
            print(f"  {f.name} -> {ins_path.name}  "
                  f"(N={len(t)}, T={durations[-1]:.2f}s, dt={dt:.4f}s)")
        else:
            print(f"  {f.name}  (N={len(t)}, T={durations[-1]:.2f}s, dt={dt:.4f}s)")

    # dt unico (verifica coerenza tra demo)
    dt_arr = np.asarray(dt_values, float)
    if not np.allclose(dt_arr, dt_arr[0], rtol=DT_RTOL, atol=1e-6):
        print(
            f"  [warn] dt non costante tra le demo: min={dt_arr.min():.6f}, "
            f"max={dt_arr.max():.6f}. Salvo dt_mean."
        )
    dt_out = float(dt_arr.mean())

    # Stack: I, O in R^{M x 6} (eq. 5 del paper)
    I = np.vstack(S_list).astype(float)
    O = np.vstack(A_list).astype(float)

    # z-score per-colonna; std=1 dove std ~ 0 per evitare divisioni instabili
    i_mean = I.mean(axis=0)
    i_std = I.std(axis=0)
    i_std = np.where(i_std < 1e-9, 1.0, i_std)
    o_mean = O.mean(axis=0)
    o_std = O.std(axis=0)
    o_std = np.where(o_std < 1e-9, 1.0, o_std)

    I_norm = (I - i_mean) / i_std
    O_norm = (O - o_mean) / o_std

    out_npz = out_dir / f"{args.task}_bc_dataset.npz"
    np.savez(
        out_npz,
        # dataset di training (raw + normalizzato)
        I=I,
        O=O,
        I_norm=I_norm,
        O_norm=O_norm,
        # statistiche di normalizzazione (riusate a inference)
        i_mean=i_mean,
        i_std=i_std,
        o_mean=o_mean,
        o_std=o_std,
        # endpoints e segnale gripper per-demo (object-arrays: lunghezze diverse)
        s0_list=np.asarray(s0_list, dtype=float),     # (K, 6)
        sT_list=np.asarray(sT_list, dtype=float),     # (K, 6)
        grip_list=np.array(grip_list, dtype=object),
        t_list=np.array(t_list, dtype=object),
        # metadati
        dt=dt_out,
        T_list=np.asarray(durations, dtype=float),
        T_mean=float(np.mean(durations)),
        demo_lengths=np.asarray(n_per_demo, dtype=int),
        demo_files=np.asarray(demo_names),
        s_columns=np.asarray(S_COLS),
        s_tilde_columns=np.asarray(S_TILDE_COLS),
        a_columns=np.asarray(A_COLS),
        state_dim=int(I.shape[1]),
        action_dim=int(O.shape[1]),
        goal_conditioned=True,
        # quaternione di riferimento per la ricentratura della rotazione:
        # le componenti rx,ry,rz dello stato sono log(q_ref^{-1} * q_abs).
        q_ref=np.asarray(q_ref, float),
    )
    print(f"\nDataset BC salvato: {out_npz}")
    print(f"  K_demos={len(inputs)}  |  M_pairs={len(I)}  |  "
          f"dt={dt_out:.4f}s  |  T_mean={np.mean(durations):.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
