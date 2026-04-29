"""Training Behavioral Cloning (Section II-C del paper).

Carica il dataset prodotto da ``bc_preprocess.py`` e addestra una policy
pi_theta : s_t -> a_t come MLP a 4 hidden layers, MSE + Adam (Tabella I).

Il modello viene salvato in ``trained_models/<task>_bc.pt`` con dentro:
  - ``state_dict`` dei pesi MLP
  - configurazione architetturale (hidden, activation, dim. I/O)
  - statistiche di normalizzazione (i_mean, i_std, o_mean, o_std) come buffer
  - meta-informazioni utili a inference (dt, T_mean, s0/sT medi, demo info)

Uso:
    python3 bc_train.py --task pick
    python3 bc_train.py --task pour --hidden 256,256,128,64 --activation tanh
    python3 bc_train.py --task pick --epochs 450 --lr 5e-3 --val-split 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parents[1]  # .../lfd_pipeline
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config.robot_config import (  # noqa: E402
    BC_DEFAULT_ACTIVATION,
    BC_DEFAULT_BATCH_SIZE,
    BC_DEFAULT_EPOCHS,
    BC_DEFAULT_HIDDEN,
    BC_DEFAULT_LR,
    BC_DEFAULT_RANDOM_STATE,
    BC_DEFAULT_WEIGHT_DECAY,
    BC_PREPROCESSED_ROOT,
    TRAINED_MODELS_ROOT,
)
from learning.bc.bc_model import BCConfig, BCPolicy, parse_hidden  # noqa: E402


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_dataset(npz_path: Path):
    if not npz_path.is_file():
        raise SystemExit(f"Dataset BC non trovato: {npz_path}\n"
                         "Esegui prima bc_preprocess.py.")
    return np.load(npz_path, allow_pickle=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training BC (MLP, MSE + Adam).")
    p.add_argument("--task", required=True, help="Nome primitiva (es. pick, place, pour).")
    p.add_argument("--in-root", type=Path, default=BC_PREPROCESSED_ROOT,
                   help=f"Root dataset BC (default: {BC_PREPROCESSED_ROOT}).")
    p.add_argument("--out-root", type=Path, default=TRAINED_MODELS_ROOT,
                   help=f"Root modelli (default: {TRAINED_MODELS_ROOT}).")
    p.add_argument("--hidden", type=str,
                   default=",".join(str(h) for h in BC_DEFAULT_HIDDEN),
                   help="Lista neuroni hidden, es. '128,128,128,64'.")
    p.add_argument("--activation", choices=["relu", "tanh"],
                   default=BC_DEFAULT_ACTIVATION)
    p.add_argument("--lr", type=float, default=BC_DEFAULT_LR)
    p.add_argument("--epochs", type=int, default=BC_DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BC_DEFAULT_BATCH_SIZE)
    p.add_argument("--weight-decay", type=float, default=BC_DEFAULT_WEIGHT_DECAY)
    p.add_argument("--val-split", type=float, default=0.0,
                   help="Frazione di validation (default 0 = no split).")
    p.add_argument("--noise-std", type=float, default=0.0,
                   help="Std del rumore gaussiano applicato allo stato "
                        "normalizzato durante il training (data augmentation "
                        "contro il covariate shift). 0 = disattivato. "
                        "Tipico: 0.01-0.05.")
    p.add_argument("--device", default="cpu",
                   help="Device torch (cpu, cuda, cuda:0...).")
    p.add_argument("--random-state", type=int, default=BC_DEFAULT_RANDOM_STATE)
    p.add_argument("--log-every", type=int, default=25,
                   help="Stampa la loss ogni N epoche (0 = nessun log).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    in_npz = args.in_root / args.task / f"{args.task}_bc_dataset.npz"
    d = load_dataset(in_npz)

    I_norm = d["I_norm"].astype(np.float32)
    O_norm = d["O_norm"].astype(np.float32)
    i_mean = d["i_mean"].astype(np.float32)
    i_std = d["i_std"].astype(np.float32)
    o_mean = d["o_mean"].astype(np.float32)
    o_std = d["o_std"].astype(np.float32)
    state_dim = int(d["state_dim"])
    action_dim = int(d["action_dim"])
    dt = float(d["dt"])
    T_mean = float(d["T_mean"])

    hidden = parse_hidden(args.hidden)
    cfg = BCConfig(state_dim=state_dim, action_dim=action_dim,
                   hidden=hidden, activation=args.activation)
    model = BCPolicy(cfg)
    model.set_norm_stats(i_mean, i_std, o_mean, o_std)

    print(f"BC training su task='{args.task}'")
    print(f"  dataset : {in_npz}")
    print(f"  M_pairs : {len(I_norm)}  |  state_dim={state_dim}  action_dim={action_dim}"
          f"  (goal-conditioned={'yes' if state_dim == 12 else 'no'})")
    print(f"  hidden  : {hidden}  |  activation={args.activation}")
    print(f"  optim   : Adam(lr={args.lr}, wd={args.weight_decay})  |  "
          f"epochs={args.epochs}  batch={args.batch_size}")
    print(f"  noise   : std={args.noise_std}  "
          f"({'OFF' if args.noise_std <= 0 else 'ON, applicato in z-score space'})")
    print(f"  device  : {args.device}")

    history = model.fit(
        I_norm=I_norm,
        O_norm=O_norm,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        random_state=args.random_state,
        log_every=args.log_every,
        val_split=args.val_split,
        noise_std=args.noise_std,
    )

    # Endpoints medi (utili a inference per scegliere s0 / target)
    s0_mean = d["s0_list"].mean(axis=0).astype(np.float32)
    sT_mean = d["sT_list"].mean(axis=0).astype(np.float32)
    N_ref = int(round(T_mean / dt)) + 1

    # Profilo gripper medio sulla griglia di fase di riferimento
    # (coerente con DMP/GMM: replicato a inference, fuori dalla policy MLP).
    s_ref = np.linspace(0.0, 1.0, N_ref)
    grip_list = d["grip_list"]
    grip_on_ref = []
    for g_k in grip_list:
        g_k = np.asarray(g_k, float)
        s_k = np.linspace(0.0, 1.0, len(g_k))
        grip_on_ref.append(np.interp(s_ref, s_k, g_k))
    grip_ref = (np.stack(grip_on_ref, axis=0).mean(axis=0).astype(np.float32)
                if grip_on_ref else np.zeros(N_ref, dtype=np.float32))

    q_ref = (np.asarray(d["q_ref"], float)
             if "q_ref" in d.files
             else np.array([0.0, 0.0, 0.0, 1.0], float))

    extra = {
        "task": args.task,
        "goal_conditioned": True,
        "dt": dt,
        "T_mean": T_mean,
        "N_ref": N_ref,
        "s0_mean": s0_mean,
        "sT_mean": sT_mean,
        "s_ref": s_ref.astype(np.float32),
        "grip_ref": grip_ref,
        # quaternione di riferimento per la ricentratura della rotazione
        # (q_abs = q_ref * exp(rotvec_rel) a inferenza).
        "q_ref": q_ref.astype(np.float32),
        "s_columns": list(d["s_columns"]),
        "s_tilde_columns": list(d["s_tilde_columns"]) if "s_tilde_columns" in d.files else None,
        "a_columns": list(d["a_columns"]),
        "demo_files": list(d["demo_files"]),
        "demo_lengths": d["demo_lengths"].tolist(),
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "hyperparams": {
            "hidden": list(hidden),
            "activation": args.activation,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "val_split": args.val_split,
            "random_state": args.random_state,
            "noise_std": args.noise_std,
        },
    }

    out_path = args.out_root / f"{args.task}_bc.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path, extra=extra)

    final_train = history["train_loss"][-1]
    print(f"\nModello BC salvato: {out_path}")
    print(f"  final train MSE (norm space) = {final_train:.6e}")
    if not np.isnan(history["val_loss"][-1]):
        print(f"  final   val MSE (norm space) = {history['val_loss'][-1]:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
