"""Training DMP multi-demo con pesi medi (Section II-D del paper).

Carica il dataset prodotto da ``dmp_preprocess.py`` e per ognuna delle 6
dimensioni dello stato y = [x, y, z, rx, ry, rz] stima un vettore di pesi.

Metodi supportati:
  - ``mean-weights`` (default): fitta un DMP per ogni demo k, ottiene w_k,
    e media i pesi: W = mean_k(w_k). Funziona perche' il forcing target e'
    normalizzato per la scala demo-specifica (g_k - y0_k), quindi i w_k
    vivono nello stesso "spazio shape" indipendentemente dalla scala.
  - ``concat-ls``: stack di X_k e f_norm,k di tutte le demo e UNA sola
    closed-form LS per dimensione. Statisticamente piu' robusto, ma le
    demo lunghe pesano di piu'.

Iperparametri di default presi dalla Tabella I del paper:
  n_bfs in [50, 200], alpha_z in [12, 25], alpha_s in [1, 4], beta_z = alpha_z/4.

Salva: ``trained_models/<task>_dmp.npz``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parents[1]  # .../lfd_pipeline
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config.robot_config import (  # noqa: E402
    DMP_DEFAULT_ALPHA_S,
    DMP_DEFAULT_ALPHA_Z,
    DMP_DEFAULT_N_BFS,
    DMP_PREPROCESSED_ROOT,
    TRAINED_MODELS_ROOT,
)
from learning.dmp.dmp_model import DMP1D, make_basis  # noqa: E402

Y_COLS = ["x", "y", "z", "rx", "ry", "rz"]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_dataset(npz_path: Path):
    if not npz_path.is_file():
        raise SystemExit(f"Dataset DMP non trovato: {npz_path}\n"
                         "Esegui prima dmp_preprocess.py.")
    d = np.load(npz_path, allow_pickle=True)
    return d


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def fit_mean_weights(Y_list, dY_list, ddY_list, n_bfs, alpha_z, alpha_s, dt, T):
    """Fit per-demo, poi media dei pesi. Ritorna W (6, n_bfs)."""
    K = len(Y_list)
    W_per_demo = np.zeros((K, 6, n_bfs), dtype=float)
    scales = np.zeros((K, 6), dtype=float)
    degenerate = []

    for k in range(K):
        Y = np.asarray(Y_list[k], float)
        dY = np.asarray(dY_list[k], float)
        ddY = np.asarray(ddY_list[k], float)
        for d in range(6):
            dmp = DMP1D(n_bfs=n_bfs, alpha_z=alpha_z, alpha_s=alpha_s,
                        dt=dt, T=T)
            dmp.fit(Y[:, d], dY[:, d], ddY[:, d])
            W_per_demo[k, d, :] = dmp.w
            scales[k, d] = dmp._scale_demo
            if abs(Y[-1, d] - Y[0, d]) < 1e-6:
                degenerate.append((k, Y_COLS[d]))

    W = W_per_demo.mean(axis=0)  # (6, n_bfs)
    W_std = W_per_demo.std(axis=0)  # (6, n_bfs) - utile per diagnostica
    return W, W_per_demo, W_std, scales, degenerate


def fit_concat_ls(Y_list, dY_list, ddY_list, n_bfs, alpha_z, alpha_s, dt, T):
    """Multi-demo Locally-Weighted Regression per BF.

    Per ogni dimensione d e ogni BF i:
        W[d, i] = sum_k sum_t psi_i^k(t) * s^k(t) * f_norm^{k,d}(t)
                / (sum_k sum_t psi_i^k(t) * (s^k(t))^2 + eps)

    Equivale a fondere in un'unica regressione locale i contributi di tutte
    le demo, mantenendo i pesi limitati anche con BF strette.
    """
    from learning.dmp.dmp_model import psi_matrix
    proto = DMP1D(n_bfs=n_bfs, alpha_z=alpha_z, alpha_s=alpha_s, dt=dt, T=T)

    K = len(Y_list)
    scales = np.zeros((K, 6), dtype=float)
    degenerate = []

    num = np.zeros((6, n_bfs), dtype=float)
    den = np.zeros((n_bfs,), dtype=float)

    for k in range(K):
        Y = np.asarray(Y_list[k], float)
        dY = np.asarray(dY_list[k], float)
        ddY = np.asarray(ddY_list[k], float)
        N = len(Y)
        s = proto._phase(N)                              # (N,)
        Psi = psi_matrix(s, proto.c, proto.h)            # (N, n_bfs)
        s_col = s[:, None]
        den += (Psi * (s_col ** 2)).sum(axis=0)

        for d in range(6):
            scale = DMP1D._safe_scale(Y[-1, d] - Y[0, d])
            scales[k, d] = scale
            if abs(Y[-1, d] - Y[0, d]) < 1e-6:
                degenerate.append((k, Y_COLS[d]))
            f_target = (
                (proto.tau ** 2) * ddY[:, d]
                - proto.alpha_z * (proto.beta_z * (Y[-1, d] - Y[:, d])
                                   - proto.tau * dY[:, d])
            )
            f_norm = f_target / scale
            num[d] += (Psi * s_col * f_norm[:, None]).sum(axis=0)

    W = num / (den[None, :] + 1e-12)
    return W, scales, degenerate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training DMP multi-demo.")
    p.add_argument("--task", required=True, help="Nome primitiva (es. pick, place, pour).")
    p.add_argument("--in-root", type=Path, default=DMP_PREPROCESSED_ROOT)
    p.add_argument("--out-root", type=Path, default=TRAINED_MODELS_ROOT)
    p.add_argument("--n-bfs", type=int, default=DMP_DEFAULT_N_BFS,
                   help=f"Numero basis function (default: {DMP_DEFAULT_N_BFS}).")
    p.add_argument("--alpha-z", type=float, default=DMP_DEFAULT_ALPHA_Z,
                   help=f"alpha_z (default: {DMP_DEFAULT_ALPHA_Z}).")
    p.add_argument("--alpha-s", type=float, default=DMP_DEFAULT_ALPHA_S,
                   help=f"alpha_s (default: {DMP_DEFAULT_ALPHA_S}).")
    p.add_argument("--method", choices=["mean-weights", "concat-ls"],
                   default="mean-weights",
                   help="Strategia di combinazione multi-demo (default: mean-weights).")
    p.add_argument("--index", type=int, default=None,
                   help="Se specificato (1-based), allena il DMP usando solo "
                        "la dimostrazione con questo indice nel dataset "
                        "(le altre vengono ignorate).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    npz_in = args.in_root / args.task / f"{args.task}_dmp_dataset.npz"
    data = load_dataset(npz_in)

    Y_list = list(data["Y_list"])
    dY_list = list(data["dY_list"])
    ddY_list = list(data["ddY_list"])
    y0_list = data["y0_list"]
    g_list = data["g_list"]
    grip_list = list(data["grip_list"])
    dt = float(data["dt"])
    T_mean = float(data["T_mean"])
    demo_files = data["demo_files"]
    q_ref = (np.asarray(data["q_ref"], float)
             if "q_ref" in data.files
             else np.array([0.0, 0.0, 0.0, 1.0], float))
    K = len(Y_list)

    selected_index = None
    if args.index is not None:
        if not (1 <= args.index <= K):
            raise SystemExit(
                f"--index {args.index} fuori range: il dataset "
                f"contiene {K} dimostrazioni (indici validi: 1..{K})."
            )
        selected_index = args.index - 1
        sel = selected_index
        print(f"  [info] usando solo la demo #{args.index}: "
              f"{demo_files[sel]}")
        Y_list = [Y_list[sel]]
        dY_list = [dY_list[sel]]
        ddY_list = [ddY_list[sel]]
        grip_list = [grip_list[sel]]
        y0_list = y0_list[sel:sel + 1]
        g_list = g_list[sel:sel + 1]
        demo_files = demo_files[sel:sel + 1]
        # T_mean rimane quello del dataset originale (non lo ricalcoliamo
        # per mantenere coerenza con la fase usata in inference)
        K = 1

    print(f"DMP training '{args.task}' | method={args.method}")
    print(f"  dataset : {npz_in}")
    print(f"  K_demos : {K}  |  dt={dt:.4f}s  |  T_mean={T_mean:.3f}s")
    print(f"  hparams : n_bfs={args.n_bfs}, alpha_z={args.alpha_z}, "
          f"alpha_s={args.alpha_s}, beta_z={args.alpha_z/4.0}")

    if args.method == "mean-weights":
        W, W_per_demo, W_std, scales, degenerate = fit_mean_weights(
            Y_list, dY_list, ddY_list,
            n_bfs=args.n_bfs, alpha_z=args.alpha_z, alpha_s=args.alpha_s,
            dt=dt, T=T_mean,
        )
    else:
        W, scales, degenerate = fit_concat_ls(
            Y_list, dY_list, ddY_list,
            n_bfs=args.n_bfs, alpha_z=args.alpha_z, alpha_s=args.alpha_s,
            dt=dt, T=T_mean,
        )
        W_per_demo = None
        W_std = None

    if degenerate:
        print(f"  [warn] dimensioni degeneri (|g-y0|<1e-6): {degenerate}")

    # Diagnostica pesi
    print("  W stats per dimensione:")
    for d, name in enumerate(Y_COLS):
        line = (f"    {name:>4s}: mean={W[d].mean():+.3e}  "
                f"std={W[d].std():.3e}  "
                f"range=[{W[d].min():+.3e}, {W[d].max():+.3e}]")
        if W_std is not None:
            line += f"  | mean_std_across_demos={W_std[d].mean():.3e}"
        print(line)

    # Endpoint medi (utili come fallback in inference)
    y0_mean = y0_list.mean(axis=0)
    g_mean = g_list.mean(axis=0)

    # Profilo gripper medio sulla griglia di fase s in [0,1] (per inference).
    n_per_demo = [len(g) for g in grip_list]
    N_ref = int(round(float(np.mean(n_per_demo))))
    s_ref = np.linspace(0.0, 1.0, N_ref)
    grip_on_ref = np.stack([
        np.interp(s_ref, np.linspace(0.0, 1.0, len(g_k)),
                  np.asarray(g_k, dtype=float))
        for g_k in grip_list
    ], axis=0)
    grip_ref = grip_on_ref.mean(axis=0)

    # Basis condivise (utili a chi fa rollout senza ricalcolarle)
    C, H = make_basis(args.n_bfs, args.alpha_s, T=T_mean)

    out_dir = args.out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.task}_dmp.npz"

    save_dict = dict(
        # parametri DMP
        W=W,                     # (6, n_bfs)
        C=C,                     # (n_bfs,)
        H=H,                     # (n_bfs,)
        alpha_z=float(args.alpha_z),
        beta_z=float(args.alpha_z) / 4.0,
        alpha_s=float(args.alpha_s),
        tau=1.0,
        n_bfs=int(args.n_bfs),
        dt=dt,
        # endpoints fallback
        y0_mean=y0_mean,         # (6,)
        # profilo gripper su fase (per inference)
        s_ref=s_ref,
        grip_ref=grip_ref,
        g_mean=g_mean,           # (6,)
        y0_list=y0_list,         # (K, 6)  - per riferimento
        g_list=g_list,           # (K, 6)
        # metadati
        T_mean=T_mean,
        K_demos=K,
        method=args.method,
        y_columns=np.asarray(Y_COLS),
        demo_files=demo_files,
        scales=scales,           # (K, 6)
        # quaternione di riferimento per la ricomposizione della rotazione
        # assoluta a inferenza (q_abs = q_ref * exp(r_rel)).
        q_ref=q_ref,
    )
    if W_per_demo is not None:
        save_dict["W_per_demo"] = W_per_demo
        save_dict["W_std"] = W_std

    np.savez(out_path, **save_dict)
    print(f"\nModello DMP salvato: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
