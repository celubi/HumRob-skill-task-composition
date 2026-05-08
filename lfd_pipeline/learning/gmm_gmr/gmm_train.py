"""Training di un GMM su [s | x, y, z, rx, ry, rz] (Section II-E del paper).

Carica il dataset prodotto da ``gmm_preprocess.py`` e fa il fit di una
``sklearn.mixture.GaussianMixture`` con covarianza piena. Per default
applica una **z-score per-dimensione** (raccomandata in letteratura, vedi
Calinon, ref. [18] del paper) per equalizzare la pesatura di posizione e
orientazione nell'EM. La normalizzazione \u00e8 disattivabile con ``--no-normalize``.

Salva ``trained_models/<task>_gmm.npz`` con i parametri della GMM, le
statistiche di normalizzazione (per la GMR al rollout), e le meta-informazioni
richieste dall'inferenza (T_mean, dt, s_ref, grip_ref, ...).

Uso:
    # fit standard con K=8 (default da config) e z-score
    python3 gmm_train.py --task pick

    # ablation senza normalizzazione, K specifico
    python3 gmm_train.py --task pick --n-components 6 --no-normalize

    # esplorazione del numero di componenti (elbow del log-likelihood)
    python3 gmm_train.py --task pick --elbow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parents[1]  # .../lfd_pipeline
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config.robot_config import (  # noqa: E402
    DEFAULT_PREPROCESS_DT,
    GMM_DEFAULT_K,
    GMM_DEFAULT_MAX_ITER,
    GMM_DEFAULT_RANDOM_STATE,
    GMM_DEFAULT_REG_COVAR,
    GMM_DEFAULT_TOL,
    GMM_PREPROCESSED_ROOT,
    TRAINED_MODELS_ROOT,
)


# ---------------------------------------------------------------------------
# Normalizzazione
# ---------------------------------------------------------------------------
def compute_zscore(Z: np.ndarray):
    """Ritorna (mean, std) per-colonna; std=1 dove std \u2248 0."""
    mean = Z.mean(axis=0)
    std = Z.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return mean, std


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def fit_gmm(Z: np.ndarray, k: int, reg_covar: float, max_iter: int,
            tol: float, random_state: int) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        reg_covar=reg_covar,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        init_params="kmeans",
    )
    gmm.fit(Z)
    return gmm


def report_fit(gmm: GaussianMixture, Z: np.ndarray, k: int) -> dict:
    n = len(Z)
    logL = float(gmm.score(Z) * n)
    bic = float(gmm.bic(Z))
    aic = float(gmm.aic(Z))
    return {
        "K": k,
        "logL": logL,
        "BIC": bic,
        "AIC": aic,
        "n_iter": int(gmm.n_iter_),
        "converged": bool(gmm.converged_),
    }


def elbow_search(Z: np.ndarray, args) -> None:
    print(f"\nElbow search su K \u2208 [{args.elbow_min}, {args.elbow_max}]")
    print(f"{'K':>3} | {'logL':>14} | {'BIC':>14} | {'AIC':>14} | {'iter':>5} | conv")
    print("-" * 70)
    for k in range(args.elbow_min, args.elbow_max + 1):
        gmm = fit_gmm(Z, k, args.reg_covar, args.max_iter, args.tol, args.random_state)
        r = report_fit(gmm, Z, k)
        print(f"{r['K']:>3} | {r['logL']:>14.2f} | {r['BIC']:>14.2f} | "
              f"{r['AIC']:>14.2f} | {r['n_iter']:>5} | {r['converged']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training GMM-GMR.")
    p.add_argument("--task", required=True, help="Nome primitiva (es. pick, place, pour).")
    p.add_argument("--in-root", type=Path, default=GMM_PREPROCESSED_ROOT,
                   help=f"Root degli output di gmm_preprocess.py (default: {GMM_PREPROCESSED_ROOT}).")
    p.add_argument("--out-root", type=Path, default=TRAINED_MODELS_ROOT,
                   help=f"Cartella per i modelli (default: {TRAINED_MODELS_ROOT}).")
    p.add_argument("--n-components", type=int, default=GMM_DEFAULT_K,
                   help=f"Numero di componenti K (default: {GMM_DEFAULT_K}).")
    p.add_argument("--reg-covar", type=float, default=GMM_DEFAULT_REG_COVAR,
                   help=f"Regolarizzazione delle covarianze (default: {GMM_DEFAULT_REG_COVAR}).")
    p.add_argument("--max-iter", type=int, default=GMM_DEFAULT_MAX_ITER,
                   help=f"Max iterazioni EM (default: {GMM_DEFAULT_MAX_ITER}).")
    p.add_argument("--tol", type=float, default=GMM_DEFAULT_TOL,
                   help=f"Tolleranza convergenza EM (default: {GMM_DEFAULT_TOL}).")
    p.add_argument("--random-state", type=int, default=GMM_DEFAULT_RANDOM_STATE,
                   help=f"Seed (default: {GMM_DEFAULT_RANDOM_STATE}).")
    p.add_argument("--no-normalize", action="store_true",
                   help="Disattiva la z-score per-dimensione (usa metri+rad nativi).")
    p.add_argument("--dt", type=float, default=DEFAULT_PREPROCESS_DT,
                   help=f"dt salvato come metadato per il rollout (default: {DEFAULT_PREPROCESS_DT}).")
    # modalit\u00e0 elbow
    p.add_argument("--elbow", action="store_true",
                   help="Esegue solo elbow-search e stampa la tabella, senza salvare.")
    p.add_argument("--elbow-min", type=int, default=3, help="K minimo per elbow (default: 3).")
    p.add_argument("--elbow-max", type=int, default=10, help="K massimo per elbow (default: 10).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = args.in_root / args.task / f"{args.task}_gmm_dataset.npz"
    if not dataset_path.is_file():
        raise SystemExit(f"Dataset GMM non trovato: {dataset_path}\n"
                         f"Esegui prima:\n  python3 gmm_preprocess.py --task {args.task}")

    data = np.load(dataset_path, allow_pickle=True)
    Z = data["Z_concat"]            # (M, 7)  [s, x, y, z, rx, ry, rz]
    s_ref = data["s_ref"]
    grip_ref = data["grip_ref"]
    T_mean = float(data["T_mean"])
    y_columns = list(data["y_columns"])

    print(f"Dataset: {dataset_path}")
    print(f"  Z shape: {Z.shape}   (colonne: s + {y_columns})")
    print(f"  T_mean : {T_mean:.3f}s   |   demo: {len(data['demo_lengths'])}")

    # Normalizzazione (z-score)
    if args.no_normalize:
        Z_fit = Z
        mean = np.zeros(Z.shape[1])
        std = np.ones(Z.shape[1])
        print("Normalizzazione: disattivata (z-score=off).")
    else:
        mean, std = compute_zscore(Z)
        Z_fit = (Z - mean) / std
        print("Normalizzazione: z-score per-dimensione (attiva).")
        with np.printoptions(precision=4, suppress=True):
            print(f"  mean: {mean}")
            print(f"  std : {std}")

    # Elbow mode: stampa e basta.
    if args.elbow:
        elbow_search(Z_fit, args)
        return 0

    # Fit standard
    gmm = fit_gmm(Z_fit, args.n_components, args.reg_covar,
                  args.max_iter, args.tol, args.random_state)
    rep = report_fit(gmm, Z_fit, args.n_components)
    print("\nFit completato:")
    for k, v in rep.items():
        print(f"  {k}: {v}")

    out_dir = args.out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.task}_gmm.npz"
    np.savez(
        out_path,
        # Parametri GMM
        weights=gmm.weights_,
        means=gmm.means_,
        covariances=gmm.covariances_,
        precisions_cholesky=gmm.precisions_cholesky_,
        # Convenzioni indici per la GMR
        input_idx=np.array([0], dtype=int),
        output_idx=np.arange(1, Z.shape[1], dtype=int),
        # Normalizzazione (su tutte le colonne, inclusa s)
        norm_mean=mean,
        norm_std=std,
        normalized=np.array(not args.no_normalize),
        # Meta-dati per il rollout
        dt=float(args.dt),
        T_mean=T_mean,
        s_ref=s_ref,
        grip_ref=grip_ref,
        y_columns=np.asarray(y_columns),
        # Iperparametri / training info
        K=int(args.n_components),
        reg_covar=float(args.reg_covar),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        random_state=int(args.random_state),
        logL=rep["logL"],
        BIC=rep["BIC"],
        AIC=rep["AIC"],
        n_iter=rep["n_iter"],
        converged=rep["converged"],
    )
    print(f"\nModello salvato: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
