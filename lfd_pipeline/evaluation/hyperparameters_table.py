"""Tabella iperparametri di training (analoga alla Table I del paper ETFA2026)
costruita leggendo i valori effettivamente usati in lfd_pipeline/config/robot_config.py.

Stampa una tabella su console e opzionalmente la salva in LaTeX (booktabs +
multirow per la colonna del metodo).

Uso:
    python3 hyperparameters_table.py
    python3 hyperparameters_table.py --save-latex paper/hp_table.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config import robot_config as cfg  # noqa: E402


# Ogni riga e' (label_console, label_latex, value_str)
def _bc_rows() -> list[tuple[str, str, str]]:
    hidden = cfg.BC_DEFAULT_HIDDEN
    act = cfg.BC_DEFAULT_ACTIVATION
    act_pretty = {"relu": "ReLU", "tanh": "tanh"}.get(act, act)
    return [
        ("MLP layers",          "MLP layers",                 f"N = {len(hidden)}"),
        ("Neurons per layer",   "Neurons per layer",          " / ".join(str(n) for n in hidden)),
        ("Activation function", "Activation function",        act_pretty),
        ("Learning rate",       "Learning rate",              f"{cfg.BC_DEFAULT_LR:g}"),
        ("Training epochs",     "Training epochs",            str(cfg.BC_DEFAULT_EPOCHS)),
        ("Batch size",          "Batch size",                 str(cfg.BC_DEFAULT_BATCH_SIZE)),
        ("Weight decay",        "Weight decay",               f"{cfg.BC_DEFAULT_WEIGHT_DECAY:g}"),
    ]


def _dmp_rows() -> list[tuple[str, str, str]]:
    az = cfg.DMP_DEFAULT_ALPHA_Z
    return [
        ("Temporal scaling tau",    r"Temporal scaling $\tau$",        f"{cfg.DMP_DEFAULT_TAU:g}"),
        ("alpha_z",                 r"$\alpha_z$",                     f"{az:g}"),
        ("beta_z = alpha_z/4",      r"$\beta_z = \alpha_z / 4$",       f"{az / 4:g}"),
        ("alpha_s",                 r"$\alpha_s$",                     f"{cfg.DMP_DEFAULT_ALPHA_S:g}"),
        ("Basis functions N",       r"Basis functions $N$",            str(cfg.DMP_DEFAULT_N_BFS)),
    ]


def _gmm_rows() -> list[tuple[str, str, str]]:
    return [
        ("Mixture components K",    r"Mixture components $K$",         str(cfg.GMM_DEFAULT_K)),
        ("reg_covar",               r"reg\_covar",                     f"{cfg.GMM_DEFAULT_REG_COVAR:g}"),
        ("Max EM iterations",       "Max iterations of EM algorithm",  str(cfg.GMM_DEFAULT_MAX_ITER)),
        ("Convergence tolerance",   "Convergence tolerance",           f"{cfg.GMM_DEFAULT_TOL:g}"),
    ]


def _all_blocks() -> list[tuple[str, list[tuple[str, str, str]]]]:
    return [
        ("BC",      _bc_rows()),
        ("DMP",     _dmp_rows()),
        ("GMM-GMR", _gmm_rows()),
    ]


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------
def print_console(blocks) -> None:
    print()
    head = f"{'Method':<10}{'Hyperparameter':<28}{'Value':<20}"
    print(head)
    print("-" * len(head))
    for i, (method, rows) in enumerate(blocks):
        for j, (lbl, _lbl_tex, val) in enumerate(rows):
            print(f"{method if j == 0 else '':<10}{lbl:<28}{val:<20}")
        if i < len(blocks) - 1:
            print()


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
def latex_table(blocks) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Motion-encoders hyperparameters used.}",
        r"\label{tab:hyperparameters}",
        r"\centering",
        r"\begin{tabular}{l l l}",
        r"\toprule",
        r" & Hyperparameter & Value \\",
        r"\midrule",
    ]
    for i, (method, rows) in enumerate(blocks):
        n = len(rows)
        for j, (_lbl_console, lbl_tex, val) in enumerate(rows):
            method_cell = (r"\multirow{" + str(n) + r"}{*}{\textbf{" + method + r"}}"
                           if j == 0 else "")
            lines.append(f"{method_cell} & {lbl_tex} & {val} \\\\")
        if i < len(blocks) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tabella iperparametri di training effettivi (Table I).")
    p.add_argument("--save-latex", type=Path, default=None,
                   help="Se fornito, salva la tabella LaTeX (booktabs+multirow).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    blocks = _all_blocks()
    print_console(blocks)
    if args.save_latex is not None:
        args.save_latex.parent.mkdir(parents=True, exist_ok=True)
        content = (
            "% Auto-generated by hyperparameters_table.py\n"
            "% Requires: \\usepackage{booktabs, multirow}\n\n"
            + latex_table(blocks) + "\n"
        )
        args.save_latex.write_text(content)
        print(f"\n[hp] tabella LaTeX salvata in {args.save_latex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
