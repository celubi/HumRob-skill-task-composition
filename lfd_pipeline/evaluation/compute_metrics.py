"""Calcola le metriche di riproduzione del paper ETFA2026 e stampa una
tabella comparativa BC/DMP/GMM-GMR per ogni task (Tables II-IV).

Per ogni demo i:
    demo  : preprocessed_demonstrations/<task>/processed_<task>_<idx>.csv
    gen_m : evaluation_results/<task>/<m>/generated_<task>_<idx>.csv

Metriche calcolate (definizioni: paper Sec. II.D):
    RMSE_pos [m]      eq.(10), errore RMS punto-a-punto su (x,y,z)
    RMSE_ori [deg]    eq.(10), RMS dell'angolo geodetico fra quaternioni
    d_H     [m]       eq.(12), Hausdorff simmetrico fra le due nuvole 3D
    J     [m^2/s^5]   eq.(13), jerk al quadrato integrato sul generato
                      (il paper riporta [m/s^3], ma dimensionalmente
                      l'integrale di ||x'''||^2 ha unita' m^2/s^5)

TRA_pos/TRA_ori (eq. 11) NON sono incluse: nei rollout di reproduce_demos.py
gli endpoint sono clampati sul demo (goal passato come parametro), quindi
TRA collassa a rumore numerico (~5e-7) per tutti i metodi e non discrimina.

Allineamento temporale: i CSV demo e gen hanno la stessa dt ma durate
diverse (es. demo ~720 campioni, gen ~386). RMSE_* viene quindi calcolato
dopo normalizzazione di fase s = (t - t0) / (T - t0) in [0,1] e
ricampionamento su un grigliato comune (lineare per xyz, SLERP per i
quaternioni). TRA usa direttamente l'ultimo campione (start/end matched
per costruzione, sec. II.E del paper). d_H e J sono geometrici/dinamici e
non richiedono allineamento punto-punto.

Aggregazione: la tabella stampa la **media** sulle demo disponibili (e
opzionalmente la std). Il numero di demo aggregate per cella e' riportato
in coda alla tabella.

Uso:
    python3 compute_metrics.py
    python3 compute_metrics.py --task pick
    python3 compute_metrics.py --methods bc dmp --save-csv /tmp/metrics.csv
    python3 compute_metrics.py --per-demo
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
_PKG_ROOT = _THIS_DIR.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config.robot_config import PREPROCESSED_ROOT  # noqa: E402

EVALUATION_ROOT = _PKG_ROOT / "evaluation_results"

METHODS_ALL = ("bc", "dmp", "gmm")
METHOD_LABELS = {"bc": "BC", "dmp": "DMP", "gmm": "GMM-GMR"}

DEMO_HEADER = ["t", "x", "y", "z", "qx", "qy", "qz", "qw",
               "rx", "ry", "rz", "gripper"]
GEN_HEADER = ["t", "x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]

METRIC_ORDER = ("RMSE_pos", "RMSE_ori", "d_H", "J")
METRIC_UNITS = {
    "RMSE_pos": "m",
    "RMSE_ori": "deg",
    "d_H": "m",
    "J": "m^2/s^5",
}
# Tutte le metriche scelte sono "lower is better": il migliore e' il min.
METRIC_BEST_IS_MIN = {m: True for m in METRIC_ORDER}

# LaTeX symbol per metrica (label di riga della tabella)
LATEX_METRIC = {
    "RMSE_pos": r"RMSE\textsubscript{pos}",
    "RMSE_ori": r"RMSE\textsubscript{ori}",
    "d_H":      r"$d_H$",
    "J":        r"$J$",
}
LATEX_UNIT = {
    "m":       "[m]",
    "deg":     "[deg]",
    "m^2/s^5": r"[m\textsuperscript{2}/s\textsuperscript{5}]",
}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _load_csv(path: Path, required_cols: list[str]) -> np.ndarray:
    rows = []
    with open(path) as f:
        rdr = csv.DictReader(f)
        missing = [c for c in required_cols if c not in (rdr.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path}: colonne mancanti {missing}")
        for row in rdr:
            rows.append([float(row[k]) for k in required_cols])
    if not rows:
        raise SystemExit(f"{path}: CSV vuoto.")
    return np.asarray(rows, float)


def load_traj(path: Path, header: list[str]) -> dict:
    a = _load_csv(path, header)
    return {"t": a[:, 0], "xyz": a[:, 1:4], "quat": a[:, 4:8]}


# ---------------------------------------------------------------------------
# Quaternion utils
# ---------------------------------------------------------------------------
def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return q / n


def _quat_make_continuous(quat: np.ndarray) -> np.ndarray:
    """Hemisphere flip: q(t).q(t+1) > 0 (paper eq. (3))."""
    q = quat.copy()
    for i in range(1, len(q)):
        if float(q[i] @ q[i - 1]) < 0.0:
            q[i] = -q[i]
    return q


def _slerp_pair(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        out = q0 + u * (q1 - q0)
        return out / max(float(np.linalg.norm(out)), 1e-12)
    th = math.acos(dot)
    s0 = math.sin((1.0 - u) * th) / math.sin(th)
    s1 = math.sin(u * th) / math.sin(th)
    return s0 * q0 + s1 * q1


def _resample_quat(s_src: np.ndarray, q_src: np.ndarray,
                   s_query: np.ndarray) -> np.ndarray:
    out = np.zeros((len(s_query), 4), float)
    for i, s in enumerate(s_query):
        if s <= s_src[0]:
            out[i] = q_src[0]
        elif s >= s_src[-1]:
            out[i] = q_src[-1]
        else:
            j = int(np.searchsorted(s_src, s)) - 1
            s0, s1 = s_src[j], s_src[j + 1]
            u = (s - s0) / max(s1 - s0, 1e-12)
            out[i] = _slerp_pair(q_src[j], q_src[j + 1], float(u))
    return _quat_normalize(out)


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angolo geodetico fra quaternioni unitari (riga per riga). Robust to
    hemisphere thanks to abs()."""
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _phase(t: np.ndarray) -> np.ndarray:
    t0, t1 = float(t[0]), float(t[-1])
    if t1 - t0 < 1e-12:
        return np.linspace(0.0, 1.0, len(t))
    return (t - t0) / (t1 - t0)


def _hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """Hausdorff simmetrico fra due nuvole (M,3) e (K,3)."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def _jerk_integral(t: np.ndarray, xyz: np.ndarray) -> float:
    """∫ ||d^3 x / dt^3||^2 dt via differenze finite."""
    if len(t) < 4:
        return float("nan")
    dt = np.diff(t)
    v = np.diff(xyz, axis=0) / np.maximum(dt[:, None], 1e-12)
    a = np.diff(v, axis=0) / np.maximum(dt[1:, None], 1e-12)
    j = np.diff(a, axis=0) / np.maximum(dt[2:, None], 1e-12)
    j_sq = np.sum(j * j, axis=1)
    dt_eff = float(np.mean(dt))
    return float(np.sum(j_sq) * dt_eff)


def compute_pair_metrics(demo: dict, gen: dict,
                         n_resample: int | None = None) -> dict:
    s_d = _phase(demo["t"])
    s_g = _phase(gen["t"])
    qd = _quat_make_continuous(_quat_normalize(demo["quat"]))
    qg = _quat_make_continuous(_quat_normalize(gen["quat"]))

    if n_resample is None:
        n_resample = max(len(demo["t"]), len(gen["t"]))
    s_q = np.linspace(0.0, 1.0, n_resample)

    xyz_d = np.stack([np.interp(s_q, s_d, demo["xyz"][:, k]) for k in range(3)],
                     axis=1)
    xyz_g = np.stack([np.interp(s_q, s_g, gen["xyz"][:, k]) for k in range(3)],
                     axis=1)
    qd_r = _resample_quat(s_d, qd, s_q)
    qg_r = _resample_quat(s_g, qg, s_q)

    err_pos = np.linalg.norm(xyz_d - xyz_g, axis=1)
    rmse_pos = float(np.sqrt(np.mean(err_pos ** 2)))

    err_ori_deg = _quat_angle_deg(qd_r, qg_r)
    rmse_ori = float(np.sqrt(np.mean(err_ori_deg ** 2)))

    d_h = _hausdorff(demo["xyz"], gen["xyz"])
    j_int = _jerk_integral(gen["t"], gen["xyz"])

    return {
        "RMSE_pos": rmse_pos,
        "RMSE_ori": rmse_ori,
        "d_H": d_h,
        "J": j_int,
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def _discover_tasks(eval_root: Path) -> list[str]:
    if not eval_root.is_dir():
        raise SystemExit(f"Cartella non trovata: {eval_root}")
    return sorted(p.name for p in eval_root.iterdir() if p.is_dir())


def _demo_indices(in_root: Path, task: str) -> list[str]:
    pat = re.compile(rf"^processed_{re.escape(task)}_(\d+)\.csv$")
    d = in_root / task
    if not d.is_dir():
        return []
    return sorted(pat.match(p.name).group(1)
                  for p in d.iterdir() if pat.match(p.name))


def _gen_path(eval_root: Path, task: str, method: str, idx: str) -> Path:
    return eval_root / task / method / f"generated_{task}_{idx}.csv"


# ---------------------------------------------------------------------------
# Aggregation + printing
# ---------------------------------------------------------------------------
def _aggregate(per_demo: list[dict]) -> dict:
    if not per_demo:
        return {m: float("nan") for m in METRIC_ORDER}
    out = {}
    for m in METRIC_ORDER:
        vals = np.array([d[m] for d in per_demo if np.isfinite(d[m])], float)
        out[m] = float(vals.mean()) if vals.size else float("nan")
        out[m + "_std"] = float(vals.std(ddof=0)) if vals.size > 1 else 0.0
        out[m + "_n"] = int(vals.size)
    return out


def _fmt(v: float, metric: str) -> str:
    if not np.isfinite(v):
        return "n/a"
    if metric == "J":
        return f"{v:.2f}"
    if metric == "RMSE_ori":
        return f"{v:.2f}"
    return f"{v:.4f}"


def _best_method(metric: str, methods: list[str],
                 agg_by_method: dict[str, dict]) -> str | None:
    """Ritorna il nome del metodo con il valore migliore per `metric`.
    Lower-is-better per tutte le metriche correnti."""
    best_m, best_v = None, None
    for m in methods:
        v = agg_by_method[m].get(metric, float("nan"))
        if not np.isfinite(v):
            continue
        if best_v is None or v < best_v:
            best_v, best_m = v, m
    return best_m


def _print_task_table(task: str, methods: list[str],
                      agg_by_method: dict[str, dict],
                      include_std: bool) -> None:
    print()
    print(f"=== TASK: {task.upper()} ===")
    header = f"{'Metric':<14} {'Unit':<8}" + \
             "".join(f"{METHOD_LABELS[m]:>14}" for m in methods)
    print(header)
    print("-" * len(header))
    for metric in METRIC_ORDER:
        best = _best_method(metric, methods, agg_by_method)
        row = f"{metric:<14} {'[' + METRIC_UNITS[metric] + ']':<8}"
        for m in methods:
            v = agg_by_method[m].get(metric, float("nan"))
            if include_std:
                std = agg_by_method[m].get(metric + "_std", 0.0)
                cell = f"{_fmt(v, metric)}±{_fmt(std, metric)}"
            else:
                cell = _fmt(v, metric)
            if m == best:
                cell = "*" + cell + "*"
            row += f"{cell:>14}"
        print(row)
    counts = ", ".join(
        f"{METHOD_LABELS[m]}={agg_by_method[m].get('RMSE_pos_n', 0)}"
        for m in methods)
    print(f"  (n demo aggregate: {counts}; * = best per riga)")


# ---------------------------------------------------------------------------
# LaTeX export (formato booktabs, in stile Tables II-IV del paper)
# ---------------------------------------------------------------------------
def _latex_table_for_task(task: str, methods: list[str],
                          agg_by_method: dict[str, dict],
                          include_std: bool) -> str:
    cols = "l c " + " ".join(["c"] * len(methods))
    label = f"tab:{task}_metrics"
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{" + task.upper() + r" -- Reproduction metrics.}",
        r"\label{" + label + r"}",
        r"\centering",
        r"\begin{tabular}{" + cols + "}",
        r"\toprule",
        " & Unit & " + " & ".join(METHOD_LABELS[m] for m in methods) + r" \\",
        r"\midrule",
    ]
    for metric in METRIC_ORDER:
        best = _best_method(metric, methods, agg_by_method)
        cells = []
        for m in methods:
            v = agg_by_method[m].get(metric, float("nan"))
            s = _fmt(v, metric)
            if include_std:
                std = agg_by_method[m].get(metric + "_std", 0.0)
                s = f"{s} $\\pm$ {_fmt(std, metric)}"
            if m == best:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        unit = LATEX_UNIT.get(METRIC_UNITS[metric], "[" + METRIC_UNITS[metric] + "]")
        lines.append(LATEX_METRIC[metric] + " & " + unit + " & "
                     + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def write_latex(path: Path, tasks: list[str], methods: list[str],
                agg_by_task: dict[str, dict[str, dict]],
                include_std: bool) -> None:
    parts = [
        r"% Auto-generated by compute_metrics.py",
        r"% Requires: \usepackage{booktabs}",
        "",
    ]
    for task in tasks:
        if task in agg_by_task:
            parts.append(_latex_table_for_task(
                task, methods, agg_by_task[task], include_std))
            parts.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tabella metriche riproduzione (paper ETFA2026).")
    p.add_argument("--task", default=None,
                   help="Limita a una singola primitiva (default: tutte "
                        "quelle in evaluation_results/).")
    p.add_argument("--methods", nargs="+", choices=METHODS_ALL,
                   default=list(METHODS_ALL),
                   help=f"Metodi da confrontare (default: {' '.join(METHODS_ALL)}).")
    p.add_argument("--in-root", type=Path, default=PREPROCESSED_ROOT,
                   help=f"Root demo preprocessate (default: {PREPROCESSED_ROOT}).")
    p.add_argument("--eval-root", type=Path, default=EVALUATION_ROOT,
                   help=f"Root traiettorie generate (default: {EVALUATION_ROOT}).")
    p.add_argument("--save-csv", type=Path, default=None,
                   help="Se fornito, salva l'aggregato (mean/std/n) in CSV.")
    p.add_argument("--save-latex", type=Path, default=None,
                   help="Se fornito, salva le tabelle LaTeX (booktabs, una "
                        "tabella per task) nel file indicato.")
    p.add_argument("--per-demo", action="store_true",
                   help="Salva anche un CSV per-demo accanto a --save-csv "
                        "(suffisso _per_demo).")
    p.add_argument("--include-std", action="store_true",
                   help="Stampa media+/-std nelle celle (default: solo media).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tasks = ([args.task] if args.task is not None
             else _discover_tasks(args.eval_root))

    print(f"[metrics] in-root  : {args.in_root}")
    print(f"[metrics] eval-root: {args.eval_root}")
    print(f"[metrics] tasks    : {' '.join(tasks)}")
    print(f"[metrics] methods  : {' '.join(args.methods)}")

    per_demo_rows: list[dict] = []   # {task, method, idx, metric, value}
    agg_rows: list[dict] = []        # {task, method, metric, mean, std, n}
    agg_by_task: dict[str, dict[str, dict]] = {}
    tasks_with_data: list[str] = []

    for task in tasks:
        demo_idxs = _demo_indices(args.in_root, task)
        if not demo_idxs:
            print(f"[metrics] [warn] nessuna demo in {args.in_root / task}; skip.")
            continue
        agg_by_method: dict[str, dict] = {}
        for method in args.methods:
            per: list[dict] = []
            for idx in demo_idxs:
                demo_p = args.in_root / task / f"processed_{task}_{idx}.csv"
                gen_p = _gen_path(args.eval_root, task, method, idx)
                if not demo_p.is_file() or not gen_p.is_file():
                    continue
                demo = load_traj(demo_p, DEMO_HEADER)
                gen = load_traj(gen_p, GEN_HEADER)
                m = compute_pair_metrics(demo, gen)
                per.append(m)
                if args.per_demo:
                    for k, v in m.items():
                        per_demo_rows.append({
                            "task": task, "method": method, "idx": idx,
                            "metric": k, "value": v,
                        })
            agg = _aggregate(per)
            agg_by_method[method] = agg
            for k in METRIC_ORDER:
                agg_rows.append({
                    "task": task, "method": method, "metric": k,
                    "mean": agg[k], "std": agg[k + "_std"], "n": agg[k + "_n"],
                })
        agg_by_task[task] = agg_by_method
        tasks_with_data.append(task)
        _print_task_table(task, args.methods, agg_by_method, args.include_std)

    if args.save_latex is not None:
        write_latex(args.save_latex, tasks_with_data, args.methods,
                    agg_by_task, args.include_std)
        print(f"\n[metrics] tabelle LaTeX salvate in {args.save_latex}")

    if args.save_csv is not None:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["task", "method", "metric",
                                              "mean", "std", "n"])
            w.writeheader()
            w.writerows(agg_rows)
        print(f"\n[metrics] aggregato salvato in {args.save_csv}")
        if args.per_demo:
            pd_path = args.save_csv.with_name(
                args.save_csv.stem + "_per_demo" + args.save_csv.suffix)
            with open(pd_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["task", "method", "idx",
                                                  "metric", "value"])
                w.writeheader()
                w.writerows(per_demo_rows)
            print(f"[metrics] per-demo salvato in {pd_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
