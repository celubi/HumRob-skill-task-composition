"""Confronto visivo demo vs traiettorie generate (BC/DMP/GMM-GMR).

Per ogni demo selezionata mostra:
  1) un plot 3D con la traiettoria dimostrata (nera) e quelle generate dai
     modelli (un colore per metodo), sovrapposte per evidenziare deviazioni
     spaziali; start (verde) ed end (rosso) della demo come marker;
  2) una griglia 3x3 dei segnali nel tempo:
         x(t)  y(t)  z(t)
         rx(t)  ry(t)  rz(t)
         gripper(t)  ||v||(t)  vuoto
     dove rx/ry/rz sono il rotvec ASSOLUTO (log-map del quaternione del CSV),
     ricostruito sia per la demo sia per i generati per confrontare le
     rotazioni sullo stesso spazio.

Uso:
    python3 visualize_comparison.py --task pick
    python3 visualize_comparison.py --task pour --index 3
    python3 visualize_comparison.py --task place --first-k 4 --method dmp
    python3 visualize_comparison.py --task pour --index 1 --save-dir /tmp/viz
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

from config.robot_config import PREPROCESSED_ROOT  # noqa: E402

# Riusa la costante di output di reproduce_demos
EVALUATION_ROOT = _PKG_ROOT / "evaluation_results"

METHODS_ALL = ("bc", "dmp", "gmm")
METHOD_COLORS = {"bc": "tab:blue", "dmp": "tab:orange", "gmm": "tab:green"}

DEMO_HEADER = ["t", "x", "y", "z", "qx", "qy", "qz", "qw",
               "rx", "ry", "rz", "gripper"]
GEN_HEADER = ["t", "x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]


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


def load_demo(path: Path) -> dict:
    a = _load_csv(path, DEMO_HEADER)
    return {
        "name": path.name,
        "t": a[:, 0], "xyz": a[:, 1:4],
        "quat": a[:, 4:8],          # qx,qy,qz,qw assoluto (per il plot 3D)
        "rotvec": a[:, 8:11],       # rx,ry,rz assoluto (preprocessato)
        "grip": a[:, 11],
    }


def load_generated(path: Path) -> dict:
    a = _load_csv(path, GEN_HEADER)
    return {
        "name": path.name,
        "t": a[:, 0], "xyz": a[:, 1:4],
        "quat": a[:, 4:8], "grip": a[:, 8],
    }


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------
def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Quaternione (qx,qy,qz,qw) -> rotvec assoluto (axis*angle, in [-pi, pi])."""
    q = np.asarray(q, float)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.zeros(3)
    q = q / n
    qw = float(np.clip(q[3], -1.0, 1.0))
    theta = 2.0 * math.acos(qw)
    sin_h = math.sqrt(max(0.0, 1.0 - qw * qw))
    if sin_h < 1e-12:
        return np.zeros(3)
    return q[:3] / sin_h * theta


def _quat_array_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """(N,4) -> (N,3). Garantisce continuita' temporale del rotvec."""
    out = np.zeros((len(quat), 3), float)
    for i in range(len(quat)):
        out[i] = _quat_to_rotvec(quat[i])
    # continuity: scegli il ramo (r vs (1 - 2pi/||r||)*r) piu' vicino al precedente
    for i in range(1, len(out)):
        r = out[i]
        norm = float(np.linalg.norm(r))
        if norm < 1e-9:
            continue
        r_alt = (1.0 - 2.0 * np.pi / norm) * r
        if np.linalg.norm(r_alt - out[i - 1]) < np.linalg.norm(r - out[i - 1]):
            out[i] = r_alt
    return out


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """Quaternione (qx,qy,qz,qw) -> matrice di rotazione 3x3."""
    qx, qy, qz, qw = q
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw),     s * (qx * qz + qy * qw)],
        [s * (qx * qy + qz * qw),     1 - s * (qx * qx + qz * qz), s * (qy * qz - qx * qw)],
        [s * (qx * qz - qy * qw),     s * (qy * qz + qx * qw),     1 - s * (qx * qx + qy * qy)],
    ], float)


def _draw_frame(ax, p: np.ndarray, R: np.ndarray, length: float,
                alpha: float = 1.0, lw: float = 1.2) -> None:
    """Disegna una terna TCP (x rosso, y verde, z blu) di lunghezza `length`."""
    colors = ("r", "g", "b")
    for k in range(3):
        v = R[:, k] * length
        ax.plot([p[0], p[0] + v[0]],
                [p[1], p[1] + v[1]],
                [p[2], p[2] + v[2]],
                color=colors[k], lw=lw, alpha=alpha)


def _frame_indices(n: int, n_frames: int) -> np.ndarray:
    """Indici uniformemente campionati lungo una traiettoria di n campioni.
    Start e end sono SEMPRE inclusi."""
    if n_frames <= 0 or n <= 0:
        return np.array([], dtype=int)
    if n_frames >= n:
        return np.arange(n)
    idxs = np.linspace(0, n - 1, n_frames).round().astype(int)
    return np.unique(np.concatenate([[0], idxs, [n - 1]]))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_3d(demo: dict, gen_by_method: dict[str, dict],
            task: str, idx_str: str, save_path: Path | None,
            n_frames: int = 6, frame_len: float = 0.03) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    # margini stretti -> il box 3D occupa quasi tutta la figura
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.10)

    xyz_d = demo["xyz"]
    quat_d = demo["quat"]
    ax.plot(xyz_d[:, 0], xyz_d[:, 1], xyz_d[:, 2],
            "-", color="black", lw=1.6, label="demo")
    ax.scatter(*xyz_d[0], color="green", s=50, marker="o", edgecolors="k",
               zorder=5, label="start (demo)")
    ax.scatter(*xyz_d[-1], color="red", s=50, marker="X", edgecolors="k",
               zorder=5, label="goal (demo)")

    # Terne TCP della demo (alpha pieno, lw piu' spesso)
    if n_frames > 0 and frame_len > 0.0:
        for k in _frame_indices(len(xyz_d), n_frames):
            _draw_frame(ax, xyz_d[k], _quat_to_R(quat_d[k]),
                        length=frame_len, alpha=1.0, lw=1.5)

    all_xyz = [xyz_d]
    for method, g in gen_by_method.items():
        c = METHOD_COLORS.get(method, "tab:purple")
        xyz = g["xyz"]
        quat = g["quat"]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                "--", color=c, lw=1.4, label=method.upper())
        ax.scatter(*xyz[-1], color=c, s=30, marker="s", edgecolors="k",
                   zorder=5)
        # Terne TCP del generato (alpha ridotto per non sovrapporsi al demo)
        if n_frames > 0 and frame_len > 0.0:
            for k in _frame_indices(len(xyz), n_frames):
                _draw_frame(ax, xyz[k], _quat_to_R(quat[k]),
                            length=frame_len, alpha=0.55, lw=1.0)
        all_xyz.append(xyz)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"[{task} #{idx_str}] traiettoria 3D: demo vs generate  "
                 "(terne TCP: x=R, y=G, z=B; demo full alpha, gen 0.55)")
    ax.legend(fontsize=11, loc="upper center",
              bbox_to_anchor=(0.5, -0.04), ncol=4)
    try:
        pts = np.vstack(all_xyz)
        ranges = pts.max(axis=0) - pts.min(axis=0)
        ranges = np.where(ranges < 1e-6, 1.0, ranges)
        # evita aspect troppo piatti: nessun asse < 60% del piu' grande,
        # cosi' i tick non si schiacciano lungo le direzioni corte
        ranges = np.maximum(ranges, 0.6 * ranges.max())
        ax.set_box_aspect(tuple(ranges))
    except Exception:
        pass
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"  saved: {save_path}")


def plot_signals(demo: dict, gen_by_method: dict[str, dict],
                 task: str, idx_str: str, save_path: Path | None) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=False,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.35})

    titles = [
        ["x [m]", "y [m]", "z [m]"],
        ["rx [rad]", "ry [rad]", "rz [rad]"],
        ["gripper", "||v|| [m/s]", ""],
    ]

    def _plot_curve(ax, t, y, color, ls, lw, label):
        ax.plot(t, y, ls, color=color, lw=lw, label=label)

    # demo: rotvec preso DIRETTAMENTE dal CSV preprocessato (assoluto)
    rv_demo = demo["rotvec"]
    t_d = demo["t"]
    xyz_d = demo["xyz"]
    grip_d = demo["grip"]
    for k in range(3):
        _plot_curve(axes[0, k], t_d, xyz_d[:, k], "black", "-", 1.4,
                    "demo" if k == 0 else None)
    for k in range(3):
        _plot_curve(axes[1, k], t_d, rv_demo[:, k], "black", "-", 1.4, None)
    _plot_curve(axes[2, 0], t_d, grip_d, "black", "-", 1.4, None)
    if len(t_d) > 1:
        dt_d = np.diff(t_d)
        v_d = np.linalg.norm(np.diff(xyz_d, axis=0), axis=1) / np.maximum(dt_d, 1e-12)
        _plot_curve(axes[2, 1], t_d[1:], v_d, "black", "-", 1.4, None)

    # generated: rotvec assoluto calcolato dal quaternione (stesso spazio della demo)
    for method, g in gen_by_method.items():
        c = METHOD_COLORS.get(method, "tab:purple")
        rv = _quat_array_to_rotvec(g["quat"])
        t = g["t"]
        xyz = g["xyz"]
        grip = g["grip"]
        for k in range(3):
            _plot_curve(axes[0, k], t, xyz[:, k], c, "--", 1.2,
                        method.upper() if k == 0 else None)
        for k in range(3):
            _plot_curve(axes[1, k], t, rv[:, k], c, "--", 1.2, None)
        # gripper: nei rollout puo' essere NaN (es. pour scripts), evita errori
        if not np.all(np.isnan(grip)):
            _plot_curve(axes[2, 0], t, grip, c, "--", 1.2, None)
        if len(t) > 1:
            dt = np.diff(t)
            v = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / np.maximum(dt, 1e-12)
            _plot_curve(axes[2, 1], t[1:], v, c, "--", 1.2, None)

    for r in range(3):
        for k in range(3):
            axes[r, k].set_title(titles[r][k])
            axes[r, k].grid(True, alpha=0.3)
            axes[r, k].set_xlabel("t [s]")
    axes[2, 2].axis("off")

    fig.suptitle(f"[{task} #{idx_str}] segnali: demo (nero) vs generate "
                 "(rotazioni nello spazio rotvec assoluto)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, fontsize=12, loc="lower center",
                   ncol=max(1, len(handles)),
                   bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93,
                        bottom=0.09, hspace=0.55, wspace=0.3)
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"  saved: {save_path}")


# ---------------------------------------------------------------------------
# Selezione input
# ---------------------------------------------------------------------------
def collect_demo_indices(in_root: Path, task: str, index: int | None,
                         first_k: int | None) -> list[str]:
    """Ritorna gli indici (stringhe zero-padded) delle demo da visualizzare."""
    task_dir = in_root / task
    if not task_dir.is_dir():
        raise SystemExit(f"Cartella demo non trovata: {task_dir}")
    if index is not None and first_k is not None:
        raise SystemExit("Usa --index oppure --first-k, non entrambi.")
    pat = re.compile(rf"^processed_{re.escape(task)}_(\d+)\.csv$")
    files = sorted(p for p in task_dir.iterdir() if pat.match(p.name))
    if not files:
        raise SystemExit(f"Nessuna demo in {task_dir}")
    if index is not None:
        idx_str_target = pat.match(files[0].name).group(1)
        # reuse zfill from filenames
        zfill = len(idx_str_target)
        idx_str = str(index).zfill(zfill)
        target = task_dir / f"processed_{task}_{idx_str}.csv"
        if not target.is_file():
            raise SystemExit(f"Demo non trovata: {target}")
        return [idx_str]
    indices = [pat.match(f.name).group(1) for f in files]
    if first_k is not None:
        if first_k <= 0 or first_k > len(indices):
            raise SystemExit(f"--first-k fuori range (1..{len(indices)}).")
        indices = indices[:first_k]
    return indices


def find_generated(eval_root: Path, task: str, method: str,
                   idx_str: str) -> Path | None:
    p = eval_root / task / method / f"generated_{task}_{idx_str}.csv"
    return p if p.is_file() else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Confronto visivo demo vs traiettorie generate.")
    p.add_argument("--task", required=True,
                   help="Nome primitiva (es. pick, place, pour).")
    p.add_argument("--methods", nargs="+", choices=METHODS_ALL,
                   default=list(METHODS_ALL),
                   help=f"Metodi da sovrapporre (default: {' '.join(METHODS_ALL)}).")
    p.add_argument("--method", choices=METHODS_ALL, default=None,
                   help="Shortcut: visualizza solo questo metodo "
                        "(equivalente a --methods <method>).")
    p.add_argument("--index", type=int, default=None,
                   help="Visualizza solo questa demo.")
    p.add_argument("--first-k", type=int, default=None,
                   help="Visualizza solo le prime K demo.")
    p.add_argument("--in-root", type=Path, default=PREPROCESSED_ROOT,
                   help=f"Cartella demo pre-processate (default: {PREPROCESSED_ROOT}).")
    p.add_argument("--eval-root", type=Path, default=EVALUATION_ROOT,
                   help=f"Cartella traiettorie generate (default: {EVALUATION_ROOT}).")
    p.add_argument("--save-dir", type=Path, default=None,
                   help="Se fornito, salva i PNG in questa cartella invece "
                        "di mostrare le finestre interattive.")
    p.add_argument("--no-3d", action="store_true",
                   help="Salta il plot 3D.")
    p.add_argument("--no-signals", action="store_true",
                   help="Salta il plot dei segnali per-asse.")
    p.add_argument("--n-frames", type=int, default=6,
                   help="Numero di terne TCP campionate per traiettoria nel "
                        "plot 3D (start ed end sempre inclusi). 0 disattiva. "
                        "Default: 6.")
    p.add_argument("--frame-len", type=float, default=0.02,
                   help="Lunghezza degli assi delle terne TCP [m] (default: 0.03).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    methods = [args.method] if args.method is not None else list(args.methods)

    indices = collect_demo_indices(args.in_root, args.task,
                                   args.index, args.first_k)
    print(f"[viz] task    : {args.task}")
    print(f"[viz] metodi  : {' '.join(methods)}")
    print(f"[viz] {len(indices)} demo da visualizzare: {indices}")

    save_dir = args.save_dir
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    if save_dir is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    for idx_str in indices:
        demo_path = args.in_root / args.task / f"processed_{args.task}_{idx_str}.csv"
        if not demo_path.is_file():
            print(f"[viz] [warn] demo mancante: {demo_path} -- skip.")
            continue
        demo = load_demo(demo_path)

        gen_by_method: dict[str, dict] = {}
        for m in methods:
            p = find_generated(args.eval_root, args.task, m, idx_str)
            if p is None:
                print(f"[viz] [warn] generato {m} #{idx_str} mancante "
                      f"({args.eval_root / args.task / m}); skip.")
                continue
            gen_by_method[m] = load_generated(p)

        if not gen_by_method:
            print(f"[viz] [warn] nessun generato disponibile per #{idx_str}; skip.")
            continue

        print(f"\n[viz] demo #{idx_str}: N={len(demo['t'])} T={demo['t'][-1]:.2f}s")
        for m, g in gen_by_method.items():
            print(f"      {m.upper()}: N={len(g['t'])} T={g['t'][-1]:.2f}s")

        if not args.no_3d:
            out = (save_dir / f"{args.task}_{idx_str}_3d.png"
                   if save_dir else None)
            plot_3d(demo, gen_by_method, args.task, idx_str, out,
                    n_frames=args.n_frames, frame_len=args.frame_len)
        if not args.no_signals:
            out = (save_dir / f"{args.task}_{idx_str}_signals.png"
                   if save_dir else None)
            plot_signals(demo, gen_by_method,
                         args.task, idx_str, out)

    if save_dir is None:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
