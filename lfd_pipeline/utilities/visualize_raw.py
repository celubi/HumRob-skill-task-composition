"""Visualizzazione delle dimostrazioni RAW (cartella ``demonstrations/``).

Legge i CSV in ``demonstrations/<task>/<task>_NN.csv``
(header: ``t,x,y,z,qx,qy,qz,qw,gripper``) e produce 3 figure:

  1) traiettoria 3D in spazio cartesiano (una linea per demo) con marker
     di start (verde) e end (rosso). Le terne TCP sono campionate lungo
     la traiettoria e disegnate nella convenzione assi GRIPPER:
     x_g = rosso, y_g = verde, z_g = blu (z_g = direzione di approccio
     della pinza). L'orientamento e' ricostruito dai quaternioni del CSV
     (R = R_base^gripper).

  2) griglia 3x3: x(t), y(t), z(t) | qx(t), qy(t), qz(t) |
     qw(t), gripper(t), ||v||(t).

  3) sanity-check sui quaternioni: norma e prodotto interno consecutivo.

Differenze rispetto a ``visualize_preprocessed.py``:
  - pattern file: ``<task>_NN.csv`` invece di ``processed_<task>_NN.csv``;
  - niente colonne rx,ry,rz: l'orientamento si plotta dai quaternioni;
  - gripper in unita' SDK (~0..850), senza normalizzazione.

Uso:
    python3 visualize_raw.py --task pick
    python3 visualize_raw.py --task place --index 3
    python3 visualize_raw.py --task pour --first-k 4
    python3 visualize_raw.py --task place --save-dir /tmp/viz_raw
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parent  # .../lfd_pipeline
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config.robot_config import DEMO_ROOT  # noqa: E402

HEADER = ["t", "x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_csv(path: Path) -> dict:
    rows = []
    with open(path) as f:
        rdr = csv.DictReader(f)
        missing = [c for c in HEADER if c not in (rdr.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path}: colonne mancanti {missing}")
        for row in rdr:
            rows.append([float(row[k]) for k in HEADER])
    if not rows:
        raise SystemExit(f"{path}: CSV vuoto.")
    arr = np.asarray(rows, float)
    return {
        "name": path.name,
        "t": arr[:, 0],
        "xyz": arr[:, 1:4],
        "quat": arr[:, 4:8],   # qx, qy, qz, qw
        "grip": arr[:, 8],
    }


def collect_inputs(in_root: Path, task: str, index: int | None,
                   first_k: int | None, zfill: int) -> list[Path]:
    task_dir = in_root / task
    if not task_dir.is_dir():
        raise SystemExit(f"Cartella non trovata: {task_dir}")
    if index is not None and first_k is not None:
        raise SystemExit("Usa --index oppure --first-k, non entrambi.")
    if index is not None:
        f = task_dir / f"{task}_{str(index).zfill(zfill)}.csv"
        if not f.is_file():
            raise SystemExit(f"Demo non trovata: {f}")
        return [f]
    pattern = re.compile(rf"^{re.escape(task)}_(\d+)\.csv$")
    files = sorted(p for p in task_dir.iterdir() if pattern.match(p.name))
    if not files:
        raise SystemExit(f"Nessuna demo in {task_dir}")
    if first_k is not None:
        if first_k <= 0 or first_k > len(files):
            raise SystemExit(f"--first-k fuori range (1..{len(files)}).")
        files = files[:first_k]
    return files


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """Quaternione (qx,qy,qz,qw) -> matrice di rotazione 3x3.
    R = R_base^gripper, quindi le sue colonne sono x_g, y_g, z_g espressi
    in frame base."""
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


def _draw_gripper_frame(ax, p: np.ndarray, R: np.ndarray, length: float,
                        alpha: float = 1.0, lw: float = 1.2):
    """Disegna la terna del GRIPPER nel frame base.
    Convenzione: colonne di R = (x_g, y_g, z_g).
      x_g rosso  (asse di apertura/chiusura dita)
      y_g verde
      z_g blu    (direzione di approccio della pinza)
    """
    colors = ("r", "g", "b")
    for k in range(3):
        v = R[:, k] * length
        ax.plot([p[0], p[0] + v[0]],
                [p[1], p[1] + v[1]],
                [p[2], p[2] + v[2]],
                color=colors[k], lw=lw, alpha=alpha)


def plot_3d(demos: list[dict], task: str, save_path: Path | None,
            n_frames: int = 8, frame_len: float = 0.03):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab10")
    all_xyz = []
    for i, d in enumerate(demos):
        c = cmap(i % 10)
        xyz = d["xyz"]
        quat = d["quat"]
        all_xyz.append(xyz)
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "-", color=c,
                lw=1.2, label=d["name"])
        ax.scatter(*xyz[0], color="green", s=40, marker="o",
                   edgecolors="k", zorder=5)
        ax.scatter(*xyz[-1], color="red", s=40, marker="X",
                   edgecolors="k", zorder=5)

        # terne gripper campionate uniformemente lungo la traiettoria.
        # Start ed end sono SEMPRE inclusi (alpha pieno).
        if n_frames > 0 and frame_len > 0.0:
            n = len(xyz)
            if n_frames >= n:
                idxs = np.arange(n)
            else:
                idxs = np.linspace(0, n - 1, n_frames).round().astype(int)
                idxs = np.unique(np.concatenate([[0], idxs, [n - 1]]))
            for k in idxs:
                Rm = _quat_to_R(quat[k])
                is_endpoint = (k == 0 or k == n - 1)
                _draw_gripper_frame(ax, xyz[k], Rm,
                                    length=frame_len,
                                    alpha=1.0 if is_endpoint else 0.55,
                                    lw=1.6 if is_endpoint else 1.0)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"[{task}] RAW: traiettorie 3D + pose gripper (da quat)  "
                 f"x_g=R, y_g=G, z_g=B (z_g = approach); "
                 f"verde=start, rosso=end")
    ax.legend(fontsize=7, loc="best")
    # box aspect proporzionale al range dei dati per non distorcere le terne
    try:
        pts = np.vstack(all_xyz)
        ranges = pts.max(axis=0) - pts.min(axis=0)
        ranges = np.where(ranges < 1e-6, 1.0, ranges)
        ax.set_box_aspect(tuple(ranges))
    except Exception:
        pass
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"  saved: {save_path}")


def plot_signals(demos: list[dict], task: str, save_path: Path | None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(13, 8), sharex=True)
    cmap = plt.get_cmap("tab10")

    titles = [
        ["x [m]",  "y [m]",  "z [m]"],
        ["qx",     "qy",     "qz"],
        ["qw",     "gripper [SDK]", "||v|| [m/s]"],
    ]

    for i, d in enumerate(demos):
        c = cmap(i % 10)
        t = d["t"]
        xyz = d["xyz"]
        q = d["quat"]
        grip = d["grip"]

        # row 0: x, y, z
        for k in range(3):
            axes[0, k].plot(t, xyz[:, k], "-", color=c, lw=1.0,
                            label=d["name"] if k == 0 else None)
        # row 1: qx, qy, qz
        for k in range(3):
            axes[1, k].plot(t, q[:, k], "-", color=c, lw=1.0)
        # row 2: qw, gripper, ||v||
        axes[2, 0].plot(t, q[:, 3], "-", color=c, lw=1.0)
        axes[2, 1].plot(t, grip, "-", color=c, lw=1.0)
        if len(t) > 1:
            dt = np.diff(t)
            v = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / np.maximum(dt, 1e-12)
            axes[2, 2].plot(t[1:], v, "-", color=c, lw=1.0)

    for r in range(3):
        for k in range(3):
            axes[r, k].set_title(titles[r][k])
            axes[r, k].grid(True, alpha=0.3)
    for k in range(3):
        axes[2, k].set_xlabel("t [s]")
    axes[0, 0].legend(fontsize=7, loc="best")

    fig.suptitle(f"[{task}] RAW: segnali")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"  saved: {save_path}")


def plot_quat_check(demos: list[dict], task: str, save_path: Path | None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cmap = plt.get_cmap("tab10")

    for i, d in enumerate(demos):
        c = cmap(i % 10)
        q = d["quat"]
        t = d["t"]
        norms = np.linalg.norm(q, axis=1)
        axes[0].plot(t, norms, "-", color=c, lw=1.0,
                     label=d["name"])
        if len(q) > 1:
            dots = np.einsum("ij,ij->i", q[:-1], q[1:])
            axes[1].plot(t[1:], dots, "-", color=c, lw=1.0)

    axes[0].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[0].set_title("||q(t)||  (atteso ~1)")
    axes[0].set_xlabel("t [s]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")

    axes[1].axhline(0.0, color="k", ls="--", lw=0.8)
    axes[1].set_title("q(t)·q(t+1)  (raw: puo' cambiare segno tra emisferi)")
    axes[1].set_xlabel("t [s]")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"[{task}] RAW: sanity-check quaternioni")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"  saved: {save_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(demos: list[dict]) -> None:
    print(f"\n{'name':<32s} {'N':>5s} {'T[s]':>8s} {'dt[s]':>8s} "
          f"{'start xyz [m]':>34s} -> {'end xyz [m]':>34s}")
    for d in demos:
        t = d["t"]
        xyz = d["xyz"]
        dt = float(np.mean(np.diff(t))) if len(t) > 1 else 0.0
        s = "[" + ", ".join(f"{v:+.3f}" for v in xyz[0]) + "]"
        e = "[" + ", ".join(f"{v:+.3f}" for v in xyz[-1]) + "]"
        print(f"{d['name']:<32s} {len(t):>5d} {t[-1]-t[0]:>8.3f} "
              f"{dt:>8.4f} {s:>34s} -> {e:>34s}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualizza le demo RAW.")
    p.add_argument("--task", required=True,
                   help="Nome task (es. pick, place, pour).")
    p.add_argument("--index", type=int, default=None,
                   help="Visualizza solo questa demo.")
    p.add_argument("--first-k", type=int, default=None,
                   help="Visualizza solo le prime K demo.")
    p.add_argument("--in-root", type=Path, default=DEMO_ROOT,
                   help=f"Root delle demo raw (default: {DEMO_ROOT}).")
    p.add_argument("--zfill", type=int, default=2,
                   help="Zero-padding indice (default: 2).")
    p.add_argument("--save-dir", type=Path, default=None,
                   help="Se fornito, salva i PNG in questa cartella invece di "
                        "mostrare le finestre interattive.")
    p.add_argument("--no-3d", action="store_true",
                   help="Salta il plot 3D.")
    p.add_argument("--n-frames", type=int, default=8,
                   help="Numero di terne gripper campionate per demo nel plot "
                        "3D (start ed end sempre inclusi). 0 disattiva. "
                        "Default: 8.")
    p.add_argument("--frame-len", type=float, default=0.03,
                   help="Lunghezza degli assi delle terne gripper [m] "
                        "(default: 0.03).")
    p.add_argument("--no-signals", action="store_true",
                   help="Salta il plot dei segnali.")
    p.add_argument("--no-quat-check", action="store_true",
                   help="Salta il sanity-check sui quaternioni.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    files = collect_inputs(args.in_root, args.task, args.index,
                           args.first_k, args.zfill)
    print(f"Trovate {len(files)} demo in {args.in_root / args.task}:")
    for f in files:
        print(f"  - {f.name}")

    demos = [load_csv(f) for f in files]
    print_summary(demos)

    save_dir = args.save_dir
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    if save_dir is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    if not args.no_3d:
        out = save_dir / f"{args.task}_raw_3d.png" if save_dir else None
        plot_3d(demos, args.task, out,
                n_frames=args.n_frames, frame_len=args.frame_len)
    if not args.no_signals:
        out = save_dir / f"{args.task}_raw_signals.png" if save_dir else None
        plot_signals(demos, args.task, out)
    if not args.no_quat_check:
        out = save_dir / f"{args.task}_raw_quat_check.png" if save_dir else None
        plot_quat_check(demos, args.task, out)

    if save_dir is None:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
