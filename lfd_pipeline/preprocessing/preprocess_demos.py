"""Pre-processing comune (Section II-B del paper) per le demo della pipeline LfD.

Per ogni file CSV grezzo in ``demonstrations/<task>/<task>_NN.csv`` produce
``preprocessed_demonstrations/<task>/processed_<task>_NN.csv`` applicando in
sequenza:

  1. Hemisphere continuity sui quaternioni grezzi  (q(t)·q(t+1) > 0).
  2. Normalizzazione  ||q|| = 1.
  3. Resampling temporale su una griglia uniforme con passo ``--dt``:
     - posizione (x,y,z) e gripper -> interpolazione lineare;
     - quaternione -> SLERP.
  4. Re-normalizzazione e re-applicazione della continuity dopo il resampling.
  5. **Ricentratura della rotazione** rispetto a un quaternione di riferimento
     ``q_ref`` (default: primo campione della prima demo del batch). Per ogni
     campione si calcola la rotazione relativa ``q_rel = q_ref^{-1} * q_abs``
     e si applica la log-map per ottenere ``(rx, ry, rz)``. Lavorando nel
     frame relativo le rotazioni vivono in un intorno dell'identita': il
     log-map e' lontano dalla singolarita' a theta=pi e non si presentano
     piu' problemi di rami antipodali fra demo o di componenti scalari
     quasi-degeneri sul DMP.

Il CSV di output mantiene **entrambe** le rappresentazioni dell'orientazione:
  - ``qx,qy,qz,qw`` = rotazione ASSOLUTA del TCP (per replay/visualizzazione);
  - ``rx,ry,rz``    = log-map della rotazione RELATIVA a ``q_ref``.

Il quaternione di riferimento viene persistito come sidecar JSON
``preprocessed_demonstrations/<task>/qref.json``: i preprocess di DMP/GMM/BC
lo leggono e lo propagano fino al modello allenato, cosicche' a inferenza
sia possibile ri-comporre la rotazione assoluta come ``q_abs = q_ref * exp(r_rel)``.

Header del CSV in uscita:
    t, x, y, z, qx, qy, qz, qw, rx, ry, rz, gripper
    (qx..qw assoluti; rx..rz relativi a q_ref)

Uso:
    python3 preprocess_demos.py --task pick
    python3 preprocess_demos.py --task pour --index 3
    python3 preprocess_demos.py --task place --dt 0.01
    python3 preprocess_demos.py --task place --ref-from preprocessed_demonstrations/place/processed_place_01.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config.robot_config import (  # noqa: E402
    DEFAULT_PREPROCESS_DT,
    DEMO_ROOT,
    PREPROCESSED_ROOT,
)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
RAW_HEADER = ["t", "x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]
OUT_HEADER = ["t", "x", "y", "z", "qx", "qy", "qz", "qw",
              "rx", "ry", "rz", "gripper"]


def load_demo(path: Path):
    """Carica una demo grezza. Ritorna array numpy di shape (N, 9)."""
    rows = []
    with open(path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            rows.append([float(row[k]) for k in RAW_HEADER])
    if not rows:
        raise ValueError(f"CSV vuoto: {path}")
    return np.asarray(rows, dtype=float)


def save_demo(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(OUT_HEADER)
        for row in arr:
            w.writerow([f"{v:.9f}" for v in row])


# ---------------------------------------------------------------------------
# Pre-processing steps
# ---------------------------------------------------------------------------
def enforce_hemisphere(quat: np.ndarray) -> np.ndarray:
    """Garantisce q(t)\u00b7q(t+1) > 0 invertendo segno dove necessario.

    quat: (N, 4) in ordine (qx, qy, qz, qw).
    """
    out = quat.copy()
    for i in range(1, len(out)):
        if np.dot(out[i - 1], out[i]) < 0.0:
            out[i] = -out[i]
    return out


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return quat / norms


def resample(arr: np.ndarray, dt: float):
    """Resampling su griglia uniforme [0, dt, 2dt, ..., T].

    Input: arr (N, 9) con colonne RAW_HEADER. Si assume continuity gi\u00e0
    applicata sui quaternioni.
    Ritorna (t_grid, xyz_g, quat_g, grip_g).
    """
    t_raw = arr[:, 0]
    # Forza monotonia stretta su t_raw (gestione duplicati o jitter al ribasso).
    for i in range(1, len(t_raw)):
        if t_raw[i] <= t_raw[i - 1]:
            t_raw[i] = t_raw[i - 1] + 1e-9

    t0, tN = float(t_raw[0]), float(t_raw[-1])
    duration = tN - t0
    if duration <= 0.0:
        raise ValueError("Durata della demo non positiva.")

    n_steps = int(np.floor(duration / dt)) + 1
    t_grid = t0 + np.arange(n_steps) * dt
    # garantisci di non oltrepassare l'ultimo campione (per scipy interp1d-like)
    t_grid = np.clip(t_grid, t0, tN)

    # xyz, gripper: interp lineare componente per componente.
    xyz = arr[:, 1:4]
    grip = arr[:, 8]
    xyz_g = np.column_stack([np.interp(t_grid, t_raw, xyz[:, k]) for k in range(3)])
    grip_g = np.interp(t_grid, t_raw, grip)

    # Quaternione: SLERP. scipy si aspetta (qx, qy, qz, qw).
    rot_raw = R.from_quat(arr[:, 4:8])
    slerp = Slerp(t_raw, rot_raw)
    rot_g = slerp(t_grid)
    quat_g = rot_g.as_quat()

    return t_grid, xyz_g, quat_g, grip_g


def quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """log map del quaternione -> angle-axis (rx, ry, rz)."""
    return R.from_quat(quat).as_rotvec()


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverso di un quaternione unitario (qx, qy, qz, qw)."""
    q = np.asarray(q, float)
    out = q.copy()
    out[..., :3] = -out[..., :3]
    return out


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Prodotto di Hamilton (qx, qy, qz, qw). Supporta broadcasting su (..., 4)."""
    q1 = np.asarray(q1, float)
    q2 = np.asarray(q2, float)
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    out = np.empty(np.broadcast_shapes(q1.shape, q2.shape), float)
    out[..., 0] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    out[..., 1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    out[..., 2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    out[..., 3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return out


def recenter_quat_stream(quat_abs: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Calcola lo stream di quaternioni RELATIVI ``q_rel = q_ref^{-1} * q_abs``.

    La rappresentazione relativa porta tutte le rotazioni in un intorno di
    ``q_ref`` (quindi vicine all'identita' nel frame relativo). In log-map
    cio' significa rotvec con norma piccola, lontana dalla singolarita' a
    ``theta = pi``: niente piu' ambiguita' antipodale, niente collassi a
    rotazione nulla quando si mediano demo diverse.
    """
    quat_abs = np.asarray(quat_abs, float)
    if quat_abs.size == 0:
        return quat_abs.copy()
    q_ref_inv = quat_inv(np.asarray(q_ref, float))
    return quat_mul(np.broadcast_to(q_ref_inv, quat_abs.shape), quat_abs)


def enforce_rotvec_continuity(rotvec: np.ndarray) -> np.ndarray:
    """Continuita' temporale del rotvec dentro una singola demo.

    Quando ||r|| ~ pi la mappa quaternione->rotvec resta ambigua
    (anche nel frame relativo, se la rotazione effettiva si avvicina a pi).
    Confrontiamo ||r_{t+1} - r_t|| con la rappresentazione equivalente
    ``(1 - 2*pi/||r_{t+1}||) * r_{t+1}`` e teniamo quella piu' vicina.

    NOTA: la consistenza INTER-DEMO non e' demandata a questa funzione:
    viene garantita dalla ricentratura globale rispetto a ``q_ref`` (vedi
    ``recenter_quat_stream``), che porta tutte le demo nello stesso intorno
    dell'identita' nel frame relativo.
    """
    rotvec = np.asarray(rotvec, float).copy()
    n = len(rotvec)
    if n < 2:
        return rotvec

    for i in range(1, n):
        r_prev = rotvec[i - 1]
        r_cur = rotvec[i]
        norm = float(np.linalg.norm(r_cur))
        if norm < 1e-9:
            continue
        r_alt = (1.0 - 2.0 * np.pi / norm) * r_cur
        if np.linalg.norm(r_alt - r_prev) < np.linalg.norm(r_cur - r_prev):
            rotvec[i] = r_alt
    return rotvec


def savgol_smooth(arr: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Savitzky–Golay zero-phase smoothing applicato componente per componente.

    arr: (N,) o (N, D). Bordi gestiti in modalità 'mirror' per non distorcere
    start/goal della traiettoria. La finestra viene clampata a un valore
    dispari <= N e > polyorder.
    """
    n = arr.shape[0]
    win = int(window)
    if win % 2 == 0:
        win += 1
    if win > n:
        win = n if n % 2 == 1 else n - 1
    if win <= polyorder:
        # Finestra troppo piccola per il grado richiesto: niente smoothing.
        return arr.copy()
    return savgol_filter(arr, window_length=win, polyorder=polyorder,
                         axis=0, mode="mirror")


# ---------------------------------------------------------------------------
# Pipeline su singola demo
# ---------------------------------------------------------------------------
def preprocess_one_stage1(in_path: Path, dt: float,
                          smooth: bool = False, smooth_window: int = 9,
                          smooth_polyorder: int = 3):
    """Stadio 1: clean + resample + smooth + continuity intra-demo.

    Restituisce ``(n_in, t_grid, xyz_g, quat_g, grip_g)`` *senza* fissare
    ancora il segno globale del quaternione e *senza* calcolare il rotvec.
    L'allineamento inter-demo (sign flip globale rispetto a ``q_ref``) e
    la conversione a rotvec avvengono in ``finalize_and_save``.
    """
    arr = load_demo(in_path)
    n_in = len(arr)

    # 1) hemisphere continuity sui dati grezzi (PRIMA del resampling)
    arr[:, 4:8] = enforce_hemisphere(arr[:, 4:8])
    # 2) normalizzazione quaternione
    arr[:, 4:8] = normalize_quat(arr[:, 4:8])

    # 3) resampling temporale su griglia uniforme
    t_grid, xyz_g, quat_g, grip_g = resample(arr, dt)

    # 3b) smoothing opzionale (Savitzky-Golay) su xyz e quaternione
    #     componente-per-componente. Gripper non filtrato per preservare gli step.
    if smooth:
        xyz_g = savgol_smooth(xyz_g, smooth_window, smooth_polyorder)
        quat_g = savgol_smooth(quat_g, smooth_window, smooth_polyorder)

    # 4) re-normalizzazione + continuity post-SLERP (precauzione)
    quat_g = normalize_quat(quat_g)
    quat_g = enforce_hemisphere(quat_g)

    # azzeriamo l'origine temporale per coerenza
    t_grid = t_grid - t_grid[0]

    return n_in, t_grid, xyz_g, quat_g, grip_g


def finalize_and_save(out_path: Path, n_in: int, t_grid: np.ndarray,
                      xyz_g: np.ndarray, quat_g: np.ndarray,
                      grip_g: np.ndarray, q_ref: np.ndarray,
                      dt: float,
                      smooth: bool = False, smooth_window: int = 9,
                      smooth_polyorder: int = 3,
                      verbose: bool = True,
                      in_name: str = "") -> None:
    """Stadio 2: ricentratura della rotazione su ``q_ref``, log-map,
    continuity rotvec, salvataggio.

    - ``qx,qy,qz,qw`` salvati nel CSV restano la rotazione ASSOLUTA del TCP
      (utile per replay / visualizzazione 3D).
    - ``rx,ry,rz`` salvati nel CSV sono la log-map della rotazione RELATIVA
      a ``q_ref``: e' questo lo spazio in cui DMP / GMM / BC apprendono.
    """
    # rotazione relativa a q_ref (stream): q_rel = q_ref^{-1} * q_abs
    quat_rel = recenter_quat_stream(quat_g, q_ref)
    # garantiamo continuita' anche dello stream relativo (per la SLERP
    # interna alla log-map non e' detto che si conservi automaticamente)
    quat_rel = enforce_hemisphere(quat_rel)
    quat_rel = normalize_quat(quat_rel)

    # log-map sul frame relativo: norme tipicamente piccole, lontane da pi.
    rotvec_rel = quat_to_rotvec(quat_rel)
    rotvec_rel = enforce_rotvec_continuity(rotvec_rel)

    out = np.column_stack([t_grid, xyz_g, quat_g, rotvec_rel, grip_g])
    save_demo(out_path, out)

    if verbose:
        smooth_info = (f", smoothing=savgol(window={smooth_window},"
                       f" polyorder={smooth_polyorder})") if smooth else ""
        max_norm = float(np.linalg.norm(rotvec_rel, axis=1).max())
        print(f"  {in_name}: {n_in} -> {len(out)} campioni "
              f"(durata {t_grid[-1]:.2f}s, dt={dt:g}s{smooth_info}; "
              f"max ||r_rel||={max_norm:.3f} rad) -> {out_path}")


def save_qref_sidecar(out_dir: Path, q_ref: np.ndarray,
                      source_demo: str | None = None) -> Path:
    """Persiste ``q_ref`` come JSON sidecar nella cartella delle demo
    processate. Letto in seguito da DMP/GMM/BC preprocess per propagarlo
    fino al modello allenato; usato a inferenza per ri-comporre la
    rotazione assoluta dal rotvec relativo prodotto dal modello.
    """
    import json
    q = np.asarray(q_ref, float).reshape(4)
    out = {
        "qx": float(q[0]), "qy": float(q[1]),
        "qz": float(q[2]), "qw": float(q[3]),
        "convention": "scalar_last (qx,qy,qz,qw)",
        "meaning": "q_ref t.c. q_rel(t) = q_ref^-1 * q_abs(t); "
                   "rx,ry,rz nei CSV sono log(q_rel).",
    }
    if source_demo:
        out["source"] = source_demo
    path = out_dir / "qref.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


def load_qref_sidecar(task_dir: Path) -> np.ndarray:
    """Carica ``q_ref`` (qx,qy,qz,qw) dal sidecar JSON in ``task_dir``.

    Solleva ``FileNotFoundError`` se il sidecar non esiste: i moduli a valle
    devono usare un fallback (q_ref=identita') solo per modelli legacy.
    """
    import json
    path = task_dir / "qref.json"
    if not path.is_file():
        raise FileNotFoundError(f"qref.json non trovato in {task_dir}.")
    with open(path) as f:
        d = json.load(f)
    q = np.array([d["qx"], d["qy"], d["qz"], d["qw"]], float)
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        raise ValueError(f"q_ref degenere in {path}.")
    return q / n


def load_ref_quat_from_csv(path: Path) -> np.ndarray:
    """Estrae il quaternione del primo campione di un CSV gia' processato.

    Utile per ri-processare una singola demo (--index) mantenendo la stessa
    convenzione di ricentratura usata in un batch precedente.
    """
    with open(path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            q = np.array([float(row["qx"]), float(row["qy"]),
                          float(row["qz"]), float(row["qw"])], float)
            n = float(np.linalg.norm(q))
            if n < 1e-9:
                raise ValueError(f"Quaternione di riferimento degenere in {path}.")
            return q / n
    raise ValueError(f"CSV di riferimento vuoto: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def collect_inputs(in_root: Path, task: str, index: int | None, zfill: int):
    task_dir = in_root / task
    if not task_dir.is_dir():
        raise SystemExit(f"Cartella demo non trovata: {task_dir}")
    if index is not None:
        fname = f"{task}_{str(index).zfill(zfill)}.csv"
        f = task_dir / fname
        if not f.is_file():
            raise SystemExit(f"Demo non trovata: {f}")
        return [f]
    pattern = re.compile(rf"^{re.escape(task)}_(\d+)\.csv$")
    files = sorted(p for p in task_dir.iterdir() if pattern.match(p.name))
    if not files:
        raise SystemExit(f"Nessuna demo trovata in {task_dir}")
    return files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-processing comune delle demo LfD.")
    p.add_argument("--task", required=True, help="Nome primitiva (es. pick, place, pour).")
    p.add_argument("--index", type=int, default=None,
                   help="Processa solo la demo con questo indice. Se omesso, le processa tutte.")
    p.add_argument("--dt", type=float, default=DEFAULT_PREPROCESS_DT,
                   help=f"Passo della time-grid [s] (default: {DEFAULT_PREPROCESS_DT}).")
    p.add_argument("--in-root", type=Path, default=DEMO_ROOT,
                   help=f"Cartella radice delle demo grezze (default: {DEMO_ROOT}).")
    p.add_argument("--out-root", type=Path, default=PREPROCESSED_ROOT,
                   help=f"Cartella radice di output (default: {PREPROCESSED_ROOT}).")
    p.add_argument("--zfill", type=int, default=2, help="Zero-padding indice (default: 2).")
    p.add_argument("--smooth", action="store_true",
                   help="Abilita smoothing Savitzky-Golay su xyz e quaternione (post-resampling).")
    p.add_argument("--smooth-window", type=int, default=9,
                   help="Finestra (campioni) per Savitzky-Golay; resa dispari se pari (default: 9).")
    p.add_argument("--smooth-polyorder", type=int, default=3,
                   help="Grado del polinomio per Savitzky-Golay (default: 3).")
    p.add_argument("--ref-from", type=Path, default=None,
                   help="CSV gia' processato da cui prendere il quaternione "
                        "di riferimento per la ricentratura della rotazione. "
                        "Se omesso, il riferimento e' il primo campione della "
                        "prima demo processata in questo run.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inputs = collect_inputs(args.in_root, args.task, args.index, args.zfill)
    out_dir = args.out_root / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pre-processing di {len(inputs)} demo per '{args.task}' (dt={args.dt}s)")
    print(f"  in : {args.in_root / args.task}")
    print(f"  out: {out_dir}")

    # --- stadio 1: clean + resample + smooth + continuity intra-demo --------
    stage1: list[tuple] = []
    for f in inputs:
        try:
            n_in, t_grid, xyz_g, quat_g, grip_g = preprocess_one_stage1(
                f, dt=args.dt,
                smooth=args.smooth,
                smooth_window=args.smooth_window,
                smooth_polyorder=args.smooth_polyorder,
            )
        except Exception as e:
            print(f"  [ERR-stage1] {f.name}: {e}")
            continue
        stage1.append((f, n_in, t_grid, xyz_g, quat_g, grip_g))

    if not stage1:
        print("Nessuna demo processata.")
        return 0

    # --- scelta del quaternione di riferimento per la ricentratura ----------
    ref_source = None
    if args.ref_from is not None:
        if not args.ref_from.is_file():
            raise SystemExit(f"--ref-from non trovato: {args.ref_from}")
        q_ref = load_ref_quat_from_csv(args.ref_from)
        ref_source = str(args.ref_from)
        print(f"  q_ref (from {args.ref_from.name}): {q_ref.round(4).tolist()}")
    else:
        q_ref = stage1[0][4][0].copy()
        q_ref = q_ref / max(float(np.linalg.norm(q_ref)), 1e-12)
        ref_source = stage1[0][0].name + " (primo campione)"
        print(f"  q_ref (da {stage1[0][0].name}, primo campione): "
              f"{q_ref.round(4).tolist()}")

    # --- stadio 2: ricentratura + log-map + continuity rotvec + save --------
    for f, n_in, t_grid, xyz_g, quat_g, grip_g in stage1:
        out_path = out_dir / f"processed_{f.name}"
        try:
            finalize_and_save(
                out_path, n_in, t_grid, xyz_g, quat_g, grip_g,
                q_ref=q_ref, dt=args.dt,
                smooth=args.smooth,
                smooth_window=args.smooth_window,
                smooth_polyorder=args.smooth_polyorder,
                verbose=True, in_name=f.name,
            )
        except Exception as e:
            print(f"  [ERR-stage2] {f.name}: {e}")

    # sidecar qref.json: riferimento globale del task per i moduli a valle.
    sidecar = save_qref_sidecar(out_dir, q_ref, source_demo=ref_source)
    print(f"  q_ref persistito in: {sidecar}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
