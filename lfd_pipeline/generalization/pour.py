"""Generalizzazione POUR: 6 ArUco x 3 modelli = 18 traiettorie.

Per pour, sia start sia goal dipendono ESCLUSIVAMENTE dalla posa del tag:
    start = pour_pose_from_tag(T,  x=POUR_OFFSET_X, y=POUR_OFFSET_Y,
                                   z=POUR_OFFSET_Z, rot_z_deg=0)
    goal  = pour_pose_from_tag(T,  x=POUR_OFFSET_X, y=POUR_OFFSET_Y + GOAL_Y,
                                   z=POUR_OFFSET_Z, rot_z_deg=GOAL_ROT_Z_DEG)

Quindi non servono start manuali: per ogni tag si producono 1 start e 1 goal,
poi ogni modello genera 1 traiettoria.

Output:
    generalization_results/pour/
        inputs.json   (tags 4x4 + starts xyzrpy + goals xyzrpy)
        metadata.json
        bc/   generated_pour_g01.csv ... g06.csv
        dmp/  ...
        gmm/  ...

Uso:
    python3 pour.py
    python3 pour.py --methods dmp gmm --n-tags 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config.robot_config import HOME_JOINT_DEG, GRIPPER_OPEN_POS  # noqa: E402
from generalization._common import (  # noqa: E402
    acquire_tags,
    add_common_cli,
    dump_json,
    go_home,
    init_robot,
    load_generators,
    open_gripper,
    resolve_model_path,
    save_generated_csv,
    shutdown,
)


POUR_MARKER_ID_DEFAULT = 2
POUR_OFFSET_X_M = 0.20
POUR_OFFSET_Y_M = 0.08
POUR_OFFSET_Z_M = 0.0
POUR_GOAL_OFFSET_Y_M = +0.02
POUR_GOAL_ROT_Z_DEG = -120.0


def pour_pose_from_tag(T_base_tag: np.ndarray,
                       offset_x_m: float, offset_y_m: float,
                       offset_z_m: float,
                       rot_z_deg: float = 0.0) -> list[float]:
    """Posa di pour: orientamento base z_g=-z_t, x_g=+x_t, eventualmente
    ruotato di ``rot_z_deg`` attorno a +Z del tag (R_axis · R_g, world).
    Identica alla helper degli script pour_*.py / pick_place_pour.py."""
    R_t = np.asarray(T_base_tag, float)[:3, :3]
    p_t = np.asarray(T_base_tag, float)[:3, 3]
    x_t = R_t[:, 0]; y_t = R_t[:, 1]; z_t = R_t[:, 2]

    x_g = x_t / np.linalg.norm(x_t)
    z_g = -z_t / np.linalg.norm(z_t)
    x_g = x_g - np.dot(x_g, z_g) * z_g
    x_g = x_g / np.linalg.norm(x_g)
    y_g = np.cross(z_g, x_g)
    R_g = np.column_stack([x_g, y_g, z_g])

    if abs(rot_z_deg) > 1e-9:
        theta = float(np.deg2rad(rot_z_deg))
        k = z_t / np.linalg.norm(z_t)
        K = np.array([[0.0, -k[2], k[1]],
                      [k[2], 0.0, -k[0]],
                      [-k[1], k[0], 0.0]], float)
        R_axis = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
        R_g = R_axis @ R_g

    p_g = p_t + offset_x_m * x_t + offset_y_m * y_t + offset_z_m * z_t

    sp = float(np.clip(-R_g[2, 0], -1.0, 1.0))
    pitch = float(np.arcsin(sp))
    if abs(np.cos(pitch)) > 1e-8:
        roll = float(np.arctan2(R_g[2, 1], R_g[2, 2]))
        yaw = float(np.arctan2(R_g[1, 0], R_g[0, 0]))
    else:
        roll = 0.0
        yaw = float(np.arctan2(-R_g[0, 1], R_g[1, 1]))
    return [float(p_g[0]), float(p_g[1]), float(p_g[2]), roll, pitch, yaw]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generalizzazione POUR su 6 ArUco x 3 modelli.")
    add_common_cli(p, default_marker_id=POUR_MARKER_ID_DEFAULT)
    p.add_argument("--n-tags", type=int, default=6,
                   help="Numero di pose ArUco da acquisire (default: 6).")
    p.add_argument("--pour-offset-x", type=float, default=POUR_OFFSET_X_M,
                   help=f"Offset POUR lungo x_t [m] (default: {POUR_OFFSET_X_M}).")
    p.add_argument("--pour-offset-y", type=float, default=POUR_OFFSET_Y_M,
                   help=f"Offset POUR lungo y_t [m] (default: {POUR_OFFSET_Y_M}).")
    p.add_argument("--pour-offset-z", type=float, default=POUR_OFFSET_Z_M,
                   help=f"Offset POUR lungo z_t [m] (default: {POUR_OFFSET_Z_M}).")
    p.add_argument("--goal-offset-y", type=float, default=POUR_GOAL_OFFSET_Y_M,
                   help=f"Offset goal-vs-start lungo y_t [m] "
                        f"(default: {POUR_GOAL_OFFSET_Y_M}).")
    p.add_argument("--goal-rot-z-deg", type=float, default=POUR_GOAL_ROT_Z_DEG,
                   help=f"Rotazione goal-vs-start attorno a +Z del tag [deg] "
                        f"(default: {POUR_GOAL_ROT_Z_DEG}).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    task = "pour"

    out_dir = Path(args.out_root) / task
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{task}] METODI / MODELLI")
    for m in args.methods:
        print(f"  {m:>3s} -> {resolve_model_path(args, m, task)}")
    print(f"[{task}] marker id={args.marker_id}  N_tags={args.n_tags}")
    print(f"[{task}] start offset (tag): "
          f"dx={args.pour_offset_x:+.3f} dy={args.pour_offset_y:+.3f} "
          f"dz={args.pour_offset_z:+.3f} m")
    print(f"[{task}] goal vs start    : "
          f"dy={args.goal_offset_y:+.3f} m  "
          f"rot_z={args.goal_rot_z_deg:+.1f} deg")
    print(f"[{task}] output -> {out_dir}")

    arm = init_robot(args.robot_ip)
    try:
        print(f"\n[{task}] HOME (deg): {HOME_JOINT_DEG}")
        go_home(arm, HOME_JOINT_DEG)
        open_gripper(arm, GRIPPER_OPEN_POS)

        tags = acquire_tags(arm, args.marker_id, args.n_tags,
                            args.marker_len, args.marker_timeout,
                            args.n_stable, task)

        # start e goal derivati dal tag
        starts = [pour_pose_from_tag(T,
                                     offset_x_m=args.pour_offset_x,
                                     offset_y_m=args.pour_offset_y,
                                     offset_z_m=args.pour_offset_z,
                                     rot_z_deg=0.0)
                  for T in tags]
        goals = [pour_pose_from_tag(T,
                                    offset_x_m=args.pour_offset_x,
                                    offset_y_m=args.pour_offset_y + args.goal_offset_y,
                                    offset_z_m=args.pour_offset_z,
                                    rot_z_deg=args.goal_rot_z_deg)
                 for T in tags]
        print(f"\n[{task}] start/goal calcolati ({len(tags)} tag):")
        for j, (s, g) in enumerate(zip(starts, goals), start=1):
            print(f"  g{j:02d} start xyz={[round(v, 4) for v in s[:3]]}  "
                  f"goal xyz={[round(v, 4) for v in g[:3]]}")

        # generatori
        generators = load_generators(args, task)

        # generazione + salvataggio
        n_total = 0
        for j, (s, g) in enumerate(zip(starts, goals), start=1):
            for method, gen in generators.items():
                try:
                    traj = gen.generate(
                        start_xyzrpy=s, goal_xyzrpy=g,
                        duration_scale=args.duration_scale,
                    )
                except Exception as e:
                    print(f"  [{method}] g{j:02d} [error] "
                          f"generate fallito: {e}")
                    continue
                out_path = out_dir / method / f"generated_{task}_g{j:02d}.csv"
                save_generated_csv(out_path, traj)
                n_total += 1
                print(f"  [{method}] g{j:02d}: N={traj.n} "
                      f"T={traj.t[-1]:.2f}s  -> {out_path}")

        # inputs.json + metadata.json
        dump_json(out_dir / "inputs.json", {
            "task": task,
            "marker_id": args.marker_id,
            "tags_T_base_tag": [T.tolist() for T in tags],
            "starts_xyzrpy": [[float(v) for v in s] for s in starts],
            "goals_xyzrpy": [[float(v) for v in g] for g in goals],
        })
        dump_json(out_dir / "metadata.json", {
            "task": task,
            "methods": list(generators.keys()),
            "models": {m: str(resolve_model_path(args, m, task))
                       for m in generators},
            "marker_id": args.marker_id,
            "marker_len_m": args.marker_len,
            "offsets_m": {
                "pour_x": args.pour_offset_x,
                "pour_y": args.pour_offset_y,
                "pour_z": args.pour_offset_z,
                "goal_y": args.goal_offset_y,
            },
            "goal_rot_z_deg": args.goal_rot_z_deg,
            "n_tags": args.n_tags,
            "duration_scale": args.duration_scale,
            "n_trajectories": n_total,
        })

        print(f"\n[{task}] DONE. Salvate {n_total} traiettorie in {out_dir}.")
    finally:
        shutdown(arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
