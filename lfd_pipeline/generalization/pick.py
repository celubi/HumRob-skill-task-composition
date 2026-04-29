"""Generalizzazione PICK: 3 start manuali x 5 ArUco x 3 modelli = 45 traiettorie.

Flusso:
    1. HOME, gripper aperto.
    2. Acquisizione manuale di N start in gravity-comp (default 3).
    3. Acquisizione di M pose ArUco (default 5, marker id default 3) -> goal
       calcolato come ``grasp_pose_from_tag`` (convenzione standalone pick).
    4. Per ogni (start_i, goal_j, metodo) genera la traiettoria e la salva.

Output:
    generalization_results/pick/
        inputs.json   (starts xyzrpy + tags 4x4 + goals xyzrpy)
        metadata.json
        bc/   generated_pick_s01_g01.csv ... s03_g05.csv
        dmp/  ...
        gmm/  ...

Uso:
    python3 pick.py
    python3 pick.py --methods bc dmp --n-starts 3 --n-tags 5
    python3 pick.py --marker-id 3 --pick-offset-z -0.005
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config.robot_config import HOME_JOINT_DEG, GRIPPER_OPEN_POS  # noqa: E402
from generalization._common import (  # noqa: E402
    acquire_starts_manual,
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
from inference.vision.tag_to_grasp import grasp_pose_from_tag  # noqa: E402


PICK_MARKER_ID_DEFAULT = 3
PICK_OFFSET_Z_M = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generalizzazione PICK su 3 start x 5 ArUco x 3 modelli.")
    add_common_cli(p, default_marker_id=PICK_MARKER_ID_DEFAULT)
    p.add_argument("--n-starts", type=int, default=3,
                   help="Numero di pose di partenza manuali (default: 3).")
    p.add_argument("--n-tags", type=int, default=5,
                   help="Numero di pose ArUco da acquisire (default: 5).")
    p.add_argument("--pick-offset-z", type=float, default=PICK_OFFSET_Z_M,
                   help=f"Offset PICK lungo z del tag [m] "
                        f"(default: {PICK_OFFSET_Z_M}).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    task = "pick"

    out_dir = Path(args.out_root) / task
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{task}] METODI / MODELLI")
    for m in args.methods:
        print(f"  {m:>3s} -> {resolve_model_path(args, m, task)}")
    print(f"[{task}] marker id={args.marker_id}  "
          f"offset_z={args.pick_offset_z:+.3f} m  "
          f"N_starts={args.n_starts}  N_tags={args.n_tags}")
    print(f"[{task}] output -> {out_dir}")

    arm = init_robot(args.robot_ip)
    try:
        print(f"\n[{task}] HOME (deg): {HOME_JOINT_DEG}")
        go_home(arm, HOME_JOINT_DEG)
        open_gripper(arm, GRIPPER_OPEN_POS)

        # 1) tag da HOME (visibilita' su tutto il workspace)
        tags = acquire_tags(arm, args.marker_id, args.n_tags,
                            args.marker_len, args.marker_timeout,
                            args.n_stable, task)
        # 2) start manuali in gravity-comp (DOPO i tag, cosi' la HOME non
        #    e' piu' necessaria per la visione)
        starts = acquire_starts_manual(arm, args.n_starts, task)

        # goal = grasp_pose_from_tag (convenzione pick standalone)
        goals = [grasp_pose_from_tag(T, offset_z_m=args.pick_offset_z)
                 for T in tags]
        print(f"\n[{task}] goal calcolati ({len(goals)}):")
        for j, g in enumerate(goals, start=1):
            print(f"  g{j:02d}: xyz={[round(v, 4) for v in g[:3]]} "
                  f"rpy={[round(v, 4) for v in g[3:]]}")

        # generatori
        generators = load_generators(args, task)

        # generazione + salvataggio
        n_total = 0
        for i, s in enumerate(starts, start=1):
            for j, g in enumerate(goals, start=1):
                for method, gen in generators.items():
                    try:
                        traj = gen.generate(
                            start_xyzrpy=s, goal_xyzrpy=g,
                            duration_scale=args.duration_scale,
                        )
                    except Exception as e:
                        print(f"  [{method}] s{i:02d}_g{j:02d} "
                              f"[error] generate fallito: {e}")
                        continue
                    out_path = (out_dir / method
                                / f"generated_{task}_s{i:02d}_g{j:02d}.csv")
                    save_generated_csv(out_path, traj)
                    n_total += 1
                    print(f"  [{method}] s{i:02d}_g{j:02d}: N={traj.n} "
                          f"T={traj.t[-1]:.2f}s  -> {out_path}")

        # inputs.json + metadata.json
        dump_json(out_dir / "inputs.json", {
            "task": task,
            "marker_id": args.marker_id,
            "starts_xyzrpy": [[float(v) for v in s] for s in starts],
            "tags_T_base_tag": [T.tolist() for T in tags],
            "goals_xyzrpy": [[float(v) for v in g] for g in goals],
        })
        dump_json(out_dir / "metadata.json", {
            "task": task,
            "methods": list(generators.keys()),
            "models": {m: str(resolve_model_path(args, m, task))
                       for m in generators},
            "marker_id": args.marker_id,
            "marker_len_m": args.marker_len,
            "pick_offset_z_m": args.pick_offset_z,
            "n_starts": args.n_starts,
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
