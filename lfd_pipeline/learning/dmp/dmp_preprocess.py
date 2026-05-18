""" 
    DMP preprocessing module.
    Per la generazione di traiettorie utilizzando il modello DMP,
    in questa repo vengono utilizzate due versioni di DMP:
        - DMP classico, per le compoenti cartesiane che rappresentano
          la posizione del TCP nello spazio. Questo modello è implementato
          utilizzando la formulazione originale proposta da Ijspeert (2013).
        - DMP basato su quaternioni unitari per rappresentare l'orientamento
          del TCP nello spazio. Questo modello è implementato utilizzando 
          la formulazione di Koutras et al. (2020)

    Questo file di prepocessing effettua le operazioni necessarie
    per entrambe le formulazioni di DMP.

    Le demo necessarie per il training dei DMP sono contenute in
    file .csv (output degli step di preprocessing comuni) con struttura:
    [t, x, y, z, qx, qy, qz, qw, rx, ry, rz, gripper]

    A partire dai dati nei .csv, per allenare i DMP è necessario:
        - calcolare le derivate prima e seconda della posizione per
          il DMP classico -> xdot, ydot, zdot, xddot, yddot, zddot
        - calcolare la velocità angolare e la sua derivata per il
          DMP per l'orientamento basatu sui quaternioni
          -> wx, wy, wz, wdotx, wdoty, wdotz

    Questo file produce due file .csv per ogni demo:
        - dmp_pos_<task>_<num>.csv:
            contiene le colonne [t, x, y, z, xdot, ydot, zdot, xddot, yddot, zddot]
        - dmp_quat_<task>_<num>.csv:
            contiene le colonne [t, qx, qy, qz, qw, wx, wy, wz, wdotx, wdoty, wdotz]

     Questi file vengono poi utilizzati per allenare i rispettivi modelli DMP.
"""
import numpy as np
import argparse
import csv
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parents[1]

if _PKG_ROOT not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config.robot_config import (
    PREPROCESSED_ROOT,
    DMP_PREPROCESSED_ROOT
)

INPUT_FILE_HEADER = ["t",
                     "x", "y", "z", 
                     "qx", "qy", "qz", "qw", 
                     "rx", "ry", "rz",
                     "gripper"]

OUT_FILE_POS_HEADER = ["t", "dt",
                       "x", "y", "z",
                       "xdot", "ydot", "zdot",
                       "xddot", "yddot", "zddot"]

OUT_FILE_QUAT_HEADER = ["t", "dt",
                       "qx", "qy", "qz", "qw",
                       "qxdot", "qydot", "qzdot", "qwdot",
                       "wx", "wy", "wz",
                       "wdotx", "wdoty", "wdotz",
                       "eqx", "eqy", "eqz",
                       "eqxdot", "eqydot", "eqzdot",
                       "eqxddot", "eqyddot", "eqzddot"]

def pos_preprocessing(t:np.ndarray, pos: np.ndarray):
    dt = float(t[1]-t[0])
    pos_dot = np.gradient(pos, dt, axis=0, edge_order=2)
    pos_ddot = np.gradient(pos_dot, dt, axis=0, edge_order=2)

    dt_vec = dt * np.ones(pos_ddot.shape[0], dtype=float)

    return dt_vec, pos_dot, pos_ddot
    
def quat_norm(q: np.ndarray):
    return np.linalg.norm(q, ord=2)

def quat_conj(q:np.ndarray):
    q_c = q.copy()
    q_c[:3] = -q[:3]

    return q_c

def quat_prod(q1: np.ndarray, q2:np.ndarray):
    q1x, q1y, q1z, q1w = q1[0], q1[1], q1[2], q1[3]
    q2x, q2y, q2z, q2w = q2[0], q2[1], q2[2], q2[3]

    q1v = np.array([q1x, q1y, q1z])
    q2v = np.array([q2x, q2y, q2z])

    # q = [qx, qy, qz, qw]
    prod = np.zeros([4,])
    prod[:3] = q1w*q2v + q2w*q1v + np.cross(q1v, q2v)
    prod[3] = float(q1w*q2w - np.dot(q1v, q2v))

    return prod

def quat_log_map(q: np.ndarray, tol=1e-6):
    v = q[:3]
    w = q[3]

    v_norm = np.linalg.norm(v, ord=2)
    if v_norm < tol:
        return np.zeros(3, dtype=float)

    w_clipped = np.clip(w, -1.0, 1.0)
    theta = np.arctan2(v_norm, w_clipped)

    if theta < tol:
        return np.array([0, 0, 0])
    
    n = v / v_norm

    return theta * n

def quat_force_unit_norm(quat: np.ndarray, tol = 1e-9):
    q_normalized = quat.copy()
    for i in range(quat.shape[0]):
        q = quat[i,:]
        q_norm = quat_norm(q)
        
        if abs(1 - q_norm) < tol:
            continue

        q_normalized[i,:] = q / q_norm
    
    return q_normalized

def quat_force_hemisphere_continuity(quat: np.ndarray):
    quat_cont = quat.copy()

    for i in range(quat_cont.shape[0]-1):
        qk = quat_cont[i,:]
        qk1 = quat_cont[i+1,:]

        if np.dot(qk, qk1)<0:
            quat_cont[i+1,:] = -quat_cont[i+1, :]

    return quat_cont

def quat_preprocessing(t: np.ndarray, quat: np.ndarray):
    quat = quat_force_unit_norm(quat)
    quat = quat_force_hemisphere_continuity(quat)

    dt = abs(t[0] - t[1])

    # compute qdot
    qdot = np.gradient(quat, dt, axis=0, edge_order=2)
    
    # step 3: compute w, wdot
    w = np.zeros([quat.shape[0], 3], dtype=float)
    for i in range(quat.shape[0]):
        q = quat[i,:]
        q_conj = quat_conj(q)

        delta_q = quat_prod(qdot[i,:], q_conj)

        w[i,:] = 2 * delta_q[:3]

    wdot = np.gradient(w, dt, axis=0, edge_order=2)

    # step 4: calcolo eq, eqdot, eqddot
    q_goal = quat[-1,:]

    eq = np.zeros([quat.shape[0], 3], dtype=float)
    for i in range(quat.shape[0]):
        q = quat[i,:]
        q_conj = quat_conj(q)

        eq[i,:] = 2 * quat_log_map(quat_prod(q_goal, q_conj))

    eqdot = np.gradient(eq, dt, axis=0, edge_order=2)
    eqddot = np.gradient(eqdot, dt, axis=0, edge_order=2)

    return qdot, w, wdot, eq, eqdot, eqddot

def read_csv(csv_path: Path):
    # read the csv
    rows = []
    with open(csv_path) as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            rows.append([float(row[key]) for key in INPUT_FILE_HEADER])
    
    # extract desired data
    arr = np.asarray(rows, dtype=float)

    t = arr[:,0]
    pos = np.vstack(arr[:,1:4])
    quat = np.vstack(arr[:,4:8])

    return t, pos, quat

def collect_inputs(in_root: Path, task: str):
    # build the path of the folder
    folder = in_root / task

    # check if it is a folder
    if not folder.exists():
        raise SystemError(f"Folder not found: '{folder}'")
    
    # collect all files
    files_path = sorted(f for f in folder.iterdir())

    return files_path

def process_demo_file(args: argparse, file: Path, index: int):
    # read the csv file
    t, pos, quat = read_csv(file)

    # create file paths for the output csv files
    out_root = Path(args.out_root)
    out_csv_folder_pos = out_root / args.task / "pos"
    out_csv_folder_quat = out_root / args.task / "rot"

    out_csv_folder_pos.mkdir(parents=True, exist_ok=True)
    out_csv_folder_quat.mkdir(parents=True, exist_ok=True)
    

    # apply preprocessing steps for the position DMP
    dt, pos_dot, pos_ddot = pos_preprocessing(t, pos)

    x, y, z = pos[:,0], pos[:,1], pos[:,2]
    xdot, ydot, zdot = pos_dot[:,0], pos_dot[:,1], pos_dot[:,2]
    xddot, yddot, zddot = pos_ddot[:,0], pos_ddot[:,1], pos_ddot[:,2]

    # store the preprocessed data of the position
    data = np.column_stack([t, dt, x, y, z, xdot, ydot, zdot, xddot, yddot, zddot])
    out_csv_file_pos = out_csv_folder_pos / f"prepocessed_dmp_pos_{args.task}_{index}.csv"
    with open(out_csv_file_pos, mode="w") as f:
        csv_writer = csv.DictWriter(f, OUT_FILE_POS_HEADER)
        csv_writer.writeheader()
        for i in range(data.shape[0]):
            csv_writer.writerow(dict(zip(OUT_FILE_POS_HEADER, data[i,:])))

    # apply preprocessing steps for the quaternion DMP
    qdot, w, wdot, eq, eqdot, eqddot = quat_preprocessing(t, quat)

    qx, qy, qz, qw = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
    qxdot, qydot, qzdot, qwdot, = qdot[:,0], qdot[:,1], qdot[:,2], qdot[:,3]
    wx, wy, wz = w[:,0], w[:,1], w[:,2]
    wxdot, wydot, wzdot = wdot[:,0], wdot[:,1], wdot[:,2]
    eqx, eqy, eqz = eq[:,0], eq[:,1], eq[:,2]
    eqxdot, eqydot, eqzdot = eqdot[:,0], eqdot[:,1], eqdot[:,2]
    eqxddot, eqyddot, eqzddot = eqddot[:,0], eqddot[:,1], eqddot[:,2]

    data = np.column_stack([t, dt, qx, qy, qz, qw, qxdot, qydot, qzdot, qwdot,
                            wx, wy, wz, wxdot, wydot, wzdot,
                            eqx, eqy, eqz,
                            eqxdot, eqydot, eqzdot, eqxddot, eqyddot, eqzddot])
    
    out_csv_file_pos = out_csv_folder_quat / f"prepocessed_dmp_quat_{args.task}_{index}.csv"
    with open(out_csv_file_pos, mode="w") as f:
        csv_writer = csv.DictWriter(f, OUT_FILE_QUAT_HEADER)
        csv_writer.writeheader()
        for i in range(data.shape[0]):
            csv_writer.writerow(dict(zip(OUT_FILE_QUAT_HEADER, data[i,:])))

def args_parse():
    parser = argparse.ArgumentParser(description="Apply the preprocessing pipeline required to train the DMP models")
    
    # add all desired parameters
    parser.add_argument("--task",
                        required=True,
                        help="Name of the primitive to train a DMP for (e.g. pick, place, pour)")
    parser.add_argument("--index",
                        default= None,
                        help="Number of the deomnstration used to train the models")
    parser.add_argument("--in-root",
                        type=str,
                        default=PREPROCESSED_ROOT,
                        help="Path of the .csv files containing global preprocessed demos")
    parser.add_argument("--out-root",
                        type=str,
                        default=DMP_PREPROCESSED_ROOT,
                        help="Path of the folder where to save the output .csv files of this processing")

    # parse the arguments and return
    return parser.parse_args()

def main():
    # parse CLI arguments
    args = args_parse()

    # collect all csv files with demonstrations of desired task
    files_path = collect_inputs(args.in_root, args.task)

    # if no index is provided, process all demo files of that task
    if args.index is None:
        for index, file in enumerate(files_path):
            process_demo_file(args, file, index)
    else:
        file = files_path[int(args.index)]
        process_demo_file(args, file, args.index)

    





if __name__ == "__main__":
    main()





