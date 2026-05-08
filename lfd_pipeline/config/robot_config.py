"""Configurazione condivisa per la pipeline LfD su xArm 6.

I valori qui sotto sono pensati per essere modificati a mano:
- ROBOT_IP: IP del controller xArm.
- HOME_JOINT_DEG: configurazione ai giunti (in gradi) assunta all'avvio
  di ogni script. Sei valori, uno per giunto del xArm 6.
- DEFAULT_RECORD_RATE_HZ: frequenza nominale di campionamento delle demo.
- GRIPPER_SPEED: velocità del gripper (unità SDK).
- DEMO_ROOT: cartella radice in cui salvare le dimostrazioni.
"""

from pathlib import Path

# --- Robot ---
ROBOT_IP = "192.168.1.221"

# Configurazione "home" ai giunti (gradi). Modifica i valori a piacere.
HOME_JOINT_DEG = [-97.8, 23.3, -83.7, -323, 109.1, 213]
# PLACE_JOINT_DEG = [27.6, 39.9, -49.1, -151.6, 84.3, 174.1]
PLACE_JOINT_DEG = [-57, 15.2, -32.7, -237.3, 82.9, 195.4]

# --- Recorder ---
DEFAULT_RECORD_RATE_HZ = 50.0

# --- Gripper ---
GRIPPER_SPEED = 2000  # unità SDK (velocità di default, es. apertura)
GRIPPER_CLOSE_SPEED = 500  # unità SDK, più lenta per registrazione a 50 Hz
GRIPPER_OPEN_POS = 850   # apertura massima (unità SDK)
GRIPPER_CLOSED_POS = 0   # chiusura completa (unità SDK)

# --- Output ---
# .../lfd_pipeline/demonstrations
DEMO_ROOT = Path(__file__).resolve().parent.parent / "demonstrations"
# .../lfd_pipeline/preprocessed_demonstrations
PREPROCESSED_ROOT = Path(__file__).resolve().parent.parent / "preprocessed_demonstrations"

# --- Preprocessing ---
DEFAULT_PREPROCESS_DT = 0.02  # passo della time-grid uniforme [s] (50 Hz)

# --- GMM-GMR ---
# Cartella per gli output del preprocessing GMM-specifico.
GMM_PREPROCESSED_ROOT = Path(__file__).resolve().parent.parent / "gmm_preprocessed_demonstrations"
# Cartella per i modelli addestrati (tutti i metodi).
TRAINED_MODELS_ROOT = Path(__file__).resolve().parent.parent / "trained_models"
# Iperparametri di default (range del paper, Tabella I).
GMM_DEFAULT_K = 3
GMM_DEFAULT_REG_COVAR = 1e-6
GMM_DEFAULT_MAX_ITER = 300
GMM_DEFAULT_TOL = 1e-4
GMM_DEFAULT_RANDOM_STATE = 0

# --- DMP ---
# Cartella per gli output del preprocessing DMP-specifico.
DMP_PREPROCESSED_ROOT = Path(__file__).resolve().parent.parent / "dmp_preprocessed_demonstrations"
# Iperparametri di default (range del paper, Tabella I).
DMP_DEFAULT_N_BFS = 100
DMP_DEFAULT_ALPHA_Z = 25.0
DMP_DEFAULT_ALPHA_S = 1.0
DMP_DEFAULT_TAU = 1.0
# Soglia (per asse) sotto cui un asse e' considerato "quasi statico" nelle
# dimostrazioni: per quelle dimensioni il diagonal scaling Ijspeert
# f = (g - y0) * f_norm viene disabilitato (si fitta la forzante in unita'
# assolute) per evitare che pesi enormi - dovuti a |g_demo - y0_demo| ~ 0 -
# generino traiettorie folli quando l'inferenza usa target generici con
# spostamento non trascurabile. Soglie distinte per posizione (m) e
# orientazione (rad). Vedi Park 2008 / Pastor 2009 / Ijspeert 2013 III-C.
DMP_STATIC_AXIS_POS_THRESH = 0.02   # m  (~2 cm)
DMP_STATIC_AXIS_ROT_THRESH = 0.10   # rad (~5.7 deg)

# --- BC ---
# Cartella per gli output del preprocessing BC-specifico.
BC_PREPROCESSED_ROOT = Path(__file__).resolve().parent.parent / "bc_preprocessed_demonstrations"
# Iperparametri di default (range del paper, Tabella I).
BC_DEFAULT_HIDDEN = (128, 128, 128, 64)   # 4 hidden layers, neuroni in [32, 256]
BC_DEFAULT_ACTIVATION = "relu"            # "relu" oppure "tanh"
BC_DEFAULT_LR = 1e-3                      # learning rate in [1e-3, 1e-2]
BC_DEFAULT_EPOCHS = 4000                  # epoche in [100, 450]
BC_DEFAULT_BATCH_SIZE = 128
BC_DEFAULT_WEIGHT_DECAY = 0.0
BC_DEFAULT_RANDOM_STATE = 0

# --- Inference / esecuzione ---
# Velocita' / accelerazione di default per l'esecuzione cartesiana del rollout.
EXEC_DEFAULT_SPEED = 100.0          # mm/s
EXEC_DEFAULT_ACC = 100.0           # mm/s^2
EXEC_DEFAULT_BLEND_RADIUS = 3.0     # mm

# --- RealSense / ArUco (vision wrist-mounted) ---
import numpy as _np  # solo per la matrice di calibrazione

REALSENSE_WIDTH = 1280
REALSENSE_HEIGHT = 720
REALSENSE_FPS = 30

# Dizionario e dimensione del marker (lato in metri).
ARUCO_DICT_NAME = "DICT_6X6_50"
ARUCO_MARKER_LEN_M = 0.030

# ID marker dell'oggetto (default per il task pick).
ARUCO_OBJECT_ID = 2

# Calibrazione hand-in-eye: T_ee^cam (gripper -> camera) in metri.
# Stessa matrice usata in lfd_evaluate/generalization_pick.py.
# Valori da calibrazione ufficiale UFactory per la "Camera Stand RealSense D435"
# (xyzrpy flangia->color-optical = [0.0671, -0.0311, 0.0216, -0.0042, -0.0085, 1.5899]),
# riportati al TCP del gripper attivo sul controller (tcp_offset_z = 172 mm):
#   tz_TCP->cam = 0.0216 - 0.172 = -0.1504 m
T_EE_CAM = _np.array([
    [+0.013372, -0.999875, -0.008474, +0.062742],
    [+0.999910, +0.013368, +0.000607, -0.030241],
    [-0.000493, -0.008481, +0.999964, -0.145046],
    [+0.000000, +0.000000, +0.000000, +1.000000]
], dtype=float)

# Offset lungo l'asse z del tag per portare il centro del marker tra le pinze
# del gripper (positivo = piu' in alto rispetto al tag, in metri).
GRASP_OFFSET_Z_M = -0.02

# Offset lungo +Z del tag per la posa di place (in metri). Il gripper si
# posiziona con +x allineato a +Z del tag e a questa distanza lungo +Z.
PLACE_OFFSET_Z_M = 0.2

# Offset lungo z del tag per la primitiva POUR (in metri).
POUR_OFFSET_Z_M = -0.04

