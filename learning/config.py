import os
import torch
from pathlib import Path

LEARNING_DIR = Path(__file__).parent
PROJECT_ROOT = LEARNING_DIR.parent

# --- Симуляция ---
HEADLESS = True
CAVE_CHANGE_INTERVAL = 5
EPISODE_TIMEOUT_SEC = 300
DT = 0.05
SIM_STEP_TIMEOUT = 2

# --- Топики ---
TOPIC_STEREO_DISTANCES = "/navigation_node/stereo_distances"
TOPIC_ODOM = "/odom"
TOPIC_IMU = "/imu/data"
TOPIC_CMD_VEL = "/cmd_vel"
TOPIC_COLLISIONS = "/drone/collisions"
SERVICE_SET_ENTITY_STATE = "/gazebo/set_entity_state"

# --- BC (Behavior Cloning) ---
BC_EXPERT_SAMPLES = 50_000
BC_EPOCHS = 500
BC_LR = 1e-3
BC_LR_PATIENCE = 30
BC_LR_FACTOR = 0.5
BC_MIN_LR = 1e-6
BC_BATCH_SIZE = 512
BC_HIDDEN = [128, 128]
BC_VAL_SPLIT = 0.15

# --- PPO ---
TOTAL_TIMESTEPS = 1_000_000
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
POLICY_KWARGS = {
    "net_arch": [64, 64],
    "activation_fn": torch.nn.Tanh,
}

# --- Чекпоинты ---
CHECKPOINT_DIR = LEARNING_DIR / "checkpoints"
SAVE_FREQ = 20_000
LOG_DIR = LEARNING_DIR / "tensorboard_logs"
EXPERT_DIR = LEARNING_DIR / "expert_data"

# --- Пути к скриптам проекта ---
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CAVE_SCRIPT = SCRIPTS_DIR / "procedural_cave.py"
CAVE_WORLD_PATH = (
    Path(__file__).parent.parent
    / "src"
    / "drone_simulation"
    / "worlds"
    / "cave.world"
)

# --- Награды (per-step) ---
R_SURVIVE = 0.005                    # базовая награда за выживание
R_SPEED_COEFF = 0.02                 # награда за скорость (× vx м/с)
R_PROXIMITY_COEFF = -0.3             # штраф за близость к препятствию
R_PROXIMITY_THRESHOLD = 1.0          # метров — активация штрафа
R_APPROACH_COEFF = -2.0              # штраф за скорость сближения (× м/шаг)
R_APPROACH_THRESHOLD = 0.005         # м/шаг — мин. сближение для штрафа
R_OPEN_SPACE_PENALTY = -0.08         # штраф вне пещеры (оба борта > 8м)
R_CENTERING_COEFF = -0.005           # штраф за асимметрию (× |l-r| или |t-b|)
R_CENTERING_THRESHOLD = 4.0          # метров — активация центрирования
R_VERTICAL_BOUNCE_COEFF = -0.005     # штраф за болтанку по высоте
R_VERTICAL_BOUNCE_THRESHOLD = 3.0    # метров — оба канала чисты
R_ANTICIPATION_COEFF = 0.02          # награда за упреждающий манёвр (× |vz|)
R_PASSAGE_REWARD = 0.15              # разовая награда за проход препятствия

# --- Награды (терминальные, однократно) ---
R_COLLISION = -15.0
R_COMPLETION = 100.0
R_STUCK = -5.0
R_TIMEOUT = -5.0
R_OUT_OF_BOUNDS = -15.0

# --- Сброс ---
DRONE_NAME = "rescue_drone"
DRONE_SPAWN_Z = 1.0
DRONE_MIN_Z = 0.1
DRONE_MAX_Z = 3.5
DRONE_MAX_YAW = 1.3
BOUNDS_XY = 50.0

# --- Параметры наблюдения ---
OBS_STEREO_MAX = 10.0
OBS_POS_MAX = 50.0
OBS_Z_MAX = 3.5

# --- Action limits ---
ACTION_VX_MIN = 0.0
ACTION_VX_MAX = 2.0
ACTION_VZ_MIN = -0.5
ACTION_VZ_MAX = 0.5
ACTION_YAW_MIN = -1.0
ACTION_YAW_MAX = 1.0


def normalize_action(raw_action):
    """Map raw cmd_vel (physical) to normalized [-1, 1] for BC→PPO weight transfer."""
    import numpy as np
    vx = np.interp(raw_action[..., 0], [ACTION_VX_MIN, ACTION_VX_MAX], [-1.0, 1.0])
    vz = np.interp(raw_action[..., 1], [ACTION_VZ_MIN, ACTION_VZ_MAX], [-1.0, 1.0])
    yaw = np.interp(raw_action[..., 2], [ACTION_YAW_MIN, ACTION_YAW_MAX], [-1.0, 1.0])
    return np.stack([vx, vz, yaw], axis=-1).astype(np.float32)


def denormalize_action(norm_action):
    """Map normalized [-1, 1] back to raw cmd_vel (physical)."""
    import numpy as np
    vx = np.interp(norm_action[..., 0], [-1.0, 1.0], [ACTION_VX_MIN, ACTION_VX_MAX])
    vz = np.interp(norm_action[..., 1], [-1.0, 1.0], [ACTION_VZ_MIN, ACTION_VZ_MAX])
    yaw = np.interp(norm_action[..., 2], [-1.0, 1.0], [ACTION_YAW_MIN, ACTION_YAW_MAX])
    return np.stack([vx, vz, yaw], axis=-1).astype(np.float32)
