import os
import torch
from pathlib import Path

LEARNING_DIR = Path(__file__).parent
PROJECT_ROOT = LEARNING_DIR.parent

# --- Simulation ---
HEADLESS = True
CAVE_CHANGE_INTERVAL = 20
EPISODE_TIMEOUT_SEC = 600
DT = 0.05
SIM_STEP_TIMEOUT = 2

# --- ROS 2 Topics ---
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
BC_HIDDEN = [256, 256]
BC_VAL_SPLIT = 0.15

# --- PPO ---
TOTAL_TIMESTEPS = 2_000_000
LEARNING_RATE = 1e-4
N_STEPS = 4096
BATCH_SIZE = 128
N_EPOCHS = 20
GAMMA = 0.995
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.05
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
POLICY_KWARGS = {
    "net_arch": [256, 256],
    "activation_fn": torch.nn.ReLU,
}

# --- Checkpoints ---
CHECKPOINT_DIR = LEARNING_DIR / "checkpoints"
SAVE_FREQ = 20_000
LOG_DIR = LEARNING_DIR / "tensorboard_logs"
EXPERT_DIR = LEARNING_DIR / "expert_data"

# --- Paths to project scripts ---
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CAVE_WORLD_PATH = (
    Path(__file__).parent.parent
    / "src"
    / "drone_simulation"
    / "worlds"
    / "cave.world"
)

# --- Rewards (per-step) ---
R_SURVIVE = 0.01
R_SPEED_COEFF = 0.03
R_PROGRESS_COEFF = 5.0
R_PROXIMITY_COEFF = -0.05
R_PROXIMITY_THRESHOLD = 0.5
R_APPROACH_COEFF = -0.2
R_APPROACH_THRESHOLD = 0.005
R_CENTERING_COEFF = 0.01
R_CENTERING_THRESHOLD = 4.0

# --- Rewards (terminal) ---
R_COLLISION = -15.0
R_COMPLETION = 50.0
R_STUCK = -5.0
R_TIMEOUT = -3.0
R_OUT_OF_BOUNDS = -10.0

# --- Reset ---
DRONE_NAME = "rescue_drone"
DRONE_SPAWN_Z = 1.0
DRONE_MIN_Z = 0.1
DRONE_MAX_Z = 3.5
DRONE_MAX_YAW = 3.14  # Allow full rotation (π rad = 180°)
BOUNDS_XY = 110.0

# --- Observation normalization ---
OBS_STEREO_MAX = 10.0
OBS_POS_MAX = 50.0
OBS_Z_MAX = 3.5

# --- Action limits ---
ACTION_VX_MIN = 0.0
ACTION_VX_MAX = 1.5
ACTION_VZ_MIN = -0.5
ACTION_VZ_MAX = 0.5
ACTION_YAW_MIN = -1.0
ACTION_YAW_MAX = 1.0

# --- Curriculum ---
CURRICULUM_STAGES = [
    {"name": "straight", "cave_script": "straight_cave.py", "progress_coeff": 1.0, "proximity_coeff": -0.05, "proximity_threshold": 0.5},
    {"name": "gentle", "cave_script": "gentle_cave.py", "progress_coeff": 0.7, "proximity_coeff": -0.10, "proximity_threshold": 0.5},
    {"name": "full", "cave_script": "procedural_cave.py", "progress_coeff": 0.5, "proximity_coeff": -0.15, "proximity_threshold": 0.5},
]
CURRICULUM_STAGE_STEPS = 500_000
CURRICULUM_ADVANCE_THRESHOLD = 0.8
CURRICULUM_REGRESS_THRESHOLD = 0.3


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
