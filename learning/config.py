"""
Configuration for Drone RL Training (MAVRL architecture).

Architecture: CNN encoder → LSTM → Actor/Critic
Input: depth map 256×256 + 7-dim goal-oriented state
Action: 4-dim velocity commands (vx, vy, vz, yaw_rate)
"""

from pathlib import Path
import numpy as np

LEARNING_DIR = Path(__file__).parent
PROJECT_ROOT = LEARNING_DIR.parent

# --- Simulation ---
HEADLESS = True
CAVE_CHANGE_INTERVAL = 1  # Change cave every rollout (matching MAVRL)
EPISODE_TIMEOUT_SEC = 600
DT = 0.1  # 10Hz (matching MAVRL sim_dt=0.1)
SIM_STEP_TIMEOUT = 2

# --- ROS 2 Topics ---
TOPIC_DEPTH_MAP = "/navigation_node/depth_map"
TOPIC_STEREO_DISTANCES = "/navigation_node/stereo_distances"
TOPIC_ODOM = "/odom"
TOPIC_IMU = "/imu/data"
TOPIC_CMD_VEL = "/cmd_vel"
TOPIC_COLLISIONS = "/drone/collisions"
SERVICE_SET_ENTITY_STATE = "/gazebo/set_entity_state"

# --- Depth Map ---
DEPTH_WIDTH = 256
DEPTH_HEIGHT = 256
DEPTH_MIN = 0.1    # meters
DEPTH_MAX = 12.0   # meters
DEPTH_CHANNELS = 1  # grayscale

# --- Observation Space ---
# Image: depth map 256×256 (uint8, 0-255)
# State: 7-dim goal-oriented (MAVRL style)
#   [log_distance, horizon_vel, theta, horizon_vel_dire, delta_z, vel_body_z, yaw]
STATE_DIM = 7

# --- Action Space (body-frame accelerations, matching MAVRL) ---
ACTION_DIM = 4
# Acceleration limits (m/s² for linear, rad/s for angular)
# MAVRL: act_max=[4.0, 4.0, 1.0, 0.6], act_min=[-4.0, -4.0, -1.0, -0.6]
ACTION_ACC_MAX = np.array([4.0, 4.0, 1.0])
ACTION_YAW_RATE_MAX = 0.6
# For normalization: action = raw * ACTION_STD + ACTION_MEAN
ACTION_MEAN = np.array([0.0, 0.0, 0.0, 0.0])
ACTION_STD = np.array([4.0, 4.0, 1.0, 0.6])

# --- Goal-point ---
# Goal = 80% of cave length along entrance heading
GOAL_DISTANCE_RATIO = 0.8
CAVE_LENGTH = 100.0  # meters (all generators produce 100m caves)
GOAL_Z = 1.75  # target altitude
GOAL_REACHED_THRESHOLD = 2.0  # meters

# --- Network Architecture (MAVRL) ---
FEATURES_DIM = 64       # VAE latent dim
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 1
ACTOR_HIDDEN = [256, 256]
CRITIC_HIDDEN = [512, 512]
ACTIVATION_FN = "relu"

# --- RecurrentPPO Hyperparameters ---
TOTAL_TIMESTEPS = 8_000_000
LEARNING_RATE = 1e-4
LEARNING_RATE_END = 0.0  # Linear decay: start 1e-4, end 0 (matching MAVRL)
N_STEPS = 1000          # steps per rollout per env
N_SEQ = 1               # sequence length for LSTM
BATCH_SIZE = 4000
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.0
VF_COEF = 0.2
MAX_GRAD_NORM = 0.5
USE_TANH_ACT = True

# --- Checkpoints ---
CHECKPOINT_DIR = LEARNING_DIR / "checkpoints"
SAVE_FREQ = 20_000
EVAL_FREQ = 200_000     # Evaluate every 200K steps (matching MAVRL)
EVAL_EPISODES = 3       # Number of eval episodes per evaluation
LOG_DIR = LEARNING_DIR / "tensorboard_logs"
DATA_DIR = LEARNING_DIR / "depth_data"

# --- Paths ---
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CAVE_WORLD_PATH = (
    Path(__file__).parent.parent
    / "src"
    / "drone_simulation"
    / "worlds"
    / "cave.world"
)

# --- Rewards (per-step, matching MAVRL) ---
# MAVRL config.yaml coefficients:
#   distance_coeff: -0.000, angle_vel_coeff: -0.000, input_coeff: -0.0003
#   yaw_coeff: -0.000, vert_coeff: -0.002, vel_coeff: -0.000, colli_coeff: -0.00
# MAVRL uses ONLY vert_coeff + input_coeff. All others = 0.
# We keep R_GOAL_COEFF because we don't have AvoidBench's built-in goal reward.

R_GOAL_COEFF = 5.0        # progress toward goal (our addition, MAVRL has built-in)

# MAVRL action penalties (smooth flight)
R_ANGULAR_PENALTY = 0.0   # angle_vel_coeff: MAVRL = 0.0
R_INPUT_PENALTY = -0.0003 # input_coeff: MAVRL = -0.0003
R_YAW_PENALTY = 0.0       # yaw_coeff: MAVRL = 0.0
R_VERTICAL_PENALTY = -0.002  # vert_coeff: MAVRL = -0.002

# --- Rewards (terminal) ---
R_COLLISION = 0.0         # MAVRL: reset_if_collide=true, no penalty
R_COMPLETION = 50.0       # reached goal (our addition)
R_STUCK = -5.0            # our addition (MAVRL has no stuck detection)
R_TIMEOUT = 0.0           # MAVRL: max_t=5.0s, no penalty, just reset
R_OUT_OF_BOUNDS = -10.0   # our addition (MAVRL has bounding box)

# --- Reset ---
DRONE_NAME = "rescue_drone"
DRONE_SPAWN_Z = 1.0
DRONE_MIN_Z = 0.1
DRONE_MAX_Z = 3.5
BOUNDS_XY = 120.0

# --- Curriculum (parametric, MAVRL-style) ---
CURRICULUM_STAGES = [
    {
        "name": "easy",
        "cave_script": "straight_cave.py",
        "cave_width": 8.0,
        "turn_angle_max": 0,
        "obstacle_density": 0.0,
    },
    {
        "name": "medium",
        "cave_script": "gentle_cave.py",
        "cave_width": 5.0,
        "turn_angle_max": 30,
        "obstacle_density": 0.3,
    },
    {
        "name": "hard",
        "cave_script": "procedural_cave.py",
        "cave_width": 3.5,
        "turn_angle_max": 55,
        "obstacle_density": 0.6,
    },
]
CURRICULUM_STAGE_STEPS = 2_000_000
CURRICULUM_ADVANCE_THRESHOLD = 0.8
CURRICULUM_REGRESS_THRESHOLD = 0.3
