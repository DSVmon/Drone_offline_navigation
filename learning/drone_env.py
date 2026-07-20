"""
Gymnasium environment for drone RL training (MAVRL architecture).

Observation: Dict{'image': depth map 256×256, 'state': 7-dim goal-oriented}
Action: 4-dim body-frame accelerations (ax, ay, az, yaw_rate)
Reward: goal-oriented with adaptive speed
"""

import time
import math
import threading
from collections import deque

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ContactsState, EntityState
from gazebo_msgs.srv import SetEntityState
from cv_bridge import CvBridge

import config


class DroneEnv(gym.Env):
    """Gymnasium environment wrapping ROS 2 / Gazebo for MAVRL training."""

    def __init__(self, headless=None, seed=None, node_name="drone_env_node"):
        super().__init__()

        self.headless = headless if headless is not None else config.HEADLESS
        self.episode_count = 0
        self.gazebo_proc = None
        self.bridge = CvBridge()

        # --- Observation space (Dict) ---
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255,
                shape=(config.DEPTH_CHANNELS, config.DEPTH_HEIGHT, config.DEPTH_WIDTH),
                dtype=np.uint8,
            ),
            'state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(config.STATE_DIM,),
                dtype=np.float64,
            ),
        })

        # --- Action space (body-frame accelerations) ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(config.ACTION_DIM,),
            dtype=np.float32,
        )

        # --- Internal state ---
        self.depth_image = np.zeros(
            (config.DEPTH_HEIGHT, config.DEPTH_WIDTH), dtype=np.uint8
        )
        self.stereo_distances = [10.0, 10.0, 10.0, 10.0, 10.0]
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = config.DRONE_SPAWN_Z
        self.current_yaw = 0.0
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_quat = np.array([0.0, 0.0, 0.0, 1.0])  # Full quaternion (x,y,z,w)
        self.odom_vx = 0.0
        self.odom_vy = 0.0
        self.odom_vz = 0.0
        self.prev_z = config.DRONE_SPAWN_Z  # For vz estimation (planar_move has no Z velocity)
        self.vz_estimated = 0.0  # Estimated vertical velocity from position change
        self.collision_detected = False
        self.collision_type = "NONE"
        self._collision_countdown = 0

        # Velocity tracking (for acceleration-based control)
        self.vel_world = np.array([0.0, 0.0, 0.0])

        # Goal-point
        self.goal_point = np.array([config.CAVE_LENGTH * config.GOAL_DISTANCE_RATIO, 0.0, config.GOAL_Z])
        self.entrance_heading = None

        # Previous state
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_yaw = 0.0

        # Step tracking
        self._step_count = 0
        self._episode_start_time = None
        self._obs_timestamp = 0
        self._last_step_timestamp = 0
        self._near_wall_count = 0

        # Stuck detection
        self._pos_history = deque(maxlen=100)

        # Image memory (for LSTM sequences)
        self._image_memory = deque(maxlen=10)
        self._state_memory = deque(maxlen=10)

        # Gazebo restart
        self._gazebo_restart_needed = False
        self._gazebo_restart_count = 0
        self._gazebo_stale_steps = 0

        # Previous action
        self.last_vx_cmd = 0.0
        self.last_vz_cmd = 0.0
        self.last_yaw_cmd = 0.0

        # Action tracking for penalties (MAVRL-style)
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_angular_vel = 0.0
        self.prev_vz_input = 0.0
        self._saved_prev_action = np.zeros(4, dtype=np.float32)
        self._saved_prev_angular_vel = 0.0
        self._saved_prev_vz_input = 0.0

        # ROS 2
        if not rclpy.ok():
            rclpy.init()
        self.node = Node(node_name)

        # Subscribers
        self._sub_depth = self.node.create_subscription(
            Image, config.TOPIC_DEPTH_MAP, self._depth_callback, 10
        )
        self._sub_stereo = self.node.create_subscription(
            Float32MultiArray, config.TOPIC_STEREO_DISTANCES,
            self._stereo_callback, 10
        )
        self._sub_odom = self.node.create_subscription(
            Odometry, config.TOPIC_ODOM, self._odom_callback, 10
        )
        self._sub_collisions = self.node.create_subscription(
            ContactsState, config.TOPIC_COLLISIONS, self._collision_callback, 10
        )

        # Publishers
        self._cmd_vel_pub = self.node.create_publisher(Twist, config.TOPIC_CMD_VEL, 10)

        # Gazebo service for Z-axis
        self._gz_set_state_client = self.node.create_client(
            SetEntityState, config.SERVICE_SET_ENTITY_STATE
        )

        # Background ROS spin
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_running = True
        self._spin_thread.start()

        self._lock = threading.Lock()

    def _spin(self):
        while self._spin_running and rclpy.ok():
            try:
                rclpy.spin_once(self.node, timeout_sec=0.01)
            except Exception:
                pass

    # --- Callbacks ---

    def _depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            # depth is uint8 [0,255] from navigation_node (MAVRL format)
            # Resize to target size if needed
            if depth.shape != (config.DEPTH_HEIGHT, config.DEPTH_WIDTH):
                depth = cv2.resize(depth, (config.DEPTH_WIDTH, config.DEPTH_HEIGHT))
            with self._lock:
                self.depth_image = depth
                self._obs_timestamp += 1
        except Exception as e:
            self.node.get_logger().warn(f"Depth callback error: {e}")

    def _stereo_callback(self, msg):
        if len(msg.data) >= 5:
            with self._lock:
                self.stereo_distances = list(msg.data[:5])

    def _odom_callback(self, msg):
        with self._lock:
            self.current_x = msg.pose.pose.position.x
            self.current_y = msg.pose.pose.position.y
            self.current_z = msg.pose.pose.position.z
            self.odom_vx = msg.twist.twist.linear.x
            self.odom_vy = msg.twist.twist.linear.y
            # planar_move doesn't publish Z velocity — estimate from position change
            self.vz_estimated = (self.current_z - self.prev_z) / config.DT if config.DT > 0 else 0.0
            self.prev_z = self.current_z
            self.vel_world = np.array([self.odom_vx, self.odom_vy, self.vz_estimated])

            q = msg.pose.pose.orientation
            sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
            cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
            self.current_roll = math.atan2(sinr_cosp, cosr_cosp)

            sinp = 2.0 * (q.w * q.y - q.z * q.x)
            if abs(sinp) >= 1.0:
                self.current_pitch = math.copysign(math.pi / 2.0, sinp)
            else:
                self.current_pitch = math.asin(sinp)

            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

            # Store full quaternion for world2body (matching MAVRL)
            self.current_quat = np.array([q.x, q.y, q.z, q.w])

            self._obs_timestamp += 1

    def _collision_callback(self, msg):
        with self._lock:
            if len(msg.states) > 0:
                self.collision_detected = True
                self.collision_type = "CONTACT"
                self._collision_countdown = 20
            else:
                if self._collision_countdown > 0:
                    self._collision_countdown -= 1
                else:
                    self.collision_detected = False

    def _check_depth_collision(self):
        """
        Detect collision using depth map analysis.
        Fallback for when bumper sensor doesn't work (planar_move limitation).

        Checks ALL regions of depth map:
        - Left/Right: walls
        - Top/Bottom: ceiling/floor
        - Center: obstacles ahead
        Priority: edges first (walls/ceiling/floor), then center (obstacles)
        """
        with self._lock:
            depth = self.depth_image.copy()

        if depth is None or depth.size == 0:
            return False

        h, w = depth.shape
        COLLISION_THRESHOLD = 10  # uint8, ~0.47m

        # Check all zones
        center_min = depth[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)].min()
        left_min = depth[h//4:3*h//4, :w//3].min()
        right_min = depth[h//4:3*h//4, 2*w//3:].min()
        top_min = depth[:h//4, w//4:3*w//4].min()
        bottom_min = depth[3*h//4:, w//4:3*w//4].min()

        all_min = min(center_min, left_min, right_min, top_min, bottom_min)

        if all_min < COLLISION_THRESHOLD:
            # Priority: edges first, then center
            if top_min < COLLISION_THRESHOLD:
                self.collision_type = "CEILING"
            elif bottom_min < COLLISION_THRESHOLD:
                self.collision_type = "FLOOR"
            elif left_min < COLLISION_THRESHOLD:
                self.collision_type = "WALL"
            elif right_min < COLLISION_THRESHOLD:
                self.collision_type = "WALL"
            else:
                self.collision_type = "OBSTACLE"
            return True

        return False

    # --- Observation building ---

    def _world2body(self, world_vel):
        """Convert world-frame velocity to body-frame (RFU). Matching MAVRL."""
        from scipy.spatial.transform import Rotation
        # Use full quaternion from odometry (matching MAVRL RobotState.world2body)
        rot = Rotation.from_quat(self.current_quat)  # [x, y, z, w]
        # World RFU → FLU
        world_flu = np.array([world_vel[1], -world_vel[0], world_vel[2]])
        # Rotate to body FLU
        body_flu = rot.inv().apply(world_flu)
        # FLU → RFU
        return np.array([body_flu[1], -body_flu[0], body_flu[2]])

    def _build_observation(self):
        """Build MAVRL-style observation: Dict{image, state}."""
        with self._lock:
            pos = np.array([self.current_x, self.current_y, self.current_z])
            # Use estimated vz (planar_move doesn't publish Z velocity)
            vel = np.array([self.odom_vx, self.odom_vy, self.vz_estimated])
            yaw = self.current_yaw

        # Goal-oriented state (7-dim, MAVRL style)
        delta_p = self.goal_point - pos
        horizon_dist = math.sqrt(delta_p[0] ** 2 + delta_p[1] ** 2)
        log_distance = math.log(horizon_dist + 1.0)

        vel_body = self._world2body(vel)
        horizon_vel = math.sqrt(vel_body[0] ** 2 + vel_body[1] ** 2)

        theta = math.atan2(-delta_p[0], delta_p[1])
        horizon_vel_dire = math.atan2(vel_body[1], vel_body[0])

        state = np.array([
            log_distance,
            horizon_vel,
            theta,
            horizon_vel_dire,
            delta_p[2],
            vel_body[2],
            yaw,
        ], dtype=np.float64)

        return {'image': self.depth_image, 'state': state}

    # --- Action application ---

    def _apply_action(self, action):
        """Denormalize action and apply as body-frame acceleration.
        Matching MAVRL: no velocity clipping."""
        action_arr = np.array(action, dtype=np.float32)
        cmd = action_arr * config.ACTION_STD + config.ACTION_MEAN

        acc_body = cmd[:3]  # body-frame acceleration
        yaw_rate = cmd[3]

        # Integrate: vel_world += R(body→world) * acc_body * dt
        # Matching MAVRL: self.vel_world = self.vel + acc_world * duration
        acc_world = self._body2world(acc_body)
        self.vel_world = self.vel_world + acc_world * config.DT

        # Publish velocity command (no clipping, like MAVRL)
        # Note: planar_move only handles X/Y. Z velocity is published but ignored by plugin.
        msg = Twist()
        msg.linear.x = float(self.vel_world[0])
        msg.linear.y = float(self.vel_world[1])
        msg.linear.z = float(self.vel_world[2])
        msg.angular.z = float(yaw_rate)
        self._cmd_vel_pub.publish(msg)

        self.last_vx_cmd = float(self.vel_world[0])
        self.last_vz_cmd = float(self.vel_world[2])
        self.last_yaw_cmd = float(yaw_rate)

        # Save action for penalty computation (MAVRL-style)
        self.prev_action = action_arr.copy()
        self.prev_angular_vel = yaw_rate
        self.prev_vz_input = float(cmd[2])

    def _body2world(self, acc_body):
        """Body-frame acceleration → world-frame (RFU). Matching MAVRL."""
        from scipy.spatial.transform import Rotation
        # Use full quaternion from odometry (matching MAVRL RobotState.body2world)
        rot = Rotation.from_quat(self.current_quat)  # [x, y, z, w]
        # Body RFU → FLU
        flu = np.array([acc_body[1], -acc_body[0], acc_body[2]])
        # Rotate to world FLU
        world_flu = rot.apply(flu)
        # FLU → RFU
        return np.array([-world_flu[1], world_flu[0], world_flu[2]])

    def _set_gazebo_z(self, vz):
        """Set Z-axis via Gazebo service with speed limiting."""
        if not self._gz_set_state_client.service_is_ready():
            return

        with self._lock:
            cx, cy, cz = self.current_x, self.current_y, self.current_z
            quat = self.current_quat  # Full quaternion [x, y, z, w]

        # Limit Z speed: max 0.3m per step (≈3m/s at 10Hz)
        max_z_step = 0.3
        z_step = np.clip(vz * config.DT, -max_z_step, max_z_step)
        target_z = cz + z_step
        target_z = max(config.DRONE_MIN_Z, min(config.DRONE_MAX_Z, target_z))

        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = config.DRONE_NAME
        req.state.pose.position.x = cx
        req.state.pose.position.y = cy
        req.state.pose.position.z = target_z
        req.state.pose.orientation.x = float(quat[0])
        req.state.pose.orientation.y = float(quat[1])
        req.state.pose.orientation.z = float(quat[2])
        req.state.pose.orientation.w = float(quat[3])
        req.state.twist.linear.z = float(vz)
        req.state.reference_frame = "world"

        try:
            future = self._gz_set_state_client.call_async(req)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=0.1)
        except Exception:
            pass

    # --- Stuck detection ---

    def _check_stuck(self, vx):
        """Check if drone is stuck."""
        with self._lock:
            cx, cy = self.current_x, self.current_y

        self._pos_history.append((cx, cy))

        is_moving_forward = vx > 0.1

        if not is_moving_forward:
            self._near_wall_count += 1
        else:
            self._near_wall_count = max(0, self._near_wall_count - 5)

        # Stuck = not moving forward for >10 seconds (200 steps)
        return self._near_wall_count > 200

    # --- Step ---

    def step(self, action):
        # Save previous action BEFORE applying new one (for reward penalties)
        self._saved_prev_action = self.prev_action.copy()
        self._saved_prev_angular_vel = self.prev_angular_vel
        self._saved_prev_vz_input = self.prev_vz_input
        
        self._apply_action(action)
        time.sleep(config.DT)

        if not self._wait_for_new_obs():
            self._gazebo_stale_steps += 1
            if not rclpy.ok() or self._gazebo_stale_steps >= 10:
                self._gazebo_restart_needed = True
                self._gazebo_stale_steps = 0
                obs = self._build_observation()
                self._step_count += 1
                return obs, 0.0, True, False, {"termination_reason": "gazebo_crash"}
        else:
            self._gazebo_stale_steps = 0

        obs = self._build_observation()
        self._step_count += 1

        with self._lock:
            collision = self.collision_detected
            cx, cy, cz = self.current_x, self.current_y, self.current_z
            vx = self.odom_vx

        # Depth-based collision detection (backup for bumper)
        if not collision:
            collision = self._check_depth_collision()
            if collision:
                with self._lock:
                    self.collision_detected = True
                    self.collision_countdown = 20

        stuck = self._check_stuck(vx)
        elapsed = time.time() - self._episode_start_time if self._episode_start_time else 0.0

        info = {
            "termination_reason": None,
            "collision": collision,
            "x": cx, "y": cy, "z": cz,
            "yaw": self.current_yaw,
            "stuck": stuck,
            "elapsed": elapsed,
            "step": self._step_count,
        }

        # Compute reward
        reward, terminated, info = self._compute_reward(
            pos=np.array([cx, cy, cz]),
            prev_pos=np.array([self.prev_x, self.prev_y]),
            vel_world=self.vel_world.copy(),
            yaw=self.current_yaw,
            collision=collision,
            stuck=stuck,
            elapsed=elapsed,
            info=info,
        )

        with self._lock:
            if collision:
                self.collision_detected = False

        self.prev_x = cx
        self.prev_y = cy
        self.prev_yaw = self.current_yaw

        truncated = False
        if not terminated and elapsed > config.EPISODE_TIMEOUT_SEC:
            truncated = True
            info["termination_reason"] = "timeout"

        if terminated or truncated:
            success = info.get("termination_reason") == "completed_lap"
            reason = info.get("termination_reason", "")
            self.node.get_logger().info(
                f"[EP {self.episode_count}] {reason} | steps={self._step_count} | reward={reward:.2f}"
            )

        return obs, reward, terminated, truncated, info

    # --- Reward ---

    def _compute_reward(self, pos, prev_pos, vel_world, yaw, collision, stuck, elapsed, info):
        terminated = False
        reward = 0.0

        # 2D distance for goal progress (ignore Z — drone may fly at different altitudes)
        dist_to_goal = math.sqrt(
            (pos[0] - self.goal_point[0])**2 + (pos[1] - self.goal_point[1])**2
        )
        prev_dist = math.sqrt(
            (prev_pos[0] - self.goal_point[0])**2 + (prev_pos[1] - self.goal_point[1])**2
        )
        progress = prev_dist - dist_to_goal
        reward += config.R_GOAL_COEFF * progress

        # 2. Action penalties (matching MAVRL config.yaml exactly)
        # input_coeff = -0.0003: penalty for action changes between steps
        action_delta = np.linalg.norm(self.prev_action - self._saved_prev_action)
        reward += config.R_INPUT_PENALTY * action_delta

        # vert_coeff = -0.002: penalty for vertical input
        reward += config.R_VERTICAL_PENALTY * abs(self._saved_prev_vz_input)

        # Note: MAVRL has angle_vel_coeff=0, yaw_coeff=0, vel_coeff=0
        # We removed our extra angular/yaw penalties to match MAVRL

        # 3. Collision → reset (MAVRL: reset_if_collide=true, no reward penalty)
        if collision:
            info["termination_reason"] = "collision"
            terminated = True

        # 4. Stuck → reset (our addition, MAVRL has timeout instead)
        if stuck:
            reward += config.R_STUCK
            info["termination_reason"] = "stuck"
            terminated = True

        # 5. Out of bounds → reset (MAVRL has bounding_box check)
        out_of_bounds = (
            pos[2] < config.DRONE_MIN_Z
            or pos[2] > config.DRONE_MAX_Z
            or abs(pos[0]) > config.BOUNDS_XY
            or abs(pos[1]) > config.BOUNDS_XY
        )
        if out_of_bounds:
            reward += config.R_OUT_OF_BOUNDS
            info["termination_reason"] = "out_of_bounds"
            terminated = True

        # 6. Goal reached → success (MAVRL: terminal observation, positive reward)
        if dist_to_goal < config.GOAL_REACHED_THRESHOLD:
            reward += config.R_COMPLETION
            info["termination_reason"] = "completed_lap"
            terminated = True

        # 7. Timeout → reset (MAVRL: max_t=5.0, no penalty)
        if elapsed > config.EPISODE_TIMEOUT_SEC:
            info["termination_reason"] = "timeout"
            terminated = True

        return reward, terminated, info

    # --- Reset ---

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)

        self._step_count = 0
        self._near_wall_count = 0
        self._pos_history.clear()
        self.vel_world = np.array([0.0, 0.0, 0.0])
        self.prev_z = config.DRONE_SPAWN_Z
        self.vz_estimated = 0.0

        with self._lock:
            self.collision_detected = False
            self.collision_type = "NONE"
            self._collision_countdown = 0

        self.node.get_logger().info(f"[RESET] Episode {self.episode_count} starting...")

        # Determine cave script from curriculum
        cave_script = self._get_cave_script()

        # Reset or relaunch Gazebo
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self._gazebo_restart_needed or self.episode_count == 0:
                    self._gazebo_restart_needed = False
                    self._gazebo_restart_count += 1
                    if self._gazebo_restart_count > 5:
                        raise RuntimeError("Gazebo crashed too many times")
                    import utils
                    utils.kill_gazebo()
                    utils.generate_cave(cave_script=cave_script)
                    self.gazebo_proc = utils.launch_gazebo(headless=self.headless)
                    utils.wait_for_gazebo()
                    self._wait_for_odom(timeout=60.0)
                elif self.episode_count % config.CAVE_CHANGE_INTERVAL == 0:
                    import utils
                    if self.gazebo_proc is not None:
                        self.gazebo_proc.kill()
                        self.gazebo_proc.wait(timeout=5)
                    utils.kill_gazebo()
                    utils.generate_cave(cave_script=cave_script)
                    self.gazebo_proc = utils.launch_gazebo(headless=self.headless)
                    utils.wait_for_gazebo()
                    self._wait_for_odom(timeout=60.0)
                else:
                    import utils
                    try:
                        utils.reset_drone(self.node)
                    except RuntimeError:
                        utils.kill_gazebo()
                        if self.gazebo_proc is not None:
                            try:
                                self.gazebo_proc.kill()
                                self.gazebo_proc.wait(timeout=5)
                            except Exception:
                                pass
                        utils.generate_cave(cave_script=cave_script)
                        self.gazebo_proc = utils.launch_gazebo(headless=self.headless)
                        utils.wait_for_gazebo()
                        self._wait_for_odom(timeout=60.0)
                break
            except (TimeoutError, RuntimeError) as e:
                self.node.get_logger().error(f"Reset attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    import utils
                    utils.kill_gazebo()
                    time.sleep(5.0)
                    self._gazebo_restart_needed = True
                else:
                    raise

        self._wait_for_new_obs(timeout=15.0)

        # Set goal-point
        self.goal_point = np.array([
            config.CAVE_LENGTH * config.GOAL_DISTANCE_RATIO,
            0.0,
            config.GOAL_Z,
        ])

        # Set entrance heading
        with self._lock:
            self.entrance_heading = (
                math.cos(self.current_yaw),
                math.sin(self.current_yaw),
            )
            self.prev_x = self.current_x
            self.prev_y = self.current_y

        self.episode_count += 1
        self._episode_start_time = time.time()

        return self._build_observation(), {}

    def _get_cave_script(self):
        """Get cave script path based on curriculum stage."""
        # Simple: always use procedural cave for now
        # TODO: integrate curriculum manager
        return config.SCRIPTS_DIR / "procedural_cave.py"

    # --- Helpers ---

    def _wait_for_odom(self, timeout=30.0):
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if abs(self.current_z - config.DRONE_SPAWN_Z) < 0.5:
                    return True
            time.sleep(0.5)
        raise TimeoutError(f"Drone odometry not received after {timeout}s")

    def _wait_for_new_obs(self, timeout=None):
        if timeout is None:
            timeout = config.SIM_STEP_TIMEOUT
        start_ts = self._obs_timestamp
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._obs_timestamp > start_ts:
                return True
            time.sleep(0.005)
        return False

    def close(self):
        self._spin_running = False
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        if self.node is not None:
            self.node.destroy_node()
