import time
import math
import threading
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from gazebo_msgs.msg import ContactsState, EntityState
from gazebo_msgs.srv import SetEntityState

import config
from reward import compute_reward
import utils


class DroneEnv(gym.Env):
    """Gymnasium environment wrapping ROS 2 / Gazebo for drone RL training."""

    def __init__(self, headless=None, seed=None):
        super().__init__()

        self.headless = headless if headless is not None else config.HEADLESS
        self.episode_count = 0
        self.gazebo_proc = None

        # --- Observation space ---
        # 20-dim: prev_stereo[5], stereo[5], x, y, z, sin(yaw), cos(yaw), vx, vz, roll/pi, pitch/pi
        obs_low = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            -1.0, -1.0, 0.0,
            -1.0, -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
        ], dtype=np.float32)
        obs_high = np.array([
            1.5, 1.5, 1.5, 1.5, 1.5,
            1.5, 1.5, 1.5, 1.5, 1.5,
            1.0, 1.0, 1.0,
            1.0, 1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # --- Action space ---
        # linear.x, linear.z, angular.z
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # --- Internal state ---
        self.stereo_distances = [10.0, 10.0, 10.0, 10.0, 10.0]
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = config.DRONE_SPAWN_Z
        self.current_yaw = 0.0
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.odom_vx = 0.0
        self.odom_vy = 0.0
        self.odom_vz = 0.0
        self.collision_detected = False
        self.collision_type = "NONE"

        self.prev_x = 0.0
        self.prev_y = 0.0
        self.entrance_heading = None
        self.elapsed_time = 0.0
        self.stuck_start_time = None
        self.completed_lap = False

        self._obs_timestamp = 0
        self._last_step_timestamp = 0
        self._step_count = 0
        self._episode_start_time = None

        self._pos_history = deque(maxlen=100)
        self._near_wall_count = 0

        self._prev_stereo_distances = [10.0, 10.0, 10.0, 10.0, 10.0]

        self.last_vx_cmd = 0.0
        self.last_vz_cmd = 0.0
        self.last_yaw_cmd = 0.0

        # ROS 2
        if not rclpy.ok():
            rclpy.init()
        self.node = Node("drone_env_node")

        self._sub_stereo = self.node.create_subscription(
            Float32MultiArray,
            config.TOPIC_STEREO_DISTANCES,
            self._stereo_callback,
            10,
        )
        self._sub_odom = self.node.create_subscription(
            Odometry,
            config.TOPIC_ODOM,
            self._odom_callback,
            10,
        )
        self._sub_collisions = self.node.create_subscription(
            ContactsState,
            config.TOPIC_COLLISIONS,
            self._collision_callback,
            10,
        )

        self._cmd_vel_pub = self.node.create_publisher(
            Twist, config.TOPIC_CMD_VEL, 10
        )

        self._gz_set_state_client = self.node.create_client(
            SetEntityState, config.SERVICE_SET_ENTITY_STATE
        )

        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_running = True
        self._spin_thread.start()

        self._lock = threading.Lock()

    def _spin(self):
        """Background thread spinning ROS 2."""
        while self._spin_running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)

    def _stereo_callback(self, msg):
        if len(msg.data) >= 5:
            with self._lock:
                self.stereo_distances = list(msg.data[:5])
                self._obs_timestamp += 1

    def _odom_callback(self, msg):
        with self._lock:
            self.current_x = msg.pose.pose.position.x
            self.current_y = msg.pose.pose.position.y
            self.current_z = msg.pose.pose.position.z
            self.odom_vx = msg.twist.twist.linear.x
            self._obs_timestamp += 1
            self.odom_vy = msg.twist.twist.linear.y
            self.odom_vz = msg.twist.twist.linear.z
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

    def _collision_callback(self, msg):
        with self._lock:
            if len(msg.states) > 0:
                if not self.collision_detected:
                    self.collision_detected = True
                    self.collision_type = "CONTACT"
            else:
                self.collision_detected = False

    def _build_observation(self):
        with self._lock:
            p = self._prev_stereo_distances[:5]
            d = self.stereo_distances[:5]
            x = self.current_x
            y = self.current_y
            z = self.current_z
            yaw = self.current_yaw
            roll = self.current_roll
            pitch = self.current_pitch
            vx = self.odom_vx
            vz = self.odom_vz

        obs = np.array([
            p[0] / config.OBS_STEREO_MAX,
            p[1] / config.OBS_STEREO_MAX,
            p[2] / config.OBS_STEREO_MAX,
            p[3] / config.OBS_STEREO_MAX,
            p[4] / config.OBS_STEREO_MAX,
            d[0] / config.OBS_STEREO_MAX,
            d[1] / config.OBS_STEREO_MAX,
            d[2] / config.OBS_STEREO_MAX,
            d[3] / config.OBS_STEREO_MAX,
            d[4] / config.OBS_STEREO_MAX,
            x / config.OBS_POS_MAX,
            y / config.OBS_POS_MAX,
            z / config.OBS_Z_MAX,
            math.sin(yaw),
            math.cos(yaw),
            np.clip(vx, -1.0, 1.0),
            np.clip(vz, -1.0, 1.0),
            roll / math.pi,
            pitch / math.pi,
        ], dtype=np.float32)

        return obs

    def _apply_action(self, action):
        """Scale normalized action [-1,1] to physical commands and publish."""
        vx = np.interp(action[0], [-1.0, 1.0],
                        [config.ACTION_VX_MIN, config.ACTION_VX_MAX])
        vz = np.interp(action[1], [-1.0, 1.0],
                        [config.ACTION_VZ_MIN, config.ACTION_VZ_MAX])
        yaw = np.interp(action[2], [-1.0, 1.0],
                         [config.ACTION_YAW_MIN, config.ACTION_YAW_MAX])

        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.z = float(vz)
        msg.angular.z = float(yaw)

        self._cmd_vel_pub.publish(msg)

        self._set_gazebo_z(float(vz))

        self.last_vx_cmd = vx
        self.last_vz_cmd = vz
        self.last_yaw_cmd = yaw

    def _set_gazebo_z(self, vz):
        """Set Z-axis via Gazebo service (planar_move workaround)."""
        if not self._gz_set_state_client.wait_for_service(timeout_sec=0.1):
            return

        with self._lock:
            cx = self.current_x
            cy = self.current_y
            cz = self.current_z
            yaw = self.current_yaw

        descent_factor = max(0.0, 1.0 - abs(min(vz, 0.0)) / 0.3)
        hover_correction = (config.DRONE_SPAWN_Z - cz) * 0.3 * descent_factor
        target_z = cz + vz * config.DT + hover_correction
        target_z = max(config.DRONE_MIN_Z, min(config.DRONE_MAX_Z, target_z))

        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = config.DRONE_NAME
        req.state.pose.position.x = cx
        req.state.pose.position.y = cy
        req.state.pose.position.z = target_z
        q = self._yaw_to_quaternion(yaw)
        req.state.pose.orientation.x = q[0]
        req.state.pose.orientation.y = q[1]
        req.state.pose.orientation.z = q[2]
        req.state.pose.orientation.w = q[3]
        req.state.reference_frame = "world"

        try:
            self._gz_set_state_client.call(req)
        except Exception:
            pass

    def _yaw_to_quaternion(self, yaw):
        return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]

    def _wait_for_odom(self, timeout=None):
        """Wait until odometry reports drone at spawn position (drone exists)."""
        if timeout is None:
            timeout = 30.0
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if abs(self.current_z - config.DRONE_SPAWN_Z) < 0.1:
                    return True
            time.sleep(0.5)
        raise TimeoutError(f"Drone odometry not received after {timeout}s")

    def _wait_for_new_obs(self, timeout=None):
        """Block until a new observation arrives or timeout."""
        if timeout is None:
            timeout = config.SIM_STEP_TIMEOUT
        start_ts = self._obs_timestamp
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._obs_timestamp > start_ts:
                return True
            time.sleep(0.005)
        return False

    def _check_stuck(self, dist):
        """Check if drone is stuck using position history + wall proximity."""
        with self._lock:
            cx = self.current_x
            cy = self.current_y

        self._pos_history.append((cx, cy))
        min_wall = min(dist)

        # Wall-proximity stuck: drone against a wall for >2s (40 steps)
        if min_wall < 0.35:
            self._near_wall_count += 1
        else:
            self._near_wall_count = max(0, self._near_wall_count - 2)
        near_wall_stuck = self._near_wall_count > 40

        # Position-progress stuck: less than 0.15m traveled in last 100 steps
        pos_stuck = False
        if len(self._pos_history) == self._pos_history.maxlen:
            first_x, first_y = self._pos_history[0]
            last_x, last_y = self._pos_history[-1]
            dx = last_x - first_x
            dy = last_y - first_y
            if math.sqrt(dx*dx + dy*dy) < 0.15:
                pos_stuck = True

        return near_wall_stuck or pos_stuck

    def step(self, action):
        prev_dist = self._prev_stereo_distances

        self._apply_action(action)
        time.sleep(config.DT)
        self._wait_for_new_obs()

        obs = self._build_observation()
        self._step_count += 1
        self._last_step_timestamp = self._obs_timestamp

        with self._lock:
            dist = self.stereo_distances[:5]
            collision = self.collision_detected
            ctype = self.collision_type
            cx = self.current_x
            cy = self.current_y
            cz = self.current_z
            yaw = self.current_yaw
            vx = self.odom_vx
            vy = self.odom_vy
            vz = self.odom_vz

        stuck = self._check_stuck(dist)
        elapsed = time.time() - self._episode_start_time if self._episode_start_time else 0.0

        info = {
            "termination_reason": None,
            "collision": collision,
            "collision_type": ctype,
            "stereo_distances": dist,
            "x": cx,
            "y": cy,
            "z": cz,
            "yaw": yaw,
            "stuck": stuck,
            "elapsed": elapsed,
            "step": self._step_count,
        }

        reward, terminated, info = compute_reward(
            distances=dist,
            prev_distances=prev_dist,
            current_x=cx,
            current_y=cy,
            current_z=cz,
            prev_x=self.prev_x,
            prev_y=self.prev_y,
            odom_vx=vx,
            odom_vy=vy,
            odom_vz=vz,
            yaw=yaw,
            collision_detected=collision,
            collision_type=ctype,
            elapsed_time=elapsed,
            entrance_heading=self.entrance_heading,
            completed_lap=self.completed_lap,
            stuck=stuck,
            info=info,
        )

        with self._lock:
            if collision:
                self.collision_detected = False

        self.prev_x = cx
        self.prev_y = cy
        self._prev_stereo_distances = dist

        truncated = False

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)

        self._step_count = 0
        self.completed_lap = False
        self.stuck_start_time = None
        self.entrance_heading = None
        self.prev_x = 0.0
        self.prev_y = 0.0
        self._pos_history.clear()
        self._near_wall_count = 0

        with self._lock:
            self.collision_detected = False
            self.collision_type = "NONE"

        self.node.get_logger().info(
            f"[RESET] Episode {self.episode_count} starting..."
        )

        if self.episode_count == 0 or self.episode_count % config.CAVE_CHANGE_INTERVAL == 0:
            if self.gazebo_proc is not None:
                self.gazebo_proc.kill()
                self.gazebo_proc.wait(timeout=5)
            utils.kill_gazebo()
            utils.generate_cave()
            self.gazebo_proc = utils.launch_gazebo(headless=self.headless)
            utils.wait_for_gazebo()
            self._wait_for_odom(timeout=45.0)
        else:
            utils.reset_drone(self.node)

        self._wait_for_new_obs(timeout=15.0)

        obs = self._build_observation()

        with self._lock:
            self.entrance_heading = (
                math.cos(self.current_yaw),
                math.sin(self.current_yaw),
            )
            self.prev_x = self.current_x
            self.prev_y = self.current_y
            self._prev_stereo_distances = self.stereo_distances[:5].copy()

        self.episode_count += 1
        self._episode_start_time = time.time()
        self._last_step_timestamp = self._obs_timestamp

        info = {}
        return obs, info

    def close(self):
        self._spin_running = False
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        if self.node is not None:
            self.node.destroy_node()
