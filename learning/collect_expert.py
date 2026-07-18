#!/usr/bin/env python3
"""
Collect expert demonstration data from the running simulation.

Run this alongside the existing simulation with straight cave:
    Terminal 1: ./run_drone.sh straight_cave.py
    Terminal 2: python3 learning/collect_expert.py

Usage:
    python3 learning/collect_expert.py
"""

import time
import math
import signal
import sys
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

import config


class ExpertCollector(Node):
    def __init__(self):
        super().__init__("expert_collector")
        self.buffer_obs = []
        self.buffer_acts = []
        self.batch_size = 200
        self.total_collected = 0
        self.target_samples = config.BC_EXPERT_SAMPLES

        # Latest observations
        self.stereo_distances = [10.0, 10.0, 10.0, 10.0, 10.0]
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = config.DRONE_SPAWN_Z
        self.current_yaw = 0.0
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.odom_vx = 0.0

        # Latest action
        self.last_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.have_odom = False
        self.have_stereo = False

        # Subscribers
        self.create_subscription(
            Float32MultiArray,
            config.TOPIC_STEREO_DISTANCES,
            self._stereo_cb,
            10,
        )
        self.create_subscription(
            Odometry,
            config.TOPIC_ODOM,
            self._odom_cb,
            10,
        )
        self.create_subscription(
            Twist,
            config.TOPIC_CMD_VEL,
            self._cmd_vel_cb,
            10,
        )

        self.create_timer(0.05, self._collect_sample)  # 20Hz

        self.get_logger().info(
            f"[EXPERT] Collecting {self.target_samples} expert samples..."
        )

    def _stereo_cb(self, msg):
        if len(msg.data) >= 5:
            self.stereo_distances = list(msg.data[:5])
            self.have_stereo = True

    def _odom_cb(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_z = msg.pose.pose.position.z
        self.odom_vx = msg.twist.twist.linear.x
        self.odom_vz = msg.twist.twist.linear.z
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.y * q.x)
        cosy_cosp = 1.0 - 2.0 * (q.x * q.x + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.z)
        self.current_roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1.0:
            self.current_pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            self.current_pitch = math.asin(sinp)
        self.have_odom = True

    def _cmd_vel_cb(self, msg):
        raw = np.array([
            msg.linear.x,
            msg.linear.z,
            msg.angular.z,
        ], dtype=np.float32)
        self.last_action = config.normalize_action(raw[np.newaxis, :])[0]

    def _build_observation(self):
        d = self.stereo_distances[:5]
        return np.array([
            d[0] / config.OBS_STEREO_MAX,
            d[1] / config.OBS_STEREO_MAX,
            d[2] / config.OBS_STEREO_MAX,
            d[3] / config.OBS_STEREO_MAX,
            d[4] / config.OBS_STEREO_MAX,
            self.current_x / config.OBS_POS_MAX,
            self.current_y / config.OBS_POS_MAX,
            self.current_z / config.OBS_Z_MAX,
            math.sin(self.current_yaw),
            math.cos(self.current_yaw),
            np.clip(self.odom_vx, -1.0, 1.0),
            np.clip(self.odom_vz, -1.0, 1.0),
            self.current_roll / math.pi,
            self.current_pitch / math.pi,
        ], dtype=np.float32)

    def _collect_sample(self):
        if not self.have_stereo or not self.have_odom:
            return

        obs = self._build_observation()
        act = self.last_action.copy()

        self.buffer_obs.append(obs)
        self.buffer_acts.append(act)

        if len(self.buffer_obs) >= self.batch_size:
            self._save_batch()
            self.buffer_obs.clear()
            self.buffer_acts.clear()

            self.get_logger().info(
                f"[EXPERT] Collected {self.total_collected}/{self.target_samples} samples"
            )

            if self.total_collected >= self.target_samples:
                self.get_logger().info(
                    f"[EXPERT] Target reached. Saved to {config.EXPERT_DIR}"
                )
                rclpy.shutdown()
                sys.exit(0)

    def _save_batch(self):
        obs_arr = np.array(self.buffer_obs)
        act_arr = np.array(self.buffer_acts)
        self.total_collected += len(obs_arr)

        save_dir = Path(config.EXPERT_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"batch_{timestamp}.npz"
        np.savez(save_path, observations=obs_arr, actions=act_arr)


def main(args=None):
    print("=" * 60)
    print("[EXPERT] Expert Data Collection")
    print("=" * 60)
    print("[EXPERT] Make sure simulation is running with straight cave:")
    print("  Terminal 1: ./run_drone.sh straight_cave.py")
    print("=" * 60)

    rclpy.init(args=args)
    node = ExpertCollector()

    def sigint_handler(sig, frame):
        node.get_logger().info("[EXPERT] Interrupted, saving remaining data...")
        if node.buffer_obs:
            node._save_batch()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
