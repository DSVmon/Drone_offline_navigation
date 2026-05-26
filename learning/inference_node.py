#!/usr/bin/env python3
"""
ROS 2 inference node for trained NN policy.

Replaces control_node in the simulation pipeline.
Loads a trained PPO model and runs it at 20Hz.

Usage:
    python3 learning/inference_node.py --model learning/checkpoints/final_model.zip
"""

import argparse
import math
import signal
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState

import config


class InferenceNode(Node):
    def __init__(self, model_path):
        super().__init__("inference_node")
        self.get_logger().info(f"Loading model from {model_path}")

        from stable_baselines3 import PPO
        self.model = PPO.load(model_path)

        # State buffers
        self.stereo_distances = [10.0, 10.0, 10.0, 10.0, 10.0]
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = config.DRONE_SPAWN_Z
        self.current_yaw = 0.0
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.odom_vx = 0.0
        self.have_data = False

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

        # Publisher
        self._cmd_vel_pub = self.create_publisher(Twist, config.TOPIC_CMD_VEL, 10)

        # Gazebo Z service
        self._gz_client = self.create_client(
            SetEntityState, config.SERVICE_SET_ENTITY_STATE
        )

        # Timer at 20Hz
        self.create_timer(config.DT, self._control_loop)

        self.get_logger().info("Inference node ready. Waiting for sensor data...")

    def _stereo_cb(self, msg):
        if len(msg.data) >= 5:
            self.stereo_distances = list(msg.data[:5])

    def _odom_cb(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_z = msg.pose.pose.position.z
        self.odom_vx = msg.twist.twist.linear.x
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
        self.have_data = True

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
            self.current_roll / math.pi,
            self.current_pitch / math.pi,
        ], dtype=np.float32)

    def _control_loop(self):
        if not self.have_data:
            return

        obs = self._build_observation()
        action, _ = self.model.predict(obs, deterministic=True)

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

    def _set_gazebo_z(self, vz):
        if not self._gz_client.service_is_ready():
            return

        target_z = self.current_z + vz * config.DT
        target_z = max(config.DRONE_MIN_Z, min(config.DRONE_MAX_Z, target_z))

        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = config.DRONE_NAME
        req.state.pose.position.x = self.current_x
        req.state.pose.position.y = self.current_y
        req.state.pose.position.z = target_z
        qz = math.sin(self.current_yaw / 2.0)
        qw = math.cos(self.current_yaw / 2.0)
        req.state.pose.orientation.x = 0.0
        req.state.pose.orientation.y = 0.0
        req.state.pose.orientation.z = float(qz)
        req.state.pose.orientation.w = float(qw)
        req.state.twist.linear.z = float(vz)
        req.state.reference_frame = "world"

        try:
            future = self._gz_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.3)
        except Exception:
            pass


def main(args=None):
    parser = argparse.ArgumentParser(description="Drone NN Inference Node")
    parser.add_argument(
        "--model",
        type=str,
        default=str(config.CHECKPOINT_DIR / "final_model.zip"),
        help="Path to trained PPO model (.zip)",
    )
    parsed_args, unknown = parser.parse_known_args()

    rclpy.init(args=unknown if unknown else None)
    node = InferenceNode(parsed_args.model)

    def sigint_handler(sig, frame):
        node.get_logger().info("Inference node shutting down.")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
