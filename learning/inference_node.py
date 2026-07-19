#!/usr/bin/env python3
"""
ROS 2 inference node for MAVRL-trained policy.

Loads a trained RecurrentPPO model and runs it at 20Hz.
Input: depth map 256×256 + 7-dim goal-oriented state
Output: 4-dim body-frame accelerations (ax, ay, az, yaw_rate)

Usage:
    python3 learning/inference_node.py --model learning/checkpoints/final_model.zip
"""

import argparse
import math
import signal
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from cv_bridge import CvBridge

import config


class InferenceNode(Node):
    def __init__(self, model_path):
        super().__init__("inference_node")
        self.get_logger().info(f"Loading model from {model_path}")

        self.bridge = CvBridge()

        # Load model
        from stable_baselines3 import PPO
        self.model = PPO.load(model_path)

        # State buffers
        self.depth_image = np.zeros((config.DEPTH_HEIGHT, config.DEPTH_WIDTH), dtype=np.uint8)
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = config.DRONE_SPAWN_Z
        self.current_yaw = 0.0
        self.vel_world = np.array([0.0, 0.0, 0.0])
        self.prev_z = config.DRONE_SPAWN_Z
        self.vz_estimated = 0.0
        self.goal_point = np.array([config.CAVE_LENGTH * config.GOAL_DISTANCE_RATIO, 0.0, config.GOAL_Z])
        self.have_data = False

        # Subscribers
        self.create_subscription(Image, config.TOPIC_DEPTH_MAP, self._depth_cb, 10)
        self.create_subscription(Odometry, config.TOPIC_ODOM, self._odom_cb, 10)

        # Publisher
        self._cmd_vel_pub = self.create_publisher(Twist, config.TOPIC_CMD_VEL, 10)

        # Gazebo Z service
        self._gz_client = self.create_client(SetEntityState, config.SERVICE_SET_ENTITY_STATE)

        # Timer at 20Hz
        self.create_timer(config.DT, self._control_loop)

        self.get_logger().info("Inference node ready. Waiting for sensor data...")

    def _depth_cb(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            # navigation_node publishes mono8: pixel = depth_m * 255/12
            # Use directly as uint8 [0,255] — same as drone_env.py
            if depth.shape != (config.DEPTH_HEIGHT, config.DEPTH_WIDTH):
                depth = cv2.resize(depth, (config.DEPTH_WIDTH, config.DEPTH_HEIGHT))
            self.depth_image = depth
        except Exception as e:
            self.get_logger().warn(f"Depth callback error: {e}")

    def _odom_cb(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_z = msg.pose.pose.position.z
        # Estimate vz from position change (planar_move doesn't publish Z velocity)
        self.vz_estimated = (self.current_z - self.prev_z) / config.DT if config.DT > 0 else 0.0
        self.prev_z = self.current_z
        self.vel_world = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            self.vz_estimated,
        ])
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.x * q.x + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.have_data = True

    def _world2body(self, world_vel):
        cy, sy = math.cos(self.current_yaw), math.sin(self.current_yaw)
        flu_x = world_vel[1]
        flu_y = -world_vel[0]
        flu_z = world_vel[2]
        body_x = cy * flu_x + sy * flu_y
        body_y = -sy * flu_x + cy * flu_y
        return np.array([body_x, body_y, flu_z])

    def _build_observation(self):
        pos = np.array([self.current_x, self.current_y, self.current_z])
        delta_p = self.goal_point - pos
        horizon_dist = math.sqrt(delta_p[0] ** 2 + delta_p[1] ** 2)
        log_distance = math.log(horizon_dist + 1.0)

        vel_body = self._world2body(self.vel_world)
        horizon_vel = math.sqrt(vel_body[0] ** 2 + vel_body[1] ** 2)

        theta = math.atan2(-delta_p[0], delta_p[1])
        horizon_vel_dire = math.atan2(vel_body[1], vel_body[0])

        state = np.array([
            log_distance, horizon_vel, theta, horizon_vel_dire,
            delta_p[2], vel_body[2], self.current_yaw,
        ], dtype=np.float64)

        return {'image': self.depth_image, 'state': state}

    def _control_loop(self):
        if not self.have_data:
            return

        obs = self._build_observation()
        action, _ = self.model.predict(obs, deterministic=True)

        # Denormalize action
        cmd = np.array(action) * config.ACTION_STD + config.ACTION_MEAN
        acc_body = cmd[:3]
        yaw_rate = cmd[3]

        # Integrate to velocity
        acc_world = self._body2world(acc_body)
        self.vel_world = self.vel_world + acc_world * config.DT
        speed = np.linalg.norm(self.vel_world[:2])
        if speed > 3.0:
            self.vel_world[:2] = self.vel_world[:2] / speed * 3.0
        self.vel_world[2] = np.clip(self.vel_world[2], -1.5, 1.5)

        # Publish
        # Note: planar_move only handles X/Y. Z velocity is published but ignored by plugin.
        msg = Twist()
        msg.linear.x = float(self.vel_world[0])
        msg.linear.y = float(self.vel_world[1])
        msg.linear.z = float(self.vel_world[2])
        msg.angular.z = float(yaw_rate)
        self._cmd_vel_pub.publish(msg)

    def _body2world(self, acc_body):
        cy, sy = math.cos(self.current_yaw), math.sin(self.current_yaw)
        flu_x = acc_body[1]
        flu_y = -acc_body[0]
        flu_z = acc_body[2]
        world_flu_x = cy * flu_x - sy * flu_y
        world_flu_y = sy * flu_x + cy * flu_y
        return np.array([-world_flu_y, world_flu_x, flu_z])

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
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.1)
        except Exception:
            pass


def main(args=None):
    parser = argparse.ArgumentParser(description="Drone MAVRL Inference Node")
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
