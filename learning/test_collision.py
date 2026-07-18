#!/usr/bin/env python3
"""
Test collision detection in Gazebo.
Checks if contact sensor works and logs collision events.
"""

import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from gazebo_msgs.msg import ContactsState


class CollisionTest(Node):
    def __init__(self):
        super().__init__("collision_test")
        
        # Subscribe to collision topic
        self.sub_collisions = self.create_subscription(
            ContactsState,
            "/drone/collisions",
            self.collision_callback,
            10
        )
        
        # Subscribe to stereo distances
        self.sub_stereo = self.create_subscription(
            Float32MultiArray,
            "/navigation_node/stereo_distances",
            self.stereo_callback,
            10
        )
        
        self.collision_count = 0
        self.msg_count = 0
        self.min_distance = 10.0
        
        self.get_logger().info("[TEST] Collision test started")
        self.get_logger().info("[TEST] Subscribed to /drone/collisions")
        self.get_logger().info("[TEST] Subscribed to /navigation_node/stereo_distances")
        self.get_logger().info("[TEST] Waiting for data...")
        
        # Timer for status report
        self.create_timer(2.0, self.report_status)
        
    def collision_callback(self, msg):
        self.msg_count += 1
        if len(msg.states) > 0:
            self.collision_count += 1
            self.get_logger().warn(
                f"[COLLISION DETECTED] #{self.collision_count} | "
                f"States: {len(msg.states)} | "
                f"Contact 1: {msg.states[0].contact_names if msg.states else 'none'}"
            )
        else:
            if self.msg_count % 50 == 0:
                self.get_logger().info(
                    f"[TEST] No collision (msg #{self.msg_count})"
                )
    
    def stereo_callback(self, msg):
        if len(msg.data) >= 5:
            min_d = min(msg.data[:5])
            if min_d < self.min_distance:
                self.min_distance = min_d
            if min_d < 0.5:
                self.get_logger().warn(
                    f"[STERO] Very close! min={min_d:.3f}m | "
                    f"all={[f'{d:.2f}' for d in msg.data[:5]]}"
                )
    
    def report_status(self):
        self.get_logger().info(
            f"[STATUS] Messages: {self.msg_count} | "
            f"Collisions: {self.collision_count} | "
            f"Min distance: {self.min_distance:.3f}m"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CollisionTest()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
