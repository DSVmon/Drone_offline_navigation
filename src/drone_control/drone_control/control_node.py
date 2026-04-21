import rclpy
from rclpy.node import Node
from enum import Enum
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class Direction(Enum):
    FORWARD = 1
    BACKWARD = -1

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.get_logger().info('Drone Shuttle Mode (with Virtual Walls) started.')

        self.current_direction = Direction.FORWARD
        self.obstacle_front = False
        self.obstacle_back = False
        self.current_x = 0.0
        
        # Subscribers
        self.create_subscription(Bool, '/perception_node/obstacle_front', self.front_callback, 10)
        self.create_subscription(Bool, '/perception_node/obstacle_back', self.back_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)

    def front_callback(self, msg):
        self.obstacle_front = msg.data

    def back_callback(self, msg):
        self.obstacle_back = msg.data

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x

    def control_loop(self):
        if self.get_clock().now().nanoseconds == 0:
            return

        msg = Twist()
        
        # Hybrid Logic: Real wall in front, Virtual wall at start (0m)
        if self.current_direction == Direction.FORWARD:
            # Change to BACKWARD ONLY if physical obstacle detected in front
            if self.obstacle_front:
                self.get_logger().warn("Obstacle in FRONT! Turning BACK to start.")
                self.current_direction = Direction.BACKWARD
                msg.linear.x = 0.0
            else:
                msg.linear.x = 1.0
                self.get_logger().info(f"Flying to real wall... Pos X: {self.current_x:.2f}", throttle_duration_sec=2.0)
        
        elif self.current_direction == Direction.BACKWARD:
            # Change to FORWARD if physical obstacle behind OR reached start point (0m)
            if self.obstacle_back or self.current_x < 0.0:
                reason = "PHYSICAL WALL" if self.obstacle_back else "START POINT (Virtual Wall)"
                self.get_logger().warn(f"Reached start! Turning FORWARD again. Reason: {reason}")
                self.current_direction = Direction.FORWARD
                msg.linear.x = 0.0
            else:
                msg.linear.x = -1.0
                self.get_logger().info(f"Returning to start... Pos X: {self.current_x:.2f}", throttle_duration_sec=2.0)

        self.cmd_vel_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
