import rclpy
from rclpy.node import Node
from enum import Enum
import math
from datetime import datetime
from pathlib import Path
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ContactsState

class MissionState(Enum):
    SEARCHING = 1    # Exploring the cave, moving forward
    INSPECTING = 5   # Stopped to analyze complex obstacle
    TURNING = 2      # Rotating 180 degrees at the dead end
    RETURNING = 3    # Flying back to the start point
    COMPLETED = 4    # Mission finished, drone stopped at (0,0)

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.get_logger().info('Drone 3D Pathfinding Node started.')

        self.state = MissionState.SEARCHING
        self.obstacle_front_dtof = False
        self.stereo_distances = [10.0, 10.0, 10.0, 10.0, 10.0] # Left, Center, Right, Top, Bottom
        self.current_x = 0.0
        self.current_z = 0.0
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.target_yaw = 0.0
        self.collision_detected = False
        
        # Inspection logic
        self.inspection_start_time = None
        self.best_way_vector = [0.0, 0.0] # [Yaw_error, Z_error]
        self.inspect_reference_center = 10.0
        self.inspect_force_index = 0
        self.inspect_last_force_time = None

        # Flight logging (one file per run)
        self.last_flight_log_time_ns = 0
        self.flight_log_interval_ns = int(0.2 * 1e9)  # 5 Hz
        self.flight_log_file = None
        self._init_flight_logger()
        
        # ... (rest of init same)
        self.create_subscription(Bool, '/perception_node/obstacle_front', self.front_dtof_callback, 10)
        self.create_subscription(Float32MultiArray, '/navigation_node/stereo_distances', self.stereo_distances_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(ContactsState, '/drone/collisions', self.collision_callback, 10)

        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)

    def _init_flight_logger(self):
        workspace_logs = Path("/drone_ws/logs")
        if not workspace_logs.exists():
            workspace_logs.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = workspace_logs / f"flight_{ts}.csv"
        self.flight_log_file = open(log_path, "w", encoding="utf-8")
        self.flight_log_file.write(
            "timestamp,state,x,z,roll_deg,pitch_deg,yaw_deg,left,center,right,top,bottom,cmd_vx,cmd_vz,cmd_roll,cmd_yaw\n"
        )
        self.flight_log_file.flush()
        self.get_logger().info(f"Flight log: {log_path}")

    def _log_flight_sample(self, msg: Twist, left: float, center: float, right: float, top: float, bottom: float):
        now_ns = self.get_clock().now().nanoseconds
        if (now_ns - self.last_flight_log_time_ns) < self.flight_log_interval_ns:
            return
        self.last_flight_log_time_ns = now_ns

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        self.flight_log_file.write(
            f"{timestamp},{self.state.name},{self.current_x:.3f},{self.current_z:.3f},"
            f"{math.degrees(self.current_roll):.2f},{math.degrees(self.current_pitch):.2f},{math.degrees(self.current_yaw):.2f},"
            f"{left:.3f},{center:.3f},{right:.3f},{top:.3f},{bottom:.3f},"
            f"{msg.linear.x:.3f},{msg.linear.z:.3f},{msg.angular.x:.3f},{msg.angular.z:.3f}\n"
        )
        self.flight_log_file.flush()

    def front_dtof_callback(self, msg):
        # Temporarily disabled laser rangefinder action as requested
        # self.obstacle_front_dtof = msg.data
        self.obstacle_front_dtof = False

    def stereo_distances_callback(self, msg):
        # msg.data is [left, center, right, top, bottom]
        if len(msg.data) >= 5:
            self.stereo_distances = msg.data

    def collision_callback(self, msg):
        if len(msg.states) > 0:
            if not self.collision_detected:
                self.get_logger().error("!!! COLLISION DETECTED !!!")
                self.collision_detected = True
        else:
            self.collision_detected = False

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_z = msg.pose.pose.position.z
        
        q = msg.pose.pose.orientation
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        self.current_roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            self.current_pitch = math.copysign(math.pi / 2, sinp)
        else:
            self.current_pitch = math.asin(sinp)

        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        if self.get_clock().now().nanoseconds == 0:
            return

        msg = Twist()
        now = self.get_clock().now()
        left, center, right, top, bottom = self.stereo_distances[:5]

        # Emergency escape if physical contact happened.
        if self.collision_detected:
            self.state = MissionState.INSPECTING
            self.inspection_start_time = now
            self.inspect_reference_center = center
            self.inspect_force_index = (self.inspect_force_index + 1) % 4
            self.inspect_last_force_time = now
            msg.linear.x = -0.15
            msg.linear.z = 0.45 if top > bottom else -0.45
            msg.angular.z = 0.8 if left > right else -0.8
            msg.angular.x = 0.6 if msg.angular.z > 0.0 else -0.6
            self._log_flight_sample(msg, left, center, right, top, bottom)
            self.cmd_vel_pub.publish(msg)
            return
        
        # 1. SEARCHING: Moving forward and reacting to easy obstacles
        if self.state == MissionState.SEARCHING:
            min_dist = min(center, left, right, top, bottom)
            
            # If something is VERY close and blocking the path, enter INSPECTION mode
            if center < 1.0 or (left < 0.6 and right < 0.6) or (top < 0.6 and bottom < 0.6):
                self.get_logger().warn(f"[MISSION] Obstacle too close ({min_dist:.2f}m). Starting INSPECTION...")
                self.state = MissionState.INSPECTING
                self.inspection_start_time = now
                self.inspect_reference_center = center
                self.inspect_force_index = 0
                self.inspect_last_force_time = now
                msg.linear.x = 0.0
            else:
                # Normal 3D reactive steering (same as before but more aggressive)
                base_speed = 0.7
                msg.linear.x = base_speed * (min(center, 2.0) / 2.0)
                
                # Horizontal steering (Yaw)
                l_inv = 1.0 / max(0.1, left)
                r_inv = 1.0 / max(0.1, right)
                msg.angular.z = (r_inv - l_inv) * 0.9
                
                # Vertical steering (Z)
                t_inv = 1.0 / max(0.1, top)
                b_inv = 1.0 / max(0.1, bottom)
                msg.linear.z = (b_inv - t_inv) * 0.6
                
                # Roll for coordination
                msg.angular.x = msg.angular.z * 0.5

        # 2. INSPECTING: Stop and find the best way through
        elif self.state == MissionState.INSPECTING:
            msg.linear.x = 0.0
            msg.linear.z = 0.0
            msg.angular.z = 0.0
            
            # Simple pathfinding: which direction has the most space?
            # We check 4 diagonal possibilities + current center
            # Weight CENTER less to encourage dodging
            options = {
                "CENTER": center * 0.8, 
                "TOP_LEFT": min(top, left),
                "TOP_RIGHT": min(top, right),
                "BOTTOM_LEFT": min(bottom, left),
                "BOTTOM_RIGHT": min(bottom, right)
            }
            
            best_dir = max(options, key=options.get)
            max_space = options[best_dir]
            if best_dir == "CENTER": max_space = center # Restore actual value
            
            self.get_logger().info(f"Inspecting... Best option: {best_dir} ({max_space:.2f}m)", throttle_duration_sec=0.5)

            elapsed = (now - self.inspection_start_time).nanoseconds / 1e9

            # If CENTER is chosen but it's still < 1.0m, we MUST move somewhere else
            if best_dir == "CENTER" and center < 1.0:
                # Force pick the next best non-center option
                non_center_options = {k: v for k, v in options.items() if k != "CENTER"}
                best_dir = max(non_center_options, key=non_center_options.get)
                max_space = non_center_options[best_dir]
                self.get_logger().warn(f"CENTER blocked ({center:.2f}m). FORCING maneuver to {best_dir}")

            # Anti-hang: if no center improvement after short inspect time, force exploration pattern.
            if elapsed > 1.0 and center <= (self.inspect_reference_center + 0.08):
                force_dirs = ["TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT"]
                force_elapsed = (now - self.inspect_last_force_time).nanoseconds / 1e9 if self.inspect_last_force_time else 999.0
                if force_elapsed > 0.8:
                    self.inspect_force_index = (self.inspect_force_index + 1) % len(force_dirs)
                    self.inspect_last_force_time = now
                best_dir = force_dirs[self.inspect_force_index]
                max_space = options[best_dir]
                self.get_logger().warn(f"ANTI-HANG forced maneuver: {best_dir}")

            # If even the best way is blocked (< 0.5m), it's a REAL dead end
            if max_space < 0.5:
                if elapsed > 2.0: 
                    self.get_logger().error("[MISSION] NO WAY THROUGH! Initiating 180 turn.")
                    self.target_yaw = self.current_yaw + math.pi
                    self.target_yaw = (self.target_yaw + math.pi) % (2 * math.pi) - math.pi
                    self.state = MissionState.TURNING
            else:
                # Execution of the chosen maneuver
                # Increase movement speeds to avoid "hanging"
                move_speed_z = 0.55
                move_speed_yaw = 0.85
                
                if "TOP" in best_dir: msg.linear.z = move_speed_z
                elif "BOTTOM" in best_dir: msg.linear.z = -move_speed_z
                
                if "LEFT" in best_dir: msg.angular.z = move_speed_yaw
                elif "RIGHT" in best_dir: msg.angular.z = -move_speed_yaw
                
                # Coordinated roll for the maneuver
                msg.angular.x = msg.angular.z * 0.65

                # Constant forward nudge to keep moving through the gap
                msg.linear.x = 0.35
                
                # If the chosen path is now clear enough, resume searching
                if max_space > 1.3:
                    self.get_logger().info(f"Path clear ({max_space:.2f}m). Resuming search.")
                    self.state = MissionState.SEARCHING

        elif self.state == MissionState.TURNING:
            # (same as before)
            yaw_error = self.target_yaw - self.current_yaw
            while yaw_error > math.pi: yaw_error -= 2 * math.pi
            while yaw_error < -math.pi: yaw_error += 2 * math.pi
            
            self.get_logger().info(f"TURNING: Err={math.degrees(yaw_error):.1f}°", throttle_duration_sec=1.0)

            if abs(yaw_error) < 0.1: # Slightly larger tolerance for stability
                self.get_logger().info(f"Rotation OK. Heading back home.")
                self.state = MissionState.RETURNING
                msg.linear.x = 0.0
                msg.angular.z = 0.0
            else:
                msg.linear.x = 0.0
                # Smooth rotation with P-controller
                rotation_speed = 1.0 * yaw_error
                max_rot, min_rot = 0.8, 0.3
                if abs(rotation_speed) > max_rot: rotation_speed = max_rot if rotation_speed > 0 else -max_rot
                elif abs(rotation_speed) < min_rot: rotation_speed = min_rot if rotation_speed > 0 else -min_rot
                msg.angular.z = rotation_speed

        elif self.state == MissionState.RETURNING:
            # Fly back to X=0 while STILL avoiding walls in 3D!
            if self.current_x <= 0.1:
                self.get_logger().warn("[MISSION] COMPLETED! Drone returned to start.")
                self.state = MissionState.COMPLETED
                msg.linear.x = 0.0
                msg.linear.z = 0.0
            else:
                base_speed = 0.5
                min_dist_ahead = min(center, top, bottom)
                if min_dist_ahead < 1.5:
                    msg.linear.x = base_speed * (min_dist_ahead / 1.5)
                else:
                    msg.linear.x = base_speed
                
                # Yaw steering
                steering_gain_yaw = 0.8
                l_inv = 1.0 / max(0.1, left)
                r_inv = 1.0 / max(0.1, right)
                msg.angular.z = (r_inv - l_inv) * steering_gain_yaw
                msg.angular.z = max(-0.6, min(0.6, msg.angular.z))

                # Vertical steering
                steering_gain_z = 0.5
                t_inv = 1.0 / max(0.1, top)
                b_inv = 1.0 / max(0.1, bottom)
                msg.linear.z = (b_inv - t_inv) * steering_gain_z
                msg.linear.z = max(-0.4, min(0.4, msg.linear.z))

                self.get_logger().info(f"3D RETURNING: Pos={self.current_x:.1f}m | T={top:.1f} B={bottom:.1f}", throttle_duration_sec=1.0)

        elif self.state == MissionState.COMPLETED:
            # Force stop and level out
            msg.linear.x = 0.0
            msg.linear.z = 0.0
            msg.angular.z = 0.0
            msg.angular.x = 0.0

        # Safety: always level out roll if no active steering mode
        if self.state not in [MissionState.SEARCHING, MissionState.INSPECTING, MissionState.RETURNING]:
            msg.angular.x = 0.0

        self._log_flight_sample(msg, left, center, right, top, bottom)
        self.cmd_vel_pub.publish(msg)

    def destroy_node(self):
        if self.flight_log_file is not None:
            self.flight_log_file.close()
            self.flight_log_file = None
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
