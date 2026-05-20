import rclpy
from rclpy.node import Node
from enum import Enum
import math
from datetime import datetime
from pathlib import Path
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ContactsState, EntityState
from gazebo_msgs.srv import SetEntityState

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
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_q = [0.0, 0.0, 0.0, 1.0] # x, y, z, w
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.target_yaw = 0.0
        self.target_z = 1.0 # Initial spawn height
        self.collision_detected = False
        
        # Navigation memory
        self.entry_yaw = 0.0 # Yaw when started inspecting or turning
        self.previous_mission_state = MissionState.SEARCHING
        
        # Inspection logic
        self.inspection_start_time = None
        self.last_best_dir = "NONE"
        self.last_dir_time = None
        self.best_way_vector = [0.0, 0.0] # [Yaw_error, Z_error]
        self.inspect_reference_center = 10.0
        self.inspect_force_index = 0
        self.inspect_last_force_time = None
        self.last_sent_vz = 0.0
        self.last_z_call_time = None
        self.z_service_future = None
        self.last_control_time = None
        
        # Smooth speed ramping
        self.current_vx_cmd = 0.0
        self.current_vz_cmd = 0.0
        self.current_yaw_cmd = 0.0

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

        # Gazebo Service Client for Z-axis movement (Planar move plugin doesn't support Z)
        self.gz_set_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        
        # Robust service waiting
        self.get_logger().info('Waiting for Gazebo service /gazebo/set_entity_state...')
        self.service_ready = False
        # We don't block __init__ forever, but we mark if it's ready
        if self.gz_set_state_client.wait_for_service(timeout_sec=5.0):
            self.service_ready = True
            self.get_logger().info('Gazebo SetEntityState service found and ready.')
        else:
            self.get_logger().error('Gazebo SetEntityState service NOT FOUND during startup!')

        # Timer (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)

    def _set_gazebo_z_velocity(self, vz):
        """Workaround for planar_move plugin: set Z position/velocity via Gazebo service"""
        if not self.service_ready:
            if self.gz_set_state_client.service_is_ready():
                self.service_ready = True
            else:
                return

        now = self.get_clock().now()
        
        # 1. Rate limiting: Don't call more than 10Hz to avoid Gazebo queue lag
        if self.last_z_call_time is not None:
            elapsed_ms = (now - self.last_z_call_time).nanoseconds / 1e6
            if elapsed_ms < 100.0: # 10 Hz max
                return

        # 2. Prevent overlapping calls: Don't start new one if previous is pending
        if self.z_service_future is not None and not self.z_service_future.done():
            return

        # 3. Deadzone and drift correction
        # If vz is tiny and we are near target, skip to avoid X/Y snapping
        z_error = abs(self.target_z - self.current_z)
        if abs(vz) < 0.01 and z_error < 0.05:
            return

        # 4. Dynamic dt calculation
        if self.last_control_time is None:
            dt = 0.05
        else:
            dt = (now - self.last_control_time).nanoseconds / 1e9
        
        # Update target Z
        self.target_z += vz * dt
        self.target_z = max(0.1, min(3.4, self.target_z))
        
        req = SetEntityState.Request()
        req.state.name = 'rescue_drone'
        
        # Snap-prevention: Using latest available odom to minimize "jump"
        req.state.pose.position.x = self.current_x
        req.state.pose.position.y = self.current_y
        req.state.pose.position.z = self.target_z
        
        req.state.pose.orientation.x = self.current_q[0]
        req.state.pose.orientation.y = self.current_q[1]
        req.state.pose.orientation.z = self.current_q[2]
        req.state.pose.orientation.w = self.current_q[3]

        req.state.twist.linear.z = float(vz)
        
        self.last_z_call_time = now
        self.z_service_future = self.gz_set_state_client.call_async(req)

    def _init_flight_logger(self):
        # Use logs directory in the current working directory (usually workspace root)
        workspace_logs = Path("logs")
        if not workspace_logs.exists():
            try:
                workspace_logs.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.get_logger().error(f"Could not create log directory: {e}")
                # Fallback to /tmp if current directory is not writable
                workspace_logs = Path("/tmp/drone_logs")
                workspace_logs.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = workspace_logs / f"flight_{ts}.csv"
        self.flight_log_file = open(log_path, "w", encoding="utf-8")
        # Extended header for deep analysis
        header = (
            "timestamp,state,x,y,z,roll_deg,pitch_deg,yaw_deg,"
            "dist_left,dist_center,dist_right,dist_top,dist_bottom,"
            "dtof_alert,collision_event,best_dir,max_space,"
            "cmd_vx,cmd_vz,cmd_yaw\n"
        )
        self.flight_log_file.write(header)
        self.flight_log_file.flush()
        self.get_logger().info(f"Flight log initialized: {log_path}")

    def _log_flight_sample(self, msg: Twist, best_dir="NONE", max_space=0.0):
        now_ns = self.get_clock().now().nanoseconds
        if (now_ns - self.last_flight_log_time_ns) < self.flight_log_interval_ns:
            return
        self.last_flight_log_time_ns = now_ns

        left, center, right, top, bottom = self.stereo_distances[:5]
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        log_row = (
            f"{timestamp},{self.state.name},{self.current_x:.3f},{self.current_y:.3f},{self.current_z:.3f},"
            f"{math.degrees(self.current_roll):.2f},{math.degrees(self.current_pitch):.2f},{math.degrees(self.current_yaw):.2f},"
            f"{left:.3f},{center:.3f},{right:.3f},{top:.3f},{bottom:.3f},"
            f"{1 if self.obstacle_front_dtof else 0},{1 if self.collision_detected else 0},"
            f"{best_dir},{max_space:.3f},"
            f"{msg.linear.x:.3f},{msg.linear.z:.3f},{msg.angular.z:.3f}\n"
        )
        self.flight_log_file.write(log_row)
        self.flight_log_file.flush()

    def front_dtof_callback(self, msg):
        # Enable dToF laser rangefinder for critical safety
        self.obstacle_front_dtof = msg.data
        if self.obstacle_front_dtof:
            self.get_logger().warn("[SAFETY] dToF Obstacle Detected!", throttle_duration_sec=1.0)

    def stereo_distances_callback(self, msg):
        # msg.data is [left, center, right, top, bottom]
        if len(msg.data) >= 5:
            new_vals = list(msg.data)
            
            # 1. Global Sanity Check: Detect "Blindness"
            # Only override if the distance is large (blindness) and dToF says we are close
            if self.obstacle_front_dtof:
                for i in range(len(new_vals)):
                    if new_vals[i] > 3.0: # Only if it thinks it's far but it's actually close
                        new_vals[i] = 0.4 
            
            # 2. Temporal Filtering (EMA) to eliminate sensor jumps seen in logs
            # This prevents "jerky" motion caused by flickering stereo matching
            alpha = 0.3 # Reduced from 0.5 for smoother response in tight spaces
            for i in range(5):
                self.stereo_distances[i] = alpha * new_vals[i] + (1.0 - alpha) * self.stereo_distances[i]

    def collision_callback(self, msg):
        if len(msg.states) > 0:
            if not self.collision_detected:
                self.get_logger().error("!!! COLLISION DETECTED !!!")
                self.collision_detected = True
        else:
            self.collision_detected = False

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_z = msg.pose.pose.position.z
        
        q = msg.pose.pose.orientation
        self.current_q = [q.x, q.y, q.z, q.w]
        
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

    def _calculate_smooth_3d_speed(self, distances):
        """Unified logic for drone movement with Safety Repulsion (No centering)"""
        left, center, right, top, bottom = distances[:5]
        msg = Twist()
        
        # --- CONSTANTS ---
        MIN_SAFE_DIST = 0.5 # User: 50cm min by cameras
        
        # 1. Forward Speed (X)
        stable_dist = center * 0.8 + min(left, right) * 0.2
        base_speed = 0.8
        
        if stable_dist < 2.0:
            msg.linear.x = base_speed * (stable_dist / 2.0)**2
        else:
            msg.linear.x = base_speed
            
        msg.linear.x = max(0.25, msg.linear.x)
        
        # 2. Horizontal steering (Yaw) - Pure Repulsion
        # Stronger response to obstacles, no attempt to find "center"
        l_rep = 1.0 / max(MIN_SAFE_DIST, left)
        r_rep = 1.0 / max(MIN_SAFE_DIST, right)
        
        # Increased gain (0.8) for better reactive avoidance
        msg.angular.z = (r_rep - l_rep) * 0.8
        msg.angular.z = max(-1.2, min(1.2, msg.angular.z))
        
        # 3. Vertical steering (Z) - Pure Repulsion
        t_rep = 1.0 / max(MIN_SAFE_DIST, top)
        b_rep = 1.0 / max(MIN_SAFE_DIST, bottom)
        
        # Increased gain (1.2) for better reactive avoidance
        msg.linear.z = (b_rep - t_rep) * 1.2
        
        # Deadzone for Z to prevent constant service calls
        if abs(msg.linear.z) < 0.05:
            msg.linear.z = 0.0
            
        msg.linear.z = max(-0.7, min(0.7, msg.linear.z))
        
        # Visual bank/tilt
        msg.angular.x = msg.angular.z * 0.3
        
        return msg

    def control_loop(self):
        # ... (skip lines)
        if self.get_clock().now().nanoseconds == 0:
            return

        msg = Twist()
        now = self.get_clock().now()
        distances = self.stereo_distances[:5]
        left, center, right, top, bottom = distances
        
        # Diagnostics
        best_dir_log = "NONE"
        max_space_log = 0.0
        
        # GLOBAL HOME CHECK: Forced stop when exiting the tunnel (X near 0)
        dist_to_home = math.sqrt(self.current_x**2 + self.current_y**2)
        is_returning = (self.state == MissionState.RETURNING or 
                       (self.state == MissionState.INSPECTING and self.previous_mission_state == MissionState.RETURNING))
        
        # "Finish Line" logic: if we are returning and reached X ~ 0 or near (0,0)
        if is_returning and (dist_to_home < 1.0 or self.current_x < 0.1):
            self.get_logger().warn(f"FINISH LINE DETECTED (x={self.current_x:.2f}, dist={dist_to_home:.2f}m). Stopping.")
            self.state = MissionState.COMPLETED
            # Muzzle all commands immediately
            self.current_vx_cmd = 0.0
            self.current_vz_cmd = 0.0
            self.current_yaw_cmd = 0.0
            # Send hard stop to Gazebo
            stop_msg = Twist()
            for _ in range(10):
                self.cmd_vel_pub.publish(stop_msg)
            return

        # Emergency escape
        if self.collision_detected:
            if self.state != MissionState.INSPECTING:
                self.previous_mission_state = self.state
            self.state = MissionState.INSPECTING
            self.inspection_start_time = now
            self.entry_yaw = self.current_yaw
            msg.linear.x = -0.35
            msg.linear.z = 0.6 if top > bottom else -0.6
            msg.angular.z = 1.2 if left > right else -1.2
            best_dir_log = "COLLISION_ESCAPE"
        
        elif self.state == MissionState.SEARCHING:
            best_dir_log = "SEARCHING"
            msg = self._calculate_smooth_3d_speed(distances)
            if self.obstacle_front_dtof or center < 0.8:
                self.get_logger().warn(f"Path tight ({center:.2f}m). Inspecting...")
                self.previous_mission_state = self.state
                self.state = MissionState.INSPECTING
                self.inspection_start_time = now
                self.entry_yaw = self.current_yaw
                msg.linear.x = 0.0

        elif self.state == MissionState.RETURNING:
            best_dir_log = "RETURNING"
            # Point to (0,0) and fly
            
            target_yaw_home = math.atan2(-self.current_y, -self.current_x)
            yaw_error = math.atan2(math.sin(target_yaw_home - self.current_yaw), math.cos(target_yaw_home - self.current_yaw))
            
            # COMBINED LOGIC: Reactive avoidance + Homing bias
            # 1. Base reactive speed from stereo (same as SEARCHING)
            msg = self._calculate_smooth_3d_speed(distances)
            
            # 2. APPROACH BRAKING: Slow down significantly as we approach X=0
            approach_dist = max(0.0, self.current_x)
            if approach_dist < 3.0:
                # Scale speed from 100% at 3m to 20% at 0m
                speed_factor = 0.2 + 0.8 * (approach_dist / 3.0)
                msg.linear.x *= speed_factor
            
            # 3. Add homing bias to yaw if path is relatively clear
            if center > 1.5:
                # Mix homing yaw error with obstacle avoidance
                # We give 70% weight to homing if clear, 30% to obstacle avoidance
                msg.angular.z = 0.7 * (1.0 * yaw_error) + 0.3 * msg.angular.z
            
            # 3. Adjust forward speed based on homing alignment
            # Slow down if we need to turn significantly to point home
            msg.linear.x *= math.cos(yaw_error) 
            msg.linear.x = max(0.05, msg.linear.x) # Keep moving slightly
            
            if self.obstacle_front_dtof or center < 0.8:
                self.get_logger().warn(f"Path tight ({center:.2f}m). Inspecting...")
                self.previous_mission_state = self.state
                self.state = MissionState.INSPECTING
                self.inspection_start_time = now
                self.entry_yaw = self.current_yaw
                msg.linear.x = 0.0

        elif self.state == MissionState.INSPECTING:
            elapsed = (now - self.inspection_start_time).nanoseconds / 1e9
            
            options = {"CENTER": center * 0.85, "TOP": top, "BOTTOM": bottom, "LEFT": left, "RIGHT": right}
            
            # DIRECTION HYSTERESIS: Only change direction if the new one is significantly better
            if self.last_best_dir == "NONE" or self.last_best_dir not in options:
                best_dir = max(options, key=options.get)
            else:
                current_val = options[self.last_best_dir]
                potential_best = max(options, key=options.get)
                potential_val = options[potential_best]
                
                # New direction must be 25% better to switch
                if potential_val > current_val * 1.25:
                    best_dir = potential_best
                else:
                    best_dir = self.last_best_dir

            self.last_best_dir = best_dir
            best_dir_log = f"INSPECT_{best_dir}"
            
            # DYNAMIC SPEED: Move faster if the chosen path is very clear
            # min 0.1, max 0.4
            msg.linear.x = max(0.1, min(0.4, options[best_dir] * 0.2))
            
            # DEAD END LOGIC: Only turn if even the best option is very tight
            # and we have been trying for a while
            best_val = options[best_dir]
            is_really_stuck = best_val < 0.7 # 0.5m safe + 0.2m buffer
            
            if (is_really_stuck and elapsed > 4.0) or (center < 0.5) or (elapsed > 10.0):
                self.get_logger().error(f"DEAD END! Best option {best_dir} was only {best_val:.2f}m. Turning 180.")
                self.target_yaw = math.atan2(math.sin(self.entry_yaw + math.pi), math.cos(self.entry_yaw + math.pi))
                self.state = MissionState.TURNING
                return
            
            # Smoothly apply velocities in inspecting
            if "TOP" in best_dir: msg.linear.z = 0.4
            elif "BOTTOM" in best_dir: msg.linear.z = -0.4
            if "LEFT" in best_dir: msg.angular.z = 0.5
            elif "RIGHT" in best_dir: msg.angular.z = -0.5

            if not self.obstacle_front_dtof and center > 1.4 and elapsed > 0.5:
                # Return to previous mission state (SEARCHING or RETURNING)
                self.state = self.previous_mission_state
                self.last_best_dir = "NONE"

        elif self.state == MissionState.TURNING:
            best_dir_log = "TURNING"
            yaw_error = math.atan2(math.sin(self.target_yaw - self.current_yaw), math.cos(self.target_yaw - self.current_yaw))
            
            # Stricter tolerance for cleaner transition to return
            if abs(yaw_error) < 0.1: 
                self.get_logger().info("Turn complete. Transitioning to RETURN.")
                self.state = MissionState.RETURNING
                # Lock current altitude for return flight
                self.target_z = self.current_z
                # Reset memory for smooth flight
                self.inspection_start_time = None 
            else:
                msg.linear.x = 0.05
                msg.angular.z = 1.4 if yaw_error > 0 else -1.4

        elif self.state == MissionState.COMPLETED:
            best_dir_log = "COMPLETED"
            msg = Twist()
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = 0.0
            # Reset ramped commands to zero
            self.current_vx_cmd = 0.0
            self.current_vz_cmd = 0.0
            self.current_yaw_cmd = 0.0

        if self.state not in [MissionState.SEARCHING, MissionState.INSPECTING, MissionState.RETURNING]:
            msg.angular.x = 0.0

        # SPEED SMOOTHING (Ramping)
        # alpha_ramp: 1.0 = no smoothing, lower = more smoothing
        # We use a lower alpha for smoother transitions
        alpha_ramp = 0.3
        self.current_vx_cmd = alpha_ramp * msg.linear.x + (1.0 - alpha_ramp) * self.current_vx_cmd
        self.current_vz_cmd = alpha_ramp * msg.linear.z + (1.0 - alpha_ramp) * self.current_vz_cmd
        self.current_yaw_cmd = alpha_ramp * msg.angular.z + (1.0 - alpha_ramp) * self.current_yaw_cmd
        
        msg.linear.x = self.current_vx_cmd
        msg.linear.z = self.current_vz_cmd
        msg.angular.z = self.current_yaw_cmd

        # Apply Z velocity via workaround service
        self._set_gazebo_z_velocity(msg.linear.z)
        
        # Save time for next loop dt calculation
        self.last_control_time = now

        self.cmd_vel_pub.publish(msg)
        
        # LOGGING AT THE END
        self._log_flight_sample(msg, best_dir=best_dir_log, max_space=max_space_log)

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
