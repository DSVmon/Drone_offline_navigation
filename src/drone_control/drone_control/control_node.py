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
    SEARCHING = 1    # Moving forward or backward through the tunnel
    INSPECTING = 5   # Stopped to analyze complex obstacle
    TURNING = 2      # Rotating 180 degrees at the dead end or entrance
    COMPLETED = 4    # Mission finished, drone stopped at (0,0)

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.get_logger().info('Drone 3D Pathfinding Node started.')
        self.get_logger().warn('dToF laser rangefinder DISABLED — stereo cameras only')

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
        self.returned_home = False
        self.entrance_heading = None
        self.entrance_capture_timer = 0
        self.last_searching_entry_time = None
        self.wall_hit_guard = False

        # Tracking counters
        self.lap_count = 0
        self.collision_count = 0
        self.collision_in_lap = 0
        
        # Inspection logic — Active Scene Probing
        self.inspection_start_time = None
        self.inspect_probe_phase = 1      # 0=MOVE_TO_ENTRY, 1=INIT, 2=PROBE_UP, 3=SWEEP_L_HI, 4=RET_YAW_HI, 5=SWEEP_R_HI, 6=RET_YAW_HI, 7=GO_MID, 8=SWEEP_L_MID, 9=RET_YAW_MID, 10=SWEEP_R_MID, 11=RET_YAW_MID, 12=PROBE_DOWN, 13=SWEEP_L_LO, 14=RET_YAW_LO, 15=SWEEP_R_LO, 16=RET_YAW_LO, 17=RET_Z, 18=BACKUP, 19=DECIDE
        self.inspect_phase_start_time = None
        self.inspect_entry_z = 0.0
        self.inspect_entry_yaw = 0.0
        self.inspect_data = {}            # snapshots from probes
        self.inspect_best_z = None
        self.inspect_retry_count = 0
        self.inspect_sweep_angle = 1.571  # rad, starts 90°, ×1.45 for pass 1
        self.inspect_entry_x = 0.0
        self.inspect_entry_y = 0.0
        self.inspect_stuck_count = 0
        self.inspect_last_entry_x = 0.0
        self.inspect_last_entry_y = 0.0
        self.inspect_pass = 0              # 0=entry pass, 1=backup pass
        self._last_inspect_phase = None    # tracks phase changes for Z-ramp reset
        self.inspect_bu_x = 0.0            # backup position X for pass 1
        self.inspect_bu_y = 0.0            # backup position Y for pass 1
        self.inspect_bu_z = 0.0            # backup position Z for pass 1
        self.inspect_bu_yaw = 0.0          # backup position yaw for pass 1
        self.last_best_dir = "NONE"
        self.last_sent_vz = 0.0
        self.last_z_call_time = None
        self.z_service_future = None
        self.last_control_time = None
        self.last_roll = 0.0
        self.last_pitch = 0.0
        
        # Odometry velocity (from twist)
        self.odom_vx = 0.0
        self.odom_vy = 0.0
        self.odom_vz = 0.0
        
        # Collision detection counters (frame-based)
        self.vel_disc_frames = 0       # velocity discrepancy sustained frames
        self.z_jam_frames = 0          # Z-axis stuck frames
        self.prox_frames = 0           # sustained proximity frames
        self.z_jam_last_z = 0.0        # last Z for jam detection
        
        # Collision type tracking
        self.collision_type = "NONE"   # NONE / WALL / CEILING / FLOOR / OBSTACLE / CONTACT
        
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
        # self.create_subscription(Bool, '/perception_node/obstacle_front', self.front_dtof_callback, 10) # dToF disabled
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
        # Separate timer for status logging (every 5s, avoids throttle overhead in control loop)
        self.create_timer(5.0, self._log_status)

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
        
        # Snap-prevention: freeze X/Y to entry position during INSPECTING probes
        if self.state == MissionState.INSPECTING and self.inspect_probe_phase >= 2:
            if self.inspect_pass == 1:
                req.state.pose.position.x = self.inspect_bu_x
                req.state.pose.position.y = self.inspect_bu_y
            else:
                req.state.pose.position.x = self.inspect_entry_x
                req.state.pose.position.y = self.inspect_entry_y
        else:
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

        log_path = workspace_logs / "flight_last.csv"
        self.flight_log_file = open(log_path, "w", encoding="utf-8")
        # Extended header for deep analysis
        header = (
            "timestamp,state,x,y,z,roll_deg,pitch_deg,yaw_deg,"
            "dist_left,dist_center,dist_right,dist_top,dist_bottom,"
            "dtof_alert,collision_type,best_dir,max_space,"
            "cmd_vx,cmd_vz,cmd_yaw,lap,total_hits\n"
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
            f"{1 if self.obstacle_front_dtof else 0},{self.collision_type},"
            f"{best_dir},{max_space:.3f},"
            f"{msg.linear.x:.3f},{msg.linear.z:.3f},{msg.angular.z:.3f},"
            f"{self.lap_count},{self.collision_count}\n"
        )
        self.flight_log_file.write(log_row)
        self.flight_log_file.flush()

    def _log_status(self):
        dist = math.sqrt(self.current_x**2 + self.current_y**2)
        self.get_logger().info(
            f"[STATUS] Lap #{self.lap_count} | {self.state.name} | "
            f"(x={self.current_x:+.1f}, y={self.current_y:+.1f}, z={self.current_z:+.1f}) | "
            f"dist={dist:.1f}m | lap_hits: {self.collision_in_lap} | total_hits: {self.collision_count}")

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
                self.collision_count += 1
                self.collision_in_lap += 1
                self.collision_type = "CONTACT"
                self.get_logger().warn(f"[HIT #{self.collision_count}] CONTACT | Lap #{self.lap_count}")
                self.collision_detected = True
        else:
            self.collision_detected = False

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_z = msg.pose.pose.position.z
        
        # Read actual velocities from odometry for collision detection
        self.odom_vx = msg.twist.twist.linear.x
        self.odom_vy = msg.twist.twist.linear.y
        self.odom_vz = msg.twist.twist.linear.z
        
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
        
        # Distance from start (0,0)
        dist_to_home = math.sqrt(self.current_x**2 + self.current_y**2)

        # --- ORIENTATION-BASED IMPACT DETECTION ---
        # Catches grazing contacts in SEARCHING (disabled in INSPECTING to avoid false positives)
        if self.state != MissionState.INSPECTING and self.last_control_time is not None:
            dt = (now - self.last_control_time).nanoseconds / 1e9
            if dt > 0:
                roll_rate = abs(self.current_roll - self.last_roll) / dt
                pitch_rate = abs(self.current_pitch - self.last_pitch) / dt
                if (roll_rate > 1.2 or pitch_rate > 1.2):
                    if not self.collision_detected:
                        self.collision_count += 1
                        self.collision_in_lap += 1
                        # Determine type from commanded movement direction
                        if self.current_vz_cmd > 0.2:
                            self.collision_type = "CEILING"
                        elif self.current_vz_cmd < -0.2:
                            self.collision_type = "FLOOR"
                        else:
                            self.collision_type = "WALL"
                        self.get_logger().warn(f"[HIT #{self.collision_count}] {self.collision_type} | Lap #{self.lap_count}")
                        self.collision_detected = True
        self.last_roll = self.current_roll
        self.last_pitch = self.current_pitch

        # --- ADVANCED COLLISION DETECTION (SEARCHING/TURNING only — disabled in INSPECTING) ---
        # Multi-detector: velocity discrepancy, Z-jam, sustained proximity
        if self.state != MissionState.INSPECTING and not self.collision_detected:
            collision_now = False

            # 1. Velocity discrepancy: commanded speed >> actual speed → wall/obstacle
            cmd_speed = math.sqrt(self.current_vx_cmd**2 + abs(self.current_vz_cmd)**2)
            odom_speed = math.sqrt(self.odom_vx**2 + self.odom_vy**2 + self.odom_vz**2)
            if cmd_speed > 0.3 and cmd_speed - odom_speed > 0.4:
                self.vel_disc_frames += 1
            else:
                self.vel_disc_frames = 0
            if self.vel_disc_frames >= 6:  # 0.3s at 20Hz
                collision_now = True
                self.collision_type = "WALL"
                self.get_logger().warn(f"[HIT #{self.collision_count+1}] WALL (vel disc) | Lap #{self.lap_count}")

            # 2. Z-axis jam: commanded Z movement but odometry shows no vertical motion
            if abs(self.current_vz_cmd) > 0.3:
                if abs(self.current_z - self.z_jam_last_z) < 0.005:
                    self.z_jam_frames += 1
                else:
                    self.z_jam_frames = 0
                self.z_jam_last_z = self.current_z
            else:
                self.z_jam_frames = 0
                self.z_jam_last_z = self.current_z
            if self.z_jam_frames >= 10:  # 0.5s
                collision_now = True
                self.collision_type = "CEILING" if self.current_vz_cmd > 0 else "FLOOR"
                self.get_logger().warn(f"[HIT #{self.collision_count+1}] {self.collision_type} (z-jam) | Lap #{self.lap_count}")

            # 3. Sustained proximity: stereo distance below critical for multiple frames
            min_dist = min(left, center, right, top, bottom)
            if min_dist < 0.15:
                self.prox_frames += 1
            else:
                self.prox_frames = 0
            if self.prox_frames >= 6:  # 0.3s
                collision_now = True
                self.collision_type = "OBSTACLE"
                self.get_logger().warn(f"[HIT #{self.collision_count+1}] OBSTACLE (proximity) | Lap #{self.lap_count}")

            if collision_now:
                self.collision_count += 1
                self.collision_in_lap += 1
                self.collision_detected = True

        # --- VIRTUAL WALL: signed distance along cave direction ---
        signed_along_cave = 0.0
        if self.entrance_heading is not None:
            hx, hy = self.entrance_heading
            signed_along_cave = self.current_x * hx + self.current_y * hy

        # --- HARD VIRTUAL WALL: coordinate-based override ---
        if self.wall_hit_guard and signed_along_cave > 0.5:
            self.wall_hit_guard = False
        if self.entrance_heading is not None and signed_along_cave < -0.3 and self.state != MissionState.TURNING:
            if self.wall_hit_guard:
                pass
            else:
                self.wall_hit_guard = True
                self.get_logger().error(f"[WALL] signed={signed_along_cave:.2f} → TURNING")
                self.target_yaw = math.atan2(math.sin(self.current_yaw + math.pi),
                                              math.cos(self.current_yaw + math.pi))
                self.state = MissionState.TURNING
                self.returned_home = True
                self.current_vx_cmd = 0.0
                self.current_vz_cmd = 0.0
                self.current_yaw_cmd = 0.0
                stop_msg = Twist()
                for _ in range(10):
                    self.cmd_vel_pub.publish(stop_msg)
                return

        # --- PROXIMITY FINISH DETECTION (backup: catches drone near start in SEARCHING) ---
        if self.entrance_heading is not None and self.state == MissionState.SEARCHING and self.last_searching_entry_time is not None:
            searching_elapsed = (now - self.last_searching_entry_time).nanoseconds / 1e9
            if searching_elapsed > 3.0 and (dist_to_home < 1.5 or signed_along_cave < 0.5):
                self.get_logger().warn(f"[HOME] dist={dist_to_home:.2f}m \u2192 TURNING")
                self.target_yaw = math.atan2(math.sin(self.current_yaw + math.pi),
                                              math.cos(self.current_yaw + math.pi))
                self.state = MissionState.TURNING
                self.returned_home = True
                self.current_vx_cmd = 0.0
                self.current_vz_cmd = 0.0
                self.current_yaw_cmd = 0.0
                return

        # Emergency escape
        if self.collision_detected:
            if self.state != MissionState.INSPECTING:
                self.previous_mission_state = self.state
                self.state = MissionState.INSPECTING
                self.inspection_start_time = now
                self.entry_yaw = self.current_yaw
                self.inspect_entry_x = self.current_x
                self.inspect_entry_y = self.current_y
                self.inspect_probe_phase = 1
                self.inspect_phase_start_time = now
                self.inspect_retry_count = 0
                self.inspect_stuck_count = 0
                self.inspect_sweep_angle = 1.571
                self.inspect_data = {}
            else:
                # Collision during INSPECTING: skip blocked phase instead of restarting
                if self._last_inspect_phase is not None and self._last_inspect_phase <= 6:
                    self.inspect_probe_phase = 7  # skip to GO_MID
                elif self._last_inspect_phase is not None and self._last_inspect_phase <= 11:
                    self.inspect_probe_phase = 12  # skip to PROBE_DOWN
                elif self._last_inspect_phase is not None:
                    self.inspect_probe_phase = 18  # skip to BACKUP
                else:
                    self.inspect_probe_phase = 7
                self.inspect_phase_start_time = now
            self.target_z = self.current_z
            self.current_vz_cmd = 0.0
            self.current_vx_cmd = 0.0
            self.current_yaw_cmd = 0.0
            self.vel_disc_frames = 0
            self.z_jam_frames = 0
            self.prox_frames = 0
            msg.linear.x = -0.35
            msg.linear.z = 0.6 if top > bottom else -0.6
            msg.angular.z = 1.2 if left > right else -1.2
            best_dir_log = "COLLISION_ESCAPE"
        
        elif self.state == MissionState.SEARCHING:
            best_dir_log = "SEARCHING"
            if self.entrance_heading is None and abs(self.current_vx_cmd) > 0.05:
                self.entrance_capture_timer += 1
                if self.entrance_capture_timer >= 5:
                    self.entrance_heading = (math.cos(self.current_yaw), math.sin(self.current_yaw))
                    self.get_logger().info(f"Entrance heading locked: yaw={math.degrees(self.current_yaw):.1f} deg")
            msg = self._calculate_smooth_3d_speed(distances)
            if self.obstacle_front_dtof or center < 0.8:
                re_inhibit = False
                if self.last_searching_entry_time is not None:
                    se = (now - self.last_searching_entry_time).nanoseconds / 1e9
                    if se < 1.0:
                        re_inhibit = True
                if not re_inhibit:
                    # Stuck detection: repeated INSPECTING entry at same spot → DEAD END
                    dx = self.current_x - self.inspect_last_entry_x
                    dy = self.current_y - self.inspect_last_entry_y
                    same_spot = math.sqrt(dx*dx + dy*dy) < 0.2
                    if same_spot:
                        self.inspect_stuck_count += 1
                    else:
                        self.inspect_stuck_count = 0
                    self.inspect_last_entry_x = self.current_x
                    self.inspect_last_entry_y = self.current_y

                    if self.inspect_stuck_count >= 2:
                        self.get_logger().error(f"[DEAD END] stuck at ({self.current_x:.2f},{self.current_y:.2f}) → TURNING")
                        self.target_yaw = math.atan2(math.sin(self.current_yaw + math.pi),
                                                      math.cos(self.current_yaw + math.pi))
                        self.state = MissionState.TURNING
                        self.current_vx_cmd = 0.0
                        self.current_vz_cmd = 0.0
                        self.current_yaw_cmd = 0.0
                        stop_msg = Twist()
                        for _ in range(10):
                            self.cmd_vel_pub.publish(stop_msg)
                        return

                    self.get_logger().warn(f"[TIGHT] {center:.2f}m → INSPECTING")
                    self.previous_mission_state = self.state
                    self.state = MissionState.INSPECTING
                    self.inspection_start_time = now
                    self.entry_yaw = self.current_yaw
                    self.inspect_entry_x = self.current_x
                    self.inspect_entry_y = self.current_y
                    self.inspect_probe_phase = 1
                    self.inspect_phase_start_time = now
                    self.inspect_retry_count = 0
                    self.inspect_sweep_angle = 1.571
                    self.inspect_data = {}
                    self._last_inspect_phase = None
                    self.target_z = self.current_z
                    self.current_vz_cmd = 0.0
                    self.current_vx_cmd = 0.0
                    self.current_yaw_cmd = 0.0
                    msg.linear.x = 0.0

        elif self.state == MissionState.INSPECTING:
            elapsed = (now - self.inspection_start_time).nanoseconds / 1e9
            phase = self.inspect_probe_phase
            phase_elapsed = 0.0
            if self.inspect_phase_start_time is not None:
                phase_elapsed = (now - self.inspect_phase_start_time).nanoseconds / 1e9
            
            kp = "b_" if self.inspect_pass == 1 else ""  # key prefix for two-pass data
            
            # Phase change detection: reset ramped commands + target_z to prevent Z overshoot
            if self._last_inspect_phase is not None and self._last_inspect_phase != phase:
                self.target_z = self.current_z
                self.current_vz_cmd = 0.0
                self.current_vx_cmd = 0.0
                self.current_yaw_cmd = 0.0
            self._last_inspect_phase = phase
            
            # Hard timeout: total INSPECTING > 55s → DEAD END
            if elapsed > 55.0:
                self.get_logger().error(f"[DEAD END] timeout {elapsed:.1f}s → TURNING")
                self.target_yaw = math.atan2(math.sin(self.entry_yaw + math.pi), math.cos(self.entry_yaw + math.pi))
                self.state = MissionState.TURNING
                self.inspect_probe_phase = 1
                self.inspect_retry_count = 0
                self.inspect_stuck_count = 0
                self.inspect_sweep_angle = 1.571
                self.inspect_data = {}
                return
            
            # --- Phase 0: MOVE_TO_ENTRY (only on retry) ---
            if phase == 0:
                best_dir_log = "INSPECT_MOVE"
                dz = self.inspect_entry_z - self.current_z
                dyaw = math.atan2(math.sin(self.inspect_entry_yaw - self.current_yaw),
                                  math.cos(self.inspect_entry_yaw - self.current_yaw))
                if abs(dz) < 0.05 and abs(dyaw) < 0.05:
                    self.inspect_probe_phase = 1
                    self.inspect_phase_start_time = now
                else:
                    msg.linear.z = 0.5 if dz > 0 else -0.5
                    msg.angular.z = 0.5 if dyaw > 0 else -0.5

            # --- Phase 1: INIT — baseline snapshot ---
            elif phase == 1:
                best_dir_log = "INSPECT_INIT"
                self.inspect_pass = 0
                self.inspect_entry_z = self.current_z
                self.inspect_entry_yaw = self.current_yaw
                self.inspect_entry_x = self.current_x
                self.inspect_entry_y = self.current_y
                self.inspect_data = {
                    "baseline": distances[:5],
                    "high": None, "low": None,
                    "backup": None,
                    "entry_top": top, "entry_bottom": bottom,
                    "high_min_center": 999, "high_min_left": 999, "high_min_right": 999,
                    "low_min_center": 999, "low_min_left": 999, "low_min_right": 999
                }
                if center > 1.0 and phase_elapsed > 0.3:
                    self.get_logger().info("[INSPECT] False alarm → SEARCHING")
                    self.state = self.previous_mission_state
                    self.last_best_dir = "NONE"
                    self.inspect_probe_phase = 1
                    self.inspect_retry_count = 0
                    self.inspect_stuck_count = 0
                    self.inspect_data = {}
                else:
                    self.inspect_probe_phase = 2
                    self.inspect_phase_start_time = now

            # --- Phase 2: PROBE_UP — 70% к потолку ---
            elif phase == 2:
                best_dir_log = "INSPECT_UP"
                entry_top = self.inspect_data.get(f"{kp}entry_top", top)
                target_z = max(0.3, min(3.2, self.inspect_entry_z + entry_top * 0.7))
                dz = target_z - self.current_z
                if dz < 0.05 or phase_elapsed > 3.0:
                    self.inspect_data[f"{kp}high"] = distances[:5]
                    self.inspect_probe_phase = 3
                    self.inspect_phase_start_time = now
                else:
                    msg.linear.z = 0.5
                    if center < self.inspect_data.get(f"{kp}high_min_center", 999):
                        self.inspect_data[f"{kp}high_min_center"] = center
                    if left < self.inspect_data.get(f"{kp}high_min_left", 999):
                        self.inspect_data[f"{kp}high_min_left"] = left
                    if right < self.inspect_data.get(f"{kp}high_min_right", 999):
                        self.inspect_data[f"{kp}high_min_right"] = right

            # --- Phase 3: SWEEP_LEFT_HIGH at high position ---
            elif phase == 3:
                best_dir_log = "INSPECT_SW_L_HI"
                msg.linear.x = 0.0
                target_yaw = self.inspect_entry_yaw - self.inspect_sweep_angle
                dyaw = math.atan2(math.sin(target_yaw - self.current_yaw),
                                  math.cos(target_yaw - self.current_yaw))
                if abs(dyaw) < 0.05 or phase_elapsed > 2.5:
                    self.inspect_data[f"{kp}high_left_yaw"] = distances[:5]
                    self.inspect_probe_phase = 4
                    self.inspect_phase_start_time = now
                else:
                    msg.angular.z = -1.0

            # --- Phase 4: RETURN_YAW after high left sweep ---
            elif phase == 4:
                best_dir_log = "INSPECT_RY_HI_1"
                msg.linear.x = 0.0
                dyaw = math.atan2(math.sin(self.inspect_entry_yaw - self.current_yaw),
                                  math.cos(self.inspect_entry_yaw - self.current_yaw))
                if abs(dyaw) < 0.05 or phase_elapsed > 2.5:
                    self.inspect_probe_phase = 5
                    self.inspect_phase_start_time = now
                else:
                    msg.angular.z = 1.0 if dyaw > 0 else -1.0

            # --- Phase 5: SWEEP_RIGHT_HIGH at high position ---
            elif phase == 5:
                best_dir_log = "INSPECT_SW_R_HI"
                msg.linear.x = 0.0
                target_yaw = self.inspect_entry_yaw + self.inspect_sweep_angle
                dyaw = math.atan2(math.sin(target_yaw - self.current_yaw),
                                  math.cos(target_yaw - self.current_yaw))
                if abs(dyaw) < 0.05 or phase_elapsed > 2.5:
                    self.inspect_data[f"{kp}high_right_yaw"] = distances[:5]
                    self.inspect_probe_phase = 6
                    self.inspect_phase_start_time = now
                else:
                    msg.angular.z = 1.0

            # --- Phase 6: RETURN_YAW after high right sweep ---
            elif phase == 6:
                best_dir_log = "INSPECT_RY_HI_2"
                msg.linear.x = 0.0
                dyaw = math.atan2(math.sin(self.inspect_entry_yaw - self.current_yaw),
                                  math.cos(self.inspect_entry_yaw - self.current_yaw))
                if abs(dyaw) < 0.05 or phase_elapsed > 2.5:
                    self.inspect_probe_phase = 7
                    self.inspect_phase_start_time = now
                else:
                    msg.angular.z = 1.0 if dyaw > 0 else -1.0

            # --- Phase 7: GO_TO_MID — descend to entry_z ---
            elif phase == 7:
                best_dir_log = "INSPECT_GO_MID"
                dz = self.inspect_entry_z - self.current_z
                if abs(dz) < 0.05 or phase_elapsed > 3.0:
                    self.inspect_probe_phase = 8
                    self.inspect_phase_start_time = now
                else:
                    msg.linear.z = 0.5 if dz > 0 else -0.5

            # --- Phase 8: SWEEP_LEFT_MID at entry_z ---
            elif phase == 8:
                best_dir_log = "INSPECT_SW_L_MID"
                msg.linear.x = 0.0
                target_yaw = self.inspect_entry_yaw - self.inspect_sweep_angle
                dyaw = math.atan2(math.sin(target_yaw - self.current_yaw),
                                  math.cos(target_yaw - self.current_yaw))
                if abs(dyaw) < 0.05 or phase_elapsed > 2.5:
                    self.inspect_data[f"{kp}mid_left_yaw"] = distances[:5]
                    self.inspect_probe_phase = 9
                    self.inspect_phase_start_time = now
                else:
                    msg.angular.z = -1.0

            # --- Phase 9: RETURN_YAW after mid left sweep ---
            elif phase == 9:
                best_dir_log = "INSPECT_RY_MID_1"
                msg.linear.x = 0.0
                dyaw = math.atan2(math.sin(self.inspect_entry_yaw - self.current_yaw),
                                  math.cos(self.inspect_entry_yaw - self.current_yaw))
                if abs(dyaw) < 0.05 or phase_elapsed > 2.5:
                    self.inspect_probe_phase = 10
                    self.inspect_phase_start_time = now
                else:
                    msg.angular.z = 1.0 if dyaw > 0 else -1.0

            # --- Phase 10: SWEEP_RIGHT_MID at entry_z ---
            elif phase == 10:
                best_dir_log = "INSPECT_SW_R_MID"
                msg.linear.x = 0.0
                target_yaw = self.inspect_entry_yaw + self.inspect_sweep_angle
                dyaw = math.atan2(math.sin(target_yaw - self.current_yaw),
                                  math.cos(target_yaw - self.current_yaw))
                if abs(dyaw) < 0.05 or phase_elapsed > 2.5:
                    self.inspect_data[f"{kp}mid_right_yaw"] = distances[:5]
                    self.inspect_probe_phase = 11
                    self.inspect_phase_start_time = now
                else:
                    msg.angular.z = 1.0

            # --- Phase 11: RETURN_YAW after mid right sweep ---
            elif phase == 11:
                best_dir_log = "INSPECT_RY_MID_2"
                msg.linear.x = 0.0
                dyaw = math.atan2(math.sin(self.inspect_entry_yaw - self.current_yaw),
                                  math.cos(self.inspect_entry_yaw - self.current_yaw))
                if abs(dyaw) < 0.05 or phase_elapsed > 2.5:
                    self.inspect_probe_phase = 12
                    self.inspect_phase_start_time = now
                else:
                    msg.angular.z = 1.0 if dyaw > 0 else -1.0

            # --- Phase 12: PROBE_DOWN — 70% к полу ---
            elif phase == 12:
                best_dir_log = "INSPECT_DOWN"
                entry_bottom = self.inspect_data.get(f"{kp}entry_bottom", bottom)
                target_z = max(0.5, min(3.2, self.inspect_entry_z - entry_bottom * 0.7))
                dz = self.current_z - target_z
                if dz < 0.05 or phase_elapsed > 4.0:
                    self.inspect_data[f"{kp}low"] = distances[:5]
                    self.inspect_probe_phase = 13
                    self.inspect_phase_start_time = now
                else:
                    msg.linear.z = -0.5
                    if center < self.inspect_data.get(f"{kp}low_min_center", 999):
                        self.inspect_data[f"{kp}low_min_center"] = center
                    if left < self.inspect_data.get(f"{kp}low_min_left", 999):
                        self.inspect_data[f"{kp}low_min_left"] = left
                    if right < self.inspect_data.get(f"{kp}low_min_right", 999):
                        self.inspect_data[f"{kp}low_min_right"] = right

            # --- Phase 13: SWEEP_LEFT_LOW at low position ---
            elif phase == 13:
                best_dir_log = "INSPECT_SW_L_LO"
                msg.linear.x = 0.0
                target_yaw = self.inspect_entry_yaw - self.inspect_sweep_angle
                dyaw = math.atan2(math.sin(target_yaw - self.current_yaw),
                                  math.cos(target_yaw - self.current_yaw))
                if abs(dyaw) < 0.05 or phase_elapsed > 2.5:
                    self.inspect_data[f"{kp}low_left_yaw"] = distances[:5]
                    self.inspect_probe_phase = 14
                    self.inspect_phase_start_time = now
                else:
                    msg.angular.z = -1.0

            # --- Phase 14: RETURN_YAW after low left sweep ---
            elif phase == 14:
                best_dir_log = "INSPECT_RY_LO_1"
                msg.linear.x = 0.0
                dyaw = math.atan2(math.sin(self.inspect_entry_yaw - self.current_yaw),
                                  math.cos(self.inspect_entry_yaw - self.current_yaw))
                if abs(dyaw) < 0.05 or phase_elapsed > 2.5:
                    self.inspect_probe_phase = 15
                    self.inspect_phase_start_time = now
                else:
                    msg.angular.z = 1.0 if dyaw > 0 else -1.0

            # --- Phase 15: SWEEP_RIGHT_LOW at low position ---
            elif phase == 15:
                best_dir_log = "INSPECT_SW_R_LO"
                msg.linear.x = 0.0
                target_yaw = self.inspect_entry_yaw + self.inspect_sweep_angle
                dyaw = math.atan2(math.sin(target_yaw - self.current_yaw),
                                  math.cos(target_yaw - self.current_yaw))
                if abs(dyaw) < 0.05 or phase_elapsed > 2.5:
                    self.inspect_data[f"{kp}low_right_yaw"] = distances[:5]
                    self.inspect_probe_phase = 16
                    self.inspect_phase_start_time = now
                else:
                    msg.angular.z = 1.0

            # --- Phase 16: RETURN_YAW after low right sweep ---
            elif phase == 16:
                best_dir_log = "INSPECT_RY_LO_2"
                msg.linear.x = 0.0
                dyaw = math.atan2(math.sin(self.inspect_entry_yaw - self.current_yaw),
                                  math.cos(self.inspect_entry_yaw - self.current_yaw))
                if abs(dyaw) < 0.05 or phase_elapsed > 2.5:
                    self.inspect_probe_phase = 17
                    self.inspect_phase_start_time = now
                else:
                    msg.angular.z = 1.0 if dyaw > 0 else -1.0

            # --- Phase 17: RETURN_Z — back to entry_z ---
            elif phase == 17:
                best_dir_log = "INSPECT_RET_Z"
                dz = self.inspect_entry_z - self.current_z
                if abs(dz) < 0.05 or phase_elapsed > 3.0:
                    if self.inspect_pass == 1:
                        self.inspect_probe_phase = 19
                    else:
                        self.inspect_probe_phase = 18
                    self.inspect_phase_start_time = now
                else:
                    msg.linear.z = 0.5 if dz > 0 else -0.5

            # --- Phase 18: BACKUP — отступить для pass 1 с увеличенным углом ---
            elif phase == 18:
                best_dir_log = "INSPECT_BACKUP"
                if phase_elapsed > 1.5:
                    self.inspect_data["backup"] = distances[:5]
                    self.inspect_data["pass0_entry_x"] = self.inspect_entry_x
                    self.inspect_data["pass0_entry_y"] = self.inspect_entry_y
                    self.inspect_data["pass0_entry_z"] = self.inspect_entry_z
                    self.inspect_data["pass0_entry_yaw"] = self.inspect_entry_yaw
                    self.inspect_bu_x = self.current_x
                    self.inspect_bu_y = self.current_y
                    self.inspect_bu_z = self.current_z
                    self.inspect_bu_yaw = self.current_yaw
                    self.inspect_entry_x = self.current_x
                    self.inspect_entry_y = self.current_y
                    self.inspect_entry_z = self.current_z
                    self.inspect_entry_yaw = self.current_yaw
                    self.inspect_data["b_entry_top"] = top
                    self.inspect_data["b_entry_bottom"] = bottom
                    self.inspect_data["b_baseline"] = distances[:5]
                    self.inspect_data["pass0_sweep_angle"] = self.inspect_sweep_angle
                    self.inspect_sweep_angle *= 1.45
                    self.inspect_pass = 1
                    self.inspect_probe_phase = 2
                    self.inspect_phase_start_time = now
                else:
                    msg.linear.x = -0.2

            # --- Phase 19: DECIDE — forward path first, then yaw-turn fallback ---
            elif phase == 19:
                best_dir_log = "INSPECT_DECIDE"

                # Forward-facing snapshots (baseline, probe snapshots, backup)
                fwd_snaps = []
                # Yawed snapshots (left/right sweeps at all heights)
                yaw_snaps = []

                for pfx in ["", "b_"]:
                    snap = self.inspect_data.get(f"{pfx}baseline")
                    if snap is not None:
                        fwd_snaps.append((f"{pfx}baseline", snap))
                    for key in ["high", "low"]:
                        snap = self.inspect_data.get(f"{pfx}{key}")
                        if snap is not None:
                            fwd_snaps.append((f"{pfx}{key}", snap))
                            mc = self.inspect_data.get(f"{pfx}{key}_min_center", 999)
                            ml = self.inspect_data.get(f"{pfx}{key}_min_left", 999)
                            mr = self.inspect_data.get(f"{pfx}{key}_min_right", 999)
                            if mc < 999:
                                fwd_snaps.append((f"{pfx}{key}_min", [ml, mc, mr, 999, 999]))
                    for suffix in ["_left_yaw", "_right_yaw"]:
                        for prefix in ["high", "mid", "low"]:
                            snap = self.inspect_data.get(f"{pfx}{prefix}{suffix}")
                            if snap is not None:
                                yaw_snaps.append((f"{pfx}{prefix}{suffix}", snap))
                snap = self.inspect_data.get("backup")
                if snap is not None:
                    fwd_snaps.append(("backup", snap))

                def _test_snapshot(s, st_override=None, sb_override=None):
                    sl, sc, sr, st, sb = s
                    if st_override is not None:
                        st = st_override
                    if sb_override is not None:
                        sb = sb_override
                    space_ok = sc >= 0.3
                    width_ok = min(sl, sr) >= 0.4
                    height_ok = min(st, sb) >= 0.25
                    if space_ok and width_ok and height_ok:
                        score = 0.5 * sc + 0.3 * min(sl, sr) + 0.2 * min(st, sb)
                        return score
                    return None

                # 1st priority: forward-facing snapshots → fly forward
                candidates = []
                for sk, s in fwd_snaps:
                    st_override = None
                    sb_override = None
                    pfx = "b_" if "b_" in sk else ""
                    if sk in ("high", "high_min", "b_high", "b_high_min"):
                        st_override = self.inspect_data.get(f"{pfx}entry_top", 1.0)
                    elif sk in ("low", "low_min", "b_low", "b_low_min"):
                        sb_override = self.inspect_data.get(f"{pfx}entry_bottom", 1.0)
                    sc = _test_snapshot(s, st_override, sb_override)
                    if sc is not None:
                        candidates.append((sc, sk, s))

                if candidates:
                    candidates.sort(reverse=True)
                    best_score, best_key, best_s = candidates[0]
                    pass0_entry_z = self.inspect_data.get("pass0_entry_z", self.inspect_entry_z)
                    entry_top = self.inspect_data.get("entry_top", 1.0)
                    entry_bottom = self.inspect_data.get("entry_bottom", 1.0)
                    self.inspect_best_z = self.inspect_entry_z
                    if best_key in ("high", "high_min"):
                        self.inspect_best_z = pass0_entry_z + entry_top * 0.7
                    elif best_key in ("low", "low_min"):
                        self.inspect_best_z = pass0_entry_z - entry_bottom * 0.7
                    elif best_key in ("b_high", "b_high_min"):
                        self.inspect_best_z = self.inspect_bu_z + self.inspect_data.get("b_entry_top", 1.0) * 0.7
                    elif best_key in ("b_low", "b_low_min"):
                        self.inspect_best_z = self.inspect_bu_z - self.inspect_data.get("b_entry_bottom", 1.0) * 0.7
                    self.inspect_best_z = max(0.3, min(3.2, self.inspect_best_z))
                    self.last_best_dir = best_key
                    self.get_logger().info(f"[INSPECT] Forward ({best_key}, score={best_score:.2f}) → SEARCHING")
                    self.target_z = self.inspect_best_z
                    self.last_searching_entry_time = now
                    self.state = self.previous_mission_state
                    self.inspect_probe_phase = 1
                    self.inspect_sweep_angle = 1.571
                    self.inspect_retry_count = 0
                    self.inspect_stuck_count = 0
                    self.inspect_data = {}
                    return

                # 2nd priority: yawed snapshots → turn to face that direction
                candidates = []
                pass0_entry_yaw = self.inspect_data.get("pass0_entry_yaw", self.entry_yaw)
                pass0_sweep = self.inspect_data.get("pass0_sweep_angle", 1.571)
                for sk, s in yaw_snaps:
                    st_override = None
                    sb_override = None
                    pfx = "b_" if "b_" in sk else ""
                    if "high" in sk:
                        st_override = self.inspect_data.get(f"{pfx}entry_top", 1.0)
                    elif "low" in sk:
                        sb_override = self.inspect_data.get(f"{pfx}entry_bottom", 1.0)
                    sc = _test_snapshot(s, st_override, sb_override)
                    if sc is not None:
                        # Determine target yaw for this snapshot
                        if "b_" in sk:
                            base_yaw = self.entry_yaw
                            sweep = self.inspect_sweep_angle
                        else:
                            base_yaw = pass0_entry_yaw
                            sweep = pass0_sweep
                        if "_left_yaw" in sk:
                            t_yaw = base_yaw - sweep
                        else:
                            t_yaw = base_yaw + sweep
                        candidates.append((sc, sk, s, t_yaw))

                if candidates:
                    candidates.sort(reverse=True)
                    best_score, best_key, best_s, t_yaw = candidates[0]
                    pass0_entry_z = self.inspect_data.get("pass0_entry_z", self.inspect_entry_z)
                    self.inspect_best_z = self.inspect_entry_z
                    if "b_" in best_key:
                        if "high" in best_key:
                            self.inspect_best_z = self.inspect_bu_z + self.inspect_data.get("b_entry_top", 1.0) * 0.7
                        elif "low" in best_key:
                            self.inspect_best_z = self.inspect_bu_z - self.inspect_data.get("b_entry_bottom", 1.0) * 0.7
                    else:
                        if "high" in best_key:
                            self.inspect_best_z = pass0_entry_z + self.inspect_data.get("entry_top", 1.0) * 0.7
                        elif "low" in best_key:
                            self.inspect_best_z = pass0_entry_z - self.inspect_data.get("entry_bottom", 1.0) * 0.7
                    self.inspect_best_z = max(0.3, min(3.2, self.inspect_best_z))
                    self.last_best_dir = best_key
                    self.get_logger().info(f"[INSPECT] Turn to {math.degrees(t_yaw):.0f}deg ({best_key}) → TURNING")
                    self.target_yaw = t_yaw
                    self.target_z = self.inspect_best_z
                    self.returned_home = False
                    self.state = MissionState.TURNING
                    self.inspect_probe_phase = 1
                    self.inspect_sweep_angle = 1.571
                    self.inspect_retry_count = 0
                    self.inspect_stuck_count = 0
                    self.inspect_data = {}
                    return

                self.get_logger().error(f"[DEAD END] no path after two passes → TURNING")
                self.target_yaw = math.atan2(math.sin(self.entry_yaw + math.pi), math.cos(self.entry_yaw + math.pi))
                self.state = MissionState.TURNING
                self.inspect_probe_phase = 1
                self.inspect_sweep_angle = 1.571
                self.inspect_retry_count = 0
                self.inspect_stuck_count = 0
                self.inspect_data = {}
                return

        elif self.state == MissionState.TURNING:
            best_dir_log = "TURNING"
            yaw_error = math.atan2(math.sin(self.target_yaw - self.current_yaw), math.cos(self.target_yaw - self.current_yaw))
            
            if abs(yaw_error) < 0.1:
                self.last_searching_entry_time = now
                if self.returned_home:
                    self.lap_count += 1
                    self.collision_in_lap = 0
                    self.get_logger().info(f"[LAP #{self.lap_count}] Turn → SEARCHING")
                else:
                    self.get_logger().info("[TURN] Turn → SEARCHING")
                self.returned_home = False
                self.state = MissionState.SEARCHING
                self.target_z = self.current_z
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

        if self.state not in [MissionState.SEARCHING, MissionState.INSPECTING]:
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
