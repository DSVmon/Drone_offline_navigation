import subprocess
import time
import signal
import math
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
import config


def kill_gazebo():
    """Kill ALL simulation processes (gazebo, nodes, ROS 2 launch)."""
    targets = [
        "gzserver", "gzclient",
        "navigation_node", "perception_node",
        "robot_state_publisher", "spawn_entity",
        "training_launch.py",
    ]
    for proc in targets:
        try:
            subprocess.run(["pkill", "-9", "-f", proc], capture_output=True, timeout=5)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    time.sleep(3.0)
    # Verify processes are dead
    for proc in ["gzserver", "gzclient"]:
        result = subprocess.run(
            ["pgrep", "-f", proc], capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            # Force kill remaining
            for pid in result.stdout.strip().split("\n"):
                try:
                    subprocess.run(["kill", "-9", pid], capture_output=True, timeout=5)
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
    time.sleep(2.0)


def generate_cave(cave_script=None):
    """Generate a new cave world file.

    Args:
        cave_script: Path to cave generator script. Defaults to procedural_cave.py.
    """
    if cave_script is None:
        cave_script = config.SCRIPTS_DIR / "procedural_cave.py"
    cave_path = str(config.CAVE_WORLD_PATH)
    result = subprocess.run(
        ["python3", str(cave_script), cave_path],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Cave generation failed: {result.stderr}")
    return cave_path


def launch_gazebo(headless=True):
    """Launch training environment (Gazebo + nodes, no control_node, no GUI)."""
    learning_launch = str(config.LEARNING_DIR / "launch" / "training_launch.py")
    world_path = str(config.CAVE_WORLD_PATH)
    cmd = [
        "ros2", "launch", learning_launch,
        f"gui:={'false' if headless else 'true'}",
        f"world:={world_path}",
    ]

    # Source ROS 2 + workspace overlay before launching
    setup_ros = "/opt/ros/humble/setup.bash"
    setup_ws = str(config.PROJECT_ROOT / "install" / "setup.bash")
    bash_cmd = f"source {setup_ros} && source {setup_ws} && {' '.join(cmd)}"

    # Log Gazebo output for debugging
    log_path = str(config.LEARNING_DIR / "gazebo_launch.log")
    log_file = open(log_path, "w")

    process = subprocess.Popen(
        ["bash", "-c", bash_cmd],
        stdout=log_file,
        stderr=log_file,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
    )
    return process


def wait_for_gazebo(timeout=60):
    """Wait until Gazebo SetEntityState service is available."""
    start = time.time()
    gzserver_seen = False
    while time.time() - start < timeout:
        result = subprocess.run(
            ["ros2", "service", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if config.SERVICE_SET_ENTITY_STATE in result.stdout:
            return True
        # Check if gzserver died after being started
        pgrep = subprocess.run(
            ["pgrep", "-f", "gzserver"],
            capture_output=True, text=True, timeout=5,
        )
        if pgrep.stdout.strip():
            gzserver_seen = True
        elif gzserver_seen:
            raise RuntimeError("gzserver process died during startup")
        time.sleep(1.0)
    raise TimeoutError(f"Gazebo not ready after {timeout}s")


def reset_drone(node):
    """Reset drone position via Gazebo service (fire-and-forget, no blocking)."""
    client = node.create_client(SetEntityState, config.SERVICE_SET_ENTITY_STATE)
    if not client.wait_for_service(timeout_sec=3.0):
        raise RuntimeError("SetEntityState service not available")

    req = SetEntityState.Request()
    req.state = EntityState()
    req.state.name = config.DRONE_NAME
    req.state.pose.position.x = 0.0
    req.state.pose.position.y = 0.0
    req.state.pose.position.z = config.DRONE_SPAWN_Z
    req.state.pose.orientation.x = 0.0
    req.state.pose.orientation.y = 0.0
    req.state.pose.orientation.z = 0.0
    req.state.pose.orientation.w = 1.0
    req.state.twist.linear.x = 0.0
    req.state.twist.linear.y = 0.0
    req.state.twist.linear.z = 0.0
    req.state.twist.angular.x = 0.0
    req.state.twist.angular.y = 0.0
    req.state.twist.angular.z = 0.0
    req.state.reference_frame = "world"

    client.call_async(req)
    time.sleep(1.0)


def full_reset_simulation(node, episode_count, headless=True, cave_script=None):
    """Full reset: regenerate cave and relaunch if needed, otherwise just reset drone.

    Args:
        cave_script: Path to cave generator script. Defaults to procedural_cave.py.
    """
    should_change_cave = (
        episode_count > 0
        and episode_count % config.CAVE_CHANGE_INTERVAL == 0
    )

    if should_change_cave or episode_count == 0:
        if episode_count > 0:
            node.get_logger().info(
                f"[RESET] Cave change #{episode_count // config.CAVE_CHANGE_INTERVAL}"
            )
        kill_gazebo()
        generate_cave(cave_script=cave_script)
        gazebo_proc = launch_gazebo(headless=headless)
        wait_for_gazebo()
        return gazebo_proc
    else:
        reset_drone(node)
        return None
