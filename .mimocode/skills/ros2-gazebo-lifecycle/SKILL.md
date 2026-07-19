---
name: ros2-gazebo-lifecycle
description: Launch, inspect, and manage ROS 2 + Gazebo Classic simulation for drone navigation. Handles zsh environment setup, headless mode, topic inspection, and process cleanup.
---

# ROS 2 + Gazebo Lifecycle Management

Complete workflow for launching, monitoring, and cleaning up the Gazebo simulation in this WSL2 project.

## Prerequisites

- ROS 2 Humble installed at `/opt/ros/humble/`
- Project built with `colcon build` (install/setup.zsh exists)
- WSL2 with WSLg (X11 display at :0)

## Environment Setup

All ROS 2 commands MUST use zsh with this environment:

```bash
zsh -c "export DISPLAY=:0 && export LIBGL_ALWAYS_SOFTWARE=1 && export MESA_GL_VERSION_OVERRIDE=3.3 && source /opt/ros/humble/setup.zsh && source /mnt/e/Git_store/Drone_offline_navigation/install/setup.zsh && <YOUR_COMMAND>"
```

Key flags:
- `DISPLAY=:0` — WSLg X11 display
- `LIBGL_ALWAYS_SOFTWARE=1` — Software rendering (no GPU in WSL2)
- `MESA_GL_VERSION_OVERRIDE=3.3` — Force OpenGL 3.3 for Gazebo
- `gui:=false` — Headless Gazebo (no gzclient)

## Standard Workflows

### Launch Simulation

```bash
# Kill any existing processes first
kill -9 $(pgrep -f gzserver) 2>/dev/null
kill -9 $(pgrep -f navigation_node) 2>/dev/null
kill -9 $(pgrep -f perception_node) 2>/dev/null
sleep 2

# Launch in background
zsh -c "export DISPLAY=:0 && export LIBGL_ALWAYS_SOFTWARE=1 && export MESA_GL_VERSION_OVERRIDE=3.3 && source /opt/ros/humble/setup.zsh && source /mnt/e/Git_store/Drone_offline_navigation/install/setup.zsh && ros2 launch drone_simulation simulation_launch.py gui:=false 2>&1" &

# Wait for Gazebo to start (typically 30-40 seconds)
sleep 35
```

### Check Running Processes

```bash
# Check Gazebo
ps aux | grep gzserver | grep -v grep | head -1

# Check navigation + perception nodes
ps aux | grep -E "(gzserver|navigation_node|perception_node)" | grep -v grep | wc -l

# Check colcon build
ps aux | grep colcon | grep -v grep | head -1
```

### Inspect ROS 2 Topics

```bash
# List all topics
zsh -c "source /opt/ros/humble/setup.zsh && source /mnt/e/Git_store/Drone_offline_navigation/install/setup.zsh && ros2 topic list 2>&1"

# Check camera topics
zsh -c "source /opt/ros/humble/setup.zsh && source /mnt/e/Git_store/Drone_offline_navigation/install/setup.zsh && timeout 5 ros2 topic list 2>&1 | grep -E 'left|right|camera'"

# Check stereo distances
zsh -c "source /opt/ros/humble/setup.zsh && source /mnt/e/Git_store/Drone_offline_navigation/install/setup.zsh && timeout 5 ros2 topic echo /navigation_node/stereo_distances --no-daemon 2>&1 | head -10"

# Check depth map
zsh -c "source /opt/ros/humble/setup.zsh && source /mnt/e/Git_store/Drone_offline_navigation/install/setup.zsh && timeout 5 ros2 topic echo /navigation_node/depth_map --no-daemon --once 2>&1 | head -15"

# Check camera images
zsh -c "source /opt/ros/humble/setup.zsh && source /mnt/e/Git_store/Drone_offline_navigation/install/setup.zsh && timeout 3 ros2 topic echo /left/image_raw --no-daemon 2>&1 | head -10"

# Check camera info
zsh -c "source /opt/ros/humble/setup.zsh && source /mnt/e/Git_store/Drone_offline_navigation/install/setup.zsh && timeout 3 ros2 topic echo /left/camera_info --no-daemon 2>&1 | head -10"

# Check publisher count
zsh -c "source /opt/ros/humble/setup.zsh && source /mnt/e/Git_store/Drone_offline_navigation/install/setup.zsh && ros2 topic info /left/image_raw 2>&1 | grep -E 'Publisher|Subscription'"
```

### List ROS 2 Nodes

```bash
zsh -c "source /opt/ros/humble/setup.zsh && source /mnt/e/Git_store/Drone_offline_navigation/install/setup.zsh && timeout 5 ros2 node list 2>&1"
```

### Kill Simulation

```bash
# Kill all related processes
kill -9 $(pgrep -f gzserver) 2>/dev/null
kill -9 $(pgrep -f navigation_node) 2>/dev/null
kill -9 $(pgrep -f perception_node) 2>/dev/null
kill -9 $(pgrep -f drone_env) 2>/dev/null
sleep 2

# Verify cleanup
ps aux | grep -E "(gzserver|navigation_node|perception_node)" | grep -v grep
```

## Gotchas

- **ROS 2 state corruption**: After long runs, `ros2 topic list` may throw `RuntimeError: !rclpy.ok()`. Workaround: kill all processes and relaunch.
- **Duplicate URDF reference blocks**: Gazebo Classic uses only the LAST `<gazebo reference="link_name">` block. Always merge into one.
- **FOV limit**: `horizontal_fov=1.74533` (100°) causes camera rendering failure under `LIBGL_ALWAYS_SOFTWARE=1`. Use 1.3962634 (80°).
- **Planar_move is 2D only**: No Z-axis physics. SetEntityState removed from training. Drone flies at constant altitude.
- **Gazebo contact sensor broken with planar_move**: Never fires contact events. Use fallback detection (stereo proximity + velocity discrepancy) for training.
