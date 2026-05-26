#!/bin/bash
source /opt/ros/humble/setup.bash
source /mnt/e/Git_store/Drone_offline_navigation/install/setup.bash
cd /mnt/e/Git_store/Drone_offline_navigation

# Kill old processes
pkill -f gzserver 2>/dev/null || true
pkill -f gzclient 2>/dev/null || true
sleep 2

# Generate cave
python3 scripts/procedural_cave.py src/drone_simulation/worlds/cave.world

# Launch sim in background
ros2 launch learning/launch/training_launch.py gui:=false &
SIM_PID=$!
echo "Launched sim with PID $SIM_PID"

# Wait for Gazebo
for i in $(seq 1 90); do
  if ros2 service list 2>/dev/null | grep -q "set_entity_state"; then
    echo "Gazebo ready after ${i}s"
    break
  fi
  sleep 1
done

# Check topics
echo "=== Topics ==="
timeout 3 ros2 topic list 2>/dev/null || echo "topic list failed"

echo "=== /cmd_vel subscribers ==="
timeout 3 ros2 topic info /cmd_vel --verbose 2>/dev/null || echo "No /cmd_vel topic"

echo "=== /odom publishers ==="
timeout 3 ros2 topic info /odom --verbose 2>/dev/null || echo "No /odom topic"

echo "=== /navigation_node/stereo_distances ==="
timeout 3 ros2 topic info /navigation_node/stereo_distances --verbose 2>/dev/null || echo "No stereo topic"

# Test cmd_vel - publish forward command
echo "=== Publishing cmd_vel: vx=0.5 ==="
timeout 3 ros2 topic pub -1 /cmd_vel geometry_msgs/Twist '{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'

echo "=== Check odometry after cmd_vel ==="
timeout 3 ros2 topic echo /odom --once 2>/dev/null || echo "odom not available"

# Cleanup
echo "=== Cleanup ==="
kill $SIM_PID 2>/dev/null || true
sleep 2
pkill -f gzserver 2>/dev/null || true
pkill -f gzclient 2>/dev/null || true
