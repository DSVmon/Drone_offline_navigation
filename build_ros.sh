#!/bin/bash
# Build only ROS packages (learning/ excluded via COLCON_IGNORE)
source /opt/ros/humble/setup.bash
colcon build --symlink-install \
  --packages-select drone_control drone_navigation drone_perception drone_simulation
