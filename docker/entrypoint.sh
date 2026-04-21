#!/bin/bash
set -e

# Source ROS 2 environment
source "/opt/ros/$ROS_DISTRO/setup.bash"

# Source the workspace if built
if [ -f "/drone_ws/install/setup.bash" ]; then
    source "/drone_ws/install/setup.bash"
fi

exec "$@"
