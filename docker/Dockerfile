# Use ROS 2 Humble as base
FROM ros:humble-ros-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    build-essential \
    libopencv-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    git \
    wget \
    lsb-release \
    # GUI and Gazebo dependencies
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    x11-apps \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-rviz2 \
    ros-humble-rviz-imu-plugin \
    ros-humble-xacro \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep if not already initialized
RUN if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then \
        rosdep init; \
    fi

# Create a non-root user and add to video group for hardware acceleration
ARG USERNAME=drone_user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && usermod -aG video $USERNAME

# Create workspace
WORKDIR /drone_ws
COPY ./src /drone_ws/src
RUN chown -R $USERNAME:$USERNAME /drone_ws

USER $USERNAME

# Install package dependencies
RUN rosdep update && \
    sudo apt-get update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    sudo rm -rf /var/lib/apt/lists/*

# Build the workspace
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    colcon build --symlink-install

# Setup entrypoint
COPY ./docker/entrypoint.sh /entrypoint.sh
USER root
RUN chmod +x /entrypoint.sh
USER $USERNAME

ENTRYPOINT ["/entrypoint.sh"]

CMD ["bash"]
