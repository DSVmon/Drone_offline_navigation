"""
Launch file for drone training (no control_node, headless support).

Launches everything needed for RL training:
  - Gazebo (with headless option)
  - Robot State Publisher
  - Spawn Drone
  - Navigation Node
  - Perception Node
  - No control_node (env publishes cmd_vel)

Usage:
    ros2 launch learning/launch/training_launch.py gui:=false
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")
    pkg_drone_simulation = get_package_share_directory("drone_simulation")

    world_path = os.path.join(pkg_drone_simulation, "worlds", "cave.world")
    xacro_file = os.path.join(
        pkg_drone_simulation, "urdf", "rescue_drone.urdf.xacro"
    )

    # Launch arguments
    gui_arg = DeclareLaunchArgument(
        "gui",
        default_value="false",
        description="Run Gazebo with GUI (true) or headless (false)",
    )

    # 1. Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "world": world_path,
            "gui": LaunchConfiguration("gui"),
        }.items(),
    )

    # 2. Robot State Publisher (processes xacro)
    robot_description_config = xacro.process_file(xacro_file)
    robot_description = {"robot_description": robot_description_config.toxml()}

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    # 3. Spawn Drone
    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-topic", "robot_description",
                   "-entity", "rescue_drone", "-z", "1.0"],
        output="screen",
    )

    # 4. Navigation Node (headless: no OpenCV windows)
    navigation_node = Node(
        package="drone_navigation",
        executable="navigation_node",
        name="navigation_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
        additional_env={"DISPLAY": ""},
    )

    # 5. Perception Node
    perception_node = Node(
        package="drone_perception",
        executable="perception_node",
        name="perception_node",
        output="screen",
        parameters=[{"min_safe_distance": 0.5, "use_sim_time": True}],
    )

    return LaunchDescription([
        gui_arg,
        gazebo,
        robot_state_publisher,
        spawn_entity,
        navigation_node,
        perception_node,
    ])
