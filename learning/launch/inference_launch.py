"""
Launch file for drone simulation with NN control.

Same as simulation_launch.py but replaces control_node with inference_node.
All other nodes (navigation, perception, Gazebo, RViz) are unchanged.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")
    pkg_drone_simulation = get_package_share_directory("drone_simulation")

    world_path = os.path.join(pkg_drone_simulation, "worlds", "cave.world")

    # Find paths relative to workspace
    workspace_root = os.path.abspath(
        os.path.join(pkg_drone_simulation, "..", "..", "..", "..")
    )
    script_path = os.path.join(workspace_root, "scripts", "procedural_cave.py")
    if not os.path.exists(script_path):
        script_path = os.path.join(os.getcwd(), "scripts", "procedural_cave.py")

    inference_script = os.path.join(workspace_root, "learning", "inference_node.py")
    if not os.path.exists(inference_script):
        inference_script = os.path.join(os.getcwd(), "learning", "inference_node.py")

    # 0. Generate procedural cave
    generate_cave = ExecuteProcess(
        cmd=["python3", script_path, world_path],
        output="screen",
    )

    # 1. Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "world": world_path,
        }.items(),
    )

    # 2. Spawn drone
    spawn_drone = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_drone_simulation, "launch", "spawn_drone.launch.py"
            )
        )
    )

    # 3. RViz2
    rviz_config_path = os.path.join(
        pkg_drone_simulation, "rviz", "drone_config.rviz"
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_path],
    )

    # 4. Perception Node (unchanged)
    perception_node = Node(
        package="drone_perception",
        executable="perception_node",
        name="perception_node",
        output="screen",
        parameters=[{"min_safe_distance": 0.5, "use_sim_time": True}],
    )

    # 5. Navigation Node (unchanged)
    navigation_node = Node(
        package="drone_navigation",
        executable="navigation_node",
        name="navigation_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # 6. Inference Node (replaces control_node)
    inference_node = ExecuteProcess(
        cmd=["python3", inference_script],
        output="screen",
    )

    return LaunchDescription([
        generate_cave,
        gazebo,
        spawn_drone,
        rviz,
        perception_node,
        navigation_node,
        inference_node,
    ])
