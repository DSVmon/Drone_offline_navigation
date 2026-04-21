import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_drone_simulation = get_package_share_directory('drone_simulation')

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': os.path.join(pkg_drone_simulation, 'worlds', 'cave.world')}.items()
    )

    # Spawn drone
    spawn_drone = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_drone_simulation, 'launch', 'spawn_drone.launch.py')
        )
    )

    # RViz2 for visualization
    rviz_config_path = os.path.join(pkg_drone_simulation, 'rviz', 'drone_config.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path]
    )

    # Perception Node
    perception_node = Node(
        package='drone_perception',
        executable='perception_node',
        name='perception_node',
        output='screen',
        parameters=[{'min_safe_distance': 0.5, 'use_sim_time': True}]
    )

    # Navigation Node
    navigation_node = Node(
        package='drone_navigation',
        executable='navigation_node',
        name='navigation_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # Control Node
    control_node = Node(
        package='drone_control',
        executable='control_node',
        name='control_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        gazebo,
        spawn_drone,
        rviz,
        perception_node,
        navigation_node,
        control_node,
    ])
