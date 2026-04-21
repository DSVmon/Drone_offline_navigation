import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_drone_simulation = get_package_share_directory('drone_simulation')
    xacro_file = os.path.join(pkg_drone_simulation, 'urdf', 'rescue_drone.urdf.xacro')

    # Process xacro
    robot_description_config = xacro.process_file(xacro_file)
    robot_description = {'robot_description': robot_description_config.toxml()}

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': True}]
    )

    # Spawn Drone
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'rescue_drone', '-z', '1.0'],
        output='screen'
    )

    return LaunchDescription([
        robot_state_publisher,
        spawn_entity,
    ])
