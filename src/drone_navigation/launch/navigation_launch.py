from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/image_raw',
            description='Topic for RGB camera images'
        ),
        DeclareLaunchArgument(
            'dtof_topic',
            default_value='/sensor/dtof_range',
            description='Topic for dToF range data'
        ),
        DeclareLaunchArgument(
            'imu_topic',
            default_value='/imu/data',
            description='Topic for IMU data'
        ),
        Node(
            package='drone_perception',
            executable='perception_node',
            name='perception',
            parameters=[{
                'image_topic': LaunchConfiguration('image_topic'),
                'dtof_topic': LaunchConfiguration('dtof_topic'),
            }]
        ),
        Node(
            package='drone_navigation',
            executable='navigation_node',
            name='navigation',
            parameters=[{
                'image_topic': LaunchConfiguration('image_topic'),
                'imu_topic': LaunchConfiguration('imu_topic'),
            }]
        ),
        Node(
            package='drone_control',
            executable='control_node',
            name='control'
        ),
    ])
