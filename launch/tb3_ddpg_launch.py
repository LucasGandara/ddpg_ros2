from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable(name='TURTLEBOT3_MODEL', value='burger'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('turtlebot3_gazebo'),
                    'launch/turtlebot3_house.launch.py'
                )
            )
        ),
        Node(
            package='ddpg_ros2',
            executable='tb3_gym_env.py',
            name='ddpg'
        )
    ])