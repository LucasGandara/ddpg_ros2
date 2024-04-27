import os

from ament_index_python import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    return LaunchDescription(
        [
            SetEnvironmentVariable(name="TURTLEBOT3_MODEL", value="burger"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("turtlebot3_gazebo"),
                        "launch/turtlebot3_dqn_stage1.launch.py",
                    )
                )
            ),
            Node(
                package="ddpg_ros2",
                executable="respawn_goal.py",
                name="respawn_goal",
            ),
            # Node(package="ddpg_ros2", executable="tb3_ddpg.py", name="ddpg"),
        ]
    )
