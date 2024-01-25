#!/usr/bin/env python3
# turtlebot model: ~/ros2_ws/install/turtlebot3_gazebo/share/turtlebot3_gazebo/launch/spawn_turtlebot3.launch.py

import gymnasium as gym
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.executors import ExternalShutdownException
from respawn_goal import Respawn
import sys


class ENV(gym.Env):
    metadata = {"render_modes": ["gazebo"]}

    def __init__(self, node: Node) -> None:
        # ROS related stuff
        self.node = node
        self.laser_scan = np.zeros(24)
        node.create_subscription(LaserScan, "/scan", self.laser_callback, 10)

        # Goal object
        self.respawn_goal = Respawn(node)

        # Env related stuff
        self.action_space = gym.spaces.Box(
            low=np.array([0, -2.0]), high=np.array([0, 2.0]), dtype=np.float32
        )

        self.observation_space = gym.spaces.Dict(
            {
                "laser": gym.spaces.Box(low=0, high=3.5, shape=(24,)),
            }
        )

    def laser_callback(self, msg: LaserScan):
        self.laser_scan = np.array(msg.ranges)

    def _get_obs(self):
        distance_to_goal = 1
        angle_to_goal = 0
        return dict(
            {
                "laser": self.laser_scan,
                "distance_to_goal": distance_to_goal,
                "angle_to_goal": angle_to_goal,
            }
        )

    def _get_info(self):
        position = 0
        return dict({"position": position})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info


if __name__ == "__main__":
    import rclpy

    rclpy.init()
    node = rclpy.create_node("tb3_gym_env")
    env = ENV(node)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        print("\nDestroying node")
        rclpy.try_shutdown()
        node.destroy_node()
