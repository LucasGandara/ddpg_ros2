#!/usr/bin/env python3
# turtlebot model: ~/ros2_ws/install/turtlebot3_gazebo/share/turtlebot3_gazebo/launch/spawn_turtlebot3.launch.py

import math
import gymnasium as gym
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.executors import ExternalShutdownException
from respawn_goal import Respawn
from geometry_msgs.msg import Twist, Quaternion, Point
from nav_msgs.msg import Odometry
import sys
from utils import euler_from_quaternion


class Env(gym.Env):
    metadata = {"render_modes": ["gazebo"]}

    def __init__(self, node: Node) -> None:
        # ROS related stuff
        self.node = node
        self.laser_scan = np.zeros(24)
        node.create_subscription(LaserScan, "/scan", self.laser_callback, 10)
        node.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        self.cmd_publisher = node.create_publisher(Twist, "/cmd_vel", 10)

        # Init odom variable
        self.position = Point()

        # Goal object
        self.respawn_goal = Respawn(node)

        # Env related stuff
        self.min_distance = 1.5
        self.max_distance = 3.5

        self.action_space = gym.spaces.Dict(
            {
                "linear": gym.spaces.Box(low=0, high=1.5, shape=(1,)),
                "angular": gym.spaces.Box(low=0, high=1.5, shape=(1,)),
            }
        )

        self.observation_space = gym.spaces.Dict(
            {
                "laser": gym.spaces.Box(low=0, high=3.5, shape=(24,)),
                "distance_to_goal": gym.spaces.Box(low=0, high=3.5, shape=(1,)),
                "angle_to_goal": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
            }
        )

    def odom_callback(self, msg: Odometry):
        self.position = msg.pose.pose.position
        self.orientation = msg.pose.pose.orientation

        orientation_list = Quaternion()
        orientation_list.x = self.orientation.x
        orientation_list.y = self.orientation.y
        orientation_list.z = self.orientation.z
        orientation_list.w = self.orientation.w

        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(
            self.respawn_goal.goal_position.position.y - self.position.y,
            self.respawn_goal.goal_position.position.x - self.position.x,
        )

        heading = goal_angle - yaw

        if heading > math.pi:
            heading -= 2 * math.pi

        elif heading < -math.pi:
            heading += 2 * math.pi

        self.heading = round(heading, 2)

    def laser_callback(self, msg: LaserScan):
        self.min_distance = msg.range_min
        self.max_distance = msg.range_max
        self.laser_scan = np.clip(
            msg.ranges, self.min_distance - 0.1, self.max_distance
        )

    def get_goal_distance(self):
        goal_distance = round(
            math.hypot(
                self.respawn_goal.goal_position.position.x - self.position.x,
                self.respawn_goal.goal_position.position.y - self.position.y,
            ),
            2,
        )

        return goal_distance

    def set_reward(self, state: dict, done: bool):
        current_distance = state["distance_to_goal"]
        heading = state["angle_to_goal"]

        if current_distance >= 7:
            reward = -current_distance / 3 + 10 / 3
        elif current_distance >= 5 and current_distance < 7:
            reward = -current_distance / 2 + 9 / 2
        elif current_distance >= 1 and current_distance < 5:
            reward = -current_distance / 4 + 13 / 4
        elif current_distance >= 0 and current_distance < 1:
            reward = -current_distance + 4
        else:
            reward = 0

        if done:
            self.node.get_logger().info("Collision !!")
            reward = -4000
            self.cmd_publisher.publish(Twist())

        if current_distance <= self.min_distance:
            self.node.get_logger().info("Goal !!")
            self.cmd_publisher.publish(Twist())
            reward = 4000

        return reward

    def _get_obs(self):
        distance_to_goal = 1
        angle_to_goal = 0

        done = False
        if min(self.laser_scan) < self.min_distance:
            done = True
            self.node.get_logger().info("Collision !!")

        if self.get_goal_distance() <= 0.2:
            done = True
            self.node.get_logger().info("Goal !!")

        return (
            dict(
                {
                    "laser": self.laser_scan,
                    "distance_to_goal": distance_to_goal,
                    "angle_to_goal": angle_to_goal,
                }
            ),
            done,
        )

    def _get_info(self):
        position = 0
        return dict({"position": position})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        observation, _ = self._get_obs()
        info = self._get_info()

        self.respawn_goal.respawn_entity()
        self.goal_x = self.respawn_goal.goal_position.position.x
        self.goal_y = self.respawn_goal.goal_position.position.y

        return observation, info

    def step(self, action):
        linear = action[0]
        angular = action[1]

        vel_cmd = Twist()
        vel_cmd.linear.x = linear
        vel_cmd.angular.z = angular
        self.cmd_publisher.publish(vel_cmd)

        observation, done = self._get_obs()
        reward = self.set_reward(observation, done)
        info = self._get_info()
        truncated = False

        return observation, reward, done, truncated, info


if __name__ == "__main__":
    import rclpy

    rclpy.init()

    node = rclpy.create_node("tb3_gym_env")
    node.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

    node.get_logger().info("Starting tb3_gym_env node")
    env = Env(node)

    env.reset()

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
