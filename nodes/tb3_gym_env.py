#!/usr/bin/env python3
# turtlebot model: ~/ros2_ws/install/turtlebot3_gazebo/share/turtlebot3_gazebo/launch/spawn_turtlebot3.launch.py

import math
import sys

import numpy as np
import rclpy
import rclpy.callback_groups
import rclpy.executors
import rclpy.logging
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.executors import ExternalShutdownException
from rclpy.logging import LoggingSeverity
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from utils import euler_from_quaternion

from ddpg_ros2.srv import EnvironmentObservation, RespawnGoal


class Env(Node):

    def __init__(self, node_name, debug_level=LoggingSeverity.INFO) -> None:
        super().__init__(node_name)
        self.get_logger().set_level(debug_level)

        self.get_logger().info("Starting tb3_gym_env node")
        # ROS related stuff
        self.laser_scan = []

        self.odom_callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
        self.callback_group_1 = rclpy.callback_groups.ReentrantCallbackGroup()
        self.callback_group_2 = rclpy.callback_groups.ReentrantCallbackGroup()

        self.create_subscription(
            LaserScan,
            topic="/scan",
            callback=self.laser_callback,
            qos_profile=10,
            callback_group=self.callback_group_1,
        )
        self.create_subscription(
            Odometry,
            topic="/odom",
            callback=self.odom_callback,
            qos_profile=10,
            callback_group=self.odom_callback_group,
        )

        # Init odom variable
        self.position = Point()

        # Goal object
        self.respawn_goal_client = self.create_client(
            RespawnGoal,
            "/ddpg_ros2/respawn_env_goal",
            callback_group=self.callback_group_2,
        )

        self.goal_position = Pose()
        self.goal_position.position.x = 0.6

        # Env related stuff
        self.min_distance = 1.5
        self.max_distance = 3.5

        self.num_observation_space = (
            24 + 1 + 1
        )  # Laser + distance to goal + angle to goal
        self.declare_parameter(
            "num_observation_space",
            self.num_observation_space,
            descriptor=ParameterDescriptor(
                name="num_observation_space",
                type=ParameterType.PARAMETER_INTEGER,
                description="Shape of the observation space",
                additional_constraints="The value must be an integer",
            ),
        )
        self.declare_parameter(
            "upper_bound",
            1.5,
            descriptor=ParameterDescriptor(
                name="upper bound of the actions",
                type=ParameterType.PARAMETER_INTEGER,
                description="Upper bound of the action value",
                additional_constraints="The value must be an integer",
            ),
        )
        self.declare_parameter(
            "lower_bound",
            -1.5,
            descriptor=ParameterDescriptor(
                name="lower bound of the actions",
                type=ParameterType.PARAMETER_INTEGER,
                description="Lower bound of the action value",
                additional_constraints="The value must be an integer",
            ),
        )

        self.cmd_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_service(EnvironmentObservation, "reset_env", self.reset_callback)

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
            self.goal_position.position.y - self.position.y,
            self.goal_position.position.x - self.position.x,
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
        ).tolist()

    def get_goal_distance(self):
        goal_distance = round(
            math.hypot(
                self.goal_position.position.x - self.position.x,
                self.goal_position.position.y - self.position.y,
            ),
            2,
        )

        return goal_distance

    def get_angle_to_goal(self):
        goal_angle = math.atan2(
            self.goal_position.position.y - self.position.y,
            self.goal_position.position.x - self.position.x,
        )

        return goal_angle

    def set_reward(self, state: dict, is_terminal: dict):
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

        if is_terminal["collision"]:
            self.get_logger().info("Collision !!")
            reward = -4000
            self.cmd_publisher.publish(Twist())

        if is_terminal["goal"]:
            self.get_logger().info("Goal !!")
            self.cmd_publisher.publish(Twist())
            self.respawn_goal.respawn_entity()
            reward = 4000

        return reward

    def _get_obs(self):
        distance_to_goal = self.get_goal_distance()
        angle_to_goal = self.get_angle_to_goal()

        done = False
        goal = False
        if min(self.laser_scan) < self.min_distance:
            done = True
            self.get_logger().info("Collision !!")

        if self.get_goal_distance() <= 0.2:
            goal = True
            self.get_logger().info("Goal !!")

        return (
            dict(
                {
                    "laser": self.laser_scan,
                    "distance_to_goal": distance_to_goal,
                    "angle_to_goal": angle_to_goal,
                }
            ),
            dict({"collision": done, "goal": goal}),
        )

    def reset_callback(
        self,
        _: EnvironmentObservation.Request,
        response: EnvironmentObservation.Response,
    ):

        observation, _ = self._get_obs()

        while not self.respawn_goal_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service respawn goal...")

        respawn_goal_request = self.respawn_goal_client.call_async(
            RespawnGoal.Request()
        )

        rclpy.spin_until_future_complete(self, respawn_goal_request, timeout_sec=3.0)

        self.get_logger().debug(
            f" Respawn goal Response: {respawn_goal_request.result().response}"
        )

        self.goal_position.x = (
            self.get_parameter("goal_position_x").get_parameter_value().integer_value
        )
        self.goal_position.y = (
            self.get_parameter("goal_position_y").get_parameter_value().integer_value
        )

        response.observation.laser = self.laser_scan
        response.observation.distance_to_goal = observation["distance_to_goal"]
        response.observation.angle_to_goal = observation["angle_to_goal"]

        return response

    def step(self, action):
        linear = action[0]
        angular = action[1]

        vel_cmd = Twist()
        vel_cmd.linear.x = linear
        vel_cmd.angular.z = angular
        self.cmd_publisher.publish(vel_cmd)

        observation, is_terminal = self._get_obs()
        reward = self.set_reward(observation, is_terminal)
        done = is_terminal["collision"]
        truncated = is_terminal["goal"]

        return observation, reward, done, truncated


if __name__ == "__main__":
    rclpy.init()

    env_node = Env("tb3_environment")

    env_executor = rclpy.executors.MultiThreadedExecutor(2)
    env_executor.add_node(env_node)

    try:
        env_executor.spin()
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        print("\nDestroying node")
        env_executor.shutdown()
        rclpy.try_shutdown()
