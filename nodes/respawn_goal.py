#!/usr/bin/env python3
# -*- coding: utf8 -*-

# Example: https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning/blob/master/turtlebot3_dqn/src/turtlebot3_dqn/respawnGoal.py

import os
import random
import sys

import rclpy
import rclpy.callback_groups
import rclpy.executors
from gazebo_msgs.srv import DeleteEntity, GetModelList, SpawnEntity
from geometry_msgs.msg import Pose
from rclpy.executors import ExternalShutdownException
from rclpy.logging import LoggingSeverity
from rclpy.node import Node

from ddpg_ros2.srv import RespawnGoal


class Respawn(Node):
    def __init__(self, node_name, logging_level=LoggingSeverity.INFO) -> None:
        super().__init__(node_name)
        self.get_logger().set_level(logging_level)
        self.get_logger().info("Starting respawn goal node")

        self.__file_path = os.path.dirname(os.path.realpath(__file__))
        self.__model_path = self.__file_path.replace(
            "/ddpg_ros2/nodes", "/ddpg_ros2/models/goal.sdf"
        )
        self.__file = open(self.__model_path, "r")
        self.model_description = self.__file.read()
        self.get_logger().debug(f"model path: {self.__model_path}")

        self.entity_name = "goal"
        self.goal_model_check = False
        self.goal_position = Pose()
        self.goal_position.position.x = 0.6
        self.get_logger().debug(
            f"goal position: {self.goal_position.position.x}, {self.goal_position.position.y}"
        )
        self.index = random.randrange(0, 13)
        self.last_index = 0

        self.callback_group_1 = rclpy.callback_groups.ReentrantCallbackGroup()

        self.model_service_client = self.create_client(
            GetModelList, "/get_model_list", callback_group=self.callback_group_1
        )
        self.spawn_entity_client = self.create_client(
            SpawnEntity, "/spawn_entity", callback_group=self.callback_group_1
        )
        self.delete_entity_client = self.create_client(
            DeleteEntity, "/delete_entity", callback_group=self.callback_group_1
        )

        self.create_service(RespawnGoal, "respawn_env_goal", self.respawn_goal_callback)
        self.create_service(RespawnGoal, "delete_model", self.delete_model_callback)
        self.create_service(RespawnGoal, "get_position", self.get_position_callback)

    def check_model_call(self) -> None:
        self.get_logger().info("Checking model")
        self.goal_model_check = False

        while not self.model_service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")
        self.get_logger().debug("List model service available, calling service...")
        model_list_request = self.model_service_client.call_async(
            GetModelList.Request()
        )
        rclpy.spin_until_future_complete(self, model_list_request, timeout_sec=3.0)
        self.get_logger().debug("List model service response received")

        self.goal_model_check = False

        for model in model_list_request.result().model_names:
            if model == self.entity_name:
                self.goal_model_check = True
                self.get_logger().debug("List model service: Model found!")

    def respawn_goal_callback(
        self, _: RespawnGoal.Request, response: RespawnGoal.Response
    ) -> RespawnGoal.Response:
        self.get_logger().info("Respawning goal")

        if not self.goal_model_check:
            while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("service not available, waiting again...")

            self.get_logger().debug(
                "Spawn entity service available, calling service..."
            )

            request = SpawnEntity.Request()
            request.name = self.entity_name
            request.xml = self.model_description
            request.robot_namespace = ""
            request.initial_pose = self.goal_position

            spawn_entity_request = self.spawn_entity_client.call_async(request)

            rclpy.spin_until_future_complete(
                self, spawn_entity_request, timeout_sec=3.0
            )
            self.get_logger().debug("Spawn entity service response received")
            self.get_logger().info(spawn_entity_request.result().status_message)

            response.response = spawn_entity_request.result().status_message

            return response

    def delete_model_callback(
        self, _: RespawnGoal.Request, response: RespawnGoal.Response
    ) -> None:
        self.get_logger().info("Deleting goal model")
        while not self.delete_entity_client.wait_for_service(timeout_sec=2.0):
            self.node.get_logger().info("service not available, waiting again...")

        delete_entity_request = self.delete_entity_client.call_async(
            DeleteEntity.Request(name=self.entity_name)
        )
        rclpy.spin_until_future_complete(self, delete_entity_request)
        self.get_logger().info(delete_entity_request.result().status_message)

        response.response = delete_entity_request.result().status_message

        return response

    def get_position_callback(
        self, _: RespawnGoal.Request, response: RespawnGoal.Response
    ) -> None:
        self.get_logger().debug("Getting goal position")
        position_check = True
        while position_check:
            goal_x_list = [
                0.6,
                1.9,
                0.5,
                0.2,
                -0.8,
                -1.0,
                -1.9,
                0.5,
                2.0,
                0.5,
                0.0,
                -0.1,
                -2.0,
            ]
            goal_y_list = [
                0.0,
                -0.5,
                -1.9,
                1.5,
                -0.9,
                1.0,
                1.1,
                -1.5,
                1.5,
                1.8,
                -1.0,
                1.6,
                -0.8,
            ]

            self.index = random.randrange(0, 13)
            goal_x = self.goal_position.position.x
            goal_y = self.goal_position.position.y

            if self.last_index == self.index:
                position_check = True
            else:
                self.last_index = self.index
                position_check = False

            self.goal_position.position.x = goal_x_list[self.index]
            self.goal_position.position.y = goal_y_list[self.index]

            self.get_logger().debug(f"Goal position: {goal_x}, {goal_y}")

        response.response = "success"

        return response


if __name__ == "__main__":
    rclpy.init()

    respawn_node = Respawn("respawn_goal", logging_level=LoggingSeverity.DEBUG)

    respawn_executor = rclpy.executors.MultiThreadedExecutor(2)
    respawn_executor.add_node(respawn_node)

    try:
        respawn_executor.spin()
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        print("\nDestroying node")
        respawn_executor.shutdown()
        rclpy.try_shutdown()
