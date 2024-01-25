#!/usr/bin/env python3
# -*- utf-8

# Example: https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning/blob/master/turtlebot3_dqn/src/turtlebot3_dqn/respawnGoal.py

import os
from geometry_msgs.msg import Pose
from rclpy.node import Node
import rclpy
from gazebo_msgs.srv import SpawnEntity, GetModelList, DeleteEntity
from rclpy.logging import LoggingSeverity
import sys
from rclpy.executors import ExternalShutdownException


class Respawn:
    def __init__(self, node: Node) -> None:
        self.node = node
        file_path = os.path.dirname(os.path.realpath(__file__))

        self.model_path = file_path.replace(
            "/ddpg_ros2/ddpg_ros2", "/ddpg_ros2/models/goal.sdf"
        )

        self.node.get_logger().debug(f"model path: {self.model_path}")
        self.file = open(self.model_path, "r")
        self.model_description = self.file.read()

        self.model_service_client = node.create_client(GetModelList, "/get_model_list")

        # Delete entity service
        self.delete_entity_client = node.create_client(DeleteEntity, "/delete_entity")

        # goal entity
        self.spawn_entity_client = node.create_client(SpawnEntity, "/spawn_entity")
        self.entity_name = "goal"
        self.goal_position = Pose()
        self.goal_position.position.x = 0.0
        self.goal_position.position.y = 0.0
        self.goal_position.position.z = 0.0

        self.goal_position.orientation.x = 0.0
        self.goal_position.orientation.y = 0.0
        self.goal_position.orientation.z = 0.0

        self.check_model = False

    def model_subscriber_callback(self, msg):
        self.check_model = False
        for model in msg.models:
            if model.name == self.entity_name:
                self.check_model = True
                break

    def check_model_call(self, model: str = None):
        self.node.get_logger().debug(
            "List model service available, waiting until is ready..."
        )
        while not self.model_service_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("service not available, waiting again...")

        self.node.get_logger().debug("List model service available, calling service...")
        model_list_request = self.model_service_client.call_async(
            GetModelList.Request()
        )
        rclpy.spin_until_future_complete(self.node, model_list_request, timeout_sec=3.0)

        self.node.get_logger().debug("List model service response received")
        self.check_model = False
        for i in range(len(model_list_request.result().model_names)):
            if model:
                if model_list_request.result().model_names[i] == model:
                    self.check_model = True
                    self.node.get_logger().debug("List model service: Model found")
            else:
                if model_list_request.result().model_names[i] == self.entity_name:
                    self.check_model = True
                    self.node.get_logger().debug("List model service: Model found")

    def respawn_entity(self):
        self.node.get_logger().debug(
            "Respawn entity service available, waiting until is ready..."
        )

        print(self.check_model)

        if not self.check_model:
            while not self.spawn_entity_client.wait_for_service(timeout_sec=2.0):
                self.node.get_logger().info(
                    "Respawn entity service not available, waiting again..."
                )

            self.node.get_logger().debug("Respawn entity calling service...")

            request = SpawnEntity.Request()
            request.name = self.entity_name
            request.xml = self.model_description
            request.robot_namespace = ""
            request.initial_pose = self.goal_position
            spawn_entity_request = self.spawn_entity_client.call_async(request)

            response = rclpy.spin_until_future_complete(
                self.node, spawn_entity_request, timeout_sec=3.0
            )

            self.node.get_logger().debug(
                f"Respawn entity service response received: {response}"
            )

    def delete_model(self):
        while True:
            if self.check_model:
                while not self.delete_entity_client.wait_for_service(timeout_sec=2.0):
                    self.node.get_logger().info(
                        "service not available, waiting again..."
                    )
                self.delete_entity_client.call_async(
                    DeleteEntity.Request(name=self.entity_name)
                )
                break
            else:
                pass


if __name__ == "__main__":
    rclpy.init()

    client = rclpy.create_node("respawn_goal")
    client.get_logger().set_level(LoggingSeverity.DEBUG)

    respawn = Respawn(client)

    respawn.check_model_call()  # Should return false

    respawn.respawn_entity()

    respawn.check_model_call()  # Should return true
    try:
        rclpy.spin(client)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        print("\nDestroying node")
        rclpy.try_shutdown()
        client.destroy_node()
