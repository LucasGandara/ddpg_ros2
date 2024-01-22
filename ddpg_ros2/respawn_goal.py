#!/usr/bin/env python3
# -*- utf-8

# Example: https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning/blob/master/turtlebot3_dqn/src/turtlebot3_dqn/respawnGoal.py
import os
from geometry_msgs.msg import Pose
from rclpy.node import Node
import rclpy
from gazebo_msgs.srv import SpawnEntity, GetModelList, DeleteEntity


class Respawn:
    def __init__(self, node: Node) -> None:
        self.node = node
        file_path = os.path.dirname(os.path.realpath(__file__))
        print(file_path)
        self.model_path = file_path.replace(
            "/ddpg_ros2/ddpg_ros2", "/ddpg_ros2/models/goal.sdf"
        )
        self.file = open(self.model_path, "r")
        self.model_description = self.file.read()

        self.model_subscriber = node.create_subscription(
            GetModelList, "/get_model_list", self.model_subscriber_callback, 10
        )

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

    def check_model(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    def respawn_entity(self):
        while True:
            if not self.check_model:
                while not self.spawn_entity_client.wait_for_service(timeout_sec=2.0):
                    self.node.get_logger().info(
                        "service not available, waiting again..."
                    )
                self.spawn_entity_client.call_async(
                    SpawnEntity.Request(self.goal_position, "goal", "robot_description")
                )
                break
            else:
                pass

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
    Respawn(client)
