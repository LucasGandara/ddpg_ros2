#!/usr/bin/env python3
# turtlebot model: ~/ros2_ws/install/turtlebot3_gazebo/share/turtlebot3_gazebo/launch/spawn_turtlebot3.launch.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
from respawn_goal import Respawn

from rcl_interfaces.srv import ListParameters


class DDPG_NODE(Node):
    def __init__(self):
        super().__init__("ddpg")
        self.publisher_ = self.create_publisher(String, "topic", 10)

        # Wait for services
        self.respawn_service = Respawn(self)

        tb3_gazebo_service = self.create_client(
            ListParameters, "/robot_state_publisher/list_parameters"
        )
        while not tb3_gazebo_service.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("service not available, waiting again...")
        time.sleep(2)

        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = "Hello World: %d" % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = DDPG_NODE()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
