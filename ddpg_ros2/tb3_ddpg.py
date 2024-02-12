#!/usr/bin/env python3
# -*- coding: utf8 -*-

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
import sys
from tb3_gym_env import Env
import gymnasium as gym
from utils import OUActionNoise, get_actor, get_critic
import numpy as np
import tensorflow as tf
import itertools


class DDPG(Node):
    def __init__(
        self,
        buffer_capacity=100000,
        batch_size=64,
        std_dev=0.02,
        noise_object=None,
        critic_lr=0.002,
        actor_lr=0.001,
        gamma=0.99,
        tau=0.005,
        debug=False,
    ) -> None:
        super().__init__("DDPG")
        self.get_logger().debug("Starting DDPG Node")

        if debug:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

        self.env = Env(self)

        assert not isinstance(
            self.env.action_space, gym.spaces.Discrete
        ), "Action space must be continuous"

        self.num_states = (
            self.env.observation_space["laser"].shape[0]
            + self.env.observation_space["distance_to_goal"].shape[0]
            + self.env.observation_space["angle_to_goal"].shape[0]
        )
        self.get_logger().info(f"Size of states --> {self.num_states}")

        self.upper_bound = self.env.action_space["linear"].high[0]
        self.lower_bound = self.env.action_space["linear"].low[0]

        self.num_actions = (
            self.env.action_space["linear"].shape[0]
            + self.env.action_space["angular"].shape[0]
        )
        self.get_logger().info(f"Size of actions --> {self.num_actions}")

        # Hyperparameters
        std_dev = std_dev
        self.ou_noise = noise_object

        # Learning rate for actor-critic models
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr

        self.actor_model = get_actor(
            self.num_states, self.num_actions, self.upper_bound, name="actor_model"
        )
        self.critic_model = get_critic(
            self.num_states, self.num_actions, name="critic_model"
        )

        self.actor_model.summary()
        self.critic_model.summary()

        self.target_actor = get_actor(
            self.num_states, self.num_actions, self.upper_bound
        )
        self.target_critic = get_critic(self.num_states, self.num_actions)

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        self.critic_optimizer = tf.keras.optimizers.legacy.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.legacy.Adam(actor_lr)

        # Discount factor for future rewards
        self.gamma = gamma

        # Used to update target networks
        self.tau = tau

        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element

        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))

    # Takes (s,a,r,s') observation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def policy(self, state, noise_object, lower_bound, upper_bound):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

        return np.squeeze(legal_action)

    def run(self):
        total_episodes = 1

        # To store reward history of each episode
        ep_reward_list = []

        # To store average reward history of last few episodes
        avg_reward_list = []

        for ep in range(total_episodes):
            prev_state, _ = self.env.reset()
            laser = prev_state["laser"]
            distance_to_goal = [prev_state["distance_to_goal"]]
            angle_to_goal = [prev_state["angle_to_goal"]]

            prev_state = tf.concat(
                [laser, distance_to_goal, angle_to_goal],
                axis=0,
                name="concat_prev_states",
            )

            episodic_reward = 0
            step_number = 0

            while True:
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

                action = self.policy(
                    tf_prev_state, self.ou_noise, self.lower_bound, self.upper_bound
                )

                # Recieve state and reward from environment.
                state, reward, done, info, _ = self.env.step(action)

                laser = self.env.laser_scan
                distance_to_goal = state["distance_to_goal"]
                angle_to_goal = state["angle_to_goal"]

                state = tf.concat(
                    [laser, [distance_to_goal], [angle_to_goal]],
                    axis=0,
                    name="concat_states",
                )

                self.record((prev_state, action, reward, state))
                episodic_reward += reward

                break


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "--debug", type=bool, default=False)

    args = parser.parse_args()

    rclpy.init()

    # Hyperparameters
    std_dev = 0.2
    actor_lr = 0.001
    critic_lr = 0.002

    noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

    # DDPG
    buffer = DDPG(
        buffer_capacity=50000,
        batch_size=64,
        std_dev=std_dev,
        noise_object=noise,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        debug=True,
    )

    buffer.run()

    try:
        rclpy.spin(buffer)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        print("\nDestroying node")
        rclpy.try_shutdown()
        buffer.destroy_node()
    return 0


if __name__ == "__main__":
    main()
