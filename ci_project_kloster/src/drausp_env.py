import time

import gymnasium as gym
import numpy as np
from gymnasium import Env
from gymnasium.wrappers import NormalizeObservation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecEnv

from drausp_instance_data import DrauspInstanceData, get_instance_data


class DrauspEnv(gym.Env):

    def __init__(self, instance_data: DrauspInstanceData):
        self.instance = instance_data

        self.obs_size = (
            self.instance.num_dimensions * (self.instance.num_moves + 1)
            + (self.instance.num_moves + 1)
            + 1
        )
        self.num_actions = instance_data.num_moves + 1
        self.actions = [i for i in range(self.num_actions)]
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Box(
            low=np.ones(self.obs_size, dtype=np.float32) * -100,
            high=np.ones(self.obs_size, dtype=np.float32) * 100,
        )

        self.current_step = 0
        self.remaining_capacity = None
        self.epoch_request_idx = None
        self.rewards = None
        self.requests = None
        self.recent_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.recent_reward = 0.0
        self.remaining_capacity = self.instance.capacity_vector.copy()
        self.epoch_request_idx = self._np_random.integers(
            low=0,
            high=self.instance.num_requests,
            size=self.instance.num_stages,
            dtype=int,
        )
        self.rewards = self.instance.revenues[self.epoch_request_idx]
        self.requests = self.instance.requests[self.epoch_request_idx]
        return self.get_observation(), {}

    def step(self, action):
        if action == 0:
            self.recent_reward = 0.0
        else:
            self.remaining_capacity -= self.requests[self.current_step][action - 1]
            if np.min(self.remaining_capacity) < 0:
                self.recent_reward = -30.0
                return self.get_final_observation(), -30.0, True, True, {}
            else:
                self.recent_reward = self.rewards[self.current_step]
        self.current_step += 1

        if self.current_step >= self.instance.num_stages:
            # self.recent_reward = 0
            return self.get_final_observation(), self.recent_reward, True, False, {}
        else:
            return self.get_observation(), self.recent_reward, False, False, {}

    def get_observation(self):
        observation = np.empty(self.obs_size, dtype=np.float32)
        observation[: self.instance.num_dimensions] = self.remaining_capacity
        diffs = self.remaining_capacity - self.requests[self.current_step]
        observation[self.instance.num_dimensions : -self.instance.num_moves - 2] = (
            diffs.ravel()
        )
        negative_mask = diffs.min(axis=1) < 0
        observation[-self.instance.num_moves - 2] = 0.0
        mask_counter = 0
        for i in range(
            (self.instance.num_moves + 1) * self.instance.num_dimensions + 1,
            self.obs_size - 1,
        ):
            if negative_mask[mask_counter]:
                observation[i] = -10.0
            else:
                observation[i] = self.rewards[self.current_step]
            mask_counter += 1
        observation[-1] = self.instance.num_stages - self.current_step
        return observation

    def get_final_observation(self):
        observation = np.zeros(self.obs_size, dtype=np.float32)
        observation[: self.instance.num_dimensions] = self.remaining_capacity
        for i in range(
            (self.instance.num_moves + 1) * self.instance.num_dimensions + 1,
            self.obs_size - 1,
        ):
            observation[i] = self.recent_reward
        return observation


def register_env(instance_data: DrauspInstanceData) -> str:
    env_id = "DRAUSP"
    gym.register(
        id=env_id,
        entry_point=DrauspEnv,
        kwargs={"instance_data": instance_data},
    )
    return env_id


def get_vec_env(
    env_id: str, num_envs: int, seed: int, normalize: bool = True
) -> VecEnv:
    env = make_vec_env(env_id, num_envs, seed)
    if normalize:
        env = VecNormalize(env)
    return env


def get_env(env_id: str, seed: int, normalize: bool = True) -> Env:
    env = gym.make(env_id)
    if normalize:
        env = NormalizeObservation(env)
    env.reset(seed=seed)
    return env


if __name__ == "__main__":
    instance_data = get_instance_data("instances/wendtris/S-wendtris12D.txt", 20)
    env_id = register_env(instance_data)
    drausp_env = get_env(env_id, 1)
    total_rewards = []
    start_time = time.time()
    num_episodes = 100_000
    for i in range(num_episodes):
        total_reward = 0.0
        terminated = False
        drausp_env.reset()
        while not terminated:
            observation, reward, terminated, truncated, info = drausp_env.step(
                drausp_env.action_space.sample()
            )
            total_reward += reward
        total_rewards.append(total_reward)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"\nResults after {num_episodes} episodes:")
    print(f"Average total reward: {np.mean(total_rewards):.2f}")
    print(f"Runtime: {runtime:.2f} seconds")
