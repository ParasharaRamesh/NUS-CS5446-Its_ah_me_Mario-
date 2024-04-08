import os
import random
import datetime
import torch
import gym  # An OpenAI toolkit for RL
import gym_super_mario_bros  # Super Mario Bros environment for OpenAI Gym
import numpy as np

from pathlib import Path
from collections import deque
from torch import nn
from torchvision import transforms as T
from PIL import Image
from gym.spaces import Box
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace  # NES Emulator for OpenAI Gym
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage


# Initialize Super Mario environment
# <world> is a number in {1, 2, 3, 4, 5, 6, 7, 8} indicating the world
world = 1
# <stage> is a number in {1, 2, 3, 4} indicating the stage within a world
stage = 1
# <version> is a number in {0, 1, 2} indicating the version of the game
version = 0
# SuperMarioBros-<world>-<stage>-v<version>
level = f'SuperMarioBros-{world}-{stage}-v{version}'

if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make(level)
else:
    # change render_mode to 'human' to see results on the screen
    env = gym_super_mario_bros.make(level, render_mode='rgb', apply_api_compatibilities=True)

# Limit the action space to
#   0. walk right
#   1. jump right
env = JoypadSpace(env, [['right'], ['right', 'A']])

env.reset()
next_state, reward, done, info = env.step(action=0)
print(f'{next_state.shape},\n{reward},\n{done},\n{info}')


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """
        Return only every `skip`-th frame

        Args:
            env (gym.Env): Gym environment that will be wrapped
            skip (int): Number of frames to skip

        Returns:
            None
        """
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """
        Repeat action, and sum reward

        Args:
            action (int): Action to be taken

        Returns:
            tuple: (np.ndarray, float, bool, dict) Observation, reward, done, info
        """
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        """
        Make the observation gray scale

        Args:
            env (gym.Env): Gym environment that will be wrapped

        Returns:
            None
        """
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        """
        Change the orientation of the observation from (H, W, C) to (C, H, W)

        Args:
            observation (np.ndarray): Observation from the environment

        Returns:
            np.ndarray: Observation with the orientation changed
        """
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        """
        Change the observation to gray scale

        Args:
            observation (np.ndarray): Observation from the environment

        Returns:
            np.ndarray: Gray scale observation
        """
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        """
        Resize the observation to a given shape

        Args:
            env (gym.Env): Gym environment that will be wrapped
            shape: New shape of the observation

        Returns:
            None
        """
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        """
        Resize the observation

        Args:
            observation (np.ndarray): Observation from the environment

        Returns:
            np.ndarray: Resized observation
        """
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


# Apply the wrappers to the environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)


class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        """
        Initialize Mario's DNN

        Args:
            state_dim: the state dimension
            action_dim: the action dimension
            save_dir: directory to save the model
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving Mario Net

        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device('cpu')))
        self.batch_size = 32

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action

        Args:
            state (LazyFrame): A single observation of the current state, dimension is (state_dim)

        Returns:
            action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(selfself, state, next_state, action, reward, done):
        """
        Add the experience to memory

        Args:
            state (LazyFrame): A single observation of the current state, dimension is (state_dim)
            next_state (LazyFrame): A single observation of the next state, dimension is (state_dim)
            action (int): An integer representing which action Mario will perform
            reward (float): The reward received after performing the action
            done (bool): A boolean representing if the episode terminated

        Returns:

        """
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        # self.memory.append((state, next_state, action, reward, done))
        

    def recall(self):
        """
        Sample experiences from memory

        Returns:

        """
        pass

    def learn(self):
        """
        Update online action value (Q) function with a batch of experiences

        Returns:

        """
        pass