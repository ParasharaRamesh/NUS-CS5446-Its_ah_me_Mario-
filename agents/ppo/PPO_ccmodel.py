#standard packages
import gym
import numpy as np
import cv2
import torch as th
from torch import nn
import os

# mario packages
import gym_super_mario_bros
from gym_super_mario_bros import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import *

# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation

# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# Import algo
from stable_baselines3 import A2C, PPO

# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv, DummyVecEnv

class CoinCollectorSuperMarioBrosEnv(SuperMarioBrosEnv):
    #score btn 2 time frames can maybe go upto 8000 so we can just divide by 100 (reference https://www.mariowiki.com/Point)
    reward_range = (-15, 100)

    def __init__(self, rom_mode='vanilla', lost_levels=False, target=None):
        super().__init__(rom_mode=rom_mode, lost_levels=lost_levels, target=target)

        # variable to keep track of score deltas
        self._score_last = 0

    @property
    def _score_reward(self):
        _reward = self._score - self._score_last
        self._score_last = self._score
        return _reward/100

    # This should override the parent function
    def _get_reward(self):
        return self._x_reward + self._score_reward + self._time_penalty + self._death_penalty

'''
The code below registers this new environment in gym for us to reference later. Code borrowed from _registration.py of gym_super_mario_bros
'''
def _register_coin_collector_mario_stage_env(id, **kwargs):
    """
    Register a Super Mario Bros. (1/2) stage environment with OpenAI Gym.

    Args:
        id (str): id for the env to register
        kwargs (dict): keyword arguments for the SuperMarioBrosEnv initializer

    Returns:
        None

    """
    # register the environment
    gym.envs.registration.register(
        id=id,
        # entry_point='.:CoinCollectorSuperMarioBrosEnv',
        entry_point=CoinCollectorSuperMarioBrosEnv,
        max_episode_steps=9999999,
        reward_threshold=9999999,
        kwargs=kwargs,
        nondeterministic=True,
    )

def _register_all_coin_collector_envs():
    # a template for making individual stage environments
    _ID_TEMPLATE = 'CoinCollectorSuperMarioBrosEnv-{}-{}-v{}'
    # A list of ROM modes for each level environment
    _ROM_MODES = [
        'vanilla',
        'downsample',
        'pixel',
        'rectangle'
    ]

    # iterate over all the rom modes, worlds (1-8), and stages (1-4)
    for version, rom_mode in enumerate(_ROM_MODES):
        for world in range(1, 9):
            for stage in range(1, 5):
                # create the target
                target = (world, stage)
                # setup the frame-skipping environment
                env_id = _ID_TEMPLATE.format(world, stage, version)
                print(f"Registering Coin Collector {env_id} in gym for use later on.")
                _register_coin_collector_mario_stage_env(env_id, rom_mode=rom_mode, target=target)
                print(f"Successfully registered coin collector env {env_id}!")

def create_gym_env_from_level(world, stage, version, use_coin_collector_env):
    level_suffix = f"{world}-{stage}-v{version}"
    if not use_coin_collector_env:
        level = f"SuperMarioBros-{level_suffix}"
        env = gym_super_mario_bros.make(level)
    else:
        env_set = set(gym.envs.registration.registry.env_specs.copy().keys())
        level = f"CoinCollectorSuperMarioBrosEnv-{level_suffix}"
        if level not in env_set:
            # register all these custom environments for the first time
            _register_all_coin_collector_envs()

        assert level in set(
            gym.envs.registration.registry.env_specs.copy().keys()
        ), f"Looks like {level} was not registered correctly!"
        env = gym.make(level)

    return env

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class ResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, size):
        gym.ObservationWrapper.__init__(self, env)
        (oldh, oldw, oldc) = env.observation_space.shape
        newshape = (size, size, oldc)
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=newshape, dtype=np.uint8)

    def observation(self, frame):
        height, width, _ = self.observation_space.shape
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            frame = frame[:,:,None]
        return frame

def create_mario_env(world, stage, version, use_coin_collector_env):
    env = create_gym_env_from_level(world, stage, version, use_coin_collector_env)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeEnv(env, size=84)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    return env

# <world> is a number in {1, 2, 3, 4, 5, 6, 7, 8} indicating the world
world = 1
# <stage> is a number in {1, 2, 3, 4} indicating the stage within a world
stage = 1
version = 3
use_coin_collector_env = True

env = create_mario_env(world, stage, version, use_coin_collector_env)

env.reset()
state, reward, done, info = env.step([0])
print('state:', state.shape) #Color scale, height, width, num of stacks

class MarioNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim):
        super(MarioNet, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))



class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(tensorboard_logdir, f'best_model_{self.n_calls}')
            self.model.save(model_path)

            total_reward = [0] * EPISODE_NUMBERS
            total_time = [0] * EPISODE_NUMBERS
            best_reward = 0

            for i in range(EPISODE_NUMBERS):
                state = env.reset()  # reset for each new trial
                done = False
                total_reward[i] = 0
                total_time[i] = 0
                while not done and total_time[i] < MAX_TIMESTEP_TEST:
                    action, _ = model.predict(state)
                    state, reward, done, info = env.step(action)

                    # This should render it
                    # env.render()

                    total_reward[i] += reward[0]
                    total_time[i] += 1

                if total_reward[i] > best_reward:
                    best_reward = total_reward[i]
                    best_epoch = self.n_calls

                state = env.reset()  # reset for each new trial

            print('time steps:', self.n_calls, '/', TOTAL_TIMESTEP_NUMB)
            print('average reward:', (sum(total_reward) / EPISODE_NUMBERS),
                  'average time:', (sum(total_time) / EPISODE_NUMBERS),
                  'best_reward:', best_reward)

            with open(reward_log_path, 'a') as f:
                print(self.n_calls, ',', sum(total_reward) / EPISODE_NUMBERS, ',', best_reward, file=f)

        return True


policy_kwargs = dict(
    features_extractor_class=MarioNet,
    features_extractor_kwargs=dict(features_dim=512),
)

tensorboard_logdir = os.path.abspath("./mario/cc_model")
reward_log_path = os.path.join(tensorboard_logdir, 'reward_log.csv')

with open(reward_log_path, 'a') as f:
    print('timesteps,reward,best_reward', file=f)


# Model Param
# CHECK_FREQ_NUMB = 10000
CHECK_FREQ_NUMB = 1000
TOTAL_TIMESTEP_NUMB = 5000000
LEARNING_RATE = 0.0001
GAE = 1.0
ENT_COEF = 0.01
N_STEPS = 512
GAMMA = 0.9
BATCH_SIZE = 64
N_EPOCHS = 10

# Test Param
EPISODE_NUMBERS = 20
MAX_TIMESTEP_TEST = 1000

callback = TrainAndLoggingCallback(check_freq=CHECK_FREQ_NUMB, save_path=tensorboard_logdir)

model = PPO('CnnPolicy', env, verbose=2, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_logdir,
            learning_rate=LEARNING_RATE, n_steps=N_STEPS, gamma=GAMMA, gae_lambda=GAE,
            ent_coef=ENT_COEF)

model.learn(total_timesteps=TOTAL_TIMESTEP_NUMB, callback=callback, log_interval=1)
