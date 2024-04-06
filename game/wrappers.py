# Can add any gym wrappers which are going to be used here
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


def env_wrapper_transform(env):
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env
