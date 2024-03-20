import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

def create_gym_env_from_level(level, use_coin_collector_env=False):
    if not use_coin_collector_env:
        env = gym_super_mario_bros.make(level)
    else:
        #TODO something with the custom environment. To mimic the gym make function to use the custom reward environment instead
        pass

    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env