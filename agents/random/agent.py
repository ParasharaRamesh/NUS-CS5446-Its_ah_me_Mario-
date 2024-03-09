from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


if __name__ == '__main__':
    # env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    done = True
    for step in range(5000):
        if done:
            state = env.reset()
        env_step = env.step(env.action_space.sample())
        # state, reward, done, _, info = env_step # #NOTE: needs gym gym-0.10.9 that way we dont need 5 actions
        state, reward, done, info = env_step
        env.render()

    env.close()
