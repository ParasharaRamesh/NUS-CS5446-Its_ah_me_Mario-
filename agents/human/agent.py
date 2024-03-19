'''
Idea is to use this class to ensure that people can play and record their performances as well.

This code is modified from nes_py.app.play_human file from nes_py python module

'''
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py._image_viewer import ImageViewer
from nes_py.wrappers import JoypadSpace
import gym
import time
from pyglet import clock
import pickle
from tqdm import tqdm

from utils.record import save_play

# the sentinel value for "No Operation"
_NOP = 0


def record_with_human_play(env: gym.Env, level, buffer=[], record=False):
    """
    Play the environment using keyboard as a human.

    Args:
        env: the initialized gym environment to play
        callback: a callback to receive output from the environment

    Returns:
        None

    """
    # ensure the observation space is a box of pixels
    assert isinstance(env.observation_space, gym.spaces.box.Box)
    # ensure the observation space is either B&W pixels or RGB Pixels
    obs_s = env.observation_space
    is_bw = len(obs_s.shape) == 2
    is_rgb = len(obs_s.shape) == 3 and obs_s.shape[2] in [1, 3]
    assert is_bw or is_rgb
    # get the mapping of keyboard keys to actions in the environment
    if hasattr(env, 'get_keys_to_action'):
        keys_to_action = env.get_keys_to_action()
    elif hasattr(env.unwrapped, 'get_keys_to_action'):
        keys_to_action = env.unwrapped.get_keys_to_action()
    else:
        raise ValueError('env has no get_keys_to_action method')
    # create the image viewer
    viewer = ImageViewer(
        env.spec.id if env.spec is not None else env.__class__.__name__,
        env.observation_space.shape[0],  # height
        env.observation_space.shape[1],  # width
        monitor_keyboard=True,
        relevant_keys=set(sum(map(list, keys_to_action.keys()), []))
    )
    # create a done flag for the environment
    done = True
    # prepare frame rate limiting
    target_frame_duration = 1 / env.metadata['video.frames_per_second']
    last_frame_time = 0

    # start the main game loop
    try:
        i = 0
        while True:
            current_frame_time = time.time()
            # limit frame rate
            if last_frame_time + target_frame_duration > current_frame_time:
                continue
            # save frame beginning time for next refresh
            last_frame_time = current_frame_time
            # clock tick
            clock.tick()
            # reset if the environment is done
            if done:
                done = False
                state = env.reset()
                viewer.show(env.unwrapped.screen)
            # unwrap the action based on pressed relevant keys
            action = keys_to_action.get(viewer.pressed_keys, _NOP)
            next_state, reward, done, info = env.step(action)
            viewer.show(env.unwrapped.screen)
            # pass the observation data through the callback
            if record:
                buffer.append({
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "next_state": next_state,
                    "info": info
                })
            state = next_state

            # shutdown if the escape key is pressed
            if viewer.is_escape_pressed:
                if record:
                    # save it as a pickle file
                    save_play(buffer, level, agent="manual")
                break
            i += 1
    except KeyboardInterrupt as e:
        pass

    viewer.close()
    env.close()


if __name__ == '__main__':
    # <world> is a number in {1, 2, 3, 4, 5, 6, 7, 8} indicating the world
    world = 1
    # <stage> is a number in {1, 2, 3, 4} indicating the stage within a world
    stage = 1

    version = 0

    # this is the pixelated version!
    # version = 2

    # SuperMarioBros-<world>-<stage>-v<version>
    level = f"SuperMarioBros-{world}-{stage}-v{version}"

    # sample level can be changed
    env = gym_super_mario_bros.make(level)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    record_with_human_play(env, record=False, level=level)
