from tqdm import tqdm

from game.environment import create_gym_env_from_level
from utils.record import save_play


def random_agent(world, stage, version, use_coin_collector_env=False, steps=500, record=False, buffer=[]):
    print("Starting random agent..")
    # create gym environment
    env = create_gym_env_from_level(world, stage, version, use_coin_collector_env)

    # play it
    done = True
    for _ in tqdm(range(steps)):
        if done:
            state = env.reset()
        action = env.action_space.sample()
        env_step = env.step(action)
        next_state, reward, done, info = env_step

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
        env.render()

    if record:
        level_suffix = f"{world}-{stage}-v{version}"
        level = f"SuperMarioBros-{level_suffix}" if not use_coin_collector_env else f"CoinCollectorSuperMarioBrosEnv-{level_suffix}"
        # save it as a pickle file
        save_play(buffer, level, agent="random")

    env.close()
    print(f"Finished random agent..")


if __name__ == '__main__':
    # <world> is a number in {1, 2, 3, 4, 5, 6, 7, 8} indicating the world
    world = 1
    # <stage> is a number in {1, 2, 3, 4} indicating the stage within a world
    stage = 2
    version = 2

    # one attempt
    random_agent(world, stage, version, steps=10000, record=False)
    #
    # # another attempt
    # random_agent(level, record=True)
