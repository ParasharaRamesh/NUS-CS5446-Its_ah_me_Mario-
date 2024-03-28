
'''
The code below registers this new environment in gym for us to reference later. Code borrowed from _registration.py of gym_super_mario_bros
'''
import gym


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
        entry_point='NUS-CS5446-Its_ah_me_Mario.game.coincollector:CoinCollectorSuperMarioBrosEnv',
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
