# NUS-CS5446-Its_ah_me_Mario-

This repository contains project code/report for the CS5446 AI Planning course @ NUS Sem 1 24'.

## About

We plan to implement some of the popular RL algorithms on the super mario bros game and evaluate the performances of each agent comprehensively.

Refer to "report/Project Proposal.pdf" for more information

## Setup

1. Create a conda environment / virtual env with the python version "3.10.12"
2. Run the following commands:

pip install setuptools==65.5.0 pip==21
pip install wheel==0.38.0
pip -qq install stable-baselines3==1.6.0
pip install -qq gym-super-mario-bros
pip install tensorflow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Alternatively, you can create the conda environment using 'environment.yml', and
`conda env create -f environment.yml`

3. Install nes-py by following the instructions from https://github.com/Kautenja/nes-py based on your OS.
4. Try running the code under agents/random/agent.py to check if the environment is set up correctly. It should additionally save a play under the data folder.
5. Try running the code under agents/human/agent.py to play a custom game (refer to manual play controls below). It should additionally save a play under the data folder.

## Structure

1. agents/* : Contains the ipynb notebooks for training the different agents
2. logs/* : Tensorboard logs for each model run
3. evaluation/* : Contains jupyter notebooks where we analyse performance of specific models / let them play
4. report/* : Contains our project report & proposal
5. utils/* : Contains utility code which can be reused across the project
6. results/*: Contains the tensorboard logs and reward_log.csv for each model

## Tensorboard logs

To view the tensorboard logs, you can follow the following steps:
1. In terminal, instantiate 'python'
2. In python, 'import tensorflow as tf'
3. Leave python by 'quit()'
4. In terminal, run 'tensorboard --logdir=.' in the main folder
5. The terminal output will show 'TensorBoard 2.10.1 at http://localhost:6007/ (Note! Port number may be different)(Press CTRL+C to quit)'
6. Open a browser, and paste http://localhost:6007/ (Remember to change the port number accordingly)
7. Select the relevant models under 'results/' folder.

## Manual Play Controls (for human agent)

| Keyboard Key | NES Joypad |
|:-------------|:-----------|
| W            | Up         |
| A            | Left       |
| S            | Down       |
| D            | Right      |
| O            | A          |
| P            | B          |
| Enter        | Start      |
| Space        | Select     |
| Esc          | Quit game  |

## Sample plays by our agent

Checkout this <a href="https://youtube.com/playlist?list=PL_MXUE32GyYG7GM9lsfA3mRVs4JFw7MLo&si=2xfG45l30NoF_Wzm">youtube playlist</a> for some plays by our agents
