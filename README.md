# NUS-CS5446-Its_ah_me_Mario-

This repository contains project code/report for the CS5446 AI Planning course @ NUS Sem 1 24'.

## About

We plan to implement some of the popular RL algorithms on the super mario bros game and evaluate the performances of each agent comprehensively.

Refer to "report/Project Proposal.pdf" for more information

## Setup

1. Create a conda environment / virtual env with the python version "3.10.12"
2. Run the following commands
`
pip install setuptools==65.5.0 pip==21  
pip install wheel==0.38.0
pip -qq install stable-baselines3==1.6.0
pip install -qq gym-super-mario-bros
pip install tensorflow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
`
3. Install nes-py by following the instructions from https://github.com/Kautenja/nes-py based on your OS.
4. Try running the code under agents/random/agent.py to check if the environment is set up correctly. It should additionally save a play under the data folder.
5. Try running the code under agents/human/agent.py to play a custom game (refer to manual play controls below). It should additionally save a play under the data folder. 

## Structure

1. agents/* : Contains the ipynb notebooks for training the different agents
2. logs/* : Tensorboard logs for each model run
3. evaluation/* : Contains jupyter notebooks where we analyse performance of specific models / let them play
4. models/* : Contains our trained model files (or) links to them for loading into our agent
5. report/* : Contains our project report & proposal
6. utils/* : Contains utility code which can be reused across the project

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
