# NUS-CS5446-Its_ah_me_Mario-

This repository contains project code/report for the CS5446 AI Planning course @ NUS Sem 1 24'.

## About

We plan to implement some of the popular RL algorithms on the super mario bros game and evaluate the performances of each agent comprehensively.

Refer to "report/Project Proposal.pdf" for more information

## Setup

1. Create a conda environment / virtual env with the python version "3.8.18"
2. Install Gym 0.17.2 `pip install gym==0.17.2`. (This contains a compatible interface with the mario gym version)
3. Install nes-py by following the instructions from https://github.com/Kautenja/nes-py based on your OS.
4. Install super mario gym environment from https://github.com/Kautenja/gym-super-mario-bros. This repo also contains all the relevant information about how the gym environment is structured.
5. Try running the code under agents/random/agent.py to check if the environment is set up correctly. It should additionally save a play under the data folder.
6. Try running the code under agents/human/agent.py to play a custom game (refer to manual play controls below). It should additionally save a play under the data folder. 

## Structure

1. agents/* : Contains the model code for each agent
3. data/* : Saving the plays by agents
4. evaluation: Contains jupyter notebooks where we analyse performance of specific models / let them play
5. models: Contains our trained model files (or) links to them for loading into our agent
6. report: Contains our project report & proposal
7. train: Contains jupyter notebooks where we attempt to train our various RL agents
8. utils: Contains utility code which can be reused across the project

## Manual Play Contorls

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
