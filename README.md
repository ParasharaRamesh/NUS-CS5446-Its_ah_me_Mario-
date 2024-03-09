# NUS-CS5446-Its_ah_me_Mario-

This repository contains project code/report for the CS5446 AI Planning course @ NUS Sem 1 24'.

## About

We plan to implement some of the popular RL algorithms on the super mario bros game and evaluate the performances of each agent comprehensively.

Refer to "report/Project Proposal.pdf" for more information

## Setup

1. Create a conda environment / virtual env with the python version "3.8.18"
2. Install Gym 0.10.9 `pip install gym==0.10.9`. (This contains a compatible interface with the mario gym version)
3. Install nes-py by following the instructions from https://github.com/Kautenja/nes-py based on your OS.
3. Install super mario gym environment from https://github.com/Kautenja/gym-super-mario-bros. This repo also contains all the relevant information about how the gym environment is structured.
4. Try running the code under agents/random/agent.py to check if the environment is setup correctly.

## Structure

1. agents/* : Contains the model code for each agent
2. data/plays/* : Examples of interesting runs by our agents
3. data/manual/* : Examples of manual playing
4. evaluation: Contains jupyter notebooks where we analyse performance of specific models / let them play
5. models: Contains our trained model files (or) links to them for loading into our agent
6. report: Contains our project report & proposal
7. train: Contains jupyter notebooks where we attempt to train our various RL agents
8. utils: Contains utility code which can be reused across the project


