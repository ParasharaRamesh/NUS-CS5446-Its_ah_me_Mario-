IGNORE EVERYTHING IN README!!

<u> New approach for setup </u>

New Reference: https://www.kaggle.com/code/deeplyai/super-mario-bros-with-stable-baseline3-ppo

1.use python 3.10.12 - create a conda environment or something
But set pip and wheel version using:
pip install setuptools==65.5.0 pip==21  # gym 0.21 installation is broken with more recent versions
pip install wheel==0.38.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip -qq install stable-baselines3==1.6.0
pip install stable-baselines3
pip install -qq gym-super-mario-bros
pip install tensorflow (this is primarly for tensorboard)


Use this for video reply if training in colab:
https://colab.research.google.com/drive/12osEZByXOlGy8J-MSpkl3faObhzPGIrB#scrollTo=RcRLC8Wldwwf






<u> What levels are we interested in ? </u>

* world 1, stage 1
* lets record some plays

<u> References:</u>

1. DQN/DDQN
*  https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
*  https://github.com/Kautenja/playing-mario-with-deep-reinforcement-learning/tree/master

2. PPO
* https://github.com/jcwleo/mario_rl/tree/master
* https://www.youtube.com/watch?v=2eeYqJ0uBKE

3. Actor Critic
* https://github.com/uvipen/Super-mario-bros-A3C-pytorch/tree/master

4. MuZero
* https://github.com/Nebraskinator/SuperMarioBrosAI/tree/master
* https://sreeharshau.github.io/Evaluating_MuZero_Super_Mario_Bros.pdf

5. Survey papers with metrics
* https://dspace.cvut.cz/bitstream/handle/10467/101068/F8-DP-2022-Schejbal-Ondrej-thesis.pdf?sequence=-1&isAllowed=y

<u> Splits: </u>

Xian He:
- DDQN pytorch tutorial stuff

Parash & Sriram:
- Actor critic

Grace:
- PPO

