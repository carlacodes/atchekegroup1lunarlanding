import gym
from gym import spaces
import torch
import pip
import base64
# import stable_baselines3


import numpy as np
import matplotlib.pyplot as plt
import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.results_plotter import ts2xy, load_results
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env

import gym
from gym import spaces

#
# # @title Plotting/Video functions
# from IPython.display import HTML
# from pyvirtualdisplay import Display
# from IPython import display as ipythondisplay
import torch
cuda = torch.device('cuda')     # Default CUDA device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, arg1, arg2, ...):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

    def step(self, action):
        ...
        return observation, reward, done, info
    def reset(self):
        ...
        return observation  # reward, done, info can't be included
    def render(self, mode='human'):
        ...
    def close (self):
# Instantiate the env
env = CustomEnv(arg1, ...)
# Define and Train the agent
model =model_test = DQN("MlpPolicy", env,policy_kwargs = policy_kwargs,
                 learning_rate=6.3e-4,
                 batch_size=128,  #for simplicity, we are not doing batch update.
                 buffer_size=50000, #size of experience of replay buffer. Set to 1 as batch update is not done
                 learning_starts=0, #learning starts immediately!
                 gamma=0.99, #discount facto. range is between 0 and 1.
                 tau = 1,  #the soft update coefficient for updating the target network
                 target_update_interval=250, #update the target network immediately.
                 train_freq=(4,"step"), #train the network at every step.
                 #max_grad_norm = 10, #the maximum value for the gradient clipping
                 exploration_initial_eps = 0.9, #initial value of random action probability
                 exploration_fraction = 0.8, #fraction of entire training period over which the exploration rate is reduced
                 gradient_steps = -1, #number of gradient steps,
                 exploration_final_eps = 0.1,

                 # exploration_initial_eps = 1  # initial value of random action probability. Range is between 0 and 1.
                 # exploration_fraction = 0.5  # fraction of entire training period over which the exploration rate is reduced. Range is between 0 and 1.
                 # exploration_final_eps = 0.05  # (set by defualt) final value of random action probability. Range is between 0 and 1.
                 seed = 1, #seed for the pseudo random generators
                 device="cuda",
                 verbose=1)
model.learn(total_timesteps=1000)