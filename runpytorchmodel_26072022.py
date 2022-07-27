import gym
from gym import spaces

print('run pytorch model')
import gym
import torch as th
import torch.nn as nn
import numpy as np

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
from YoutubeCodeRepository.ReinforcementLearning.DeepQLearning import simple_dqn_torch_2020

print(f"Is CUDA supported by this system?{torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device:{torch.cuda.current_device()}")

print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")
import gym

cuda = torch.device('cuda')  # Default CUDA device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
from stable_baselines3 import PPO
from stable_baselines3 import DQN

# model_path = "".format('dqn_lunar')

model_path = "dqn_lunar.zip"
model_test = DQN.load(model_path)
print('loaded model')
# for key, value in model_test.get_parameters().items():
#     print(key, value.shape)

env = gym.make("LunarLander-v4").unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

paramshapes = model_test.get_parameters()


def copy_dqn_weights(baselines_model):
    torch_dqn = simple_dqn_torch_2020.DeepQNetwork(lr=0.001, n_actions=4, input_dims=[9], fc1_dims=256, fc2_dims=256)
    model_params = baselines_model.get_parameters()
    # Get only the policy parameters
    model_params = model_params['policy']
    policy_keys = [key for key in model_params.keys() if "pi" in key or "c" in key]
    policy_params = [model_params[key] for key in policy_keys]

    for (th_key, pytorch_param), key, policy_param in zip(torch_dqn.named_parameters(), policy_keys, policy_params):
        param = policy_param.copy()
        # Copy parameters from stable baselines model to pytorch model

        # Conv layer
        if len(param.shape) == 4:
            # https://gist.github.com/chirag1992m/4c1f2cb27d7c138a4dc76aeddfe940c2
            # Tensorflow 2D Convolutional layer: height * width * input channels * output channels
            # PyTorch 2D Convolutional layer: output channels * input channels * height * width
            param = np.transpose(param, (3, 2, 0, 1))

        # weight of fully connected layer
        if len(param.shape) == 2:
            param = param.T

        # bias
        if 'b' in key:
            param = param.squeeze()

        param = torch.from_numpy(param)
        pytorch_param.data.copy_(param.data.clone())

    return torch_dqn


dqn_torch_v = copy_dqn_weights(model_test)
ct = 0

for child in dqn_torch_v.children():
    ct += 1
    if ct < 2:
        for param in child.parameters():
            print(param)
            print(ct)
            param.requires_grad = False

import gym
from YoutubeCodeRepository.ReinforcementLearning.DeepQLearning.utils import plotLearning
import numpy as np


def obs_to_torch(obs):
    # TF: NHWC
    # PyTorch: NCHW
    # https://discuss.pytorch.org/t/dimensions-of-an-input-image/19439
    # obs = np.transpose(obs, (0, 3, 1, 2))
    # # Normalize
    # obs = obs / 255.0
    obs = th.tensor(obs).float()
    obs = obs.to(device)
    return obs


env = gym.make('LunarLander-v4')

episode_reward = 0
done = False
obs = env.reset()
print(next(dqn_torch_v.parameters()).device)
while not done:
    action = th.argmax(dqn_torch_v(obs_to_torch(obs))).item()
    # action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    episode_reward += reward

print(episode_reward)
