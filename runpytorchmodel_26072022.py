import gym
from gym import spaces

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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from stable_baselines3 import PPO

model_path = "{}.pkl".format('LunarLander-v4')

model_test = PPO.load(model_path, verbose=2)

for key, value in model_test.get_parameters().items():
    print(key, value.shape)

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
from YoutubeCodeRepository.ReinforcementLearning.DeepQLearning.simple_dqn_torch_2020_2 import Agent
from YoutubeCodeRepository.ReinforcementLearning.DeepQLearning.utils import plotLearning
import numpy as np

env = gym.make('LunarLander-v4')
agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
              input_dims=[8], lr=0.001)
scores, eps_history = [], []
n_games = 500

for i in range(n_games):
    score = 0
    done = False
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward,
                               observation_, done)
        agent.learn()
        observation = observation_
    scores.append(score)
    eps_history.append(agent.epsilon)

    avg_score = np.mean(scores[-100:])

    print('episode ', i, 'score %.2f' % score,
          'average score %.2f' % avg_score,
          'epsilon %.2f' % agent.epsilon)
x = [i + 1 for i in range(n_games)]
filename = 'lunar_lander.png'
plotLearning(x, scores, eps_history, filename)
