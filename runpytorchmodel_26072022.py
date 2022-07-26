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

from YoutubeCodeRepository.ReinforcementLearning.DeepQLearning import simple_dqn_torch_2020

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


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.05, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = simple_dqn_torch_2020.DeepQNetwork(lr, n_actions=n_actions,
                                                         input_dims=input_dims,
                                                         fc1_dims=256, fc2_dims=256)
        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        # print('calling machine learning phil''s function')
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(
            self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(
            self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(
            self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min


agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
              input_dims=[9], lr=0.001)
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
