import io
import os
import glob
import torch
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

import gym
nn_layers = [64,64] #This is the configuration of your neural network. Currently, we have two layers, each consisting of 64 neurons.
                    #If you want three layers with 64 neurons each, set the value to [64,64,64] and so on.

learning_rate = 0.001 #This is the step-size with which the gradient descent is carried out.
                      #Tip: Use smaller step-sizes for larger networks.
env = gym.make('LunarLander-v2')
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# Create environment

#You can also load other environments like cartpole, MountainCar, Acrobot. Refer to https://gym.openai.com/docs/ for descriptions.
#For example, if you would like to load Cartpole, just replace the above statement with "env = gym.make('CartPole-v1')".

env = stable_baselines3.common.monitor.Monitor(env, log_dir )

callback = EvalCallback(env,log_path = log_dir, deterministic=True) #For evaluating the performance of the agent periodically and logging the results.
policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=nn_layers)
model_old = DQN("MlpPolicy", env,policy_kwargs = policy_kwargs,
            learning_rate=learning_rate,
            batch_size=1,  #for simplicity, we are not doing batch update.
            buffer_size=1, #size of experience of replay buffer. Set to 1 as batch update is not done
            learning_starts=1, #learning starts immediately!
            gamma=0.99, #discount facto. range is between 0 and 1.
            tau = 1,  #the soft update coefficient for updating the target network
            target_update_interval=1, #update the target network immediately.
            train_freq=(1,"step"), #train the network at every step.
            max_grad_norm = 10, #the maximum value for the gradient clipping
            exploration_initial_eps = 0.9, #initial value of random action probability
            exploration_fraction = 0.8, #fraction of entire training period over which the exploration rate is reduced
            gradient_steps = 1, #number of gradient steps,
             exploration_final_eps = 0.05,

            # exploration_initial_eps = 1  # initial value of random action probability. Range is between 0 and 1.
            # exploration_fraction = 0.5  # fraction of entire training period over which the exploration rate is reduced. Range is between 0 and 1.
            # exploration_final_eps = 0.05  # (set by defualt) final value of random action probability. Range is between 0 and 1.
            seed = 1, #seed for the pseudo random generators
            verbose=0) #Set verbose to 1 to observe training logs. We encourage you to set the verbose to 1.
model_test = DQN("MlpPolicy", env,policy_kwargs =policy_kwargs,
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
            verbose=1) #Set verbose to 1 to observe training logs. We encourage you to set the verbose to 1.
# You can also experiment with other RL algorithms like A2C, PPO, DDPG etc. Refer to  https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
#for documentation. For example, if you would like to run DDPG, just replace "DQN" above with "DDPG".
#
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample())
#
# env.close()

"""
Utility functions to enable video recording of gym environment
and displaying it.
To enable video, just do "env = wrap_env(env)""
"""
#

test_env = (gym.make("LunarLander-v2"))
observation = test_env.reset()
total_reward = 0
while True:
  test_env.render()
  # for _ in range(1000):
  #     env.render()
  test_env.step(env.action_space.sample())
  #
  # env.close()
  action, states = model.predict(observation, deterministic=True)
  observation, reward, done, info = test_env.step(action)
  total_reward += reward
  if done:
    break

# print(total_reward)
test_env.close()
