import io
import os
import glob
import torch
import pip
import base64
import math
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


FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400
##define constant parameters

print(f"Is CUDA supported by this system?{torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device:{torch.cuda.current_device()}")

print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")
import gym
cuda = torch.device('cuda')     # Default CUDA device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# with torch.cuda.device(0):
nn_layers = [64,64] #This is the configuration of your neural network. Currently, we have two layers, each consisting of 64 neurons.
#If you want three layers with 64 neurons each, set the value to [64,64,64] and so on.

learning_rate = 0.001 #This is the step-size with which the gradient descent is carried out.
#Tip: Use smaller step-sizes for larger networks.
env = gym.make('LunarLander-v2')
log_dir = "/tmp/gym/"
log_dir='C:/Users/carla/PycharmProjects/atchekegroup1lunarlanding/gym2007/'
os.makedirs(log_dir, exist_ok=True)
from gym.envs.box2d import LunarLander
def step(self, actions):
  assert self.lander is not None

  # Update wind
  assert self.lander is not None, "You forgot to call reset()"
  if self.enable_wind and not (
      self.legs[0].ground_contact or self.legs[1].ground_contact
  ):
      # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
      # which is proven to never be periodic, k = 0.01
      wind_mag = (
          math.tanh(
              math.sin(0.02 * self.wind_idx)
              + (math.sin(math.pi * 0.01 * self.wind_idx))
          )
          * self.wind_power
      )
      self.wind_idx += 1
      self.lander.ApplyForceToCenter(
          (wind_mag, 0.0),
          True,
      )

      # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
      # which is proven to never be periodic, k = 0.01
      torque_mag = math.tanh(
          math.sin(0.02 * self.torque_idx)
          + (math.sin(math.pi * 0.01 * self.torque_idx))
      ) * (self.turbulence_power)
      self.torque_idx += 1
      self.lander.ApplyTorque(
          (torque_mag),
          True,
      )

  if self.continuous:
      action = np.clip(action, -1, +1).astype(np.float32)
  else:
      assert self.action_space.contains(
          action
      ), f"{action!r} ({type(action)}) invalid "

  # Engines
  tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
  side = (-tip[1], tip[0])
  dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

  m_power = 0.0
  if (self.continuous and action[0] > 0.0) or (
      not self.continuous and action == 2
  ):
      # Main engine
      if self.continuous:
          m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
          assert m_power >= 0.5 and m_power <= 1.0
      else:
          m_power = 1.0
      # 4 is move a bit downwards, +-2 for randomness
      ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
      oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
      impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
      p = self._create_particle(
          3.5,  # 3.5 is here to make particle speed adequate
          impulse_pos[0],
          impulse_pos[1],
          m_power,
      )  # particles are just a decoration
      p.ApplyLinearImpulse(
          (ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
          impulse_pos,
          True,
      )
      self.lander.ApplyLinearImpulse(
          (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
          impulse_pos,
          True,
      )

  s_power = 0.0
  if (self.continuous and np.abs(action[1]) > 0.5) or (
      not self.continuous and action in [1, 3]
  ):
      # Orientation engines
      if self.continuous:
          direction = np.sign(action[1])
          s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
          assert s_power >= 0.5 and s_power <= 1.0
      else:
          direction = action - 2
          s_power = 1.0
      ox = tip[0] * dispersion[0] + side[0] * (
          3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
      )
      oy = -tip[1] * dispersion[0] - side[1] * (
          3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
      )
      impulse_pos = (
          self.lander.position[0] + ox - tip[0] * 17 / SCALE,
          self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
      )
      p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
      p.ApplyLinearImpulse(
          (ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
          impulse_pos,
          True,
      )
      self.lander.ApplyLinearImpulse(
          (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
          impulse_pos,
          True,
      )

  self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

  pos = self.lander.position
  vel = self.lander.linearVelocity
  state = [
      (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
      (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
      vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
      vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
      self.lander.angle,
      20.0 * self.lander.angularVelocity / FPS,
      1.0 if self.legs[0].ground_contact else 0.0,
      1.0 if self.legs[1].ground_contact else 0.0,
  ]
  assert len(state) == 8

  reward = 0
  shaping = (
      -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
      - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
      - 100 * abs(state[4])
      + 10 * state[6]
      + 10 * state[7]
  )  # And ten points for legs contact, the idea is if you
  # lose contact again after landing, you get negative reward
  if self.prev_shaping is not None:
      reward = shaping - self.prev_shaping
  self.prev_shaping = shaping

  reward -= (
      m_power * 0.30
  )  # less fuel spent is better, about -30 for heuristic landing. You should modify these values.
  reward -= s_power * 0.03

  done = False
  if self.game_over or abs(state[0]) >= 1.0:
      done = True
      reward = -100
  if not self.lander.awake:
      done = True
      reward = +100
  return np.array(state, dtype=np.float32), reward, done, {}


class Custom_LunarLander_wind(LunarLander):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        LunarLander.__init__(self);

    def step(self, actions):
        #assert self.enable_wind

        # Update wind
        self.reset()
        assert self.lander is not None, "You forgot to call reset()"
        if self.enable_wind and not (
                self.legs[0].ground_contact or self.legs[1].ground_contact
        ):
            # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            wind_mag = (
                    math.tanh(
                        math.sin(0.02 * self.wind_idx)
                        + (math.sin(math.pi * 0.01 * self.wind_idx))
                    )
                    * self.wind_power
            )
            self.wind_idx += 1
            self.lander.ApplyForceToCenter(
                (wind_mag, 0.0),
                True,
            )

            # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            torque_mag = math.tanh(
                math.sin(0.02 * self.torque_idx)
                + (math.sin(math.pi * 0.01 * self.torque_idx))
            ) * (self.turbulence_power)
            self.torque_idx += 1
            self.lander.ApplyTorque(
                (torque_mag),
                True,
            )

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid "

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (
                not self.continuous and action == 2
        ):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            # 4 is move a bit downwards, +-2 for randomness
            ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(
                3.5,  # 3.5 is here to make particle speed adequate
                impulse_pos[0],
                impulse_pos[1],
                m_power,
            )  # particles are just a decoration
            p.ApplyLinearImpulse(
                (ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (
                not self.continuous and action in [1, 3]
        ):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                direction = action - 2
                s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * (
                    3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                    3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse(
                (ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8

        reward = 0
        shaping = (
                -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
                - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
                - 100 * abs(state[4])
                + 10 * state[6]
                + 10 * state[7]
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= (
                m_power * 0.30
        )  # less fuel spent is better, about -30 for heuristic landing. You should modify these values.
        reward -= s_power * 0.03

        done = False
        if self.game_over or abs(state[0]) >= 1.0:
            done = True
            reward = -100
        if not self.lander.awake:
            done = True
            reward = +100
        return np.array(state, dtype=np.float32), reward, done, {}

    def reset(self):
        pass  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close(self):
        pass


env=Custom_LunarLander_wind()


callback = EvalCallback(env,log_path = log_dir, deterministic=True) #For evaluating the performance of the agent periodically and logging the results.
policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=nn_layers)

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
                 device="cuda",
                 verbose=1)

model_test.learn(total_timesteps=100000, log_interval=10, callback=callback)

x, y = ts2xy(load_results(log_dir), 'timesteps')  # Organising the logged results in to a clean format for plotting.
plt.plot(x, y)
plt.ylim([-300, 300])
plt.xlabel('Timesteps')
plt.ylabel('Episode Rewards')
plt.title('Carl parameters model')
plt.show()