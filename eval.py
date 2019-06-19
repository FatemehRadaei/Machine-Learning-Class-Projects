from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import csv
from dqn import QLearner, compute_td_loss, ReplayBuffer


"""setting up the environment"""
env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)
num_frames = 100000
#num_frames = 20000
batch_size = 32
gamma = 0.99
replay_initial = 10000
replay_buffer = ReplayBuffer(100000)
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


"""loading the saved model"""
device = torch.device("cuda")
model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
filename = 'newdqnModel.pt'
model.load_state_dict(torch.load(filename))
model.to(device)
model.eval()

"""choosing 1000 frames randomly!"""
frame_range = 50000
frame_list = set(random.sample(range(1, frame_range), 1000))
vis_feature_matrix = []
vis_rewards = []
vis_actions = []

state = env.reset()
indx = 0
episode_reward = 0
for frame_idx in range(0,frame_range):
    #print(frame_idx)
    action= model.act(state, 0)
    features = model.getFeatureGlobalMatrix(state)
    next_state, reward, done, _ = env.step(action)
    #print(reward)
    if frame_idx in frame_list:
        vis_actions.append([float(action)])
        vis_feature_matrix.append(features)
        vis_rewards.append([float(episode_reward)])

        filter_matrix = []
        for i in range(84):
           filter_matrix.append(state[0][i])
        plt.matshow(filter_matrix)
        plt.colorbar()
        plt.savefig('filter_{}.png'.format(str(indx)))
        indx += 1

    if done:
        state = env.reset()
        episode_reward = 0
    else:
        state = next_state
        episode_reward += reward


#### saving data for visualization
#import IPython; IPython.embed()

with open('features.csv', 'w') as f:
   writer = csv.writer(f)
   writer.writerows(vis_feature_matrix)

with open('actions.csv', 'w') as f:
   writer = csv.writer(f)
   writer.writerows(vis_actions)

with open('rewards.csv', 'w') as f:
   writer = csv.writer(f)
   writer.writerows(vis_rewards)
