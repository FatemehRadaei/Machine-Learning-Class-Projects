from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
USE_CUDA = torch.cuda.is_available()


class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n
        self.vis_features = []

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        old_x = self.fc(x)
        #self.vis_features = old_x[0]
        x = self.fc2(old_x)
        return x

    def getFeatureGlobalMatrix(self,state):
        state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
        x = state
        x = self.features(x)
        x = x.view(x.size(0), -1)
        old_x = self.fc(x)
        return old_x[0]

    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            ######## YOUR CODE HERE! ########
            # TODO: Given state, you should write code to get the Q value and chosen action
            # Complete the R.H.S. of the following 2 lines and uncomment them
            # q_value =
            # action =
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
            #import IPython; IPython.embed()
            ######## YOUR CODE HERE! ########

        else:
            action = random.randrange(self.env.action_space.n)

            #q_value = random.randrange()
        return action#, q_value

    def save_model(self,filename,model):
    	print('[deepRL]  saving model checkpoint to ' + filename)
#        checkpoint = {
#            'dqn': model.state_dict(),
#        }
#        torch.save(checkpoint, filename)
    	torch.save(model.state_dict(), filename)



def compute_td_loss(model, batch_size, gamma, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), requires_grad=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    ######## YOUR CODE HERE! ########
    # TODO: Implement the Temporal Difference Loss
    # loss =
    q_values = model(state)
    next_q_values = model(next_state)
    next_q_state_values = model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    #optimizer.zero_grad()
    #loss.backward()
    #optimizer.step()
    ######## YOUR CODE HERE! ########
    return loss


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        ######## YOUR CODE HERE! ########
        # TODO: Randomly sampling data with specific batch size from the buffer
        # Hint: you may use the python library "random".
        # If you are not familiar with the "deque" python library, please google it.
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
#        return np.concatenate(state), action, reward, np.concatenate(next_state), done

        ######## YOUR CODE HERE! ########
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)
