from torch import nn
import torch
import gymnasium as gym
from collections import deque
import itertools
import numpy as np
import random

BATCH_SIZE = 32
GAMMA = 0.99
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END=0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ=1000

class Network(nn.Module):
    def __init__ (self, env):
        super().__init__()

        # computing number of inputs to the network by taking product of observation space shape
        # cartpole is 1D, but traci will be 2d
        # essentially how many neurons are in input layer of nn
        in_features = int(np.prod(env.observation_space.shape))

        # rest of network is standard (64 hidden units)
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            # determines number of ouput layers
            # q learning CAN ONLY be used with action spaces that have FINITE actions (discrete action space)
            nn.Linear(64, env.action_space.n)
        )
    def forward(self, x):
        return self.net(x)
    
    # intelligent action
    def act(self, obs):
        # turn obs to torch tensor
        obs_t = torch.as_tensor(obs, dtype = torch.float32)
        # compute q values for obs (q values with every possible action)
        q_values = self(obs_t.unsqueeze(0))

        # get action with highest q value
        max_q_index = torch.argmax(q_values, dim = 1)[0]
        # turn pytorch tensor into integer
        action = max_q_index.detach().item()

        # return action index (from 0 to actions - 1)
        return action 


# create env
env = gym.make('CartPole-v1')

# standard deque for python
replay_buffer = deque(maxlen=BUFFER_SIZE)
# store rewards earned by agent in single episode - tracks improvement of agent as it trains
rew_buffer = deque([0,0], maxlen=100)

episode_reward = 0.0

# creating networks
online_net = Network(env)
target_net = Network(env)

#set target network params equal to online
target_net.load_state_dict(online_net.state_dict())

# create optimizer
optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

# INITIALIZE REPLAY BUFFER
obs = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    # new_obs is state of env, rew = reward returned by env by the action, done = whether env info is done
    new_obs, rew, done, _, _ = env.step(action)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    # if env needs to be reset
    if done:
        obs = env.reset()



# MAIN TRAINING LOOP
obs = env.reset()

# similar to while true
for step in itertools.count():
    # select an action to take in env, but we are using epsilon greedy policy
    # .interp() --> going from epsilon_start (100% random actions) to epsilon_end (2% random actions)
    # this is done to facilitate exploration of the agent in the env
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    rnd_sample = random.random()

    if rnd_sample <= epsilon:
        # take random action
        action = env.action_space.sample()
    else:
        # intelligently select action using nn
        action = online_net.act(obs)
    
    new_obs, rew, done, _, _ = env.step(action)
    # transition tuple
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    # update reward for this episode
    episode_reward += rew

    if done:
        obs = env.reset()

        # add to reward buffer and reset the current episode's reward
        rew_buffer.append(episode_reward)
        episode_reward = 0.0

    # START GRADIENT STEP
    # sample transitions from replay buffer
    transitions = random.sample(replay_buffer, BATCH_SIZE)

    #extract each element from transition buffer as it's own array
    # using np.asarray because pytorch is FASTER in making tensor from numpy array than python array
    obses = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rews = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions])

    # convert each array into pytorch tensor
    obses_t = torch.as_tensor(obses, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

    # COMPUTER TRAGETS FOR LOSS FUNCTION

    #set of q values FOR EACH observation
    target_q_values = target_net(new_obses_t)

    # need to collapse to one highest q val per observation
    # each observation is batch dimention and q values are dimension 1
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

    # compute targets
    targets = rews_t + GAMMA  * (1 - dones_t) * max_target_q_values

    # comptue loss
    q_values = online_net(obses_t)

    action_q_values = torch.gather(input = q_values, dim = 1, index = actions_t)

    # hubert loss function
    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    # After solved, watch it play
    if len(rew_buffer) >= 100:
        if np.mean(rew_buffer) >= 195:
            while True:
                action = online_net.act(obs)

                obs, _, done, _ = env.step(action)
                env.render()
                if done:
                    env.reset()

    # GRADIENT DESCENT
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update target network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Logging
    if step % 1000 == 0:
        print()
        print('Step', step)
        print('Avg Rew', np.mean(rew_buffer))