import gym
from gym.spaces import Box, Discrete
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from utils.dqn import extract_tensors, QValues, plot, Experience
from utils.common import mlp
from utils.env_managers.cartpole import CartPoleEnvManager

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import argparse


parser = argparse.ArgumentParser(description = "DQN algorithm usage")

parser.add_argument(
    "--episodes",
    "-e",
    type=int,
    help="Number of episodes to train",
    default=1200
)

parser.add_argument(
    "--batch_size",
    "-bs",
    type=int,
    help="Batch size",
    default=512
)

parser.add_argument(
    "--memory_size",
    "-m",
    type=int,
    help="Capacity of the memory buffer",
    default=100000
)

parser.add_argument(
    "--target_update",
    "-tu",
    type=int,
    help="Episodes step for the synchronization",
    default=10
)

parser.add_argument(
    "--gamma",
    "-gam",
    type=float,
    help="Discount factor",
    default=0.99
)

parser.add_argument(
    "--lr",
    type=float,
    help="Learning rate",
    default=1e-3,
)

parser.add_argument(
    "--eps_vals",
    "-ev",
    type=tuple,
    help="Exploration hyperparameters (start, end, decay)",
    default=(1,0.01, 0.001)
)

parser.add_argument(
    "--hidden_sizes",
    "-hs",
    type=tuple,
    help="Shape of each FC hidden layer",
    default=(32,32)
)

parser.add_argument(
    "--observation_type",
    "-obs_type",
    type=str,
    help="Either image or the default provided by the OpenAI Gym API",
    default="img",
    choices=["img", "default"]
)

parser.add_argument(
    "--env",
    type=str,
    help="OpenAI Gym environment",
    default="CartPole-v0"
)


parser.add_argument(
    "--device",
    type=str,
    help="cuda or cpu",
    default="cpu",
    choices=["cuda", "cpu"]
)

args = parser.parse_args()




def DQN(obs_type, observation_space, action_space, hidden_sizes, activation = nn.ReLU, last_activation = nn.Identity, device = torch.device("cpu")):
    if obs_type == "img": # image
        dims = [np.prod(observation_space.shape)] + list(hidden_sizes)
    elif obs_type=="default":
        dims = [observation_space.shape[0]] + list(hidden_sizes)
        
    if isinstance(action_space, Box):
        act_dim = action_space.shape[0]
    else:
        act_dim = action_space.n
    
    return mlp(obs_type,
               dims + [act_dim], 
               activation, 
               last_activation).to(device)
    

# Replay buffer
class ReplayMemory():
    """Replay buffer object
    
    Attributes:
        capacity (int): max number of experiences that the buffer can store
        memory (list): list of experiences
        push_count (int): current index in the buffer
    Methods:
        push(experience): append experiences to the buffer according to the current index
        sample(batch_size): returns a random sample from the buffer
        can_provide_sample(batch_size): returns True or False according to the storage availability 
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else: # replace from the beginning of the buffer
            self.memory[self.push_count%self.capacity] = experience
        self.push_count += 1
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size
    
# Epsilon greedy strategy
class EpsilonGreedyStrategy():
    """Epsilon-greedy handler
    
    Attributes:
        start (float): starting point for the exploration
        end (float): ending point
        decay (float): decaying rate of the exploration
    """
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)
    

# RL Agent
class Agent():
    """ Highlevel Agent handler selection actions function
    """
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device
        
    def select_action(self, state, policy_net):
        rate = strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        
        if rate > random.random():
            action = random.randrange(self.num_actions)
            return  torch.tensor([action]).to(self.device) # exploration
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim = 1).to(self.device) # exploitation
            


if args.device == "cuda" and not torch.cuda.is_available():
    raise ("raise GPU error, please selec --device 'cpu'")

device = torch.device(args.device)
env = gym.make(args.env)


if args.observation_type == "img":
    em = CartPoleEnvManager(env, device)
    obs_sp = Box(low=0, 
                 high=255, 
                 shape=(3, em.get_screen_height(), em.get_screen_width()), 
                 dtype=np.uint8)
    obs = em.get_state()
    n_actions = em.num_actions_available()
else:
    obs_sp = env.observation_space
    n_actions = env.action_space.n

strategy = EpsilonGreedyStrategy(args.eps_vals[0], args.eps_vals[1], args.eps_vals[2])
agent = Agent(strategy, n_actions, args.device)
memory = ReplayMemory(args.memory_size)

action_sp = env.action_space
policy_net = DQN(obs_type = args.observation_type, 
                 observation_space = obs_sp, 
                 action_space = action_sp, 
                 hidden_sizes = args.hidden_sizes, 
                 device = args.device)
target_net = DQN(obs_type = args.observation_type, 
                 observation_space = obs_sp, 
                 action_space = action_sp, 
                 hidden_sizes = args.hidden_sizes, 
                 device = args.device)

target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(params = policy_net.parameters(), lr = args.lr)

ep_ret, ep_len = 0, 0
all_durations = []
all_returns = []
for episode in range(args.episodes):
    if args.observation_type == "img":
        em.reset()
        state = em.get_state()
    else:
        state = torch.from_numpy(env.reset()).float().unsqueeze(dim=0)
    for timestep in count():
        action = agent.select_action(state, policy_net)
        if args.observation_type == "img":
            reward = em.take_action(action)
            next_state = em.get_state() 
            done = em.done
        else:
            next_state, reward, done , _ = env.step(action.item())
            next_state = torch.tensor([next_state], device = args.device).float()
            reward = torch.tensor([reward], device = args.device)
        
        memory.push(Experience(state, action, next_state, reward, torch.tensor([done])))
        state = next_state 
        ep_ret += reward.item() 
        
        if memory.can_provide_sample(args.batch_size):
            experiences = memory.sample(args.batch_size)
            states, actions, rewards, next_states, dones = extract_tensors(experiences)
            
            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states, dones, device)
            target_q_values = (next_q_values * args.gamma) + rewards
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if done:
            all_returns.append(ep_ret)
            all_durations.append(timestep)
            plot(all_returns, 100)
            ep_ret = 0
            break
            
    if episode % args.target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
if args.observation_type == "img":
    em.close() 
