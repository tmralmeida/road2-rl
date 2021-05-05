import gym
from gym.spaces import Box, Discrete
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from utils.vpg import combined_shape, discount_cumsum, plot, stats
from utils.common import mlp
from utils.env_managers.cartpole import CartPoleEnvManager

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical, Normal
import argparse


parser = argparse.ArgumentParser(description = "VPG algorithm usage")

parser.add_argument(
    "--epochs",
    "-e",
    type=int,
    help="Number of epochs to train",
    default=1200
)

parser.add_argument(
    "--steps_per_epoch",
    "-spe",
    type=int,
    help="Max number of (s,a) per epoch",
    default=4000
)

parser.add_argument(
    "--gamma",
    "-gam",
    type=float,
    help="Discount factor",
    default=0.99
)


parser.add_argument(
    "--lam",
    type=float,
    help="Hyperparameter for the baseline. See baselines for more info",
    default=0.97
)


parser.add_argument(
    "--pi_lr",
    type=float,
    help="Learning rate for the training of the policy net",
    default=3e-4,
)


parser.add_argument(
    "--v_lr",
    type=float,
    help="Learning rate for the training of the value function net",
    default=1e-3,
)


parser.add_argument(
    "--train_v_iters",
    "-tvi",
    type=int,
    help="Number of iterations to run the value function optimizer in each epoch",
    default=80,
)

parser.add_argument(
    "--max_ep_len",
    type=int,
    help="Max len of a trajectory / episode / rollout",
    default=1000
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
    default="default",
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

    
# Experience object
Experience = namedtuple("Experience", (
    "state, action, reward"
))


# Replay buffer
class ReplayMemory():
    """Replay buffer object
    
    Attributes:
        capacity (int): max number of experiences that the buffer can store
        gamma (float): discount factor
        lam (float): baseline parameter
        push_count (int): current index in the buffer
        obs_buf, act_buf, rew_buf, val_buf, logp_buf, adv_buf (np.array): buffers
        
    Methods:
        reset_count(): reset the counter
        fill_buffer(obs,act,rew,v,logp): store experiences in the buffer
        push(experience, v): appends an experience to the buffer and the respective value and logp 
        get(): returns the full buffer
        
    """
    def __init__(self, capacity, obs_dim, act_dim, gamma=0.99, lam=0.95):
        self.capacity = capacity
        self.gamma = gamma
        self.lam = lam
        self.obs_buf = np.zeros(combined_shape(capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(capacity, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(capacity, dtype=np.float32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.ret_buf = np.zeros(capacity, dtype=np.float32)
        self.val_buf = np.zeros(capacity, dtype=np.float32)
        self.push_count, self.path_start_idx = 0, 0
        
    def reset_count(self):
        self.push_count = 0
        
    def fill_buffer(self, obs, act, rew, v):
        self.obs_buf[self.push_count] = obs.cpu() if isinstance(obs, torch.Tensor) else obs
        self.act_buf[self.push_count] = act
        self.rew_buf[self.push_count] = rew
        self.val_buf[self.push_count] = v
        
        
    def push(self, experience, v):
        obs, act, rew = experience.state, experience.action, experience.reward
        if self.push_count >= self.capacity: self.reset_count()
        self.fill_buffer(obs, act, rew, v)    
        self.push_count += 1   


    def finish_path(self, last_val =0):
        """Based on: https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/vpg/core.py#L29
        """
        path_slice = slice(self.path_start_idx, self.push_count)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.push_count
        
    def get(self):
        #adv_mean, adv_std = np.mean(np.array(self.adv_buf)), np.std(np.array(self.adv_buf))
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.rew_buf,
                    adv=self.adv_buf)
        for k, v in data.items():
            {k: torch.as_tensor(v, dtype=torch.float32)}
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


# Policy and value functions
class MLPCategoricalPolicy(nn.Module):
    """ Handles the full operation of an policy of discrete actions

    Attributes:
        policy_net (nn.Sequential): NN that maps observations into the actions' logits
    Methods:
        dist(obs): computes the logits of the observation and returns a categorical dist based on that
        log_prob_from_dist(pi, act): computes the log likelihood of an observation in a certain policy dist
        forward(obs, optional act): handles the full process: get a dist and computes the log likelihood 
    """
    def __init__(self, obs_type, sizes, activation = nn.Tanh, last_activation = nn.Identity, device = torch.device("cpu")):
        super().__init__()
        self.policy_net = mlp(obs_type, sizes, activation, last_activation).to(device)
        
    def dist(self, obs):
        logits = self.policy_net(obs)
        return Categorical(logits = logits)
    
    def log_prob_from_dist(self, pi, act):
        return pi.log_prob(act)
    
    def forward(self, obs, act = None):
        pi = self.dist(obs)
        logp_a = None
        if act is not None:
            logp_a = self.log_prob_from_dist(pi, act)
        return pi, logp_a


class MLPGaussianPolicy():
    """Handles the full operation of a policy of continuous actions
    
    Attributes:
        mu_net (nn.Sequential): NN that maps observation into the mean of each action
        log_std (float): logarithm of the standard deviation
        
    Methods:
        dist(obs): returns a normal distribution based on the observation
        log_prob_from_dist(pi, act): returns the logits of an action of a certain policy

    """
    def __init__(self, obs_type, sizes, act_dim, activation = nn.Tanh, last_activation = nn.Identity, device = torch.device("cpu")):
        self.mu_net = mlp(obs_type, sizes, activation, last_activation).to(device)
        log_std = -0.5 *np.ones(act_dim, dtype = np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        
    def dist(self, obs):
        return Normal(self.mu_net(obs), torch.exp(self.log_std))
    
    def log_prob_from_dist(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)
    
class PolicyValue(nn.Module):
    """ Highlevel handler of the Policy and Value functions. Although several res define REINFORCE as an AC, I don't agree
    """
    def __init__(self, obs_type, observation_space, action_space, hidden_sizes, activation = nn.Tanh, last_activation = nn.Identity, device = torch.device("cpu")):
        super().__init__()
        
        if obs_type == "img": # image
            dims = [np.prod(observation_space.shape)] + list(hidden_sizes)
        elif obs_type=="default":
            dims = [observation_space.shape[0]] + list(hidden_sizes)
            
        # Value Function
        self.v_pi = mlp(obs_type, dims + [1], activation, last_activation).to(device)
        
        if isinstance(action_space, Box):
            self.pi = MLPGaussianPolicy(obs_type, dims + [action_space.shape[0]], action_space.n, activation, last_activation, device)
        else:
            self.pi = MLPCategoricalPolicy(obs_type, dims + [action_space.n], activation, last_activation, device)
            
                
    def step(self, obs):
        with torch.no_grad():
            pi = self.pi.dist(obs)
            a = pi.sample()
            v = self.v_pi(obs).squeeze()
        return a.cpu().numpy(), v.cpu().numpy()
  

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
    em.reset()
    obs = em.get_state()
else:
    obs_sp = env.observation_space
    obs = env.reset()
act_sp = env.action_space
pv = PolicyValue(obs_type = args.observation_type, observation_space = obs_sp, action_space = act_sp, hidden_sizes = args.hidden_sizes, device = device)
memory = ReplayMemory(args.steps_per_epoch, 
                      obs_sp.shape, 
                      act_sp.shape, 
                      gamma = args.gamma, 
                      lam = args.lam)

pi_optimizer = optim.Adam(pv.pi.parameters(), lr = args.pi_lr)
v_optimizer = optim.Adam(pv.v_pi.parameters(), lr = args.v_lr)

ep_ret, ep_len = 0, 0
stats_return, stats_return["mean"], stats_return["max"], stats_return["min"] = {}, [], [], []
all_durations = []
for epoch in range(args.epochs):
    epoch_returns  = []
    for st in range(args.steps_per_epoch):
        action, value = pv.step(torch.as_tensor(obs, dtype = torch.float32).to(device))
        if args.observation_type == "img":
            reward = em.take_action(action)
            ep_ret += reward.item()
        else:
            next_obs, reward, done, info = env.step(action)
            ep_ret += reward
        ep_len += 1
        memory.push(Experience(obs, action, reward), value)
        if args.observation_type == "img":
            obs = em.get_state()
        else:
            obs = next_obs 
        
        timeout = ep_len == args.max_ep_len
        terminal = em.done if args.observation_type =="img" else done or timeout # trajectory finished
        epoch_ended = st == args.steps_per_epoch - 1 
        
        if terminal or epoch_ended:
            if epoch_ended and not(terminal):
                print('Warning: trajectory cut off by episode at %d steps.'%ep_len, flush=True)
            
            # trajectory didn't reach terminal state (not done) --> bootstrap
            if timeout or epoch_ended:
                _, value = pv.step(torch.as_tensor(obs, dtype = torch.float32).to(device))
            else:
                value = 0
            memory.finish_path(value)
            
            if terminal:
                epoch_returns.append(ep_ret)
                all_durations.append(ep_len)
            if args.observation_type == "img":
                em.reset()     
                obs = em.get_state()
            else:
                env.reset()
            ep_ret, ep_len = 0, 0
            
                
            
    # Update VPG
    data = memory.get()
    _obs, _act, _ret, _adv = data["obs"].to(device), data["act"].to(device), data["ret"].to(device), data["adv"].to(device)
    # Train policy
    pi_optimizer.zero_grad()
    pi, logp = pv.pi(_obs, _act)
    loss_pi = -(logp * _adv).mean()
    loss_pi.backward()
    pi_optimizer.step()
    mean_, max_, min_ = stats(epoch_returns)
    stats_return["mean"].append(mean_)
    stats_return["min"].append(min_)
    stats_return["max"].append(max_)
    
    # Train value function
    for i in range(args.train_v_iters):
        v_optimizer.zero_grad()
        loss_v_pi = ((pv.v_pi(_obs) - _ret)**2).mean()
        loss_v_pi.backward()
        v_optimizer.step()
    plot(epoch + 1, stats_return)
if args.observation_type == "img":
    em.close()