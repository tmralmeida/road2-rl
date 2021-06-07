import gym
from gym.spaces import Box, Discrete
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import matplotlib.pyplot as plt
from collections import deque
from utils.common import mlp
from utils.env_managers.cartpole import CartPoleEnvManager

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical, Normal
import argparse


parser = argparse.ArgumentParser(description = "A3C algorithm usage")

parser.add_argument(
    "--epochs",
    "-e",
    type=int,
    help="Number of epochs to train",
    default=10000
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
    help="Hyperparameter for GAE.",
    default=0.95
)


parser.add_argument(
    "--lr",
    type=float,
    help="Learning rate",
    default=1e-3,
)

parser.add_argument(
    "--betas",
    type=tuple,
    help="Hyperparameters for the shared optimizer",
    default=(0.9, 0.99)
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
    help="Either image or the default provided by the OpenAI Gym API. Only available for now: default",
    default="default",
    choices=["img", "default"]
)

parser.add_argument(
    "--env",
    type=str,
    help="OpenAI Gym environment",
    default="CartPole-v0"
)


args = parser.parse_args()

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
    
       
class ActorCritic(nn.Module):
    """ Highlevel Actor-Critic handler
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
    
    
# Shared Optimizer
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                
class Worker(mp.Process):
    def __init__(self,
                shared_ac, 
                optim,
                global_ep,
                global_ep_ret,
                res_queue,
                problem_params,
                net_params,
                name):
        super(Worker, self).__init__()
        self.shared_ac = shared_ac
        self.optim = optim
        self.global_ep = global_ep
        self.global_ep_ret = global_ep_ret
        self.res_queue = res_queue
        self.name = 'w%02i' % name
        self._params = problem_params
        self._net_params = net_params
        
        self.env = gym.make(self._params.env)
        self.local_ac = ActorCritic(self._params.observation_type, 
                                    self._params.observation_space, 
                                    self.env.action_space, 
                                    self._net_params.hidden_sizes, 
                                    self._net_params.activation,
                                    self._net_params.last_activation)
        self.em = CartPoleEnvManager(self.env, "cpu") if self._params.observation_type == "img" else None 
    def run(self):
        if self.em is not None:
            self.em.reset()
            obs = self.em.get_state()
        else:
            obs = self.env.reset()
        obs = torch.as_tensor(obs, dtype = torch.float32)
        while self.global_ep.value < self._params.epochs:
            b_obs, b_act, b_v, b_r, ep_ret = [], [], deque(), deque(), 0
            while True:
                action, value = self.local_ac.step(obs)
                if self.em is not None:
                    reward = em.take_action(action)
                    ep_ret += reward.item()
                    next_obs = em.get_state()
                    done = em.done
                else:
                    next_obs, reward, done, _ = self.env.step(action)
                    ep_ret += reward
                b_obs.append(obs)
                b_act.append(torch.from_numpy(action))
                if done:
                    next_obs = self.env.reset()
                obs = torch.as_tensor(next_obs, dtype = torch.float32)
                b_v.appendleft(value)
                b_r.appendleft(reward)
                if done:
                    value = 0
                    with self.global_ep.get_lock():
                        self.global_ep.value += 1

                    with self.global_ep_ret.get_lock():
                        if self.global_ep_ret.value == 0: # first
                            self.global_ep_ret.value += ep_ret
                        else:
                            self.global_ep_ret.value = self.global_ep_ret.value * 0.99 + ep_ret * 0.01 # moving average
                        print(f"worker {self.name}|Ep:{self.global_ep.value}|Ep_r: {self.global_ep_ret.value}|")
                    self.res_queue.put(self.global_ep_ret.value)
                    break
            
            if not done:
                _, value = self.local_ac.step(obs)
                
            b_v.appendleft(value)
            loss_pi, loss_v, gae = 0, 0, 0
            R = value
            pi, logp = self.local_ac.pi(torch.stack(b_obs), torch.stack(b_act))
            logp_r = torch.flip(logp, dims=[logp.dim()-1])
            for i in range(len(b_r)):
                R *= self._params.gamma  + b_r[i]
                adv = R - b_v[i]
                loss_v += 0.5 * adv**2
                
                # Generalized Advantage Estimation
                delta_t = b_r[i] + self._params.gamma * b_v[i + 1] - b_v[i] # error
                gae = gae * self._params.gamma * self._params.lam + delta_t
                
                loss_pi -=  logp_r[i] * gae
            
            self.optim.zero_grad()
            ((loss_pi + loss_v)/2).backward()
            for local_param, shared_param in zip(self.local_ac.parameters(),self.shared_ac.parameters()):
                shared_param._grad = local_param.grad
            self.optim.step()
            self.local_ac.load_state_dict(self.shared_ac.state_dict())
        self.res_queue.put(None)


device = torch.device("cpu")
env = gym.make(args.env)

if args.observation_type == "img":
    raise RuntimeError("Open Ai render method is not prepared for multiprocessing.")
    em = CartPoleEnvManager(env, device)
    obs_sp = Box(low=0, 
                 high=255, 
                 shape=(3, em.get_screen_height(), em.get_screen_width()), 
                 dtype=np.uint8)
else:
    obs_sp = env.observation_space

act_sp = env.action_space
shared_actor_critic = ActorCritic(args.observation_type, 
                                  obs_sp, 
                                  act_sp,
                                  args.hidden_sizes)


ProblemParams = namedtuple("ProblemParams",
                           ("epochs", "gamma", "lam","observation_type", "observation_space","env"))   

NetParams = namedtuple("NetParams",
                      ("hidden_sizes, activation, last_activation"))

params = ProblemParams(
    args.epochs, args.gamma, args.lam, args.observation_type, obs_sp, args.env
)    

net_params = NetParams(
    args.hidden_sizes, nn.Tanh, nn.Identity
)

optim = SharedAdam(shared_actor_critic.parameters(), args.lr, args.betas)
global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

workers = [Worker(shared_actor_critic, 
                  optim, global_ep, 
                  global_ep_r, 
                  res_queue, 
                  params, 
                  net_params, i) for i in range(mp.cpu_count())]


[w.start() for w in workers]
rewards = []
while True:
    reward = res_queue.get()
    if reward is not None:
        rewards.append(reward)
    else:
        break
[w.join() for w in workers]

plt.plot(rewards);
plt.pause(5)
