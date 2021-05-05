import torch
import torch.nn as nn


def mlp(obs_type, sizes, activation = nn.Tanh, last_activation = nn.Identity):
    """ Function that returns a multi layer perceptron pytorch object. It's the policy network

    Args:
        obs_type (str): type of the input: {"img", "default"}, image or OpenAi Gym default
        sizes (iterator): hidden sizes 
        activation (class 'torch.nn.modules.activation): activation of the hidden layers. Defaults to nn.Tanh.
        last_activation (class 'torch.nn.modules.activation): [activation of the last layer]. Defaults to nn.Identity.
    """
    s = list(zip(sizes[::1], sizes[1::])) # pairs together the respectiive sizes of each FC layer
    layers = [nn.Flatten()] if obs_type == "img" else []
    for shape in s[:-1]:
        layers.extend([nn.Linear(shape[0], shape[1]), activation()])
    layers.extend([nn.Linear(s[-1][0], s[-1][1]), last_activation()])
    return nn.Sequential(*layers)