import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import scipy.signal

def combined_shape(length, shape=None):
    """Returns the correct shape based on the type of oobservation and actions spaces
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def stats(arr):
    return np.mean(arr), np.max(arr), np.min(arr)

def plot(ep, stats_return):
    mean_ = stats_return["mean"]
    max_ = stats_return["max"]
    min_ = stats_return["min"]
    plt.figure(2)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("# Epochs")
    plt.ylabel("Avg Returns")
    plt.plot(mean_, c = "r")
    plt.fill_between(np.arange(len(mean_)), max_, min_, facecolor = "blue", alpha = 0.3)
    plt.pause(0.001)
    print("Epoch", ep, "\n", "Avg Return", mean_[-1])
    display.clear_output(wait = True)
    