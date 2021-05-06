import torch
from collections import namedtuple
from IPython import display
import matplotlib.pyplot as plt

Experience = namedtuple(
    "Experience", ("state", "action", "next_state", "reward", "done")
)


class QValues():
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim = 1, index=actions.unsqueeze(-1))
    
    @staticmethod
    def get_next(target_net, next_states, dones, device):
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(device)
        values[dones == False] = target_net(next_states[dones == False]).max(dim = 1)[0].detach()
        return values

def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))
    
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)
    t5 = torch.cat(batch.done)
    return (t1,t2,t3,t4, t5)


def get_moving_average(period, values):
    values = torch.tensor(values, dtype = torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension = 0, size = period, step = 1).mean(dim = 1).flatten(start_dim = 0)
        moving_avg = torch.cat((torch.zeros(period - 1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()


def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("# Episode")
    plt.ylabel("Rewards")
    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    print("Episode", len(values), "\n", moving_avg_period, "episode moving avg:", moving_avg[-1])
    display.clear_output(wait = True)