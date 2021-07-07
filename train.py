import gym
import torch.cuda

from torch.optim import Adam

from models import *
from vpg import *

def train():

    env_name = 'CartPole-v1'

    epochs = 2000
    max_ep_length = 2000
    save_freq = 200

    value_updates_per_trajectory = 80

    actor_hidden_sizes = [64, 64]
    value_hidden_sizes = [64, 64]
    actor_lr = 0.001
    value_lr = 0.001

    gamma = 0.99
    lambd = 0.97

    env = gym.make(env_name)

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n

    actor_net = ActorNet(obs_size, actor_hidden_sizes, act_size).float()
    value_net = ValueNet(obs_size, value_hidden_sizes).float()

    actor_optim = Adam(actor_net.parameters(), actor_lr)
    value_optim = Adam(value_net.parameters(), value_lr)

    for i in range(epochs):
        print("EPOCH {}".format(i))
        train_one_epoch(env, actor_net, value_net, actor_optim, value_optim, gamma, lambd,
                        value_updates_per_trajectory=value_updates_per_trajectory,
                        max_epoch_length=max_ep_length, render=False)

        if (i+1) % save_freq == 0:
            torch.save(actor_net.state_dict(), "models/actor_{}".format(i + 1))
            torch.save(value_net.state_dict(), "models/value_{}".format(i + 1))


if __name__=='__main__':
    train()
