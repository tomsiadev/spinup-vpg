import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical


class FFNet(nn.Module):
    def __init__(self, in_size, hidden_sizes):
        super(FFNet, self).__init__()
        assert hidden_sizes != 0

        self.input = nn.Linear(in_size, hidden_sizes[0])

        hiddens = []
        last_hidden_size = hidden_sizes[0]
        for hidden_size in hidden_sizes[1:]:
            hiddens.append(nn.Linear(last_hidden_size, hidden_size))
            hiddens.append(nn.ReLU())

            last_hidden_size = hidden_size

        self._last_hidden_size = last_hidden_size
        self.hiddens = nn.Sequential(*hiddens)


class ActorNet(FFNet):
    def __init__(self, in_size, hidden_sizes, num_actions):
        super(ActorNet, self).__init__(in_size, hidden_sizes)

        self.output = nn.Linear(self._last_hidden_size, num_actions)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)

        x = self.hiddens(x)

        x = self.output(x)

        return F.softmax(x, dim=0)

    def get_policy(self, obs, train=False):
        with torch.set_grad_enabled(train):
            return Categorical(self.forward(obs))

    def get_action(self, obs, train=False):
        return self.get_policy(obs, train).sample().item()


class ValueNet(FFNet):
    def __init__(self, in_size, hidden_sizes):
        super(ValueNet, self).__init__(in_size, hidden_sizes)

        self.output = nn.Linear(self._last_hidden_size, 1)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)

        x = self.hiddens(x)

        return self.output(x)

    def get_value(self, obs, train=True):
        with torch.set_grad_enabled(train):
            return self.forward(obs)
