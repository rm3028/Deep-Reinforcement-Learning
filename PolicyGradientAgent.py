
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyGradient(nn.Module):
    def __init__(self, observation_size, action_num):
        super(PolicyGradient, self).__init__()
        hidden_size = observation_size << 1

        self.fc1 = nn.Linear(observation_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_num)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)

        x = self.fc2(x)
        x = torch.tanh(x)

        x = self.fc3(x)
        x = F.softmax(x, -1)
        return x


class PolicyGradientAgent:
    def __init__(self, observation_size, action_num, lr=0.001, device='cpu'):
        self.network = PolicyGradient(observation_size, action_num)
        self.network.to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def learn(self, epoch_pg):
        loss = -epoch_pg

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, state):
        action_prob = self.network(state)
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
