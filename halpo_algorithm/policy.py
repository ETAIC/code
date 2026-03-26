from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class SharedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(128, 64)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_sizes[1], act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mean = self.mean_head(h)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        logp = dist.log_prob(action).sum(dim=-1)
        return action, logp

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            a, _ = self.sample(obs)
        return a

    def act_with_grad(self, obs: torch.Tensor) -> torch.Tensor:
        a, _ = self.sample(obs)
        return a

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1) 
