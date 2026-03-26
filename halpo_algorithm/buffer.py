from typing import Dict, Any

import torch


class OnPolicyBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, act_dim: int, device: torch.device):
        self.T = num_steps
        self.N = num_envs
        self.obs = torch.zeros(self.T, self.N, obs_dim, device=device)
        self.next_obs = torch.zeros(self.T, self.N, obs_dim, device=device)
        self.actions = torch.zeros(self.T, self.N, act_dim, device=device)
        self.rewards = torch.zeros(self.T, self.N, device=device)
self.team_rewards = torch.zeros(self.T, self.N, device=device)  #
        self.dones = torch.zeros(self.T, self.N, device=device, dtype=torch.bool)
        self.values = torch.zeros(self.T, self.N, device=device)
        self.logprobs = torch.zeros(self.T, self.N, device=device)
        self.ptr = 0
        self.device = device

    def add(self, obs, action, reward, done, value, logprob, next_obs, team_reward=None):
        t = self.ptr
        self.obs[t].copy_(obs)
        self.actions[t].copy_(action)
        self.rewards[t].copy_(reward)
        if team_reward is not None:
            self.team_rewards[t].copy_(team_reward)
        self.dones[t].copy_(done)
        self.values[t].copy_(value)
        self.logprobs[t].copy_(logprob)
        self.next_obs[t].copy_(next_obs)
        self.ptr += 1

    def ready(self) -> bool:
        return self.ptr >= self.T

    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float):
        T, N = self.T, self.N
        adv = torch.zeros(T, N, device=self.device)
        ret = torch.zeros(T, N, device=self.device)
        gae = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            nonterminal = (~self.dones[t]).float()
            next_value = last_value if t == T - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
            ret[t] = adv[t] + self.values[t]
        self.advantages = adv
        self.returns = ret

    def get_minibatches(self, num_minibatches: int):
        T, N = self.T, self.N
        total = T * N
        indices = torch.randperm(total, device=self.device)
        mb_size = total // num_minibatches
        for i in range(num_minibatches):
            mb_idx = indices[i * mb_size : (i + 1) * mb_size]
            t_idx = mb_idx // N
            n_idx = mb_idx % N
            yield {
                "obs": self.obs[t_idx, n_idx],
                "actions": self.actions[t_idx, n_idx],
                "logprobs": self.logprobs[t_idx, n_idx],
                "values": self.values[t_idx, n_idx],
                "returns": self.returns[t_idx, n_idx],
                "advantages": self.advantages[t_idx, n_idx],
"rewards": self.rewards[t_idx, n_idx],  #
"team_reward": self.team_rewards[t_idx, n_idx],  #
                "next_obs": self.next_obs[t_idx, n_idx],
"env_idx": n_idx,  #
            }

    def reset(self):
        self.ptr = 0 
