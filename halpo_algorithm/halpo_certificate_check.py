import os
import sys

import torch
from torch.optim import SGD

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from halpo_algorithm.policy import SharedGaussianActor


def _flatten_grads(actor: torch.nn.Module) -> torch.Tensor:
    flats = []
    for p in actor.parameters():
        if p.grad is None:
            flats.append(torch.zeros_like(p).view(-1))
        else:
            flats.append(p.grad.view(-1))
    return torch.cat(flats) if flats else torch.tensor([], device=next(actor.parameters()).device)


def _policy_loss_ppo(actor, obs, actions, old_logprobs, advantages, clip_param: float, entropy_coef: float):
    new_logprobs = actor.log_prob(obs, actions)
    ratio = torch.exp(new_logprobs - old_logprobs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
    loss_pi = -(torch.min(surr1, surr2)).mean()
    with torch.no_grad():
        _, std = actor.forward(obs)
    entropy = (0.5 + 0.5 * torch.log(2 * torch.pi * std * std)).sum(dim=-1).mean()
    return loss_pi - entropy_coef * entropy


def _team_loss(actor, obs, actions, team_reward: torch.Tensor):
    new_logprobs = actor.log_prob(obs, actions)
    return -(new_logprobs * team_reward.detach()).mean()


def main():
    device = torch.device("cpu")

    batch_size = 8
    obs_dim = 4
    act_dim = 4

    clip_param = 0.3
    entropy_coef = 0.12
    sigma = 0.01
    eps = 1e-8
    lr = 1e-3

    actor = SharedGaussianActor(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=(16, 8)).to(device)
    optimizer = SGD(actor.parameters(), lr=lr)

    found = False
    for seed in range(50):
        torch.manual_seed(seed)

        obs = torch.randn(batch_size, obs_dim, device=device)
        actions = actor.act_with_grad(obs).detach()
        old_logprobs = actor.log_prob(obs, actions).detach() + 0.2 * torch.randn(batch_size, device=device)

        advantages = torch.randn(batch_size, device=device)
        team_reward = 0.7 * advantages + 0.3 * torch.randn(batch_size, device=device)

        optimizer.zero_grad(set_to_none=True)
        policy_loss = _policy_loss_ppo(actor, obs, actions, old_logprobs, advantages, clip_param, entropy_coef)
        policy_loss.backward(retain_graph=True, create_graph=True)
        u_ind = _flatten_grads(actor)

        optimizer.zero_grad(set_to_none=True)
        team_loss = _team_loss(actor, obs, actions, team_reward)
        team_loss.backward(retain_graph=True, create_graph=True)
        u_team = _flatten_grads(actor)

        u_diff = u_ind - u_team
        V = 0.5 * torch.dot(u_diff, u_diff)

        optimizer.zero_grad(set_to_none=True)
        V.backward()
        h = _flatten_grads(actor)

        u_ind_det = u_ind.detach()
        h_det = h.detach()
        V_det = V.detach()

        psi = torch.dot(h_det, u_ind_det) + sigma * V_det
        norm_h2 = torch.dot(h_det, h_det)
        lam = torch.clamp(psi / (norm_h2 + eps), min=0.0)

        if float(lam) <= 1e-6:
            continue

        d_star = u_ind_det - lam * h_det
        constraint_on_d_star = torch.dot(h_det, d_star) + sigma * V_det
        constraint_on_minus_d_star = torch.dot(h_det, -d_star) + sigma * V_det
        ok_projection = float(constraint_on_d_star) <= 1e-6

        print("seed =", seed)
        print("constraint(h, d_star) + sigma*V =", float(constraint_on_d_star))
        print("constraint(h, -d_star) + sigma*V =", float(constraint_on_minus_d_star))
        print("lambda_proj =", float(lam))
        print("kkt_satisfied_for_d_star =", ok_projection)
        found = True
        break

    if not found:
        print("lambda_proj remained ~0 in the sampled seeds")


if __name__ == "__main__":
    main()

