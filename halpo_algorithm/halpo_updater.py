import math
from typing import Dict, Any, Iterable, Tuple, Optional

import torch
from torch import nn
from torch.optim import Optimizer

from .mechanism_monitor import MechanismMonitor


def _flatten_grads(params: Iterable[nn.Parameter]) -> torch.Tensor:
    grads = []
    for p in params:
        if p.grad is None:
            grads.append(torch.zeros_like(p).view(-1))
        else:
            grads.append(p.grad.view(-1))
    return torch.cat(grads) if grads else torch.tensor([])


def _assign_flat_grad(params: Iterable[nn.Parameter], flat: torch.Tensor) -> None:
    """Assign a flat gradient vector back into params.grad shapes."""
    offset = 0
    for p in params:
        numel = p.numel()
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        p.grad.copy_(flat[offset : offset + numel].view_as(p))
        offset += numel


def project_and_step(
    actor: nn.Module,
    optimizer: Optimizer,
    calc_policy_loss_fn,
    batch: Dict[str, Any],
    env,
    sigma: float,
    logger: Any = None,
    halpo_stats: Dict[str, float] | None = None,
    mechanism_monitor: Optional[MechanismMonitor] = None,
    epoch: int = 0,
    batch_idx: int = 0,
step: int = 0,  #
max_grad_norm: float = 0.5,  #
    reference_states: Optional[torch.Tensor] = None,
    reference_actions: Optional[torch.Tensor] = None,
    reference_team_rewards: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Perform one actor update step with HALPO gradient projection.

    Args:
        actor: policy network (produces action distribution / actions given observations).
        optimizer: optimizer for actor.
        calc_policy_loss_fn: callable(actor, batch) -> loss tensor.
        batch: dict of training tensors; must include 'obs' and 'next_obs'.
        env: environment instance exposing calculate_lyapunov_value(state, action).
        sigma: HALPO decay rate.
        logger: optional logger with add_scalar(name, value, step) or similar.
        halpo_stats: optional dict to accumulate per-epoch ratios/means.

    Returns:
        A dict with HALPO logging metrics.
    """
    device = next(actor.parameters()).device

    optimizer.zero_grad(set_to_none=True)
    policy_loss = calc_policy_loss_fn(actor, batch)
    policy_loss.backward(retain_graph=True, create_graph=True)
    g_flat = _flatten_grads(actor.parameters())
    
    u_ind = g_flat.clone()
    
    optimizer.zero_grad(set_to_none=True)
    obs = batch["obs"].to(device)
    actions = batch["actions"]
    
    new_logprobs = actor.log_prob(obs, actions)
    
    if "team_reward" in batch:
        team_reward = batch["team_reward"]
    elif "rewards" in batch:
        rewards = batch["rewards"]
        team_reward = rewards * 2.0
    else:
        advantages = batch.get("advantages", None)
        if advantages is not None:
            team_reward = advantages * 2.0
        else:
            raise ValueError("Batch must contain 'rewards' or 'team_reward' for team gradient computation")
    
    team_loss = -(new_logprobs * team_reward.detach()).mean()
    team_loss.backward(retain_graph=True, create_graph=True)
    u_team_flat = _flatten_grads(actor.parameters())
    
    u_team = u_team_flat.clone()

    # 2) Compute Lyapunov constraint (A_bar, b_bar)
    # Re-sample actions from CURRENT policy using obs and next_obs
    obs = batch["obs"].to(device)
    next_obs = batch.get("next_obs", None)
    if next_obs is None:
        # If next_obs not provided, approximate with obs shifted by 1 in buffer, else fallback
        raise ValueError("Batch is missing 'next_obs' required for HALPO ΔV computation.")

    # Actor forward to obtain actions; support deterministic or sampling via a provided helper
    if hasattr(actor, "act"):
        with torch.no_grad():
            a_t = actor.act(obs)
            a_tp1 = actor.act(next_obs)
    else:
        # Assume actor(obs) returns action directly
        with torch.no_grad():
            a_t = actor(obs)
            a_tp1 = actor(next_obs)

    # Expect env.calculate_lyapunov_value takes (state, action) with action as dict for MARL
    # If policy outputs per-agent actions packed, pass as dict if available in batch
    # Try to construct dict actions {robot_0, robot_1} from batch if present
    action_dict_t = batch.get("action_dict_t", None)
    action_dict_tp1 = batch.get("action_dict_tp1", None)

    if action_dict_t is None or action_dict_tp1 is None:
        # Fallback: treat the policy action as shared action for both agents
        batch_size = a_t.shape[0]
        act_dim_per_agent = a_t.shape[1] // 2
        action_dict_t = {
            "robot_0": a_t[:, :act_dim_per_agent],
            "robot_1": a_t[:, act_dim_per_agent:]
        }
        action_dict_tp1 = {
            "robot_0": a_tp1[:, :act_dim_per_agent],
            "robot_1": a_tp1[:, act_dim_per_agent:]
        }

    # Compute V(s, pi(s)) and V(s', pi(s'))
    unwrapped_env = env
    while hasattr(unwrapped_env, 'unwrapped') and unwrapped_env.unwrapped is not unwrapped_env:
        unwrapped_env = unwrapped_env.unwrapped
    
    if not hasattr(unwrapped_env, 'calculate_lyapunov_value'):
        raise AttributeError(
            f"Environment {type(unwrapped_env)} does not have calculate_lyapunov_value method. "
            f"Available methods: {[m for m in dir(unwrapped_env) if not m.startswith('_')]}"
        )
    
    env_idx = batch.get("env_idx", None)
    batch_size = obs.shape[0]
    num_envs = unwrapped_env.num_envs if hasattr(unwrapped_env, 'num_envs') else 24
    
    if batch_size <= num_envs:
        if env_idx is not None:
            V_t_list = []
            V_tp1_list = []
            for env_id in range(num_envs):
                mask = (env_idx == env_id)
                if mask.sum() > 0:
                    idx = torch.where(mask)[0][0]
                    action_t_env = {
                        "robot_0": action_dict_t["robot_0"][idx:idx+1],
                        "robot_1": action_dict_t["robot_1"][idx:idx+1]
                    }
                    action_tp1_env = {
                        "robot_0": action_dict_tp1["robot_0"][idx:idx+1],
                        "robot_1": action_dict_tp1["robot_1"][idx:idx+1]
                    }
                    V_t_env = unwrapped_env.calculate_lyapunov_value(state=None, action=action_t_env)
                    V_tp1_env = unwrapped_env.calculate_lyapunov_value(state=None, action=action_tp1_env)
                    V_t_list.append(V_t_env[env_id:env_id+1].expand(mask.sum()))
                    V_tp1_list.append(V_tp1_env[env_id:env_id+1].expand(mask.sum()))
            V_t = torch.cat(V_t_list, dim=0)
            V_tp1 = torch.cat(V_tp1_list, dim=0)
        else:
            action_t_single = {
                "robot_0": action_dict_t["robot_0"][0:1].expand(num_envs, -1),
                "robot_1": action_dict_t["robot_1"][0:1].expand(num_envs, -1)
            }
            action_tp1_single = {
                "robot_0": action_dict_tp1["robot_0"][0:1].expand(num_envs, -1),
                "robot_1": action_dict_tp1["robot_1"][0:1].expand(num_envs, -1)
            }
            V_t_all = unwrapped_env.calculate_lyapunov_value(state=None, action=action_t_single)
            V_tp1_all = unwrapped_env.calculate_lyapunov_value(state=None, action=action_tp1_single)
            V_t = V_t_all.mean().expand(batch_size)
            V_tp1 = V_tp1_all.mean().expand(batch_size)
    else:
        action_t_avg = {
            "robot_0": action_dict_t["robot_0"].mean(dim=0, keepdim=True).expand(num_envs, -1),
            "robot_1": action_dict_t["robot_1"].mean(dim=0, keepdim=True).expand(num_envs, -1)
        }
        action_tp1_avg = {
            "robot_0": action_dict_tp1["robot_0"].mean(dim=0, keepdim=True).expand(num_envs, -1),
            "robot_1": action_dict_tp1["robot_1"].mean(dim=0, keepdim=True).expand(num_envs, -1)
        }
        V_t_all = unwrapped_env.calculate_lyapunov_value(state=None, action=action_t_avg)
        V_tp1_all = unwrapped_env.calculate_lyapunov_value(state=None, action=action_tp1_avg)
        V_t = V_t_all.mean().expand(batch_size)
        V_tp1 = V_tp1_all.mean().expand(batch_size)
    delta_V = V_tp1 - V_t

    delta_V_mean = delta_V.mean()
    V_t_mean = V_t.mean()

    # Compute A_bar = grad_theta delta_V_mean w.r.t. actor parameters
    # Retain graph not needed for policy graph; but recreate computational graph via actor parameters
    # We need dependence of delta_V on theta via actions; to allow autograd, we should compute actions WITH grad
    # Recompute actions with grad enabled
    if hasattr(actor, "act_with_grad"):
        a_t_grad = actor.act_with_grad(obs)
        a_tp1_grad = actor.act_with_grad(next_obs)
    else:
        a_t_grad = actor(obs)
        a_tp1_grad = actor(next_obs)

    act_dim_per_agent = a_t_grad.shape[1] // 2
    action_t_grad_dict = {
        "robot_0": a_t_grad[:, :act_dim_per_agent].mean(dim=0, keepdim=True).expand(num_envs, -1),
        "robot_1": a_t_grad[:, act_dim_per_agent:].mean(dim=0, keepdim=True).expand(num_envs, -1)
    }
    action_tp1_grad_dict = {
        "robot_0": a_tp1_grad[:, :act_dim_per_agent].mean(dim=0, keepdim=True).expand(num_envs, -1),
        "robot_1": a_tp1_grad[:, act_dim_per_agent:].mean(dim=0, keepdim=True).expand(num_envs, -1)
    }
    V_t_g_all = unwrapped_env.calculate_lyapunov_value(state=None, action=action_t_grad_dict)
    V_tp1_g_all = unwrapped_env.calculate_lyapunov_value(state=None, action=action_tp1_grad_dict)
    V_t_g = V_t_g_all.mean()
    V_tp1_g = V_tp1_g_all.mean()
    delta_V_g = (V_tp1_g - V_t_g).mean()

    actor_params = [p for p in actor.parameters() if p.requires_grad]
    
    V_theta = 0.5 * torch.dot(u_ind - u_team, u_ind - u_team)
    V_theta_value = V_theta.item()
    
    grad_V_list = []
    if V_theta.requires_grad:
        V_theta.backward(retain_graph=True, create_graph=False)
        for p in actor_params:
            if p.grad is not None:
                grad_V_list.append(p.grad.view(-1))
            else:
                grad_V_list.append(torch.zeros_like(p).view(-1))
        h = torch.cat(grad_V_list) if grad_V_list else torch.tensor([], device=device)
        optimizer.zero_grad(set_to_none=True)
    else:
        h = (u_ind - u_team).detach()
    
    A_bar_flat = h.clone()
    b_bar = (-float(sigma)) * V_theta_value

    # 3) Projection
    eps = 1e-8
    u_ind_detached = u_ind.detach()
    h_detached = A_bar_flat.detach()
    V_detached = V_theta_value
    
    psi = torch.dot(h_detached, u_ind_detached).item() + float(sigma) * V_detached
    norm_h2 = torch.dot(h_detached, h_detached).item()
    
    lambda_proj = 0.0
    if norm_h2 > eps:
        lambda_proj = max(0.0, psi / (norm_h2 + eps))
    
    projection_active = False
    grad_change_ratio = torch.tensor(0.0, device=device)
    
    if lambda_proj > 0.0:
        d_star = u_ind_detached - lambda_proj * h_detached
        projection_active = True
        denom = torch.norm(u_ind_detached) + eps
        grad_change_ratio = torch.norm(d_star - u_ind_detached) / denom
    else:
        d_star = u_ind_detached

    if mechanism_monitor is not None:
        grad_V = u_ind - u_team
        
        current_V_theta = V_theta_value
        
        lr = optimizer.param_groups[0]['lr']
        
        if projection_active and norm_h2 > eps:
            delta_V_theta_approx = lr * (-float(sigma) * V_detached)
        else:
            delta_V_theta_approx = lr * psi
        
        next_V_theta_simple = 0.5 * torch.dot(d_star - u_team.detach(), d_star - u_team.detach()).item()
        delta_V_theta_simple = next_V_theta_simple - current_V_theta
        
        delta_V_theta = delta_V_theta_approx
        
        u_ind_for_metrics = u_ind.detach()
        u_team_for_metrics = u_team.detach()
        grad_V = (u_ind - u_team).detach()
        
        metrics = mechanism_monitor.compute_metrics(
            u_ind=u_ind_for_metrics,
            u_team=u_team_for_metrics,
            d_star=d_star,
            V_t_mean=V_t_mean,
            delta_V_mean=delta_V_mean,
            V_theta=current_V_theta,
            delta_V_theta=delta_V_theta,
            grad_V=grad_V,
        )
        
        metrics['lambda_projection'] = lambda_proj
        
        episode_metrics = batch.get('episode_metrics', None)
        mechanism_monitor.record_batch(metrics, epoch=epoch, batch_idx=batch_idx, step=step, episode_metrics=episode_metrics)
        
        mechanism_monitor.export_if_needed(epoch)

    # Write back d* to param grads and step
    _assign_flat_grad(actor_params, d_star)
    
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=max_grad_norm)
    
    optimizer.step()

    # Logging
    metrics = {
        "halpo/lyapunov_value_mean": V_t_mean.detach().item(),
        "halpo/lyapunov_delta_mean": delta_V_mean.detach().item(),
        "halpo/constraint_dot_product": psi,
        "halpo/constraint_boundary": -float(sigma) * V_detached,
        "halpo/projection_active": float(projection_active),
        "halpo/gradient_change_norm": grad_change_ratio.detach().item(),
        "halpo/lambda_projection": lambda_proj,
        "halpo/V_theta_value": V_theta_value,
    }

    if halpo_stats is not None:
        # accumulate for epoch-level stats
        halpo_stats.setdefault("proj_count", 0)
        halpo_stats.setdefault("batch_count", 0)
        halpo_stats.setdefault("grad_change_sum", 0.0)
        halpo_stats["batch_count"] += 1
        if projection_active:
            halpo_stats["proj_count"] += 1
            halpo_stats["grad_change_sum"] += metrics["halpo/gradient_change_norm"]

    if logger is not None and hasattr(logger, "add_scalar"):
        if hasattr(logger, "step"):
            batch_step = logger.step + batch_idx
        else:
            batch_step = step + batch_idx
        
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
            try:
                    logger.add_scalar(f"Batch/{k}", v, batch_step)
            except Exception:
                pass

    return metrics


class HalpoUpdater:
    """A thin wrapper that holds actor, optimizer and provides an update() with HALPO."""

    def __init__(
        self,
        actor: nn.Module,
        optimizer: Optimizer,
        env,
        sigma: float,
        logger: Any = None,
        mechanism_monitor: Optional[MechanismMonitor] = None,
max_grad_norm: float = 0.5,  #
    ):
        self.actor = actor
        self.optimizer = optimizer
        self.env = env
        self.sigma = sigma
        self.logger = logger
        self.halpo_stats = {}
        self.mechanism_monitor = mechanism_monitor
        self.current_epoch = 0
        self.batch_idx = 0
        self.max_grad_norm = max_grad_norm

    def update(
        self, 
        calc_policy_loss_fn, 
        batch: Dict[str, Any], 
epoch: int = 0,  #
batch_idx: int = 0,  #
step: int = 0,  #
        reference_states: Optional[torch.Tensor] = None,
        reference_actions: Optional[torch.Tensor] = None,
        reference_team_rewards: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        if epoch is None:
            epoch = self.current_epoch
        if batch_idx is None:
            batch_idx = self.batch_idx
            self.batch_idx += 1
        
        return project_and_step(
            self.actor,
            self.optimizer,
            calc_policy_loss_fn,
            batch,
            self.env,
            self.sigma,
            logger=self.logger,
            halpo_stats=self.halpo_stats,
            mechanism_monitor=self.mechanism_monitor,
            epoch=epoch,
            batch_idx=batch_idx,
step=step,  #
max_grad_norm=self.max_grad_norm,  #
            reference_states=reference_states,
            reference_actions=reference_actions,
            reference_team_rewards=reference_team_rewards,
        )
    
    def set_epoch(self, epoch: int):
"""
        self.current_epoch = epoch
        self.batch_idx = 0

    def epoch_stats(self) -> Dict[str, float]:
        proj = float(self.halpo_stats.get("proj_count", 0))
        total = float(self.halpo_stats.get("batch_count", 0) or 1.0)
        avg_change = 0.0
        if proj > 0:
            avg_change = float(self.halpo_stats.get("grad_change_sum", 0.0) / max(proj, 1.0))
        return {
            "halpo/projection_active_ratio": proj / total,
            "halpo/gradient_change_norm_mean": avg_change,
        } 
