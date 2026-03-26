from typing import Dict, Any, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .buffer import OnPolicyBuffer
from .halpo_updater import HalpoUpdater
from .mechanism_monitor import MechanismMonitor


class LocalHALPOTrainer:
    def __init__(
        self,
        env,
        actor: nn.Module,
        lr: float,
        num_envs: int,
        obs_dim: int,
        act_dim: int,
        num_steps: int,
        ppo_epoch: int,
        num_minibatches: int,
        gamma: float,
        gae_lambda: float,
        sigma: float,
        device: torch.device,
        logger=None,
        log_dir: str = "logs",
clip_param: float = 0.3,  #
entropy_coef: float = 0.12,  #
max_grad_norm: Optional[float] = 5.0,  #
    ):
        self.env = env
        self.actor = actor.to(device)
        self.optimizer = Adam(self.actor.parameters(), lr=lr)
        self.device = device
        self.buffer = OnPolicyBuffer(num_steps, num_envs, obs_dim, act_dim, device)
        self.ppo_epoch = ppo_epoch
        self.num_minibatches = num_minibatches
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.mechanism_monitor = MechanismMonitor(log_dir=log_dir, export_interval=100)
        
# V(θ)（V(θ)θ，）
self.reference_states = None  # epoch
        self.reference_actions = None
        self.reference_team_rewards = None
        self.reference_advantages = None
        
        self.updater = HalpoUpdater(
            actor=self.actor,
            optimizer=self.optimizer,
            env=self.env,
            sigma=sigma,
            logger=logger,
            mechanism_monitor=self.mechanism_monitor,
max_grad_norm=self.max_grad_norm,  #
        )
        self.current_epoch = 0
self.total_steps = 0  #
self.num_steps = num_steps  # epoch
self.num_envs = num_envs  #

    def _policy_loss(self, actor: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
        # Simple PPO clipped objective; baseline-free (for minimal reproducibility)
        obs = batch["obs"]
        actions = batch["actions"]
        old_logprobs = batch["logprobs"]
        advantages = batch["advantages"]
        # normalize advantages
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # current logprob
        new_logprobs = self.actor.log_prob(obs, actions)
        ratio = torch.exp(new_logprobs - old_logprobs)
# clip_param（halpo）
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
        loss_pi = -(torch.min(surr1, surr2)).mean()
# entropy bonus - entropy_coef（halpo）
        with torch.no_grad():
            _, std = self.actor.forward(obs)
        entropy = (0.5 + 0.5 * torch.log(2 * torch.pi * std * std)).sum(dim=-1).mean()
        loss = loss_pi - self.entropy_coef * entropy
        return loss

    def collect_rollout(self):
        # IsaacLab MARL env returns dict obs; we concatenate two agents for shared policy
        reset_result = self.env.reset()
# gym (obs, info)
        if isinstance(reset_result, tuple):
            obs_dict, info = reset_result
        else:
            obs_dict = reset_result
        # assume keys: "robot_0", "robot_1"
# feature， [num_envs, obs_dim*2]
        obs = torch.cat([obs_dict["robot_0"], obs_dict["robot_1"]], dim=1)
        obs = obs.to(self.device)

# episode
episode_returns = []  # episode
episode_lengths = []  # episode
success_flags = []  # episode
episode_rewards_list = []  # episode（）

        for t in range(self.buffer.T):
            with torch.no_grad():
                action = self.actor.act(obs)  # action shape: [num_envs, act_dim*2]
                logprob = self.actor.log_prob(obs, action)
                # step needs per-agent dict; split back
                # obs shape: [num_envs, obs_dim*2], action shape: [num_envs, act_dim*2]
# action，batch
                num_envs = obs.shape[0]
                act_dim_per_agent = action.shape[1] // 2
                actions_dict = {
                    "robot_0": action[:, :act_dim_per_agent],
                    "robot_1": action[:, act_dim_per_agent:],
                }
            step_result = self.env.step(actions_dict)
# gym
            if len(step_result) == 4:
                next_obs_dict, reward_dict, terminated_dict, info = step_result
                truncated_dict = {k: torch.zeros_like(v, dtype=torch.bool) for k, v in terminated_dict.items()}
            elif len(step_result) == 5:
                next_obs_dict, reward_dict, terminated_dict, truncated_dict, info = step_result
            else:
                raise ValueError(f"Unexpected step result format: {len(step_result)} elements")
            # pack tensors
# ：GAE，
            reward = (reward_dict["robot_0"] + reward_dict["robot_1"]) / 2.0
# （）
#  = agent（）
            team_reward = reward_dict["robot_0"] + reward_dict["robot_1"]
            done = (terminated_dict["robot_0"] | truncated_dict["robot_0"]) | (terminated_dict["robot_1"] | truncated_dict["robot_1"])  # [N]
# feature， [num_envs, obs_dim*2]
            next_obs = torch.cat([next_obs_dict["robot_0"], next_obs_dict["robot_1"]], dim=1).to(self.device)
            # value baseline: zero (minimal); could add critic here
            value = torch.zeros_like(reward)
# store: buffer，team_reward shape[num_envs]，
            self.buffer.add(obs, action, reward, done, value, logprob, next_obs, team_reward=team_reward)
            
# episode
# （done=True）
            if torch.any(done):
                if isinstance(info, (list, tuple)) and len(info) > 0:
# info，
                    for env_idx, env_info in enumerate(info):
                        if env_idx < len(done) and done[env_idx].item():
                            if env_info is not None and isinstance(env_info, dict):
# episode
                                if 'episode' in env_info:
                                    ep_info = env_info['episode']
                                    if ep_info is not None:
                                        if 'r' in ep_info:
                                            episode_returns.append(float(ep_info['r']))
                                            episode_rewards_list.append(float(ep_info['r']))
# episode
                                        if 'l' in ep_info:
                                            episode_lengths.append(float(ep_info['l']))
# （）
                                        if 'success' in ep_info:
                                            success_flags.append(1.0 if ep_info['success'] else 0.0)
# extraslog（IsaacLab）
                                if 'log' in env_info:
                                    log_info = env_info['log']
                                    if isinstance(log_info, dict):
# total_rewardepisode return
                                        if 'total_reward' in log_info:
                                            episode_returns.append(float(log_info['total_reward']))
                                            episode_rewards_list.append(float(log_info['total_reward']))
# successes
                                        if 'successes' in log_info:
                                            success_flags.append(float(log_info['successes']))
                elif isinstance(info, dict):
# info（）
# ，log
                    if 'log' in info:
                        log_info = info['log']
                        if isinstance(log_info, dict):
                            if 'total_reward' in log_info:
# ，
                                episode_returns.append(float(log_info['total_reward']))
                                episode_rewards_list.append(float(log_info['total_reward']))
                            if 'successes' in log_info:
                                success_flags.append(float(log_info['successes']))
# episode
                    if 'episode' in info:
                        ep_info = info['episode']
                        if ep_info is not None:
                            if 'r' in ep_info:
                                episode_returns.append(float(ep_info['r']))
                                episode_rewards_list.append(float(ep_info['r']))
                            if 'l' in ep_info:
                                episode_lengths.append(float(ep_info['l']))
                            if 'success' in ep_info:
                                success_flags.append(1.0 if ep_info['success'] else 0.0)
            
            obs = next_obs

        last_value = torch.zeros_like(self.buffer.rewards[-1])
        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

# ：epoch num_steps * num_envs
        self.total_steps += self.num_steps * self.num_envs

# episode
# ：episode_return  epoch （ epoch  episode ）
        avg_episode_return = float(np.mean(episode_returns)) if episode_returns else 0.0
        self.episode_metrics = {
'episode_return': avg_episode_return,  # （ epoch ）
'epoch_return': avg_episode_return,  # ：epoch  episode
            'success_rate': float(np.mean(success_flags)) if success_flags else 0.0,
            'avg_episode_length': float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            'episode_reward_mean': float(np.mean(episode_rewards_list)) if episode_rewards_list else 0.0,
            'episode_reward_std': float(np.std(episode_rewards_list)) if len(episode_rewards_list) > 1 else 0.0,
            'num_episodes': len(episode_returns),
        }

# epoch，V(θ)
        if self.reference_states is None and self.current_epoch == 0:
# buffer（V(θ)）
# 50%，
            num_ref_samples = min(self.buffer.T * self.buffer.N // 2, 1000)
            indices = torch.randperm(self.buffer.T * self.buffer.N, device=self.device)[:num_ref_samples]
            t_idx = indices // self.buffer.N
            n_idx = indices % self.buffer.N
            
            self.reference_states = self.buffer.obs[t_idx, n_idx].clone().detach()
            self.reference_actions = self.buffer.actions[t_idx, n_idx].clone().detach()
            self.reference_team_rewards = self.buffer.team_rewards[t_idx, n_idx].clone().detach()
            self.reference_advantages = self.buffer.advantages[t_idx, n_idx].clone().detach()
            
            print(f"[INFO] Saved {num_ref_samples} reference states for V(θ) computation")

    def update(self, epoch: int = 0):
"""，epoch"""
        if epoch is None:
            epoch = self.current_epoch
        
# epoch
        epoch = int(epoch)
        
        self.updater.set_epoch(epoch)
        
        batch_idx = 0
        for _ in range(self.ppo_epoch):
            for mb in self.buffer.get_minibatches(self.num_minibatches):
# reference_advantagesbatch
                if self.reference_advantages is not None:
                    mb['reference_advantages'] = self.reference_advantages
                
# episode_metricsbatch（cumulative reward）
                if hasattr(self, 'episode_metrics') and self.episode_metrics is not None:
                    mb['episode_metrics'] = self.episode_metrics
                
# epoch、batch_idxstep
                self.updater.update(
                    self._policy_loss, 
                    mb, 
epoch=epoch,  # epoch
batch_idx=batch_idx,  # batch_idx
step=self.total_steps,  #
                    reference_states=self.reference_states,
                    reference_actions=self.reference_actions,
                    reference_team_rewards=self.reference_team_rewards,
                )
                batch_idx += 1
        
        stats = self.updater.epoch_stats()
        self.buffer.reset()
        
# epoch
        self.current_epoch = epoch + 1
        
# 10epoch
        if epoch % 10 == 0 or epoch >= 9:
            self.mechanism_monitor.export_to_csv()
        
        return stats 
