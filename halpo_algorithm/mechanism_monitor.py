"""
"""
import os
import csv
from typing import Dict, Any, Optional
from collections import defaultdict

import torch
from torch import nn


class MechanismMonitor:
"""
    
def __init__(self, log_dir: str = "logs", export_interval: int = 1):  #
        """
        Args:
log_dir:
export_interval:
        """
        self.log_dir = log_dir
        self.export_interval = export_interval
        os.makedirs(log_dir, exist_ok=True)
        
        self.csv_path = os.path.join(log_dir, "train_mechanism.csv")
self.epoch_buffer = defaultdict(list)  #
        self.current_epoch = 0
        self.batch_count = 0
        
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
'step', 'epoch', 'batch',  # step
                    'rationality_gap_V', 'alignment', 'gcr_conflict',
                    'u_ind_norm', 'u_team_norm', 'd_star_norm',
'V_t_mean', 'delta_V_mean',  #
'V_theta', 'delta_V_theta',  #
'lambda_projection',  #
'episode_return',  #
'epoch_return',  # epoch
'success_rate',  #
'avg_episode_length',  #
'episode_reward_mean',  # episode
'episode_reward_std',  # episode
'num_episodes',  #
                ])
    
    def compute_metrics(
        self,
        u_ind: torch.Tensor,
        u_team: torch.Tensor,
        d_star: torch.Tensor,
        V_t_mean: torch.Tensor,
        delta_V_mean: torch.Tensor,
        V_theta: Optional[float] = None,
        delta_V_theta: Optional[float] = None,
        grad_V: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        
        Args:
u_ind:
u_team:
d_star:
V_t_mean:
delta_V_mean:
V_theta:
delta_V_theta:
grad_V: ∇V
            
        Returns:
        """
        u_ind = u_ind.detach()
        u_team = u_team.detach()
        d_star = d_star.detach()
        
        # 1. Rationality Gap V = 0.5 × ||u_ind - u_team||²
        diff = u_ind - u_team
        rationality_gap_V = 0.5 * torch.dot(diff, diff).item()
        
        if V_theta is None:
            V_theta = rationality_gap_V
        
        u_ind_norm = torch.norm(u_ind)
        u_team_norm = torch.norm(u_team)
        if u_ind_norm.item() > 1e-8 and u_team_norm.item() > 1e-8:
            alignment = torch.dot(u_ind, u_team).item() / (u_ind_norm.item() * u_team_norm.item())
        else:
            alignment = 0.0
        
        if delta_V_theta is not None:
            gcr_conflict = 1.0 if delta_V_theta > 0 else 0.0
        else:
            if grad_V is None:
                grad_V = u_ind - u_team
            
            grad_V = grad_V.detach()
            grad_V_norm = torch.norm(grad_V)
            
            if grad_V_norm.item() > 1e-8:
                dot_product = torch.dot(d_star, grad_V).item()
                gcr_conflict = 1.0 if dot_product > 0 else 0.0
            else:
                gcr_conflict = 0.0
        
        metrics = {
            'rationality_gap_V': rationality_gap_V,
            'alignment': alignment,
            'gcr_conflict': gcr_conflict,
            'u_ind_norm': u_ind_norm.item(),
            'u_team_norm': u_team_norm.item(),
            'd_star_norm': torch.norm(d_star).item(),
            'V_t_mean': V_t_mean.detach().item() if isinstance(V_t_mean, torch.Tensor) else float(V_t_mean),
            'delta_V_mean': delta_V_mean.detach().item() if isinstance(delta_V_mean, torch.Tensor) else float(delta_V_mean),
            'V_theta': V_theta,
            'delta_V_theta': delta_V_theta if delta_V_theta is not None else 0.0,
        }
        
        return metrics
    
    def record_batch(
        self,
        metrics: Dict[str, float],
        epoch: int,
        batch_idx: int = 0,
step: int = 0,  #
        episode_metrics: Optional[Dict[str, float]] = None,
        ):
        """
        
        Args:
metrics:
epoch:
batch_idx:
step:
episode_metrics: epoch
- episode_return:
- epoch_return: epoch
- success_rate:
- avg_episode_length:
- episode_reward_mean: episode
- episode_reward_std: episode
- num_episodes:
        """
        self.current_epoch = epoch
        self.batch_count = batch_idx
        
        if episode_metrics is None:
            episode_metrics = {
'episode_return': 0.0,  #
'epoch_return': 0.0,  # epoch
                'success_rate': 0.0,
                'avg_episode_length': 0.0,
                'episode_reward_mean': 0.0,
                'episode_reward_std': 0.0,
                'num_episodes': 0,
            }
        else:
            if 'epoch_return' not in episode_metrics and 'episode_return' in episode_metrics:
                episode_metrics['epoch_return'] = episode_metrics['episode_return']
        
        metrics_with_meta = {
'step': step,  #
            'epoch': epoch,
            'batch': batch_idx,
            **metrics,
            **episode_metrics
        }
        self.epoch_buffer[epoch].append(metrics_with_meta)
    
    def export_if_needed(self, epoch: int):
"""
        if epoch % self.export_interval == 0:
            self.export_to_csv()
    
    def export_to_csv(self):
"""
        rows_to_write = []
        for epoch in sorted(self.epoch_buffer.keys()):
            for batch_data in self.epoch_buffer[epoch]:
                rows_to_write.append([
batch_data.get('step', batch_data.get('epoch', 0) * 1000),  #
                    batch_data['epoch'],
                    batch_data['batch'],
                    batch_data['rationality_gap_V'],
                    batch_data['alignment'],
                    batch_data['gcr_conflict'],
                    batch_data['u_ind_norm'],
                    batch_data['u_team_norm'],
                    batch_data['d_star_norm'],
                    batch_data['V_t_mean'],
                    batch_data['delta_V_mean'],
batch_data.get('V_theta', batch_data['rationality_gap_V']),  #
batch_data.get('delta_V_theta', batch_data.get('delta_V_mean', 0.0)),  #
batch_data.get('lambda_projection', 0.0),  #
batch_data.get('episode_return', 0.0),  #
batch_data.get('epoch_return', batch_data.get('episode_return', 0.0)),  # epoch
                    batch_data.get('success_rate', 0.0),
                    batch_data.get('avg_episode_length', 0.0),
                    batch_data.get('episode_reward_mean', 0.0),
                    batch_data.get('episode_reward_std', 0.0),
                    batch_data.get('num_episodes', 0),
                ])
        
        if rows_to_write:
            file_exists = os.path.exists(self.csv_path)
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists or os.path.getsize(self.csv_path) == 0:
                    writer.writerow([
'step', 'epoch', 'batch',  # step
                        'rationality_gap_V', 'alignment', 'gcr_conflict',
                        'u_ind_norm', 'u_team_norm', 'd_star_norm',
'V_t_mean', 'delta_V_mean',  #
'V_theta', 'delta_V_theta',  #
'lambda_projection',  #
'episode_return',  #
'epoch_return',  # epoch
'success_rate',  #
'avg_episode_length',  #
'episode_reward_mean',  # episode
'episode_reward_std',  # episode
'num_episodes',  #
                    ])
                writer.writerows(rows_to_write)
            
            exported_epochs = set(self.epoch_buffer.keys())
            for epoch in exported_epochs:
                if epoch % self.export_interval == 0:
                    del self.epoch_buffer[epoch]
    
    def get_epoch_stats(self, epoch: int) -> Dict[str, float]:
"""
        if epoch not in self.epoch_buffer:
            return {}
        
        batches = self.epoch_buffer[epoch]
        if not batches:
            return {}
        
        stats = {}
        for key in ['rationality_gap_V', 'alignment', 'gcr_conflict', 'V_t_mean']:
            values = [b[key] for b in batches]
            stats[f'{key}_mean'] = sum(values) / len(values)
            if len(values) > 1:
                mean_val = stats[f'{key}_mean']
                variance = sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)
                stats[f'{key}_std'] = variance ** 0.5
            else:
                stats[f'{key}_std'] = 0.0
        
        gcr_values = [b['gcr_conflict'] for b in batches]
        stats['gcr_rate'] = sum(gcr_values) / len(gcr_values) if gcr_values else 0.0
        
        return stats

