import numpy as np
from typing import Optional, List


class WholeBodyController:
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.num_g1_joints = 37
        
        self.default_dof_pos = np.array([
            -0.20, 0.0, 0.0, 0.42, -0.23, 0.0,
            -0.20, 0.0, 0.0, 0.42, -0.23, 0.0,
            0.0,
            0.5, 0.0, 0.462, 0.65, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.57,
            0.5, 0.0, -0.462, 0.65, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.57,
        ], dtype=np.float32)
        
        self.vx_range = (-0.5, 0.5)
        self.vy_range = (-0.4, 0.4)
        self.yaw_range = (-1.57, 1.57)
        self.height_range = (-0.5, 0.8)
        self.torso_pitch_range = (-0.52, 1.57)
        self.control_frequency = 50.0
        self.control_dt = 1.0 / self.control_frequency

    def step(self, high_level_cmd: np.ndarray, sim_joint_names: Optional[List[str]] = None) -> np.ndarray:
        vx = np.clip(high_level_cmd[0], self.vx_range[0], self.vx_range[1])
        vy = np.clip(high_level_cmd[1], self.vy_range[0], self.vy_range[1])
        yaw_dot = np.clip(high_level_cmd[2], self.yaw_range[0], self.yaw_range[1])
        com_height = np.clip(high_level_cmd[3], self.height_range[0], self.height_range[1])
        torso_pitch = np.clip(high_level_cmd[4], self.torso_pitch_range[0], self.torso_pitch_range[1])
        left_wrist_delta = high_level_cmd[5:8] if len(high_level_cmd) >= 8 else np.zeros(3)
        right_wrist_delta = high_level_cmd[8:11] if len(high_level_cmd) >= 11 else np.zeros(3)
        if len(self.default_dof_pos) >= self.num_g1_joints:
            joint_targets = self.default_dof_pos[:self.num_g1_joints].copy()
        else:
            joint_targets = np.zeros(self.num_g1_joints, dtype=np.float32)
            joint_targets[:len(self.default_dof_pos)] = self.default_dof_pos
        return joint_targets

    def reset(self):
        pass
