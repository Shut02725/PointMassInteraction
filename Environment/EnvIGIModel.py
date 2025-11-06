import torch
import torch.nn as nn
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import deque
from target_integration.partner_state_estimation import partner_estimation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DyadTrackingEnv:
    def __init__(self, dt=0.01, noise_config=None):
        self.dt = dt
        self.time = 0
        self.step_count = 0
        self.delay_steps = 12
        self.m = 1
        self.damping = 0.3
        # noise default setting
        if noise_config is None:
            noise_config = {
                'visual_pos_noise_std': 0.005,                # visual pos noise
                'visual_vel_noise_std': 0.05,                 # visual vel noise
                'visual_acc_noise_std': 0.5,                  # visual acc noise

                'haptic_weber_ratio': 0.05,                   # haptic noise 10%
                'haptic_base_noise': 0.05,                     # haptic noise 0.1N
                
                'motor_weber_ratio': 0.02,                    # motor noise 5%
                'motor_base_noise': 0.01,                     # motor noise 0.03N

                'enable_noise': True
            }
        
        self.noise_config = noise_config

        self.target_pos = torch.zeros(2, device=device)
        self.target_vel = torch.zeros(2, device=device)
        self.target_acc = torch.zeros(2, device=device)

        self.agent1_pos = torch.zeros(2, device=device)
        self.agent1_vel = torch.zeros(2, device=device)
        self.agent1_acc = torch.zeros(2, device=device)

        self.agent1_control = torch.zeros(2, device=device)
        self.agent1_control_buffer = deque(maxlen=self.delay_steps)
        self.agent1_force = torch.zeros(2, device=device)

        self.agent2_pos = torch.zeros(2, device=device)
        self.agent2_vel = torch.zeros(2, device=device)
        self.agent2_acc = torch.zeros(2, device=device)

        self.agent2_control = torch.zeros(2, device=device)
        self.agent2_control_buffer = deque(maxlen=self.delay_steps)
        self.agent2_force = torch.zeros(2, device=device)

        self.F_interaction = torch.zeros(2, device=device)

        self.k_interaction = 120
        self.c_interaction = 7

        # partner estimation
        n_obs_partner = 14

        Q = 50 * torch.tensor([[self.dt**6 / 36, self.dt**5 / 12, self.dt**4 / 6],
                                [self.dt**5 / 12, self.dt**4 / 4, self.dt**3 / 2],
                                [self.dt**4 / 6, self.dt**3 / 2, self.dt**2]], dtype=torch.float32, device=device)

        Q_p = torch.zeros((26, 26), dtype=torch.float32, device=device)
        Q_p[0:3, 0:3] = Q
        Q_p[3:6, 3:6] = Q
        Q_p[6:9, 6:9] = Q
        Q_p[9:12, 9:12] = Q
        Q_p[12:15, 12:15] = Q
        Q_p[15:18, 15:18] = Q
        Q_p[18:20, 18:20] = torch.eye(2, dtype=torch.float32, device=device)
        Q_p[20:26, 20:26] = 1e-3 * torch.eye(6, dtype=torch.float32, device=device)
        R_p = torch.eye(14, dtype=torch.float32, device=device) * 0.0001

        B_p = torch.tensor([0, 0, self.dt / self.m, 0, 0, self.dt / self.m], dtype=torch.float32, device=device).view(-1, 1)
        H_p = torch.zeros((14, 26), dtype=torch.float32, device=device)
        H_p[0:6, 0:6] = torch.eye(6, dtype=torch.float32, device=device)
        H_p[6:14, 12:20] = torch.eye(8, dtype=torch.float32, device=device)

        self.agent1_partner_estimation = partner_estimation(self.dt, B_p, self.k_interaction, self.c_interaction, Q_p, R_p, H_p)
        self.agent2_partner_estimation = partner_estimation(self.dt, B_p, self.k_interaction, self.c_interaction, Q_p, R_p, H_p)

        self.agent1_partner_obs = torch.zeros(n_obs_partner, device=device)
        self.agent2_partner_obs = torch.zeros(n_obs_partner, device=device)

        # self observation
        n_obs_self = 4

        self.agent1_self_obs = torch.zeros(n_obs_self, device=device)
        self.agent2_self_obs = torch.zeros(n_obs_self, device=device)

        self.agent1_pos_error = torch.zeros(2, device=device)
        self.agent1_vel_error = torch.zeros(2, device=device)
        self.agent1_acc_error = torch.zeros(2, device=device)

        self.agent2_pos_error = torch.zeros(2, device=device)
        self.agent2_vel_error = torch.zeros(2, device=device)
        self.agent2_acc_error = torch.zeros(2, device=device)

        # For recording trajectory
        self.trajectory_history = {
            'target': [],
            'agent1': [],
            'agent2': [],
            'time': []
        }

    def add_noise(self, tensor, noise_std):
        """add gaussian noise"""
        if not self.noise_config['enable_noise'] or noise_std == 0:
            return tensor
        noise = torch.normal(0, noise_std, size=tensor.shape, device=device)
        return tensor + noise

    def target_traj(self, t):
        """Target trajectory function using PyTorch"""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(float(t), device=device)
        
        # Position
        x = (3*torch.sin(1.8*t) + 3.4*torch.sin(1.9*t) + 
             2.5*torch.sin(1.82*t) + 4.3*torch.sin(2.34*t)) / 100
        y = (3*torch.sin(1.1*t) + 3.2*torch.sin(3.6*t) + 
             3.8*torch.sin(2.5*t) + 4.8*torch.sin(1.48*t)) / 100
        
        # Velocity (derivative of position)
        vx = (3*1.8*torch.cos(1.8*t) + 3.4*1.9*torch.cos(1.9*t) + 
              2.5*1.82*torch.cos(1.82*t) + 4.3*2.34*torch.cos(2.34*t)) / 100
        vy = (3*1.1*torch.cos(1.1*t) + 3.2*3.6*torch.cos(3.6*t) + 
              3.8*2.5*torch.cos(2.5*t) + 4.8*1.48*torch.cos(1.48*t)) / 100
        
        # Acceleration (derivative of velocity)
        ax = (-3*1.8*1.8*torch.sin(1.8*t) - 3.4*1.9*1.9*torch.sin(1.9*t) - 
              2.5*1.82*1.82*torch.sin(1.82*t) - 4.3*2.34*2.34*torch.sin(2.34*t)) / 100
        ay = (-3*1.1*1.1*torch.sin(1.1*t) - 3.2*3.6*3.6*torch.sin(3.6*t) - 
              3.8*2.5*2.5*torch.sin(2.5*t) - 4.8*1.48*1.48*torch.sin(1.48*t)) / 100
        
        pos = torch.stack([x, y])
        vel = torch.stack([vx, vy])
        acc = torch.stack([ax, ay])
        
        return pos, vel, acc

    def Agent1_FBController(self, pos_error, vel_error, acc_error):
        kp = 10
        kd = 5
        ka = 0

        FB_control = - kp*pos_error - kd*vel_error - ka*acc_error

        control_magnitude = torch.norm(FB_control)
        motor_noise_std = (self.noise_config['motor_base_noise'] + 
                          self.noise_config['motor_weber_ratio'] * control_magnitude)
        FB_control_noisy = self.add_noise(FB_control, motor_noise_std)

        self.agent1_control_buffer.append(FB_control_noisy)

        if len(self.agent1_control_buffer) == self.delay_steps:
            self.agent1_control = self.agent1_control_buffer[0].clone()
        else:
            self.agent1_control = torch.zeros(2, device=device)

    def Agent2_FBController(self, pos_error, vel_error, acc_error):
        kp = 5
        kd = 2
        ka = 0

        FB_control = - kp*pos_error - kd*vel_error - ka*acc_error

        control_magnitude = torch.norm(FB_control)
        motor_noise_std = (self.noise_config['motor_base_noise'] + 
                          self.noise_config['motor_weber_ratio'] * control_magnitude)
        FB_control_noisy = self.add_noise(FB_control, motor_noise_std)

        self.agent2_control_buffer.append(FB_control_noisy)

        if len(self.agent2_control_buffer) == self.delay_steps:
            self.agent2_control = self.agent2_control_buffer[0].clone()
        else:
            self.agent2_control = torch.zeros(2, device=device)

    def step(self):
        # target trajectory
        self.target_pos, self.target_vel, self.target_acc = self.target_traj(self.time)
        
        # interaction force
        self.F_interaction = (- self.k_interaction * (self.agent1_pos - self.agent2_pos) 
                            - self.c_interaction * (self.agent1_vel - self.agent2_vel))

        # === ノイズのある観測 ===
        target_pos_noisy = self.add_noise(self.target_pos, self.noise_config['visual_pos_noise_std'])
        target_vel_noisy = self.add_noise(self.target_vel, self.noise_config['visual_vel_noise_std'])
        target_acc_noisy = self.add_noise(self.target_acc, self.noise_config['visual_acc_noise_std'])

        agent1_pos_noisy = self.add_noise(self.agent1_pos, self.noise_config['visual_pos_noise_std'])
        agent1_vel_noisy = self.add_noise(self.agent1_vel, self.noise_config['visual_vel_noise_std'])
        agent1_acc_noisy = self.add_noise(self.agent1_acc, self.noise_config['visual_acc_noise_std'])

        agent2_pos_noisy = self.add_noise(self.agent2_pos, self.noise_config['visual_pos_noise_std'])
        agent2_vel_noisy = self.add_noise(self.agent2_vel, self.noise_config['visual_vel_noise_std'])
        agent2_acc_noisy = self.add_noise(self.agent2_acc, self.noise_config['visual_acc_noise_std'])
        
        # haptic noise
        force_magnitude = torch.norm(self.F_interaction)
        haptic_noise_std = (self.noise_config['haptic_base_noise'] + 
                           self.noise_config['haptic_weber_ratio'] * force_magnitude)
        F_interaction_noisy = self.add_noise(self.F_interaction, haptic_noise_std)

        # === Partner estimation（内部モデル）===
        self.agent1_partner_obs = torch.stack([
            agent1_pos_noisy[0], agent1_vel_noisy[0], agent1_acc_noisy[0],
            agent1_pos_noisy[1], agent1_vel_noisy[1], agent1_acc_noisy[1],
            target_pos_noisy[0], target_vel_noisy[0], target_acc_noisy[0],
            target_pos_noisy[1], target_vel_noisy[1], target_acc_noisy[1],
            F_interaction_noisy[0], F_interaction_noisy[1]
        ])

        self.agent2_partner_obs = torch.stack([
            agent2_pos_noisy[0], agent2_vel_noisy[0], agent2_acc_noisy[0],
            agent2_pos_noisy[1], agent2_vel_noisy[1], agent2_acc_noisy[1],
            target_pos_noisy[0], target_vel_noisy[0], target_acc_noisy[0],
            target_pos_noisy[1], target_vel_noisy[1], target_acc_noisy[1],
            -F_interaction_noisy[0], -F_interaction_noisy[1]
        ])

        # パートナーの目標位置を推定
        agent1_partner_target_pos = self.agent1_partner_estimation.step(self.agent1_partner_obs)
        agent2_partner_target_pos = self.agent2_partner_estimation.step(self.agent2_partner_obs)

        # === Self state estimation（制御用カルマンフィルタ）===
        # 観測値の構成(interaction goal integration)
        self.agent1_self_obs = torch.stack([
            agent1_pos_noisy[0] - target_pos_noisy[0],          # 目標との位置誤差（x軸）
            agent1_pos_noisy[0] - agent1_partner_target_pos[0], # パートナー予測目標との誤差（x軸）
            agent1_pos_noisy[1] - target_pos_noisy[1],          # 目標との位置誤差（y軸）
            agent1_pos_noisy[1] - agent1_partner_target_pos[1]  # パートナー予測目標との誤差（y軸）
        ])
        
        self.agent2_self_obs = torch.stack([
            agent2_pos_noisy[0] - target_pos_noisy[0],
            agent2_pos_noisy[0] - agent2_partner_target_pos[0],
            agent2_pos_noisy[1] - target_pos_noisy[1],
            agent2_pos_noisy[1] - agent2_partner_target_pos[1]
        ])

        # 推定状態から位置・速度・加速度を抽出
        # agent1_state_estimated: [pos_x, vel_x, acc_x, pos_y, vel_y, acc_y]
        self.agent1_pos_error = agent1_pos_noisy - target_pos_noisy
        self.agent1_vel_error = agent1_vel_noisy - target_vel_noisy
        self.agent1_acc_error = agent1_acc_noisy - target_acc_noisy

        self.agent2_pos_error = agent2_pos_noisy - target_pos_noisy
        self.agent2_vel_error = agent2_vel_noisy - target_vel_noisy
        self.agent2_acc_error = agent2_acc_noisy - target_acc_noisy

        # FB制御器（推定されたエラーを使用）
        self.Agent1_FBController(self.agent1_pos_error, self.agent1_vel_error, self.agent1_acc_error)
        self.Agent2_FBController(self.agent2_pos_error, self.agent2_vel_error, self.agent2_acc_error)

        # === 物理シミュレーション（真の状態で更新）===
        self.agent1_pos += self.agent1_vel * self.dt + self.agent1_acc * self.dt**2 / 2
        self.agent1_vel += self.agent1_acc * self.dt
        self.agent1_acc = ((self.F_interaction + self.agent1_force + self.agent1_control) / self.m)

        self.agent2_pos += self.agent2_vel * self.dt + self.agent2_acc * self.dt**2 / 2
        self.agent2_vel += self.agent2_acc * self.dt
        self.agent2_acc = ((-self.F_interaction + self.agent2_force + self.agent2_control) / self.m)

        # Record trajectory
        self.trajectory_history['target'].append([self.target_pos[0].item(), self.target_pos[1].item()])
        self.trajectory_history['agent1'].append([self.agent1_pos[0].item(), self.agent1_pos[1].item()])
        self.trajectory_history['agent2'].append([self.agent2_pos[0].item(), self.agent2_pos[1].item()])
        self.trajectory_history['time'].append(self.time)

        self.time += self.dt
        self.step_count += 1