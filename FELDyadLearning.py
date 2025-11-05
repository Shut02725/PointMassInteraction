import torch
import torch.nn as nn
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import os
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from target_integration.partner_state_estimation import partner_estimation
from target_integration.self_state_estimation import self_estimation
from LSTM_forward_dynamics import ForwardDynamicsLSTM
from LSTM_inverse_dynamics import InverseDynamicsLSTM

class DataNormalizer:
    def __init__(self):
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
        self.fitted = False

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            normalizer_data = pickle.load(f)
        self.input_mean = normalizer_data['input_mean']
        self.input_std = normalizer_data['input_std']
        self.output_mean = normalizer_data['output_mean']
        self.output_std = normalizer_data['output_std']
        self.fitted = normalizer_data['fitted']
        print(f"Normalizer loaded from {filepath}")
        return self

    def normalize_input(self, input_data):
        if not self.fitted:
            raise ValueError("Normalizer has not been fitted yet")
        # conversion of normalization parameter
        if not isinstance(self.input_mean, torch.Tensor):
            input_mean = torch.tensor(self.input_mean, dtype=torch.float32, device=input_data.device)
            input_std = torch.tensor(self.input_std, dtype=torch.float32, device=input_data.device)
        else:
            input_mean = self.input_mean.to(input_data.device)
            input_std = self.input_std.to(input_data.device)
        # normailzation（PyTorch tensor computation）
        normalized = (input_data - input_mean) / input_std
        return normalized

    def denormalize_output(self, normalized_output):
        if not self.fitted:
            raise ValueError("Normalizer has not been fitted yet")
        # 正規化パラメータをPyTorchテンソルに変換
        if not isinstance(self.output_mean, torch.Tensor):
            output_mean = torch.tensor(self.output_mean, dtype=torch.float32, device=normalized_output.device)
            output_std = torch.tensor(self.output_std, dtype=torch.float32, device=normalized_output.device)
        else:
            output_mean = self.output_mean.to(normalized_output.device)
            output_std = self.output_std.to(normalized_output.device)
        # 非正規化処理（PyTorchテンソル同士の演算）
        denormalized = normalized_output * output_std + output_mean
        return denormalized

def load_trained_model(model_dir, agent_name="Agent1"):
    model_path = os.path.join(model_dir, f"{agent_name}_model.pth")
    normalizer_path = os.path.join(model_dir, f"{agent_name}_normalizer.pkl")    
    if not os.path.exists(model_path):
        print(f"error: no model file: {model_path}")
        return None, None
    if not os.path.exists(normalizer_path):
        print(f"error: no normalilzation file: {normalizer_path}")
        return None, None
    # model initialization
    model = ForwardDynamicsLSTM(
        input_dim=14,
        output_dim=6,
        hidden_dim=512,
        num_layers=2,
        dropout=0.0,
        seed=42
    ).to(device)
    # load learned weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    # load normalization
    normalizer = DataNormalizer()
    normalizer.load(normalizer_path)
    print(f"model and normalizaer have been loaded:")
    print(f"model: {model_path}")
    print(f"normalizer: {normalizer_path}")
    return model, normalizer

class DualTrackingEnv:
    def __init__(self, dt=0.01, noise_config=None, use_trained_model=False, 
                 model_dir_agent1=None, model_dir_agent2=None):
        self.dt = dt
        self.time = 0
        self.step_count = 0
        self.delay_steps = 12
        
        # using learned model setting
        self.use_trained_model = use_trained_model
        self.model_dir_agent1 = model_dir_agent1
        self.model_dir_agent2 = model_dir_agent2
        
        # Agent1用のモデル
        self.agent1_trained_model = None
        self.agent1_normalizer = None
        
        # Agent2用のモデル
        self.agent2_trained_model = None
        self.agent2_normalizer = None
        
        # loading learned model for Agent1
        if self.use_trained_model and self.model_dir_agent1:
            self.agent1_trained_model, self.agent1_normalizer = load_trained_model(
                self.model_dir_agent1, "Agent1"
            )
            if self.agent1_trained_model is None:
                print("warning: failed to load the Agent1 forward model.")
            else:
                for param in self.agent1_trained_model.parameters():
                    param.requires_grad = False
                print("Agent1 Forward dynamics model parameters frozen for inference only")
        
        # loading learned model for Agent2
        if self.use_trained_model and self.model_dir_agent2:
            self.agent2_trained_model, self.agent2_normalizer = load_trained_model(
                self.model_dir_agent2, "Agent2"
            )
            if self.agent2_trained_model is None:
                print("warning: failed to load the Agent2 forward model.")
            else:
                for param in self.agent2_trained_model.parameters():
                    param.requires_grad = False
                print("Agent2 Forward dynamics model parameters frozen for inference only")
        
        self.m = 1
        self.damping = 0.3
        # noise default setting
        if noise_config is None:
            noise_config = {
                'visual_pos_noise_std': 0.003,                # visual pos noise
                'visual_vel_noise_std': 0.03,                 # visual vel noise
                'visual_acc_noise_std': 0.3,                  # visual acc noise

                'proprioceptive_pos_noise_std': 0.005,        # proprioceptive pos noise
                'proprioceptive_vel_noise_std': 0.02,         # proprioceptive vel noise

                'haptic_weber_ratio': 0.10,                   # haptic noise 10%
                'haptic_base_noise': 0.1,                     # haptic noise 0.1N
                
                'motor_weber_ratio': 0.05,                    # motor noise 5%
                'motor_base_noise': 0.03,                     # motor noise 0.03N

                'enable_noise': True
            }
        self.noise_config = noise_config
        # target pos, vel, acc
        self.target_pos = torch.zeros(2, device=device)
        self.target_vel = torch.zeros(2, device=device)
        self.target_acc = torch.zeros(2, device=device)
        # agent1 pos, vel, acc
        self.agent1_pos = torch.zeros(2, device=device)
        self.agent1_vel = torch.zeros(2, device=device)
        self.agent1_acc = torch.zeros(2, device=device)
        # agent1 state
        self.agent1_state = torch.zeros(6, device=device)
        self.agent1_state_pred = torch.zeros(6, device=device)
        # agent1 control, force
        self.agent1_control = torch.zeros(2, device=device)
        self.agent1_control_prev = torch.zeros(2, device=device)
        self.agent1_control_buffer = deque(maxlen=self.delay_steps)
        self.agent1_force = torch.zeros(2, device=device)
        # agent2 pos, vel, acc
        self.agent2_pos = torch.zeros(2, device=device)
        self.agent2_vel = torch.zeros(2, device=device)
        self.agent2_acc = torch.zeros(2, device=device)
        # agent2 state
        self.agent2_state = torch.zeros(6, device=device)
        self.agent2_state_pred = torch.zeros(6, device=device)
        # agent2 control, force
        self.agent2_control = torch.zeros(2, device=device)
        self.agent2_control = torch.zeros(2, device=device)
        self.agent2_control_prev = torch.zeros(2, device=device)
        self.agent2_control_buffer = deque(maxlen=self.delay_steps)
        self.agent2_force = torch.zeros(2, device=device)
        # agent2 interaction
        self.F_interaction = torch.zeros(2, device=device)
        self.k_interaction = 5
        self.c_interaction = 2
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
        R_p = torch.eye(14, dtype=torch.float32, device=device) * 0.001
        B_p = torch.tensor([0, 0, self.dt / self.m, 0, 0, self.dt / self.m], dtype=torch.float32, device=device).view(-1, 1)
        H_p = torch.zeros((14, 26), dtype=torch.float32, device=device)
        H_p[0:6, 0:6] = torch.eye(6, dtype=torch.float32, device=device)
        H_p[6:14, 12:20] = torch.eye(8, dtype=torch.float32, device=device)
        self.agent1_partner_estimation = partner_estimation(self.dt, B_p, self.k_interaction, self.c_interaction, Q_p, R_p, H_p)
        self.agent2_partner_estimation = partner_estimation(self.dt, B_p, self.k_interaction, self.c_interaction, Q_p, R_p, H_p)
        self.agent1_partner_obs = torch.zeros(n_obs_partner, device=device)
        self.agent2_partner_obs = torch.zeros(n_obs_partner, device=device)
        # LSTM data buffer
        self.sequence_length = 10  # same sequence length as training 
        self.agent1_history_buffer = deque(maxlen=self.sequence_length)
        self.agent2_history_buffer = deque(maxlen=self.sequence_length)
        # self observation
        n_state_self = 6
        A_s = torch.tensor([[1, self.dt, self.dt**2 / 2, 0, 0, 0],
                            [0, 1, self.dt, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, self.dt, self.dt**2 / 2],
                            [0, 0, 0, 0, 1, self.dt],
                            [0, 0, 0, 0, 0, 1]], dtype=torch.float32, device=device)
        B_s = torch.tensor([[0, 0],
                            [0, 0],
                            [self.dt/self.m, 0],
                            [0, 0],
                            [0, 0],
                            [0, self.dt/self.m]], dtype=torch.float32, device=device)
        H_s = torch.tensor([
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ], dtype=torch.float32, device=device)
        Q_s = torch.zeros((n_state_self, n_state_self), dtype=torch.float32, device=device)
        Q_s[0:3, 0:3] = Q
        Q_s[3:6, 3:6] = Q

        n_obs_self = 4

        self.agent1_self_obs = torch.zeros(n_obs_self, device=device)
        self.agent2_self_obs = torch.zeros(n_obs_self, device=device)

        # ノイズ設定
        agnet1_uniform_noise = np.random.rand(n_obs_self)
        agnet2_uniform_noise = np.random.rand(n_obs_self)

        MinTaskNoise = 0.005
        MaxTaskNoise = 0.1

        self.agent1_scaled_noise = MinTaskNoise + (MaxTaskNoise - MinTaskNoise) * agnet1_uniform_noise
        self.agent2_scaled_noise = MinTaskNoise + (MaxTaskNoise - MinTaskNoise) * agnet2_uniform_noise

        agent1_noise = torch.tensor(self.agent1_scaled_noise ** 2, dtype=torch.float32, device=device)
        agent2_noise = torch.tensor(self.agent2_scaled_noise ** 2, dtype=torch.float32, device=device)
        R_s1 = torch.diag(agent1_noise)
        R_s2 = torch.diag(agent2_noise)

        self.agent1_self_estimation = self_estimation(dt, A_s, B_s, Q_s, R_s1, H_s)
        self.agent2_self_estimation = self_estimation(dt, A_s, B_s, Q_s, R_s2, H_s)

        # For recording trajectory
        self.trajectory_history = {
            'target': [],
            'agent1': [],
            'agent2': [],
            'time': []
        }
        
        # ===== Agent1用の逆モデル =====
        self.agent1_inverse_model = InverseDynamicsLSTM(
            input_dim=6,
            output_dim=2,
            hidden_dim=256,
            num_layers=2,
            dropout=0.1,
            seed=42
        ).to(device)
        
        # ===== Agent2用の逆モデル =====
        self.agent2_inverse_model = InverseDynamicsLSTM(
            input_dim=6,
            output_dim=2,
            hidden_dim=256,
            num_layers=2,
            dropout=0.1,
            seed=43  # 異なるseedで初期化
        ).to(device)

        self.criterion = nn.HuberLoss(delta=1.0)  # MSELossの代わりに使用（外れ値に対してロバスト）
        
        # 各エージェント用のオプティマイザー
        self.agent1_optimizer = torch.optim.Adam(self.agent1_inverse_model.parameters(), lr=1e-3)
        # Agent2の学習率を下げる（学習の安定化のため）
        self.agent2_optimizer = torch.optim.Adam(self.agent2_inverse_model.parameters(), lr=5e-4)
        
        # 各エージェント用の学習データ
        self.agent1_episode_losses = []
        self.agent2_episode_losses = []
        
        self.agent1_episode_states = []
        self.agent2_episode_states = []
        
        self.agent1_episode_fb_controls = []
        self.agent2_episode_fb_controls = []

        self.agent1_state_history = deque(maxlen=self.sequence_length)
        self.agent2_state_history = deque(maxlen=self.sequence_length)
        
        # 各エージェント用の隠れ状態
        self.agent1_hidden = None
        self.agent2_hidden = None

        self.agent1_state_pred = torch.zeros(1, 1, 6, device=device)
        self.agent2_state_pred = torch.zeros(1, 1, 6, device=device)

        # 各エージェント用の制御履歴
        self.agent1_prev_fb_control = torch.zeros(2, device=device)
        self.agent1_current_ff_control = torch.zeros(2, device=device)
        
        self.agent2_prev_fb_control = torch.zeros(2, device=device)
        self.agent2_current_ff_control = torch.zeros(2, device=device)

        self.agent1_fb_control_history = []
        self.agent1_ff_control_history = []
        self.agent1_total_control_history = []
        
        self.agent2_fb_control_history = []
        self.agent2_ff_control_history = []
        self.agent2_total_control_history = []

    def add_noise(self, tensor, noise_std):
        """add gausian noise"""
        if not self.noise_config['enable_noise'] or noise_std == 0:
            return tensor
        noise = torch.normal(0, noise_std, size=tensor.shape, device=device)
        return tensor + noise

    def target_traj(self, t):
        """Target trajectory function using PyTorch"""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(float(t), device=device)
        
        # Position (修正: 重複項を削除)
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
        motor_noise_std = (self.noise_config['motor_base_noise'] + self.noise_config['motor_weber_ratio'] * control_magnitude)
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
        motor_noise_std = (self.noise_config['motor_base_noise'] + self.noise_config['motor_weber_ratio'] * control_magnitude)
        FB_control_noisy = self.add_noise(FB_control, motor_noise_std)

        self.agent2_control_buffer.append(FB_control_noisy)

        if len(self.agent2_control_buffer) == self.delay_steps:
            self.agent2_control = self.agent2_control_buffer[0].clone()
        else:
            self.agent2_control = torch.zeros(2, device=device)

    def update_history_buffer(self, agent_id=1):
        """
        履歴バッファを更新する（学習時と同じ入力形式）
        
        Args:
            agent_id: エージェントID (1 or 2)
        """
        if agent_id == 1:
            agent_state = self.agent1_state
            agent_control = self.agent1_control
            agent_self_obs = self.agent1_self_obs
            agent_force = self.F_interaction + self.agent1_force
            buffer = self.agent1_history_buffer
        else:
            agent_state = self.agent2_state
            agent_control = self.agent2_control
            agent_self_obs = self.agent2_self_obs
            agent_force = -self.F_interaction + self.agent2_force
            buffer = self.agent2_history_buffer
        
        # 学習時と同じ入力形式で履歴データを構築
        history_input = torch.cat([
            agent_state,           # 6次元: エージェントの状態
            agent_control,         # 2次元: 制御入力
            agent_force,          # 2次元: 力（相互作用力 + 外力）
            agent_self_obs         # 4次元: 自己観測
        ])
        
        buffer.append(history_input.clone())

    def predict_with_lstm(self, agent_id=1):
        """Forward modelで次状態を予測（勾配を保持）"""
        if agent_id == 1:
            if not self.use_trained_model or self.agent1_trained_model is None:
                return None
            trained_model = self.agent1_trained_model
            normalizer = self.agent1_normalizer
            buffer = self.agent1_history_buffer
            current_state = self.agent1_state
        else:
            if not self.use_trained_model or self.agent2_trained_model is None:
                return None
            trained_model = self.agent2_trained_model
            normalizer = self.agent2_normalizer
            buffer = self.agent2_history_buffer
            current_state = self.agent2_state
        
        if len(buffer) < self.sequence_length:
            return None
        
        try:
            sequence_data = list(buffer)
            lstm_input = torch.stack(sequence_data).unsqueeze(0)
            
            # 入力データを正規化
            lstm_input_normalized = normalizer.normalize_input(lstm_input)
            
            # Forward modelは勾配計算しない（パラメータは固定）
            with torch.no_grad():
                trained_model.eval()
                lstm_output = trained_model(lstm_input_normalized)
            
            if isinstance(lstm_output, tuple):
                predicted_output_normalized = lstm_output[0]
            else:
                predicted_output_normalized = lstm_output
            
            # 出力を非正規化
            predicted_delta = normalizer.denormalize_output(predicted_output_normalized)
            predicted_delta = predicted_delta.squeeze()
            
            if predicted_delta.dim() == 0:
                predicted_delta = predicted_delta.unsqueeze(0)
            
            if predicted_delta.numel() != 6:
                if predicted_delta.numel() >= 6:
                    predicted_delta = predicted_delta[-6:]
                else:
                    print(f"警告: Agent{agent_id}の予測出力の次元が不正です: {predicted_delta.shape}")
                    return None
            
            # 重要: detach()して新しい計算グラフを開始
            predicted_next_state = current_state + predicted_delta
            predicted_next_state = predicted_next_state.detach()
            predicted_next_state.requires_grad_(True)
            
            return predicted_next_state
            
        except Exception as e:
            print(f"Agent{agent_id} LSTM予測でエラーが発生しました: {e}")
            return None

    def step(self):
        """1ステップの物理シミュレーションと学習データ収集"""
        
        # ターゲット軌道の計算
        self.target_pos, self.target_vel, self.target_acc = self.target_traj(self.time)

        # 相互作用力の計算
        self.F_interaction = (- self.k_interaction * (self.agent1_pos - self.agent2_pos) 
                            - self.c_interaction * (self.agent1_vel - self.agent2_vel))

        # === 生物学的ノイズの追加 ===
        # 1. 視覚ノイズ（ターゲット観測）
        target_pos_noisy = self.add_noise(self.target_pos, self.noise_config['visual_pos_noise_std'])
        target_vel_noisy = self.add_noise(self.target_vel, self.noise_config['visual_vel_noise_std'])
        target_acc_noisy = self.add_noise(self.target_acc, self.noise_config['visual_acc_noise_std'])
        
        # 2. 固有受容感覚ノイズ（自己の状態）
        agent1_pos_sensed = self.add_noise(self.agent1_pos, self.noise_config['proprioceptive_pos_noise_std'])
        agent1_vel_sensed = self.add_noise(self.agent1_vel, self.noise_config['proprioceptive_vel_noise_std'])
        
        agent2_pos_sensed = self.add_noise(self.agent2_pos, self.noise_config['proprioceptive_pos_noise_std'])
        agent2_vel_sensed = self.add_noise(self.agent2_vel, self.noise_config['proprioceptive_vel_noise_std'])
        
        # 3. 触覚ノイズ（力情報）- 信号依存
        force_magnitude = torch.norm(self.F_interaction)
        haptic_noise_std = (self.noise_config['haptic_base_noise'] + self.noise_config['haptic_weber_ratio'] * force_magnitude)
        F_interaction_noisy = self.add_noise(self.F_interaction, haptic_noise_std)

        # パートナー観測の更新
        self.agent1_partner_obs = torch.stack([
            agent1_pos_sensed[0], agent1_vel_sensed[0], self.agent1_acc[0],
            agent1_pos_sensed[1], agent1_vel_sensed[1], self.agent1_acc[1],
            target_pos_noisy[0], target_vel_noisy[0], self.target_acc[0],
            target_pos_noisy[1], target_vel_noisy[1], self.target_acc[1],
            F_interaction_noisy[0], F_interaction_noisy[1]
        ])

        self.agent2_partner_obs = torch.stack([
            agent2_pos_sensed[0], agent2_vel_sensed[0], self.agent2_acc[0],
            agent2_pos_sensed[1], agent2_vel_sensed[1], self.agent2_acc[1],
            target_pos_noisy[0], target_vel_noisy[0], self.target_acc[0],
            target_pos_noisy[1], target_vel_noisy[1], self.target_acc[1],
            -F_interaction_noisy[0], -F_interaction_noisy[1]
        ])

        # パートナー推定
        agent1_partner_target_pos = self.agent1_partner_estimation.step(self.agent1_partner_obs)
        agent2_partner_target_pos = self.agent2_partner_estimation.step(self.agent2_partner_obs)

        # 自己観測の更新
        self.agent1_self_obs[0] = agent1_pos_sensed[0] - target_pos_noisy[0]
        self.agent1_self_obs[1] = agent1_pos_sensed[0] - agent1_partner_target_pos[0]
        self.agent1_self_obs[2] = agent1_pos_sensed[1] - target_pos_noisy[1]
        self.agent1_self_obs[3] = agent1_pos_sensed[1] - agent1_partner_target_pos[1]
        
        self.agent2_self_obs[0] = agent2_pos_sensed[0] - target_pos_noisy[0]
        self.agent2_self_obs[1] = agent2_pos_sensed[0] - agent2_partner_target_pos[0]
        self.agent2_self_obs[2] = agent2_pos_sensed[1] - target_pos_noisy[1]
        self.agent2_self_obs[3] = agent2_pos_sensed[1] - agent2_partner_target_pos[1]

        # 実際の状態量の更新（真の値を使用）
        self.agent1_state = torch.cat([
            self.agent1_pos - self.target_pos,
            self.agent1_vel - self.target_vel,
            self.agent1_acc - self.target_acc
        ])

        self.agent2_state = torch.cat([
            self.agent2_pos - self.target_pos,
            self.agent2_vel - self.target_vel,
            self.agent2_acc - self.target_acc
        ])

        # 前回の制御入力を保存
        self.agent1_control_prev = self.agent1_control.clone()
        self.agent2_control_prev = self.agent2_control.clone()

        # 履歴バッファを更新
        self.update_history_buffer(agent_id=1)
        self.update_history_buffer(agent_id=2)

        # ===== Agent1のFF制御の計算 =====
        agent1_state_input = None
        
        if self.use_trained_model and self.agent1_trained_model is not None:
            agent1_predicted = self.predict_with_lstm(agent_id=1)
            if agent1_predicted is not None:
                self.agent1_state_pred = agent1_predicted
                
                # Inverse modelへの入力形状を整える
                agent1_state_input = self.agent1_state_pred
                if agent1_state_input.dim() == 1:
                    agent1_state_input = agent1_state_input.unsqueeze(0).unsqueeze(0)
                elif agent1_state_input.dim() == 2:
                    agent1_state_input = agent1_state_input.unsqueeze(0)
                
                # Inverse modelでFF制御を計算（勾配あり）
                agent1_ff_control, self.agent1_hidden = self.agent1_inverse_model(
                    agent1_state_input, hidden=self.agent1_hidden
                )
                self.agent1_current_ff_control = agent1_ff_control.squeeze()
            else:
                self.agent1_current_ff_control = torch.zeros(2, device=device, requires_grad=True)
        else:
            self.agent1_current_ff_control = torch.zeros(2, device=device, requires_grad=True)

        # ===== Agent2のFF制御の計算 =====
        agent2_state_input = None
        
        if self.use_trained_model and self.agent2_trained_model is not None:
            agent2_predicted = self.predict_with_lstm(agent_id=2)
            if agent2_predicted is not None:
                self.agent2_state_pred = agent2_predicted
                
                # Inverse modelへの入力形状を整える
                agent2_state_input = self.agent2_state_pred
                if agent2_state_input.dim() == 1:
                    agent2_state_input = agent2_state_input.unsqueeze(0).unsqueeze(0)
                elif agent2_state_input.dim() == 2:
                    agent2_state_input = agent2_state_input.unsqueeze(0)
                
                # Inverse modelでFF制御を計算（勾配あり）
                agent2_ff_control, self.agent2_hidden = self.agent2_inverse_model(
                    agent2_state_input, hidden=self.agent2_hidden
                )
                self.agent2_current_ff_control = agent2_ff_control.squeeze()
            else:
                self.agent2_current_ff_control = torch.zeros(2, device=device, requires_grad=True)
        else:
            self.agent2_current_ff_control = torch.zeros(2, device=device, requires_grad=True)

        # ===== Agent1の損失計算 =====
        if (self.use_trained_model and self.step_count > self.sequence_length and 
            agent1_state_input is not None and self.agent1_trained_model is not None):
            next_time = self.time + self.dt
            next_target_pos, next_target_vel, next_target_acc = self.target_traj(next_time)
            
            # 物理変数をdetach()して独立させる（重要！）
            agent1_pos_detached = self.agent1_pos.detach()
            agent1_vel_detached = self.agent1_vel.detach()
            agent1_acc_detached = self.agent1_acc.detach()
            F_interaction_detached = self.F_interaction.detach()
            
            # Agent1: current_ff_controlを含む次時刻の状態を予測
            temp_agent1_pos = agent1_pos_detached + agent1_vel_detached * self.dt + agent1_acc_detached * self.dt**2 / 2
            temp_agent1_vel = agent1_vel_detached + agent1_acc_detached * self.dt
            # FF制御のみが勾配を持つ
            temp_agent1_acc = (F_interaction_detached + self.agent1_current_ff_control) / self.m
            
            temp_agent1_state = torch.cat([
                temp_agent1_pos - next_target_pos,
                temp_agent1_vel - next_target_vel,
                temp_agent1_acc - next_target_acc
            ])
            
            # 次時刻のFB制御を計算（勾配を保持）
            kp1 = 10
            kd1 = 5
            ka1 = 0
            pos_error1 = temp_agent1_state[0:2]
            vel_error1 = temp_agent1_state[2:4]
            acc_error1 = temp_agent1_state[4:6]
            
            agent1_fb_control_for_loss = - kp1*pos_error1 - kd1*vel_error1 - ka1*acc_error1
            
            # 損失計算（FB制御をゼロに近づける）
            target_zero = torch.zeros_like(agent1_fb_control_for_loss)
            agent1_loss = self.criterion(agent1_fb_control_for_loss, target_zero)
            
            # 損失を記録（勾配グラフを保持）
            self.agent1_episode_losses.append(agent1_loss)
            
            # デバッグ情報の記録
            self.agent1_episode_states.append(agent1_state_input.detach())
            self.agent1_episode_fb_controls.append(agent1_fb_control_for_loss.detach())

        # ===== Agent2の損失計算 =====
        if (self.use_trained_model and self.step_count > self.sequence_length and 
            agent2_state_input is not None and self.agent2_trained_model is not None):
            next_time = self.time + self.dt
            next_target_pos, next_target_vel, next_target_acc = self.target_traj(next_time)
            
            # 物理変数をdetach()して独立させる（重要！）
            agent2_pos_detached = self.agent2_pos.detach()
            agent2_vel_detached = self.agent2_vel.detach()
            agent2_acc_detached = self.agent2_acc.detach()
            F_interaction_detached = self.F_interaction.detach()
            
            # Agent2: current_ff_controlを含む次時刻の状態を予測
            temp_agent2_pos = agent2_pos_detached + agent2_vel_detached * self.dt + agent2_acc_detached * self.dt**2 / 2
            temp_agent2_vel = agent2_vel_detached + agent2_acc_detached * self.dt
            # FF制御のみが勾配を持つ
            temp_agent2_acc = (-F_interaction_detached + self.agent2_current_ff_control) / self.m
            
            temp_agent2_state = torch.cat([
                temp_agent2_pos - next_target_pos,
                temp_agent2_vel - next_target_vel,
                temp_agent2_acc - next_target_acc
            ])
            
            # 次時刻のFB制御を計算（勾配を保持）
            kp2 = 10  # 20 → 10に変更
            kd2 = 5   # 10 → 5に変更
            ka2 = 0
            pos_error2 = temp_agent2_state[0:2]
            vel_error2 = temp_agent2_state[2:4]
            acc_error2 = temp_agent2_state[4:6]
            
            agent2_fb_control_for_loss = - kp2*pos_error2 - kd2*vel_error2 - ka2*acc_error2
            
            # 損失計算（FB制御をゼロに近づける）
            target_zero = torch.zeros_like(agent2_fb_control_for_loss)
            agent2_loss = self.criterion(agent2_fb_control_for_loss, target_zero)
            
            # 損失を記録（勾配グラフを保持）
            self.agent2_episode_losses.append(agent2_loss)
            
            # デバッグ情報の記録
            self.agent2_episode_states.append(agent2_state_input.detach())
            self.agent2_episode_fb_controls.append(agent2_fb_control_for_loss.detach())

        # ===== 実際のFB制御（物理シミュレーション用） =====
        self.Agent1_FBController(self.agent1_state)
        self.Agent2_FBController(self.agent2_state)
        
        agent1_actual_fb_control = self.agent1_control.clone()
        agent2_actual_fb_control = self.agent2_control.clone()
        
        # 物理シミュレーションではdetachして使用
        agent1_total_control = agent1_actual_fb_control + self.agent1_current_ff_control
        agent2_total_control = agent2_actual_fb_control + self.agent2_current_ff_control
        
        self.agent1_control = agent1_total_control
        self.agent2_control = agent2_total_control
        
        # 記録
        self.agent1_fb_control_history.append(agent1_actual_fb_control.detach().cpu().numpy().copy())
        self.agent1_ff_control_history.append(self.agent1_current_ff_control.detach().cpu().numpy().copy())
        self.agent1_total_control_history.append(agent1_total_control.detach().cpu().numpy().copy())
        
        self.agent2_fb_control_history.append(agent2_actual_fb_control.detach().cpu().numpy().copy())
        self.agent2_ff_control_history.append(self.agent2_current_ff_control.detach().cpu().numpy().copy())
        self.agent2_total_control_history.append(agent2_total_control.detach().cpu().numpy().copy())

        # ===== 物理シミュレーション =====
        self.agent1_pos += self.agent1_vel * self.dt + self.agent1_acc * self.dt**2 / 2
        self.agent1_vel += self.agent1_acc * self.dt
        self.agent1_acc = (self.F_interaction + self.agent1_control) / self.m
        
        self.agent2_pos += self.agent2_vel * self.dt + self.agent2_acc * self.dt**2 / 2
        self.agent2_vel += self.agent2_acc * self.dt
        self.agent2_acc = (-self.F_interaction + self.agent2_control) / self.m

        # 状態量の更新
        self.agent1_state = torch.cat([
            self.agent1_pos - self.target_pos,
            self.agent1_vel - self.target_vel,
            self.agent1_acc - self.target_acc
        ])

        self.agent2_state = torch.cat([
            self.agent2_pos - self.target_pos,
            self.agent2_vel - self.target_vel,
            self.agent2_acc - self.target_acc
        ])

        # 軌跡の記録
        self.trajectory_history['target'].append([self.target_pos[0].item(), self.target_pos[1].item()])
        self.trajectory_history['agent1'].append([self.agent1_pos[0].item(), self.agent1_pos[1].item()])
        self.trajectory_history['agent2'].append([self.agent2_pos[0].item(), self.agent2_pos[1].item()])
        self.trajectory_history['time'].append(self.time)

        self.time += self.dt
        self.step_count += 1

    def reset_episode(self):
        """エピソード開始時の物理状態のみリセット（学習モデルは保持）"""
        print("  Resetting episode (preserving learned weights)...")
        
        # 時間とステップカウンターのリセット
        self.time = 0
        self.step_count = 0
        self.agent1_hidden = None
        self.agent2_hidden = None
        
        # 物理状態のリセット
        self.target_pos = torch.zeros(2, device=device)
        self.target_vel = torch.zeros(2, device=device)
        self.target_acc = torch.zeros(2, device=device)
        
        # Agent1の物理状態リセット
        self.agent1_pos = torch.tensor([0.02, 0.02], device=device)
        self.agent1_vel = torch.zeros(2, device=device)
        self.agent1_acc = torch.zeros(2, device=device)
        self.agent1_state = torch.zeros(6, device=device)
        self.agent1_state_pred = torch.zeros(1, 1, 6, device=device)
        self.agent1_control = torch.zeros(2, device=device)
        self.agent1_control_prev = torch.zeros(2, device=device)
        self.agent1_force = torch.zeros(2, device=device)
        
        # Agent2の物理状態リセット
        self.agent2_pos = torch.tensor([-0.02, -0.02], device=device)
        self.agent2_vel = torch.zeros(2, device=device)
        self.agent2_acc = torch.zeros(2, device=device)
        self.agent2_state = torch.zeros(6, device=device)
        self.agent2_state_pred = torch.zeros(1, 1, 6, device=device)
        self.agent2_control = torch.zeros(2, device=device)
        self.agent2_control_prev = torch.zeros(2, device=device)
        self.agent2_force = torch.zeros(2, device=device)
        
        # 相互作用力のリセット
        self.F_interaction = torch.zeros(2, device=device)
        
        # 観測値のリセット
        n_obs_partner = 14
        n_obs_self = 4
        self.agent1_partner_obs = torch.zeros(n_obs_partner, device=device)
        self.agent2_partner_obs = torch.zeros(n_obs_partner, device=device)
        self.agent1_self_obs = torch.zeros(n_obs_self, device=device)
        self.agent2_self_obs = torch.zeros(n_obs_self, device=device)
        
        # バッファのクリア
        self.agent1_control_buffer.clear()
        self.agent2_control_buffer.clear()
        self.agent1_history_buffer.clear()
        self.agent2_history_buffer.clear()
        self.agent1_state_history.clear()
        self.agent2_state_history.clear()
        
        # 軌跡履歴のクリア
        self.trajectory_history = {
            'target': [],
            'agent1': [],
            'agent2': [],
            'time': []
        }
        
        # エピソード用学習データのクリア
        self.agent1_episode_losses = []
        self.agent2_episode_losses = []
        
        self.agent1_episode_states = []
        self.agent2_episode_states = []
        
        self.agent1_episode_fb_controls = []
        self.agent2_episode_fb_controls = []

        # 制御履歴のリセット
        self.agent1_prev_fb_control = torch.zeros(2, device=device)
        self.agent1_current_ff_control = torch.zeros(2, device=device)
        
        self.agent2_prev_fb_control = torch.zeros(2, device=device)
        self.agent2_current_ff_control = torch.zeros(2, device=device)

        self.agent1_fb_control_history = []
        self.agent1_ff_control_history = []
        self.agent1_total_control_history = []
        
        self.agent2_fb_control_history = []
        self.agent2_ff_control_history = []
        self.agent2_total_control_history = []
        
        # パートナー推定器の状態をリセット
        try:
            if hasattr(self.agent1_partner_estimation, 'reset'):
                self.agent1_partner_estimation.reset()
            if hasattr(self.agent2_partner_estimation, 'reset'):
                self.agent2_partner_estimation.reset()
            if hasattr(self.agent1_self_estimation, 'reset'):
                self.agent1_self_estimation.reset()
            if hasattr(self.agent2_self_estimation, 'reset'):
                self.agent2_self_estimation.reset()
        except Exception as e:
            print(f"  Warning: Could not reset estimation modules: {e}")
        
        print("  Episode reset completed (learning models preserved)")

    def update_inverse_models(self):
        """
        エピソード終了時に両方のInverse Modelを更新
        合計損失アプローチ：メモリ効率的でinplace操作の問題を回避
        
        Returns:
            dict: 各エージェントの平均損失値
        """
        results = {}
        
        has_agent1_loss = len(self.agent1_episode_losses) > 0
        has_agent2_loss = len(self.agent2_episode_losses) > 0
        
        # ===== 両方のエージェントに損失がある場合：合計損失で学習 =====
        if has_agent1_loss and has_agent2_loss:
            total_loss_agent1 = torch.stack(self.agent1_episode_losses).mean()
            total_loss_agent2 = torch.stack(self.agent2_episode_losses).mean()
            
            print(f"  Optimizing both inverse models simultaneously...")
            print(f"    Agent1: {len(self.agent1_episode_losses)} losses, avg = {total_loss_agent1.item():.6f}")
            print(f"    Agent2: {len(self.agent2_episode_losses)} losses, avg = {total_loss_agent2.item():.6f}")
            
            # 合計損失を計算
            combined_loss = total_loss_agent1 + total_loss_agent2
            
            # 両方のoptimizerをゼロに
            self.agent1_optimizer.zero_grad()
            self.agent2_optimizer.zero_grad()
            
            # 一度だけbackward（retain_graph不要）
            combined_loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.agent1_inverse_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.agent2_inverse_model.parameters(), max_norm=1.0)
            
            # 両方のoptimizerでステップ
            self.agent1_optimizer.step()
            self.agent2_optimizer.step()
            
            # 統計を記録
            avg_loss_agent1 = total_loss_agent1.item()
            avg_loss_agent2 = total_loss_agent2.item()
            
            # FB制御の統計
            fb1_magnitudes = [torch.norm(fb).item() for fb in self.agent1_episode_fb_controls]
            avg_fb1 = np.mean(fb1_magnitudes) if fb1_magnitudes else 0.0
            
            fb2_magnitudes = [torch.norm(fb).item() for fb in self.agent2_episode_fb_controls]
            avg_fb2 = np.mean(fb2_magnitudes) if fb2_magnitudes else 0.0
            
            print(f"  ✓ Both models updated:")
            print(f"    Agent1: loss = {avg_loss_agent1:.6f}, FB mag = {avg_fb1:.6f}")
            print(f"    Agent2: loss = {avg_loss_agent2:.6f}, FB mag = {avg_fb2:.6f}")
            
            results['agent1'] = avg_loss_agent1
            results['agent2'] = avg_loss_agent2
        
        # ===== Agent1のみ損失がある場合 =====
        elif has_agent1_loss:
            total_loss_agent1 = torch.stack(self.agent1_episode_losses).mean()
            
            print(f"  Optimizing Agent1 inverse model only... (collected {len(self.agent1_episode_losses)} losses)")
            
            self.agent1_optimizer.zero_grad()
            total_loss_agent1.backward()
            torch.nn.utils.clip_grad_norm_(self.agent1_inverse_model.parameters(), max_norm=1.0)
            self.agent1_optimizer.step()
            
            avg_loss_agent1 = total_loss_agent1.item()
            
            fb_control_magnitudes = [torch.norm(fb).item() for fb in self.agent1_episode_fb_controls]
            avg_fb_magnitude = np.mean(fb_control_magnitudes) if fb_control_magnitudes else 0.0
            
            print(f"  ✓ Agent1 inverse model updated:")
            print(f"    - Avg loss: {avg_loss_agent1:.6f}")
            print(f"    - Avg FB control magnitude: {avg_fb_magnitude:.6f}")
            
            results['agent1'] = avg_loss_agent1
            results['agent2'] = None
        
        # ===== Agent2のみ損失がある場合 =====
        elif has_agent2_loss:
            total_loss_agent2 = torch.stack(self.agent2_episode_losses).mean()
            
            print(f"  Optimizing Agent2 inverse model only... (collected {len(self.agent2_episode_losses)} losses)")
            
            self.agent2_optimizer.zero_grad()
            total_loss_agent2.backward()
            torch.nn.utils.clip_grad_norm_(self.agent2_inverse_model.parameters(), max_norm=1.0)
            self.agent2_optimizer.step()
            
            avg_loss_agent2 = total_loss_agent2.item()
            
            fb_control_magnitudes = [torch.norm(fb).item() for fb in self.agent2_episode_fb_controls]
            avg_fb_magnitude = np.mean(fb_control_magnitudes) if fb_control_magnitudes else 0.0
            
            print(f"  ✓ Agent2 inverse model updated:")
            print(f"    - Avg loss: {avg_loss_agent2:.6f}")
            print(f"    - Avg FB control magnitude: {avg_fb_magnitude:.6f}")
            
            results['agent1'] = None
            results['agent2'] = avg_loss_agent2
        
        # ===== どちらも損失がない場合 =====
        else:
            print("  No losses to optimize for either agent")
            results['agent1'] = None
            results['agent2'] = None
        
        # 次のエピソードのために損失リストをクリア
        self.agent1_episode_losses = []
        self.agent2_episode_losses = []
        
        return results

    def get_episode_statistics(self):
        """
        現在のエピソードの両エージェントの統計情報を取得
        
        Returns:
            dict: 各エージェントの統計情報の辞書
        """
        stats = {}
        
        # Agent1の統計
        if len(self.agent1_fb_control_history) > 0:
            fb_array = np.array(self.agent1_fb_control_history)
            ff_array = np.array(self.agent1_ff_control_history)
            total_array = np.array(self.agent1_total_control_history)
            
            fb_magnitudes = np.linalg.norm(fb_array, axis=1)
            ff_magnitudes = np.linalg.norm(ff_array, axis=1)
            total_magnitudes = np.linalg.norm(total_array, axis=1)
            
            avg_fb = np.mean(fb_magnitudes)
            avg_ff = np.mean(ff_magnitudes)
            avg_total = np.mean(total_magnitudes)
            fb_ratio = avg_fb / avg_total if avg_total > 0 else 0.0
            
            stats['agent1'] = {
                'avg_fb_magnitude': avg_fb,
                'avg_ff_magnitude': avg_ff,
                'avg_total_magnitude': avg_total,
                'fb_to_total_ratio': fb_ratio,
                'num_steps': len(self.agent1_fb_control_history)
            }
        else:
            stats['agent1'] = {
                'avg_fb_magnitude': 0.0,
                'avg_ff_magnitude': 0.0,
                'avg_total_magnitude': 0.0,
                'fb_to_total_ratio': 0.0,
                'num_steps': 0
            }
        
        # Agent2の統計
        if len(self.agent2_fb_control_history) > 0:
            fb_array = np.array(self.agent2_fb_control_history)
            ff_array = np.array(self.agent2_ff_control_history)
            total_array = np.array(self.agent2_total_control_history)
            
            fb_magnitudes = np.linalg.norm(fb_array, axis=1)
            ff_magnitudes = np.linalg.norm(ff_array, axis=1)
            total_magnitudes = np.linalg.norm(total_array, axis=1)
            
            avg_fb = np.mean(fb_magnitudes)
            avg_ff = np.mean(ff_magnitudes)
            avg_total = np.mean(total_magnitudes)
            fb_ratio = avg_fb / avg_total if avg_total > 0 else 0.0
            
            stats['agent2'] = {
                'avg_fb_magnitude': avg_fb,
                'avg_ff_magnitude': avg_ff,
                'avg_total_magnitude': avg_total,
                'fb_to_total_ratio': fb_ratio,
                'num_steps': len(self.agent2_fb_control_history)
            }
        else:
            stats['agent2'] = {
                'avg_fb_magnitude': 0.0,
                'avg_ff_magnitude': 0.0,
                'avg_total_magnitude': 0.0,
                'fb_to_total_ratio': 0.0,
                'num_steps': 0
            }
        
        return stats


def run_simulation(env, num_steps=2000, num_episodes=20):
    """両エージェントの学習機能付きシミュレーション実行"""
    
    all_episode_agent1_tracking_errors = []
    all_episode_agent2_tracking_errors = []
    all_episode_agent1_losses = []
    all_episode_agent2_losses = []
    
    print(f"Starting training with {num_episodes} episodes, {num_steps} steps each")
    print("Both Agent1 and Agent2 will learn simultaneously")
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"=== Episode {episode + 1}/{num_episodes} ===")
        print(f"{'='*60}")
        
        if episode == 0:
            print("  First episode - initializing")
            env.agent1_pos = torch.tensor([0.02, 0.02], device=device)
            env.agent2_pos = torch.tensor([-0.02, -0.02], device=device)
            
            # 学習用変数の初期化
            env.agent1_episode_losses = []
            env.agent2_episode_losses = []
            env.agent1_episode_states = []
            env.agent2_episode_states = []
            env.agent1_episode_fb_controls = []
            env.agent2_episode_fb_controls = []
        else:
            env.reset_episode()
        
        # このエピソードの追跡誤差を記録
        episode_agent1_tracking_errors = []
        episode_agent2_tracking_errors = []
        
        print(f"  Running simulation loop...")
        
        # シミュレーションループ
        for i in range(num_steps):
            env.step()
            
            # 追跡誤差を記録
            agent1_tracking_error = torch.norm(env.agent1_pos - env.target_pos).item()
            agent2_tracking_error = torch.norm(env.agent2_pos - env.target_pos).item()
            
            episode_agent1_tracking_errors.append(agent1_tracking_error)
            episode_agent2_tracking_errors.append(agent2_tracking_error)
            
            # 進行状況表示
            if (i + 1) % 500 == 0:
                agent1_loss = env.agent1_episode_losses[-1].item() if len(env.agent1_episode_losses) > 0 else 0.0
                agent2_loss = env.agent2_episode_losses[-1].item() if len(env.agent2_episode_losses) > 0 else 0.0
                print(f"  Step {i+1:4d}/{num_steps}:")
                print(f"    Agent1: Error = {agent1_tracking_error:.6f}, Loss = {agent1_loss:.8f}")
                print(f"    Agent2: Error = {agent2_tracking_error:.6f}, Loss = {agent2_loss:.8f}")
        
        # 両エージェントの逆モデルを更新
        loss_results = env.update_inverse_models()
        
        # エピソード統計
        episode_avg_agent1_error = np.mean(episode_agent1_tracking_errors)
        episode_avg_agent2_error = np.mean(episode_agent2_tracking_errors)
        episode_final_agent1_error = episode_agent1_tracking_errors[-1]
        episode_final_agent2_error = episode_agent2_tracking_errors[-1]
        
        # 結果を記録
        all_episode_agent1_tracking_errors.append({
            'episode': episode,
            'time_series': episode_agent1_tracking_errors,
            'avg_error': episode_avg_agent1_error,
            'final_error': episode_final_agent1_error
        })
        all_episode_agent2_tracking_errors.append({
            'episode': episode,
            'time_series': episode_agent2_tracking_errors,
            'avg_error': episode_avg_agent2_error,
            'final_error': episode_final_agent2_error
        })
        
        all_episode_agent1_losses.append(loss_results.get('agent1', 0.0) or 0.0)
        all_episode_agent2_losses.append(loss_results.get('agent2', 0.0) or 0.0)
        
        # エピソード結果の表示
        print(f"\n--- Episode {episode+1} Results ---")
        print(f"  Agent1:")
        print(f"    Average Tracking Error: {episode_avg_agent1_error:.6f}")
        print(f"    Final Tracking Error:   {episode_final_agent1_error:.6f}")
        print(f"    Average FB Loss:        {all_episode_agent1_losses[-1]:.8f}")
        print(f"  Agent2:")
        print(f"    Average Tracking Error: {episode_avg_agent2_error:.6f}")
        print(f"    Final Tracking Error:   {episode_final_agent2_error:.6f}")
        print(f"    Average FB Loss:        {all_episode_agent2_losses[-1]:.8f}")
        
        # 改善度の表示
        if episode > 0:
            prev_avg_error1 = all_episode_agent1_tracking_errors[episode-1]['avg_error']
            error_improvement1 = ((prev_avg_error1 - episode_avg_agent1_error) / prev_avg_error1) * 100
            
            prev_avg_error2 = all_episode_agent2_tracking_errors[episode-1]['avg_error']
            error_improvement2 = ((prev_avg_error2 - episode_avg_agent2_error) / prev_avg_error2) * 100
            
            print(f"  Agent1 error improvement: {error_improvement1:+.2f}%")
            print(f"  Agent2 error improvement: {error_improvement2:+.2f}%")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    
    # 最終統計
    if len(all_episode_agent1_tracking_errors) > 1:
        # Agent1
        initial_error1 = all_episode_agent1_tracking_errors[0]['avg_error']
        final_error1 = all_episode_agent1_tracking_errors[-1]['avg_error']
        total_improvement1 = ((initial_error1 - final_error1) / initial_error1) * 100
        
        # Agent2
        initial_error2 = all_episode_agent2_tracking_errors[0]['avg_error']
        final_error2 = all_episode_agent2_tracking_errors[-1]['avg_error']
        total_improvement2 = ((initial_error2 - final_error2) / initial_error2) * 100
        
        print(f"\nOverall Performance:")
        print(f"  Agent1:")
        print(f"    Initial Error: {initial_error1:.6f} → Final Error: {final_error1:.6f}")
        print(f"    Total Improvement: {total_improvement1:.2f}%")
        print(f"  Agent2:")
        print(f"    Initial Error: {initial_error2:.6f} → Final Error: {final_error2:.6f}")
        print(f"    Total Improvement: {total_improvement2:.2f}%")
    
    # プロット
    plot_dual_learning_summary(
        all_episode_agent1_tracking_errors, all_episode_agent1_losses,
        all_episode_agent2_tracking_errors, all_episode_agent2_losses
    )
    plot_final_episode_dual(env, all_episode_agent1_tracking_errors[-1], 
                           all_episode_agent2_tracking_errors[-1])
    
    return (env.trajectory_history, 
            all_episode_agent1_tracking_errors, all_episode_agent1_losses,
            all_episode_agent2_tracking_errors, all_episode_agent2_losses)


def plot_dual_learning_summary(agent1_tracking_errors, agent1_losses,
                               agent2_tracking_errors, agent2_losses):
    """両エージェントの学習進捗を示すプロット"""
    
    episodes = list(range(len(agent1_tracking_errors)))
    
    agent1_avg_errors = [data['avg_error'] for data in agent1_tracking_errors]
    agent1_final_errors = [data['final_error'] for data in agent1_tracking_errors]
    
    agent2_avg_errors = [data['avg_error'] for data in agent2_tracking_errors]
    agent2_final_errors = [data['final_error'] for data in agent2_tracking_errors]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Agent1の追跡誤差
    axes[0, 0].plot(episodes, agent1_avg_errors, 'b-o', linewidth=2, markersize=5, label='Average Error')
    axes[0, 0].plot(episodes, agent1_final_errors, 'r--s', linewidth=1, markersize=3, label='Final Error')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Tracking Error')
    axes[0, 0].set_title('Agent1 - Tracking Error Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Agent2の追跡誤差
    axes[0, 1].plot(episodes, agent2_avg_errors, 'b-o', linewidth=2, markersize=5, label='Average Error')
    axes[0, 1].plot(episodes, agent2_final_errors, 'r--s', linewidth=1, markersize=3, label='Final Error')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Tracking Error')
    axes[0, 1].set_title('Agent2 - Tracking Error Progress')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Agent1の学習損失
    axes[1, 0].plot(episodes, agent1_losses, 'g-o', linewidth=2, markersize=5)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('FB Control Loss')
    axes[1, 0].set_title('Agent1 - Learning Loss Progress')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # 4. Agent2の学習損失
    axes[1, 1].plot(episodes, agent2_losses, 'g-o', linewidth=2, markersize=5)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('FB Control Loss')
    axes[1, 1].set_title('Agent2 - Learning Loss Progress')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('dual_learning_progress.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Dual agent learning progress saved: dual_learning_progress.png")
    plt.show()


def plot_final_episode_dual(env, agent1_final_data, agent2_final_data):
    """最終エピソードの両エージェントの結果プロット"""
    
    trajectory_history = env.trajectory_history
    agent1_time_series_error = agent1_final_data['time_series']
    agent2_time_series_error = agent2_final_data['time_series']
    
    target_traj = np.array(trajectory_history['target'])
    agent1_traj = np.array(trajectory_history['agent1'])
    agent2_traj = np.array(trajectory_history['agent2'])
    time_array = np.array(trajectory_history['time'])
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 軌跡プロット
    plt.subplot(3, 2, 1)
    plt.plot(target_traj[:, 0], target_traj[:, 1], 'r-', linewidth=3, label='Target', alpha=0.9)
    plt.plot(agent1_traj[:, 0], agent1_traj[:, 1], 'b-', linewidth=2, label='Agent 1', alpha=0.8)
    plt.plot(agent2_traj[:, 0], agent2_traj[:, 1], 'g-', linewidth=2, label='Agent 2', alpha=0.8)
    plt.plot(target_traj[0, 0], target_traj[0, 1], 'ko', markersize=10, label='Start')
    plt.plot(target_traj[-1, 0], target_traj[-1, 1], 'ks', markersize=10, label='End')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Final Episode - Trajectories')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 2. Agent1追跡誤差
    plt.subplot(3, 2, 2)
    plt.plot(time_array, agent1_time_series_error, 'b-', linewidth=2)
    plt.axhline(y=agent1_final_data['avg_error'], color='r', linestyle='--', 
                label=f"Avg: {agent1_final_data['avg_error']:.6f}")
    plt.xlabel('Time (s)')
    plt.ylabel('Tracking Error')
    plt.title('Agent1 - Tracking Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Agent2追跡誤差
    plt.subplot(3, 2, 3)
    plt.plot(time_array, agent2_time_series_error, 'g-', linewidth=2)
    plt.axhline(y=agent2_final_data['avg_error'], color='r', linestyle='--', 
                label=f"Avg: {agent2_final_data['avg_error']:.6f}")
    plt.xlabel('Time (s)')
    plt.ylabel('Tracking Error')
    plt.title('Agent2 - Tracking Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Agent1制御入力
    plt.subplot(3, 2, 4)
    agent1_fb_history = np.array(env.agent1_fb_control_history)
    agent1_ff_history = np.array(env.agent1_ff_control_history)
    agent1_fb_norms = np.linalg.norm(agent1_fb_history, axis=1)
    agent1_ff_norms = np.linalg.norm(agent1_ff_history, axis=1)
    
    plt.plot(time_array, agent1_fb_norms, 'r-', linewidth=2, label='FB Control', alpha=0.8)
    plt.plot(time_array, agent1_ff_norms, 'b-', linewidth=2, label='FF Control', alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Control Magnitude')
    plt.title('Agent1 - Control Magnitudes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Agent2制御入力
    plt.subplot(3, 2, 5)
    agent2_fb_history = np.array(env.agent2_fb_control_history)
    agent2_ff_history = np.array(env.agent2_ff_control_history)
    agent2_fb_norms = np.linalg.norm(agent2_fb_history, axis=1)
    agent2_ff_norms = np.linalg.norm(agent2_ff_history, axis=1)
    
    plt.plot(time_array, agent2_fb_norms, 'r-', linewidth=2, label='FB Control', alpha=0.8)
    plt.plot(time_array, agent2_ff_norms, 'g-', linewidth=2, label='FF Control', alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Control Magnitude')
    plt.title('Agent2 - Control Magnitudes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. 統計情報
    plt.subplot(3, 2, 6)
    plt.axis('off')
    
    agent1_avg_fb = np.mean(agent1_fb_norms)
    agent1_avg_ff = np.mean(agent1_ff_norms)
    agent1_fb_ratio = agent1_avg_fb / (agent1_avg_fb + agent1_avg_ff) if (agent1_avg_fb + agent1_avg_ff) > 0 else 0
    
    agent2_avg_fb = np.mean(agent2_fb_norms)
    agent2_avg_ff = np.mean(agent2_ff_norms)
    agent2_fb_ratio = agent2_avg_fb / (agent2_avg_fb + agent2_avg_ff) if (agent2_avg_fb + agent2_avg_ff) > 0 else 0
    
    stats_text = f"""
    FINAL EPISODE STATISTICS
    
    Agent1:
    • Avg Error: {agent1_final_data['avg_error']:.6f} m
    • Final Error: {agent1_final_data['final_error']:.6f} m
    • Avg FB mag: {agent1_avg_fb:.4f}
    • Avg FF mag: {agent1_avg_ff:.4f}
    • FB ratio: {agent1_fb_ratio:.3f}
    
    Agent2:
    • Avg Error: {agent2_final_data['avg_error']:.6f} m
    • Final Error: {agent2_final_data['final_error']:.6f} m
    • Avg FB mag: {agent2_avg_fb:.4f}
    • Avg FF mag: {agent2_avg_ff:.4f}
    • FB ratio: {agent2_fb_ratio:.3f}
    
    Duration: {time_array[-1]:.1f} s
    Steps: {len(time_array)}
    """
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('dual_final_episode_results.png', dpi=300, bbox_inches='tight')
    print(f"✓ Dual agent final episode results saved: dual_final_episode_results.png")
    plt.show()


# ===== メイン実行部分 =====
if __name__ == "__main__":
    # 各エージェント用のモデルディレクトリを指定
    model_dir_agent1 = "/home/hino/Desktop/SICE_SI/dual_noise_tracking_data/Agent1_dual_model_seed42"
    model_dir_agent2 = "/home/hino/Desktop/SICE_SI/dual_noise_tracking_data/Agent2_dual_model_seed42"
    
    print("="*80)
    print("DUAL TRACKING SIMULATION WITH DUAL AGENT FEEDFORWARD LEARNING")
    print("="*80)
    
    # 環境の初期化
    print("\nInitializing environment...")
    env = DualTrackingEnv(
        dt=0.01, 
        use_trained_model=True, 
        model_dir_agent1=model_dir_agent1,
        model_dir_agent2=model_dir_agent2
    )
    
    # 初期位置の設定
    env.agent1_pos = torch.tensor([0.02, 0.02], device=device)
    env.agent2_pos = torch.tensor([-0.02, -0.02], device=device)
    
    print(f"✓ Environment initialized:")
    print(f"  Device: {device}")
    print(f"  Agent1 Forward model: {'Loaded' if env.agent1_trained_model else 'Not available'}")
    print(f"  Agent2 Forward model: {'Loaded' if env.agent2_trained_model else 'Not available'}")
    print(f"  Agent1 Inverse model: Ready for training")
    print(f"  Agent2 Inverse model: Ready for training")
    
    # シミュレーション実行
    print("\nStarting simulation with dual agent learning...")
    num_episodes = 15
    num_steps = 2000
    
    print(f"Configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Steps per episode: {num_steps}")
    print(f"  Agent1 learning rate: {env.agent1_optimizer.param_groups[0]['lr']:.2e}")
    print(f"  Agent2 learning rate: {env.agent2_optimizer.param_groups[0]['lr']:.2e}")
    
    try:
        (trajectory_history, 
         agent1_tracking_errors, agent1_losses,
         agent2_tracking_errors, agent2_losses) = run_simulation(
            env, num_steps=num_steps, num_episodes=num_episodes
        )
        
        print("\n" + "="*80)
        print("SIMULATION COMPLETED SUCCESSFULLY")
        print("="*80)
        
        # 最終結果サマリー
        if len(agent1_tracking_errors) > 1:
            # Agent1
            initial_error1 = agent1_tracking_errors[0]['avg_error']
            final_error1 = agent1_tracking_errors[-1]['avg_error']
            improvement1 = ((initial_error1 - final_error1) / initial_error1) * 100
            
            # Agent2
            initial_error2 = agent2_tracking_errors[0]['avg_error']
            final_error2 = agent2_tracking_errors[-1]['avg_error']
            improvement2 = ((initial_error2 - final_error2) / initial_error2) * 100
            
            print(f"\nFinal Results:")
            print(f"  Agent1:")
            print(f"    Initial tracking error: {initial_error1:.6f}")
            print(f"    Final tracking error:   {final_error1:.6f}")
            print(f"    Total improvement:      {improvement1:.2f}%")
            
            print(f"  Agent2:")
            print(f"    Initial tracking error: {initial_error2:.6f}")
            print(f"    Final tracking error:   {final_error2:.6f}")
            print(f"    Total improvement:      {improvement2:.2f}%")
            
            if improvement1 > 5.0 and improvement2 > 5.0:
                print(f"  → Both agents: Learning SUCCESS!")
            elif improvement1 > 1.0 or improvement2 > 1.0:
                print(f"  → Moderate improvement observed")
            else:
                print(f"  → Limited improvement - consider tuning hyperparameters")
        
        print(f"\nGenerated files:")
        print(f"  • dual_learning_progress.png - Dual agent learning curves")
        print(f"  • dual_final_episode_results.png - Final episode analysis for both agents")
        
    except KeyboardInterrupt:
        print(f"\n\nSimulation interrupted by user")
    except Exception as e:
        print(f"\n\nSimulation failed: {e}")
        import traceback
        traceback.print_exc()