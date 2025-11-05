import torch

class self_estimation:
    def __init__(self, dt, A, B, Q, R, H):
        self.dt = dt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 状態遷移行列 A (テンソルとして渡されることを想定)
        self.A = A
        # 制御入力行列 B
        self.B = B
        # ノイズ共分散行列
        self.Q = Q   # プロセスノイズ
        self.R = R   # 観測ノイズ
        # 観測行列 H（位置のみ観測）
        self.H = H

        self.n = H.shape[1]  # 状態ベクトルの次元
        self.m = H.shape[0]  # 観測ベクトルの次元

        # 状態と共分散行列を適切なデバイスとdtypeで初期化
        self.xhat = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self.P = 0.001*torch.eye(self.n, dtype=torch.float32, device=self.device)

        self.tau = torch.zeros(2, dtype=torch.float32, device=self.device)
        self.tau_pre = torch.zeros(2, dtype=torch.float32, device=self.device)

    def system_dynamics(self, tau):
        """制御入力＋外力 を用いて状態を予測"""
        # テンソル演算
        x_next = torch.matmul(self.A, self.xhat) + torch.matmul(self.B, tau.view(-1, 1)).squeeze()

        return x_next
    
    def step(self, z, tau):
        self.tau_pre = self.tau.clone()
        # 入力を適切なデバイスとdtypeに変換 (必要に応じて)
        if isinstance(z, torch.Tensor):
            z = z.to(dtype=self.xhat.dtype, device=self.xhat.device)
        else:
            z = torch.tensor(z, dtype=self.xhat.dtype, device=self.xhat.device)
        if isinstance(tau, torch.Tensor):
            tau = tau.to(dtype=self.tau.dtype, device=self.tau.device)
        else:
            tau = torch.tensor(tau, dtype=self.tau.dtype, device=self.tau.device)

        self.tau = tau

        x_odo = self.system_dynamics(self.tau - self.tau_pre)

        # 共分散予測 (転置には .T を使用、行列積には @ または torch.matmul を使用)
        P_odo = torch.matmul(torch.matmul(self.A, self.P), self.A.T) + self.Q
        I = torch.eye(self.n, dtype=self.xhat.dtype, device=self.device)

        # 更新
        S = torch.matmul(torch.matmul(self.H, P_odo), self.H.T) + self.R
        # 逆行列計算
        S_inv = torch.linalg.inv(S)
        K = torch.matmul(torch.matmul(P_odo, self.H.T), S_inv)
        # innovation計算
        innovation = z - torch.matmul(self.H, x_odo)

        self.xhat = x_odo + torch.matmul(K, innovation)
        self.P = torch.matmul((I - torch.matmul(K, self.H)), P_odo)

        # print("self estimation P", self.P)
        # print("self", self.xhat)

        # 推定された状態ベクトルをテンソルとして返す
        return self.xhat
