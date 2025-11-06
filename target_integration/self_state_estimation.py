import torch

class self_estimation:
    def __init__(self, dt, A, B, Q, R, H):
        self.dt = dt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # state transition matrix A
        self.A = A
        # control input matrix B
        self.B = B
        # noise covariance matrix
        self.Q = Q   # process noise
        self.R = R   # observation noise
        # observation matrix H
        self.H = H
        self.n = H.shape[1]  # dim of state vector
        self.m = H.shape[0]  # dim of observation vector
        self.xhat = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self.P = 0.001*torch.eye(self.n, dtype=torch.float32, device=self.device)
        self.tau = torch.zeros(2, dtype=torch.float32, device=self.device)
        self.tau_pre = torch.zeros(2, dtype=torch.float32, device=self.device)

    def system_dynamics(self, tau):
        # calculate tensor
        x_next = torch.matmul(self.A, self.xhat) + torch.matmul(self.B, tau.view(-1, 1)).squeeze()
        return x_next
    
    def step(self, z, tau):
        self.tau_pre = self.tau.clone()
        # convert input to appropriate device and dtype
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
        # predict covariance matrixd
        P_odo = torch.matmul(torch.matmul(self.A, self.P), self.A.T) + self.Q
        I = torch.eye(self.n, dtype=self.xhat.dtype, device=self.device)
        # update
        S = torch.matmul(torch.matmul(self.H, P_odo), self.H.T) + self.R
        # inverse matrix
        S_inv = torch.linalg.inv(S)
        K = torch.matmul(torch.matmul(P_odo, self.H.T), S_inv)
        # calculate innovation
        innovation = z - torch.matmul(self.H, x_odo)
        self.xhat = x_odo + torch.matmul(K, innovation)
        self.P = torch.matmul((I - torch.matmul(K, self.H)), P_odo)
        # return estimated state vector
        return self.xhat
