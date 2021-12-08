import numpy as np

import time


class RIS_MISO(object):
    def __init__(self, num_antennas, num_RIS_elements, num_users, AWGN_var=1e-2):
        self.M = num_antennas
        self.N = num_RIS_elements
        self.K = num_users

        assert self.M == self.K

        self.awgn_var = AWGN_var

        self.state_dim = 2 * self.K * (1 + self.K + self.M + self.N) + 2 * self.N * (1 + self.M)
        self.action_dim = 2 * self.M * self.K + 2 * self.N

        self.H_1 = None
        self.H_2 = None
        self.G = np.eye(self.M, dtype=complex)
        self.Phi = np.eye(self.N, dtype=complex)

        self.state = None
        self.done = None

        self.episode_t = None

    def reset(self):
        self.H_1 = np.random.rayleigh(1, (self.N, self.M)) + 1j * np.random.rayleigh(1, (self.N, self.M))
        self.H_2 = np.random.rayleigh(1, (self.N, self.K)) + 1j * np.random.rayleigh(1, (self.N, self.K))

        init_action_real = np.hstack((np.real(self.G.reshape(1, -1)), np.real(np.diag(self.Phi)).reshape(1, -1)))
        init_action_imag = np.hstack((np.imag(self.G.reshape(1, -1)), np.imag(np.diag(self.Phi)).reshape(1, -1)))
        init_action = np.hstack((init_action_real, init_action_imag))

        power_t_real = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2
        power_t_imag = np.imag(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2
        power_t = np.hstack((power_t_real, power_t_imag))

        H_2_tilde = self.H_2.T @ self.Phi @ self.H_1 @ self.G
        power_r_real = np.real(H_2_tilde).reshape(1, -1) ** 2
        power_r_imag = np.imag(H_2_tilde).reshape(1, -1) ** 2
        power_r = np.hstack((power_r_real, power_r_imag))

        H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
        H_2_real, H_2_imag = np.real(self.H_2).reshape(1, -1), np.imag(self.H_2).reshape(1, -1)

        self.state = np.hstack((init_action, power_t, power_r, H_1_real, H_1_imag, H_2_real, H_2_imag))

        return self.state

    def compute_reward(self):
        reward = 0
        opt_reward = 0

        for k in range(self.K):
            h_2_k = self.H_2[:, k].reshape(-1, 1)
            g_k = self.G[:, k].reshape(-1, 1)

            x = np.abs(h_2_k.T @ self.Phi @ self.H_1 @ g_k) ** 2
            x = x.item()

            G_removed = np.delete(self.G, k, axis=1)

            interference = np.sum(np.abs(h_2_k.T @ self.Phi @ self.H_1 @ G_removed) ** 2)
            y = interference + (self.K - 1) * self.awgn_var

            rho_k = x / y

            reward += np.log(1 + rho_k) / np.log(2)
            opt_reward += np.log(1 + x / ((self.K - 1) * self.awgn_var)) / np.log(2)

        return reward, opt_reward

    def step(self, action):
        G_real = action[0, :self.M ** 2]
        G_imag = action[0, self.M ** 2:2 * self.M ** 2]

        Phi_real = action[0, -2 * self.N:-self.N]
        Phi_imag = action[0, -self.N:]

        self.G = G_real.reshape(self.M, self.K) + 1j * G_imag.reshape(self.M, self.K)
        self.Phi = np.eye(self.N, dtype=complex) * (Phi_real + 1j * Phi_imag)

        power_t_real = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2
        power_t_imag = np.imag(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2
        power_t = np.hstack((power_t_real, power_t_imag))

        H_2_tilde = self.H_2.T @ self.Phi @ self.H_1 @ self.G
        power_r_real = np.real(H_2_tilde).reshape(1, -1) ** 2
        power_r_imag = np.imag(H_2_tilde).reshape(1, -1) ** 2
        power_r = np.hstack((power_r_real, power_r_imag))

        H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
        H_2_real, H_2_imag = np.real(self.H_2).reshape(1, -1), np.imag(self.H_2).reshape(1, -1)

        self.state = np.hstack((action, power_t, power_r, H_1_real, H_1_imag, H_2_real, H_2_imag))

        reward, opt_reward = self.compute_reward()

        done = opt_reward == reward

        return self.state, reward, done, None
