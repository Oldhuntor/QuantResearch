import torch


def bayesian_linreg_conjugate(H, x, sigma2=1.0, tau2=1e6):
    """
    Conjugate posterior for the linear model x ~ Normal(H*theta, sigma2*I),
    with prior theta ~ Normal(0, tau2*I).

    H: Nx2 tensor (rows: [Y_t, 1])
    x: Nx1 tensor
    sigma2: known noise variance
    tau2: prior variance for beta, mu
    returns: (m_N, S_N) posterior mean and covariance (2x1, 2x2)
    """
    N, D = H.shape
    I_D = torch.eye(D)
    S0_inv = (1 / tau2) * I_D  # prior precision
    HtH = H.T @ H

    S_N = torch.inverse(S0_inv + (1 / sigma2) * HtH)
    m_N = S_N @ ((1 / sigma2) * (H.T @ x))
    return m_N, S_N


# Example usage:
N = 100
Y = torch.randn(N)  # predictor
beta_true = 2.0
mu_true = -1.0
sigma_eps = 0.1
X = beta_true * Y + mu_true + sigma_eps * torch.randn(N)

# Build design matrix H: Nx2
# row t: [Y_t, 1]
H = torch.stack([Y, torch.ones_like(Y)], dim=1)  # shape (N,2)

m_N, S_N = bayesian_linreg_conjugate(H, X, sigma2=sigma_eps ** 2, tau2=1e6)
print("Posterior mean for [beta, mu]:", m_N)
print("Posterior covariance matrix:", S_N)
