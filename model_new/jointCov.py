import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt


class JointGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        # Beta and mu can be flexible
        self.beta_mean = gpytorch.means.ZeroMean()
        self.beta_covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() +
            gpytorch.kernels.LinearKernel() +
            gpytorch.kernels.PeriodicKernel()
        )

        self.mu_mean = gpytorch.means.ZeroMean()
        self.mu_covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() +
            gpytorch.kernels.PeriodicKernel()
        )

        # Epsilon: Force stationarity with periodic oscillation around zero mean
        self.eps_mean = gpytorch.means.ZeroMean()  # Force zero mean

        # Kernel for epsilon that enforces stationarity:
        # - RBF for smoothness
        # - Periodic for oscillation
        # - Small lengthscale to allow rapid oscillation
        # - Constrained output scale to keep it bounded
        periodic_kernel = gpytorch.kernels.PeriodicKernel(
            period_length_prior=gpytorch.priors.NormalPrior(0.05, 0.01)  # Short period for oscillation
        )
        rbf_kernel = gpytorch.kernels.RBFKernel(
            lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 20.0)  # Shorter lengthscale
        )

        self.eps_covar = gpytorch.kernels.ScaleKernel(
            rbf_kernel + periodic_kernel,
            outputscale_prior=gpytorch.priors.GammaPrior(1.0, 10.0)  # Keep amplitude small
        )

    def forward(self, x):
        n = x.size(0)
        X_t = x[:, 0]

        # Compute means for each component using same input x
        beta_mean = self.beta_mean(x)
        mu_mean = self.mu_mean(x)
        eps_mean = self.eps_mean(x)

        # Build full mean and covariance
        full_mean = torch.cat([beta_mean, mu_mean, eps_mean])
        full_covar = self.build_full_covar(x, x)
        full_covar = full_covar + torch.eye(3 * n) * 1e-3

        # Create observation operator H = [X_t, 1, 1]
        H = torch.zeros(n, 3 * n)
        for i in range(n):
            H[i, i] = X_t[i]  # X_t component
            H[i, n + i] = 1.0  # mu component
            H[i, 2 * n + i] = 1.0  # eps component

        # Transform to observation space
        obs_mean = H @ full_mean
        obs_covar = H @ full_covar @ H.t()

        return gpytorch.distributions.MultivariateNormal(obs_mean, obs_covar)

    def predict_components(self, x_test):
        """
        Predict latent components using the formula:
        m_post = m_prior + K_xs^T(HK_xx H^T + R)^{-1}(y - Hm_prior)
        """
        self.eval()

        with torch.no_grad():
            n_train = self.train_inputs[0].size(0)
            n_test = x_test.size(0)

            # 1. Get prior means
            prior_beta_mean = self.beta_mean(x_test)
            prior_mu_mean = self.mu_mean(x_test)  # Using x_test instead of time indices
            prior_eps_mean = self.eps_mean(x_test)  # Keep eps with same input for consistency
            prior_mean = torch.cat([prior_beta_mean, prior_mu_mean, prior_eps_mean])

            train_beta_mean = self.beta_mean(self.train_inputs[0])
            train_mu_mean = self.mu_mean(self.train_inputs[0])  # Using X instead of time indices
            train_eps_mean = self.eps_mean(self.train_inputs[0])
            train_prior_mean = torch.cat([train_beta_mean, train_mu_mean, train_eps_mean])

            # 2. Build H matrix [X_t, 1, 1]
            H = torch.zeros(n_train, 3 * n_train)
            X_train = self.train_inputs[0][:, 0]
            for i in range(n_train):
                H[i, i] = X_train[i]  # X_t component
                H[i, n_train + i] = 1.0  # mu component
                H[i, 2 * n_train + i] = 1.0  # epsilon component

            # 3. Build covariance matrices
            K_xx = self.build_full_covar(self.train_inputs[0], self.train_inputs[0])
            K_xs = self.build_full_covar(self.train_inputs[0], x_test)
            K_ss = self.build_full_covar(x_test, x_test)

            # 4. Compute (HK_xx H^T + R)^{-1}
            noise_var = self.likelihood.noise.clone()
            R = torch.eye(n_train) * (noise_var + 1e-6)  # Added jitter to noise term
            HKH = H @ K_xx @ H.t()
            V = HKH + R
            L = torch.linalg.cholesky(V)
            V_inv = torch.cholesky_solve(torch.eye(n_train), L)

            # 5. Compute y - Hm_prior
            Hm_prior = H @ train_prior_mean
            innovation = self.train_targets - Hm_prior

            # 6. Compute posterior mean using the formula
            adjustment = K_xs.t() @ H.t() @ V_inv @ innovation
            post_mean = prior_mean + adjustment

            # 7. Compute posterior covariance
            posterior_cov = K_ss - K_xs.t() @ H.t() @ V_inv @ H @ K_xs
            # posterior_cov += torch.eye(3 * n_test) * 1e-6  # numerical stability

            # 8. Extract components
            beta_mean = post_mean[:n_test]
            mu_mean = post_mean[n_test:2 * n_test]
            eps_mean = post_mean[2 * n_test:]

            beta_cov = posterior_cov[:n_test, :n_test]
            mu_cov = posterior_cov[n_test:2 * n_test, n_test:2 * n_test]
            eps_cov = posterior_cov[2 * n_test:, 2 * n_test:]

            beta_std = torch.sqrt(torch.diag(beta_cov))
            mu_std = torch.sqrt(torch.diag(mu_cov))
            eps_std = torch.sqrt(torch.diag(eps_cov))

            # Scale beta uncertainty by X_test
            X_test_flat = x_test[:, 0]
            beta_std = beta_std * X_test_flat

            return beta_mean, mu_mean, eps_mean, beta_std, mu_std, eps_std

    def build_full_covar(self, x1, x2):
        """Build the full covariance matrix between two sets of inputs"""
        n1, n2 = x1.size(0), x2.size(0)

        # Add small jitter to diagonal for numerical stability
        jitter = 1e-6

        # Compute cross-covariances using same inputs
        beta_covar = self.beta_covar(x1, x2)
        mu_covar = self.mu_covar(x1, x2)
        eps_covar = self.eps_covar(x1, x2)

        # Build block diagonal covariance
        zeros = torch.zeros(n1, n2)
        row1 = torch.cat([beta_covar.evaluate() + jitter * torch.eye(n1, n2), zeros, zeros], dim=1)
        row2 = torch.cat([zeros, mu_covar.evaluate() + jitter * torch.eye(n1, n2), zeros], dim=1)
        row3 = torch.cat([zeros, zeros, eps_covar.evaluate() + jitter * torch.eye(n1, n2)], dim=1)
        full_covar = torch.cat([row1, row2, row3], dim=0)

        return full_covar

def train_model(train_x, train_y, n_iter=200):
    # Convert numpy arrays to torch tensors and reshape
    X = torch.from_numpy(train_x).float().unsqueeze(-1)
    y = torch.from_numpy(train_y).float()

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = JointGPModel(X, y, likelihood)

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=0.01)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training loop
    model.train()
    likelihood.train()

    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            print(f'Iter {i + 1:>3d}/{n_iter} - Loss: {loss.item():.3f}')

    return model, likelihood


def predict(model, likelihood, X_test):
    # Convert to torch tensor and reshape
    X_test = torch.from_numpy(X_test).float().unsqueeze(-1)

    # Put model in evaluation mode
    model.eval()
    likelihood.eval()

    # Make predictions
    with torch.no_grad():
        # Get component predictions with uncertainties
        beta_pred, mu_pred, eps_pred, beta_std, mu_std, eps_std = model.predict_components(X_test)

        # Get full Y predictions
        obs_dist = model(X_test)
        obs_pred = likelihood(obs_dist)
        lower_y, upper_y = obs_pred.confidence_region()

    # Convert to numpy
    return (beta_pred.numpy(), mu_pred.numpy(), eps_pred.numpy(),
            beta_std.numpy(), mu_std.numpy(), eps_std.numpy(),
            lower_y.numpy(), upper_y.numpy())


if __name__ == "__main__":
    # Generate synthetic data
    # Generate synthetic data
    np.random.seed(42)
    N = 200

    # Generate X in original range
    X_original = np.linspace(0, 200, N)

    # True parameters with realistic behavior
    beta_true = 1.0 + 0.2 * np.sin(0.5 * X_original)
    mu_true = 0.5 * np.cos(0.3 * X_original)
    eps_true = 0.1 * np.random.randn(N)

    # Generate observations
    Y_original = beta_true * X_original + mu_true + eps_true

    # Scale both X and Y
    X = (X_original - X_original.min()) / (X_original.max() - X_original.min())
    Y = (Y_original - Y_original.mean()) / Y_original.std()

    # Train model
    model, likelihood = train_model(X, Y, n_iter=500)

    # Make predictions
    beta_pred, mu_pred, eps_pred, beta_std, mu_std, eps_std, lower_y, upper_y = predict(model, likelihood, X)

    # Calculate predicted Y using the correct observation equation
    Y_pred = beta_pred * X + mu_pred + eps_pred

    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))

    # Plot Y
    axes[0].plot(X, Y, 'k.', label='Observed Y', alpha=0.6, markersize=4)
    axes[0].plot(X, Y_pred, 'b-', label='Predicted Y', linewidth=2)
    axes[0].fill_between(X, lower_y, upper_y, color='b', alpha=0.2, label='95% CI')
    rmse = np.sqrt(np.mean((Y - Y_pred) ** 2))
    axes[0].set_title(f'Observations and Predictions (RMSE: {rmse:.4f})')
    axes[0].legend()
    axes[0].grid(True)

    # Plot beta component
    axes[1].plot(X, beta_true, 'k-', label='True β(t)', linewidth=2)
    axes[1].plot(X, beta_pred, 'b-', label='Predicted β(t)', linewidth=2)
    axes[1].fill_between(X,
                         beta_pred - 1.96 * beta_std,
                         beta_pred + 1.96 * beta_std,
                         color='b', alpha=0.2, label='95% CI')
    axes[1].set_title('Beta Component')
    axes[1].legend()
    axes[1].grid(True)

    # Plot mu component
    axes[2].plot(X, mu_true, 'k-', label='True μ(t)', linewidth=2)
    axes[2].plot(X, mu_pred, 'b-', label='Predicted μ(t)', linewidth=2)
    axes[2].fill_between(X,
                         mu_pred - 1.96 * mu_std,
                         mu_pred + 1.96 * mu_std,
                         color='b', alpha=0.2, label='95% CI')
    axes[2].set_title('Mu Component')
    axes[2].legend()
    axes[2].grid(True)

    # Plot epsilon component
    axes[3].plot(X, eps_true, 'k-', label='True ε(t)', linewidth=2)
    axes[3].plot(X, eps_pred, 'b-', label='Predicted ε(t)', linewidth=2)
    axes[3].fill_between(X,
                         eps_pred - 1.96 * eps_std,
                         eps_pred + 1.96 * eps_std,
                         color='b', alpha=0.2, label='95% CI')
    axes[3].set_title('Epsilon Component')
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()

    # Print metrics
    print(f"\nEvaluation Metrics:")
    print(f"RMSE for Y: {rmse:.4f}")
    print(f"RMSE for beta: {np.sqrt(np.mean((beta_true - beta_pred) ** 2)):.4f}")
    print(f"RMSE for mu: {np.sqrt(np.mean((mu_true - mu_pred) ** 2)):.4f}")
    print(f"RMSE for epsilon: {np.sqrt(np.mean((eps_true - eps_pred) ** 2)):.4f}")