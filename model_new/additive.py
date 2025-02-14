import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt


class AdditiveGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_t, train_x, train_y, likelihood):
        super().__init__(torch.stack([train_t, train_x], dim=1), train_y, likelihood)

        # Original kernel for beta(t)
        self.beta_kernel = gpytorch.kernels.RBFKernel(active_dims=[0])
        self.scaled_beta_kernel = gpytorch.kernels.ScaleKernel(self.beta_kernel)

        # Original kernel for mu(t)
        self.mu_kernel = gpytorch.kernels.RBFKernel(active_dims=[0])
        self.scaled_mu_kernel = gpytorch.kernels.ScaleKernel(self.mu_kernel)

        # Modified kernel for mu(t)
        # Mean function
        self.mean_module = gpytorch.means.ConstantMean()

    def forward(self, x):
        mean = torch.zeros(x.shape[0])
        # Extract time and covariate
        t, x_covar = x[:, 0], x[:, 1]

        # Compute mu(t) contribution
        K_mu = self.scaled_mu_kernel(x)

        # Compute beta(t)*X contribution
        K_beta = self.scaled_beta_kernel(x)
        X_outer = torch.outer(x_covar, x_covar)
        K_beta = K_beta * X_outer

        # Total covariance
        covar = K_mu + K_beta

        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def get_components(self, t_new, x_train, y_train):
        """Get mu(t) and beta(t) components at new time points"""
        self.eval()
        with torch.no_grad():
            # Prepare training data
            train_inputs = torch.stack([
                torch.tensor(x_train[0], dtype=torch.float32),  # time
                torch.tensor(x_train[1], dtype=torch.float32)  # X
            ], dim=1)
            train_y = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)  # Add dimension

            # Prepare test points for time
            t_new_tensor = torch.tensor(t_new, dtype=torch.float32).reshape(-1, 1)
            dummy = torch.zeros_like(t_new_tensor)
            test_x = torch.cat([t_new_tensor, dummy], dim=1)

            # Get mu(t) component
            K_mu_train = self.scaled_mu_kernel(train_inputs).evaluate()
            K_mu_test = self.scaled_mu_kernel(test_x, train_inputs).evaluate()
            K_mu_test_test = self.scaled_mu_kernel(test_x).evaluate()

            # Get beta(t) component
            K_beta_train = self.scaled_beta_kernel(train_inputs).evaluate()
            K_beta_train = K_beta_train * torch.outer(train_inputs[:, 1], train_inputs[:, 1])
            K_beta_test = self.scaled_beta_kernel(test_x, train_inputs).evaluate()
            K_beta_test_test = self.scaled_beta_kernel(test_x).evaluate()

            # Total training covariance
            K_total = K_mu_train + K_beta_train + self.likelihood.noise.item() * torch.eye(len(train_y))
            L = torch.linalg.cholesky(K_total)

            # Solve the triangular systems
            temp = torch.linalg.solve_triangular(L, train_y, upper=False)
            alpha = torch.linalg.solve_triangular(L.t(), temp, upper=True)

            # Compute posterior means
            mu_t_mean = (K_mu_test @ alpha).reshape(-1)
            beta_t_mean = (K_beta_test @ alpha).reshape(-1)

            # Compute posterior variances
            v_mu = torch.linalg.solve_triangular(L, K_mu_test.t(), upper=False)
            v_beta = torch.linalg.solve_triangular(L, K_beta_test.t(), upper=False)

            mu_t_var = K_mu_test_test.diag() - torch.sum(v_mu ** 2, dim=0)
            beta_t_var = K_beta_test_test.diag() - torch.sum(v_beta ** 2, dim=0)

            return {
                'mu_t_mean': mu_t_mean.numpy(),
                'mu_t_var': mu_t_var.numpy(),
                'beta_t_mean': beta_t_mean.numpy(),
                'beta_t_var': beta_t_var.numpy()
            }


def fit_additive_gp(time, X, y, n_iter=50, lr=0.01):
    """Fit the additive GP model"""
    # Convert to torch tensors
    train_t = torch.tensor(time, dtype=torch.float32)
    train_x = torch.tensor(X, dtype=torch.float32)
    train_y = torch.tensor(y, dtype=torch.float32)

    # Initialize model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = AdditiveGPModel(train_t, train_x, train_y, likelihood)

    # Train mode
    model.train()
    likelihood.train()

    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=lr)

    # Loss function
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training loop
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(torch.stack([train_t, train_x], dim=1))
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            print(f'Iteration {i + 1}/{n_iter} - Loss: {loss.item():.3f}')
            print(f'Component variances:')
            print(f'  beta(t): {model.scaled_beta_kernel.outputscale.item():.3f}')
            print(f'  mu(t): {model.scaled_mu_kernel.outputscale.item():.3f}')
            print(f'  noise: {likelihood.noise.item():.3f}')

    return model, likelihood


def plot_components(model, time, X, y, t_new=None, beta_t=None, mu_t=None):
    """Plot all components of the additive GP model"""
    if t_new is None:
        t_new = np.linspace(time.min(), time.max(), 200)
    x_new = np.interp(t_new, time, X)  # Interpolate X to new time points

    # Set up the figure
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 1])

    # Prepare data for plotting
    model.eval()
    with torch.no_grad():
        # Stack inputs
        X_new = torch.stack([
            torch.tensor(t_new, dtype=torch.float32),
            torch.tensor(x_new, dtype=torch.float32)
        ], dim=1)

        # Get predictions
        pred = model(X_new)
        mean = pred.mean.numpy()
        lower, upper = pred.confidence_region()
        lower, upper = lower.numpy(), upper.numpy()

        # Get components
        components = model.get_components(t_new, [time, X], y)

        # Extract noise variance
        noise_var = model.likelihood.noise.item()

    # 1. Beta(t) plot
    ax1 = fig.add_subplot(gs[0, 0])
    if beta_t is not None:
        ax1.plot(time, beta_t, 'k-', label='True β(t)', alpha=0.6)
    ax1.plot(t_new, components['beta_t_mean'], 'b-', label='Estimated β(t)')
    ax1.fill_between(t_new,
                     components['beta_t_mean'] - 2 * np.sqrt(components['beta_t_var']),
                     components['beta_t_mean'] + 2 * np.sqrt(components['beta_t_var']),
                     color='b', alpha=0.2)
    ax1.set_title('Time-varying coefficient β(t)')
    ax1.legend()
    ax1.grid(True)

    # 2. Mu(t) plot
    ax2 = fig.add_subplot(gs[0, 1])
    if mu_t is not None:
        ax2.plot(time, mu_t, 'k-', label='True μ(t)', alpha=0.6)
    ax2.plot(t_new, components['mu_t_mean'], 'r-', label='Estimated μ(t)')
    ax2.fill_between(t_new,
                     components['mu_t_mean'] - 2 * np.sqrt(components['mu_t_var']),
                     components['mu_t_mean'] + 2 * np.sqrt(components['mu_t_var']),
                     color='r', alpha=0.2)
    ax2.set_title('Time-varying mean μ(t)')
    ax2.legend()
    ax2.grid(True)

    # 3. X*beta(t) effect plot
    ax3 = fig.add_subplot(gs[1, 0])
    if beta_t is not None:
        true_x_beta = X * beta_t
        ax3.plot(time, true_x_beta, 'k-', label='True X*β(t)', alpha=0.6)
    est_x_beta = x_new * components['beta_t_mean']
    ax3.plot(t_new, est_x_beta, 'g-', label='Estimated X*β(t)')
    ax3.fill_between(t_new,
                     est_x_beta - 2 * np.sqrt((x_new ** 2) * components['beta_t_var']),
                     est_x_beta + 2 * np.sqrt((x_new ** 2) * components['beta_t_var']),
                     color='g', alpha=0.2)
    ax3.set_title('Covariate effect X*β(t)')
    ax3.legend()
    ax3.grid(True)

    # 4. Epsilon plot
    ax4 = fig.add_subplot(gs[1, 1])
    if beta_t is not None and mu_t is not None:
        epsilon = y - (X * beta_t + mu_t)  # True residuals
        ax4.plot(time, epsilon, 'k.', label='True ε', alpha=0.6)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.fill_between(time, -2 * np.sqrt(noise_var), 2 * np.sqrt(noise_var),
                     color='gray', alpha=0.2, label='±2σ noise interval')
    ax4.set_title('Residuals ε')
    ax4.legend()
    ax4.grid(True)

    # 5. Final prediction plot
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(time, y, 'k.', label='Observations', alpha=0.6)
    ax5.plot(t_new, mean, 'b-', label='Predicted mean')
    ax5.fill_between(t_new, lower, upper, color='b', alpha=0.2, label='95% confidence interval')
    ax5.set_title('Final prediction y = X*β(t) + μ(t) + ε')
    ax5.legend()
    ax5.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot residual histogram
    if beta_t is not None and mu_t is not None:
        plt.figure(figsize=(6, 4))
        plt.hist(epsilon, bins=20, density=True, alpha=0.6)
        plt.axvline(x=0, color='r', linestyle='--')
        x_grid = np.linspace(epsilon.min(), epsilon.max(), 100)
        plt.plot(x_grid, 1 / np.sqrt(2 * np.pi * noise_var) *
                 np.exp(-x_grid ** 2 / (2 * noise_var)))
        plt.title('Distribution of residuals')
        plt.xlabel('ε')
        plt.ylabel('Density')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Generate synthetic data
    n_samples = 100
    time = np.linspace(0, 10, n_samples)
    X = np.random.randn(n_samples)

    # True functions
    beta_t = np.sin(time)  # time-varying coefficient
    mu_t = 0.5 * np.cos(2 * time)  # time-varying mean

    # Generate y = X*beta(t) + mu(t) + noise
    y = X * beta_t + mu_t + 0.1 * np.random.randn(n_samples)

    # Fit model
    model, likelihood = fit_additive_gp(time, X, y, n_iter=500)

    # Get components at new time points
    t_new = np.linspace(0, 10, 200)
    components = model.get_components(t_new, [time, X], y)

    print("\nLearned component variances:")
    print(f"beta(t) variance: {model.scaled_beta_kernel.outputscale.item():.3f}")
    print(f"mu(t) variance: {model.scaled_mu_kernel.outputscale.item():.3f}")
    print(f"noise variance: {likelihood.noise.item():.3f}")

    t_new = np.linspace(time.min(), time.max(), 200)
    plot_components(model, time, X, y, t_new, beta_t, mu_t)