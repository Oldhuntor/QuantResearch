import numpy as np
from scipy.linalg import block_diag
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import matplotlib.pyplot as plt


class SequentialGPTimeVaryingRegression:
    # Modify the kernel definitions in __init__:
    def __init__(self):
        # Kernel for beta - allow longer length scales
        kernel_beta = (ConstantKernel(1.0, constant_value_bounds=(1e-5, 100.0)) *
                       RBF(length_scale=1.0, length_scale_bounds=(1e-2, 100.0)) +
                       WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 10.0)))

        # Kernel for mu - similar structure but different initial values
        kernel_mu = (ConstantKernel(1.0, constant_value_bounds=(1e-5, 100.0)) *
                     RBF(length_scale=1.0, length_scale_bounds=(1e-2, 100.0)) +
                     WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 10.0)))

        # Kernel for epsilon - shorter length scales for noise
        kernel_eps = (ConstantKernel(1.0, constant_value_bounds=(1e-5, 100.0)) *
                      RBF(length_scale=0.1, length_scale_bounds=(1e-3, 10.0)) +
                      WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 10.0)))

        self.gp_beta = GaussianProcessRegressor(
            kernel=kernel_beta,
            n_restarts_optimizer=10,
            normalize_y=True  # Add normalization
        )
        self.gp_mu = GaussianProcessRegressor(
            kernel=kernel_mu,
            n_restarts_optimizer=10,
            normalize_y=True
        )
        self.gp_eps = GaussianProcessRegressor(
            kernel=kernel_eps,
            n_restarts_optimizer=10,
            normalize_y=True
        )
    def get_rolling_estimates(self, y, X, t, window_size=20):
        """Get rolling window OLS estimates"""
        beta_estimates = []
        mu_estimates = []
        eps_estimates = []
        t_points = []

        for i in range(len(t) - window_size):
            X_window = X[i:i + window_size]
            y_window = y[i:i + window_size]

            # Weighted least squares for local estimates
            weights = 1 / (np.abs(np.arange(window_size) - window_size / 2) + 1)
            weighted_X = X_window * weights
            weighted_y = y_window * weights

            # Local regression
            design_matrix = np.column_stack([weighted_X, weights])
            coeffs, *_ = np.linalg.lstsq(design_matrix, weighted_y, rcond=None)
            beta, mu = coeffs

            # Estimate residuals
            eps = np.mean(y_window - (beta * X_window + mu))

            beta_estimates.append(beta)
            mu_estimates.append(mu)
            eps_estimates.append(eps)
            t_points.append(t[i + window_size // 2])

        return (np.array(t_points), np.array(beta_estimates),
                np.array(mu_estimates), np.array(eps_estimates))

    def one_step_ahead_prediction(self, y, X, t, train_size=0.5):
        """Perform one-step-ahead predictions"""
        n = len(t)
        start_idx = int(n * train_size)
        window_size = 20

        # Storage for predictions
        beta_predictions = []
        beta_stds = []
        mu_predictions = []
        mu_stds = []
        eps_predictions = []
        eps_stds = []
        y_predictions = []
        y_stds = []

        for i in range(start_idx, n):
            # Get OLS estimates up to current point
            t_train = t[:i]
            X_train = X[:i]
            y_train = y[:i]

            t_points, beta_est, mu_est, eps_est = self.get_rolling_estimates(
                y_train, X_train, t_train, window_size)

            # Reshape for GP
            t_points = t_points.reshape(-1, 1)
            t_pred = np.array([[t[i]]])

            # Fit GPs and predict
            self.gp_beta.fit(t_points, beta_est)
            beta_pred, beta_std = self.gp_beta.predict(t_pred, return_std=True)

            self.gp_mu.fit(t_points, mu_est)
            mu_pred, mu_std = self.gp_mu.predict(t_pred, return_std=True)

            self.gp_eps.fit(t_points, eps_est)
            eps_pred, eps_std = self.gp_eps.predict(t_pred, return_std=True)

            # Store predictions
            beta_predictions.append(beta_pred[0])
            beta_stds.append(beta_std[0])
            mu_predictions.append(mu_pred[0])
            mu_stds.append(mu_std[0])
            eps_predictions.append(eps_pred[0])
            eps_stds.append(eps_std[0])

            # Calculate y prediction
            y_pred = beta_pred[0] * X[i] + mu_pred[0] + eps_pred[0]
            y_std = np.sqrt(X[i] ** 2 * beta_std[0] ** 2 + mu_std[0] ** 2 + eps_std[0] ** 2)

            y_predictions.append(y_pred)
            y_stds.append(y_std)

            if i % 50 == 0:
                print(f"Completed {i - start_idx}/{n - start_idx} predictions")

        return (np.array(beta_predictions), np.array(beta_stds),
                np.array(mu_predictions), np.array(mu_stds),
                np.array(eps_predictions), np.array(eps_stds),
                np.array(y_predictions), np.array(y_stds))


# Generate synthetic data
np.random.seed(42)
T = 500
t = np.linspace(0, 10, T)
X = np.random.randn(T)

# True parameters
beta_true = 2 * np.sin(t) + 0.5 * np.cos(2 * t) + 1
mu_true = 0.3 * t + 0.2 * np.sin(3 * t)
time_varying_noise = 0.1 + 0.3 * np.abs(np.sin(t))
magnitude_varying_noise = 0.1 * np.abs(X)
epsilon_true = np.random.normal(0, time_varying_noise + magnitude_varying_noise)

# Generate observations
y = beta_true * X + mu_true + epsilon_true

# Perform one-step-ahead predictions
model = SequentialGPTimeVaryingRegression()
(beta_pred, beta_std, mu_pred, mu_std,
 eps_pred, eps_std, y_pred, y_std) = model.one_step_ahead_prediction(y, X, t)

# Calculate indices for prediction period
start_idx = 250  # Corresponding to train_size=0.5
pred_t = t[start_idx:]
true_beta = beta_true[start_idx:]
true_mu = mu_true[start_idx:]
true_eps = epsilon_true[start_idx:]
true_y = y[start_idx:]

# Calculate performance metrics
beta_rmse = np.sqrt(np.mean((true_beta - beta_pred) ** 2))
mu_rmse = np.sqrt(np.mean((true_mu - mu_pred) ** 2))
eps_rmse = np.sqrt(np.mean((true_eps - eps_pred) ** 2))
y_rmse = np.sqrt(np.mean((true_y - y_pred) ** 2))

# Plotting
plt.style.use('default')
fig, axes = plt.subplots(4, 1, figsize=(15, 20))
fig.suptitle('One-Step-Ahead Predictions with Rolling Window OLS', fontsize=14)

# Plot beta predictions
axes[0].plot(pred_t, true_beta, 'k-', label='True β(t)', alpha=0.7)
axes[0].plot(pred_t, beta_pred, 'r-', label='Predicted β(t)')
axes[0].fill_between(pred_t,
                     beta_pred - 2 * beta_std,
                     beta_pred + 2 * beta_std,
                     color='r', alpha=0.2,
                     label='95% CI')
axes[0].set_ylabel('β(t)')
axes[0].legend()
axes[0].set_title(f'Beta Parameter (RMSE: {beta_rmse:.3f})')

# Plot mu predictions
axes[1].plot(pred_t, true_mu, 'k-', label='True μ(t)', alpha=0.7)
axes[1].plot(pred_t, mu_pred, 'g-', label='Predicted μ(t)')
axes[1].fill_between(pred_t,
                     mu_pred - 2 * mu_std,
                     mu_pred + 2 * mu_std,
                     color='g', alpha=0.2,
                     label='95% CI')
axes[1].set_ylabel('μ(t)')
axes[1].legend()
axes[1].set_title(f'Mu Parameter (RMSE: {mu_rmse:.3f})')

# Plot epsilon predictions
axes[2].plot(pred_t, true_eps, 'k-', label='True ε(t)', alpha=0.7)
axes[2].plot(pred_t, eps_pred, 'b-', label='Predicted ε(t)')
axes[2].fill_between(pred_t,
                     eps_pred - 2 * eps_std,
                     eps_pred + 2 * eps_std,
                     color='b', alpha=0.2,
                     label='95% CI')
axes[2].set_ylabel('ε(t)')
axes[2].legend()
axes[2].set_title(f'Epsilon Parameter (RMSE: {eps_rmse:.3f})')

# Plot y predictions
axes[3].plot(pred_t, true_y, 'k-', label='True y(t)', alpha=0.7)
axes[3].plot(pred_t, y_pred, 'r-', label='Predicted y(t)')
axes[3].fill_between(pred_t,
                     y_pred - 2 * y_std,
                     y_pred + 2 * y_std,
                     color='r', alpha=0.2,
                     label='95% CI')
axes[3].set_ylabel('y(t)')
axes[3].set_xlabel('Time')
axes[3].legend()
axes[3].set_title(f'Final Prediction (RMSE: {y_rmse:.3f})')

plt.tight_layout()
plt.show()

# Print detailed metrics
print("\nOne-Step-Ahead Prediction Performance Metrics:")
print(f"Beta RMSE: {beta_rmse:.3f}")
print(f"Mu RMSE: {mu_rmse:.3f}")
print(f"Epsilon RMSE: {eps_rmse:.3f}")
print(f"Y RMSE: {y_rmse:.3f}")