import numpy as np
from scipy.linalg import block_diag, solve
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

class JointGPTimeVaryingRegression:
    def __init__(self):
        # Define individual kernels
        self.kernel_beta = (ConstantKernel(1.0, constant_value_bounds=(1e-3, 10.0)) *
                            RBF(length_scale=0.5, length_scale_bounds=(1e-2, 10.0)) +
                            WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1.0)))

        self.kernel_mu = (ConstantKernel(1.0, constant_value_bounds=(1e-3, 10.0)) *
                          RBF(length_scale=0.5, length_scale_bounds=(1e-2, 10.0)) +
                          WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1.0)))

        self.kernel_eps = (ConstantKernel(1.0, constant_value_bounds=(1e-3, 10.0)) *
                           RBF(length_scale=0.1, length_scale_bounds=(1e-3, 1.0)) +
                           WhiteKernel(noise_level=1.0, noise_level_bounds=(0.1, 10.0)))

        self.jitter = 1e-6  # Small constant for numerical stability

    def compute_joint_kernel(self, t1, t2=None):
        """Compute the joint kernel matrix for all parameters"""
        if t2 is None:
            t2 = t1

        K_beta = self.kernel_beta(t1, t2)
        K_mu = self.kernel_mu(t1, t2)
        K_eps = self.kernel_eps(t1, t2)

        # Create block diagonal kernel matrix
        K = block_diag(K_beta, K_mu, K_eps)

        # Add jitter to diagonal for numerical stability
        if t1 is t2:
            K += np.eye(K.shape[0]) * self.jitter

        return K

    def compute_H_matrix(self, X, t):
        """Compute the observation matrix H"""
        n = len(t)
        H = np.zeros((n, 3 * n))

        for i in range(n):
            H[i, i] = X[i]  # beta coefficient
            H[i, n + i] = 1  # mu coefficient
            H[i, 2 * n + i] = 1  # epsilon coefficient

        return H

    def fit_predict(self, y, X, t):
        """Joint fitting and prediction using Gaussian Process regression"""
        t = t.reshape(-1, 1)
        n = len(t)

        # Compute kernel matrix
        K = self.compute_joint_kernel(t)

        # Compute observation matrix
        H = self.compute_H_matrix(X, t)

        # Compute joint covariance with improved numerical stability
        S = H @ K @ H.T + np.eye(n) * self.jitter

        try:
            # Try Cholesky decomposition for better numerical stability
            L = np.linalg.cholesky(S)
            alpha = solve(L.T, solve(L, y))
            K_gain = K @ H.T @ alpha

            # Compute posterior covariance more stably
            V = solve(L, H @ K.T)
            joint_cov = K - V.T @ V

        except np.linalg.LinAlgError:
            # Fallback to traditional inverse if Cholesky fails
            S_inv = np.linalg.inv(S + np.eye(n) * self.jitter)
            K_gain = K @ H.T @ S_inv @ y
            joint_cov = K - K @ H.T @ S_inv @ H @ K

        # Extract individual parameters
        beta_mean = K_gain[:n]
        mu_mean = K_gain[n:2 * n]
        eps_mean = K_gain[2 * n:]

        # Ensure positive variances
        diag_cov = np.diag(joint_cov)
        beta_std = np.sqrt(np.maximum(diag_cov[:n], 0))
        mu_std = np.sqrt(np.maximum(diag_cov[n:2 * n], 0))
        eps_std = np.sqrt(np.maximum(diag_cov[2 * n:], 0))

        return beta_mean, beta_std, mu_mean, mu_std, eps_mean, eps_std

    def one_step_ahead_prediction(self, y, X, t, train_size=0.5):
        """Perform one-step-ahead predictions for all parameters"""
        n = len(t)
        start_idx = int(n * train_size)

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
            # Use data up to current point
            t_train = t[:i]
            X_train = X[:i]
            y_train = y[:i]

            # Fit model and predict
            beta_mean, beta_std, mu_mean, mu_std, eps_mean, eps_std = self.fit_predict(y_train, X_train, t_train)

            # Store parameter predictions
            beta_predictions.append(beta_mean[-1])
            beta_stds.append(beta_std[-1])
            mu_predictions.append(mu_mean[-1])
            mu_stds.append(mu_std[-1])
            eps_predictions.append(eps_mean[-1])
            eps_stds.append(eps_std[-1])

            # Make one-step prediction for y
            y_pred = beta_mean[-1] * X[i] + mu_mean[-1] + eps_mean[-1]
            y_std = np.sqrt(X[i] ** 2 * beta_std[-1] ** 2 + mu_std[-1] ** 2 + eps_std[-1] ** 2)

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
beta_true = 2 * np.sin(t) + 0.5 * np.cos(2*t) + 1
mu_true = 0.3 * t + 0.2 * np.sin(3*t)
time_varying_noise = 0.1 + 0.3 * np.abs(np.sin(t))
magnitude_varying_noise = 0.1 * np.abs(X)
epsilon_true = np.random.normal(0, time_varying_noise + magnitude_varying_noise)

# Generate observations
y = beta_true * X + mu_true + epsilon_true

# Perform one-step-ahead predictions
model = JointGPTimeVaryingRegression()
(beta_pred, beta_std, mu_pred, mu_std,
 eps_pred, eps_std, y_pred, y_std) = model.one_step_ahead_prediction(y, X, t)

# Calculate indices for prediction period
start_idx = 250  # Corresponding to train_size=0.5
pred_t = t[start_idx:]
true_beta = beta_true[start_idx:]
true_mu = mu_true[start_idx:]
true_eps = epsilon_true[start_idx:]
true_y = y[start_idx:]

# Calculate performance metrics for each component
beta_rmse = np.sqrt(mean_squared_error(true_beta, beta_pred))
mu_rmse = np.sqrt(mean_squared_error(true_mu, mu_pred))
eps_rmse = np.sqrt(mean_squared_error(true_eps, eps_pred))
y_rmse = np.sqrt(mean_squared_error(true_y, y_pred))

# Plotting
plt.style.use('default')
fig, axes = plt.subplots(4, 1, figsize=(15, 20))
fig.suptitle('One-Step-Ahead Predictions for All Parameters', fontsize=14)

# Plot beta predictions
axes[0].plot(pred_t, true_beta, 'k-', label='True β(t)', alpha=0.7)
axes[0].plot(pred_t, beta_pred, 'r-', label='Predicted β(t)')
axes[0].fill_between(pred_t,
                     beta_pred - 2*beta_std,
                     beta_pred + 2*beta_std,
                     color='r', alpha=0.2,
                     label='95% CI')
axes[0].set_ylabel('β(t)')
axes[0].legend()
axes[0].set_title(f'Beta Parameter (RMSE: {beta_rmse:.3f})')

# Plot mu predictions
axes[1].plot(pred_t, true_mu, 'k-', label='True μ(t)', alpha=0.7)
axes[1].plot(pred_t, mu_pred, 'g-', label='Predicted μ(t)')
axes[1].fill_between(pred_t,
                     mu_pred - 2*mu_std,
                     mu_pred + 2*mu_std,
                     color='g', alpha=0.2,
                     label='95% CI')
axes[1].set_ylabel('μ(t)')
axes[1].legend()
axes[1].set_title(f'Mu Parameter (RMSE: {mu_rmse:.3f})')

# Plot epsilon predictions
axes[2].plot(pred_t, true_eps, 'k-', label='True ε(t)', alpha=0.7)
axes[2].plot(pred_t, eps_pred, 'b-', label='Predicted ε(t)')
axes[2].fill_between(pred_t,
                     eps_pred - 2*eps_std,
                     eps_pred + 2*eps_std,
                     color='b', alpha=0.2,
                     label='95% CI')
axes[2].set_ylabel('ε(t)')
axes[2].legend()
axes[2].set_title(f'Epsilon Parameter (RMSE: {eps_rmse:.3f})')

# Plot y predictions
axes[3].plot(pred_t, true_y, 'k-', label='True y(t)', alpha=0.7)
axes[3].plot(pred_t, y_pred, 'r-', label='Predicted y(t)')
axes[3].fill_between(pred_t,
                     y_pred - 2*y_std,
                     y_pred + 2*y_std,
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