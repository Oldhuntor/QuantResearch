import numpy as np
import matplotlib.pyplot as plt
from composite_GPs2 import ThreeComponentGP

from simulation import run_simulation

window_size = 50
n_data = 400
change_points = (70, 140, 210, 280, 350)
# 1) Generate piecewise-constant cointegration data
time, y, x, beta_true, mu_true = ThreeComponentGP.generate_cointegration_data(
    T=n_data,
    seed=42,
    noise_std=0.5,
    change_points=change_points,
    beta_values=(1, 2, 3, 4, 5, 6),
    mu_values=(1, 2, 3, 4, 5, 6),
    alpha=1.0
)

X_train = x[:window_size]
Y_train = y[:window_size]
model = ThreeComponentGP(Y_train, X_train, window_size=window_size)
model.train_model(num_epochs=50, learning_rate=0.01)
true_error = y - beta_true * x - mu_true
true_error = true_error[window_size:]
preds, actuals, ci_low, ci_high = run_simulation(
    model,
    y,
    x,
    change_points=change_points,
    true_error=true_error,
    window_size=window_size,
    num_samples=500
)


y = np.array(y)
x = np.array(x)
beta_true = np.array(beta_true)
mu_true = np.array(mu_true)
true_spread = y - beta_true * x - mu_true
noise_variance = 0.5 ** 2
# Prior parameters
prior_mean = np.array([0.0, 0.0])  # Prior means for alpha and beta
prior_cov = np.diag([10.0, 10.0])  # Prior covariance matrix

prior_mean_non_recursive = np.array([0.0, 0.0])
prior_cov_non_recursive = np.diag([10.0, 10.0])

# Store results
posterior_means_recursive = []
posterior_means_non_recursive = []
posterior_intervals_recursive = []
posterior_intervals_non_recursive = []
estimated_spread_recursive = []
estimated_spread_non_recursive = []
true_spreads = true_spread[window_size:]

for t in range(window_size, n_data):
    # Sliding window data
    x_window = x[t - window_size:t]
    y_window = y[t - window_size:t]
    X = np.vstack([np.ones(window_size), x_window]).T  # Design matrix with intercept

    # Recursive Bayesian Update
    XTX = X.T @ X
    XTy = X.T @ y_window
    posterior_cov = np.linalg.inv(np.linalg.inv(prior_cov) + XTX / noise_variance)
    posterior_mean = posterior_cov @ (np.linalg.inv(prior_cov) @ prior_mean + XTy / noise_variance)

    # Update prior for next iteration
    prior_mean = posterior_mean
    prior_cov = posterior_cov

    # Compute credible intervals
    interval_recursive = 2 * np.sqrt(np.diag(posterior_cov))
    posterior_means_recursive.append(posterior_mean)
    posterior_intervals_recursive.append(interval_recursive)

    # Estimate spread (recursive)
    y_hat_recursive = posterior_mean[0] + posterior_mean[1] * x_window
    estimated_spread_recursive.append(y_window[-1] - y_hat_recursive[-1])

    # Non-Recursive Bayesian Estimation (Batch)
    posterior_cov_non_recursive = np.linalg.inv(np.linalg.inv(prior_cov_non_recursive) + XTX / noise_variance)
    posterior_mean_non_recursive = posterior_cov_non_recursive @ (
        np.linalg.inv(prior_cov_non_recursive) @ prior_mean_non_recursive + XTy / noise_variance
    )

    # Compute credible intervals
    interval_non_recursive = 2 * np.sqrt(np.diag(posterior_cov_non_recursive))
    posterior_means_non_recursive.append(posterior_mean_non_recursive)
    posterior_intervals_non_recursive.append(interval_non_recursive)

    # Estimate spread (non-recursive)
    y_hat_non_recursive = posterior_mean_non_recursive[0] + posterior_mean_non_recursive[1] * x_window
    estimated_spread_non_recursive.append(y_window[-1] - y_hat_non_recursive[-1])

posterior_means_recursive = np.array(posterior_means_recursive)
posterior_means_non_recursive = np.array(posterior_means_non_recursive)
posterior_intervals_recursive = np.array(posterior_intervals_recursive)
posterior_intervals_non_recursive = np.array(posterior_intervals_non_recursive)

# Visualization
plt.figure(figsize=(15, 15))

# True beta vs estimated beta
plt.subplot(3, 1, 1)
plt.plot(beta_true[window_size:], label="True Beta", color="black", linestyle="--")
plt.plot(posterior_means_recursive[:, 1], label="Recursive Beta", color="blue")
plt.fill_between(
    range(len(posterior_means_recursive)),
    posterior_means_recursive[:, 1] - posterior_intervals_recursive[:, 1],
    posterior_means_recursive[:, 1] + posterior_intervals_recursive[:, 1],
    color="blue", alpha=0.2, label="Recursive Beta CI"
)
plt.plot(posterior_means_non_recursive[:, 1], label="Non-Recursive Beta", color="orange")
plt.fill_between(
    range(len(posterior_means_non_recursive)),
    posterior_means_non_recursive[:, 1] - posterior_intervals_non_recursive[:, 1],
    posterior_means_non_recursive[:, 1] + posterior_intervals_non_recursive[:, 1],
    color="orange", alpha=0.2, label="Non-Recursive Beta CI"
)
plt.title("Comparison of Recursive and Non-Recursive Beta Estimates")
plt.xlabel("Time")
plt.ylabel("Beta")
plt.legend()

# True intercept vs estimated intercept
plt.subplot(3, 1, 2)
plt.plot(mu_true[window_size:], label="True Intercept", color="black", linestyle="--")
plt.plot(posterior_means_recursive[:, 0], label="Recursive Intercept", color="blue")
plt.fill_between(
    range(len(posterior_means_recursive)),
    posterior_means_recursive[:, 0] - posterior_intervals_recursive[:, 0],
    posterior_means_recursive[:, 0] + posterior_intervals_recursive[:, 0],
    color="blue", alpha=0.2, label="Recursive Intercept CI"
)
plt.plot(posterior_means_non_recursive[:, 0], label="Non-Recursive Intercept", color="orange")
plt.fill_between(
    range(len(posterior_means_non_recursive)),
    posterior_means_non_recursive[:, 0] - posterior_intervals_non_recursive[:, 0],
    posterior_means_non_recursive[:, 0] + posterior_intervals_non_recursive[:, 0],
    color="orange", alpha=0.2, label="Non-Recursive Intercept CI"
)
plt.title("Comparison of Recursive and Non-Recursive Intercept Estimates")
plt.xlabel("Time")
plt.ylabel("Intercept")
plt.legend()

# True spread vs estimated spread
plt.subplot(3, 1, 3)
plt.plot(true_spreads, label="True Spread", color="black", linestyle="--")
plt.plot(estimated_spread_recursive, label="Recursive Estimated Spread", color="blue")
plt.plot(estimated_spread_non_recursive, label="Non-Recursive Estimated Spread", color="orange")
plt.title("True Spread vs Recursive and Non-Recursive Estimated Spread")
plt.xlabel("Time")
plt.ylabel("Spread")
plt.legend()

plt.tight_layout()
plt.show()
