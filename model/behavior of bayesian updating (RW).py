import numpy as np
import matplotlib.pyplot as plt

# Generate some example data with changing beta
# np.random.seed(42)
n_data = 500
beta_change_interval = 100  # Change beta every 100 steps

# Generate true beta with changes every interval
true_beta = np.zeros(n_data)
beta_values = [1.5, -0.5, 2.0, -1.0, 0.5]
for i in range(len(beta_values)):
    start = i * beta_change_interval
    end = min((i + 1) * beta_change_interval, n_data)
    true_beta[start:end] = beta_values[i]

# Generate cointegrated data: Y_t = beta * X_t + noise
x = np.cumsum(np.random.normal(0, 1, n_data))  # Random walk for X_t
noise_variance = 0.5 ** 2
y = true_beta * x + np.random.normal(0, np.sqrt(noise_variance), n_data)

# Sliding window size
window_size = 20

# Prior parameters for Bayesian regression
prior_mean = 0
prior_var = 10
alpha_0 = 2  # Shape parameter for inverse-gamma prior on variance
beta_0 = 0.5  # Scale parameter for inverse-gamma prior on variance

# Store posterior estimates (Recursive)
posterior_means_recursive = []
posterior_vars_recursive = []
posterior_noise_vars_recursive = []
spreads_recursive = []
spread_intervals_recursive = []

# Recursive Bayesian updates with sliding window
prior_mean_recursive = prior_mean
prior_var_recursive = prior_var
beta_0_recursive = beta_0
for t in range(window_size, n_data):
    x_window = x[t - window_size:t]
    y_window = y[t - window_size:t]

    # Compute sufficient statistics
    xTx = np.sum(x_window ** 2)
    xTy = np.sum(x_window * y_window)
    residual = y_window - prior_mean_recursive * x_window
    residual_sum_sq = np.sum(residual ** 2)

    # Update noise variance posterior (inverse-gamma)
    alpha_n = alpha_0 + window_size / 2
    beta_n = beta_0_recursive + 0.5 * residual_sum_sq
    posterior_noise_var = beta_n / (alpha_n - 1)  # Mean of inverse-gamma

    # Update beta posterior
    posterior_var = 1 / (1 / prior_var_recursive + xTx / posterior_noise_var)
    posterior_mean = posterior_var * (prior_mean_recursive / prior_var_recursive + xTy / posterior_noise_var)

    # Compute spread and credible interval
    spread = y_window - posterior_mean * x_window
    spread_interval = 2 * np.sqrt(posterior_noise_var + posterior_var * x_window ** 2)

    # Append results
    posterior_means_recursive.append(posterior_mean)
    posterior_vars_recursive.append(posterior_var)
    posterior_noise_vars_recursive.append(posterior_noise_var)
    spreads_recursive.append(spread[-1])  # Use the latest spread in the window
    spread_intervals_recursive.append(spread_interval[-1])

    # Update prior for the next window
    prior_mean_recursive = posterior_mean
    prior_var_recursive = posterior_var
    beta_0_recursive = beta_n  # Update scale parameter for variance

# Store posterior estimates (Non-Recursive)
posterior_means_non_recursive = []
posterior_vars_non_recursive = []
posterior_noise_vars_non_recursive = []
spreads_non_recursive = []
spread_intervals_non_recursive = []

# Non-Recursive Bayesian updates with sliding window
for t in range(window_size, n_data):
    x_window = x[t - window_size:t]
    y_window = y[t - window_size:t]

    # Compute sufficient statistics
    xTx = np.sum(x_window ** 2)
    xTy = np.sum(x_window * y_window)
    residual = y_window - prior_mean * x_window
    residual_sum_sq = np.sum(residual ** 2)

    # Update noise variance posterior (inverse-gamma)
    alpha_n = alpha_0 + window_size / 2
    beta_n = beta_0 + 0.5 * residual_sum_sq
    posterior_noise_var = beta_n / (alpha_n - 1)  # Mean of inverse-gamma

    # Update beta posterior
    posterior_var = 1 / (1 / prior_var + xTx / posterior_noise_var)
    posterior_mean = posterior_var * (prior_mean / prior_var + xTy / posterior_noise_var)

    # Compute spread and credible interval
    spread = y_window - posterior_mean * x_window
    spread_interval = 2 * np.sqrt(posterior_noise_var + posterior_var * x_window ** 2)

    # Append results
    posterior_means_non_recursive.append(posterior_mean)
    posterior_vars_non_recursive.append(posterior_var)
    posterior_noise_vars_non_recursive.append(posterior_noise_var)
    spreads_non_recursive.append(spread[-1])  # Use the latest spread in the window
    spread_intervals_non_recursive.append(spread_interval[-1])

# Convert to arrays for plotting
posterior_means_recursive = np.array(posterior_means_recursive)
posterior_vars_recursive = np.array(posterior_vars_recursive)
posterior_noise_vars_recursive = np.array(posterior_noise_vars_recursive)
spreads_recursive = np.array(spreads_recursive)
spread_intervals_recursive = np.array(spread_intervals_recursive)

posterior_means_non_recursive = np.array(posterior_means_non_recursive)
posterior_vars_non_recursive = np.array(posterior_vars_non_recursive)
posterior_noise_vars_non_recursive = np.array(posterior_noise_vars_non_recursive)
spreads_non_recursive = np.array(spreads_non_recursive)
spread_intervals_non_recursive = np.array(spread_intervals_non_recursive)

# Plot results
plt.figure(figsize=(15, 20))

# True beta
plt.subplot(5, 1, 1)
plt.plot(true_beta[window_size:], label="True Beta", color="black", linestyle="-", linewidth=2)
plt.title("True Beta (Changing Every 100 Steps)")
plt.xlabel("Time")
plt.ylabel("Beta")
plt.legend()

# Posterior beta mean (Recursive)
plt.subplot(5, 1, 2)
plt.plot(true_beta[window_size:], label="True Beta", color="black", linestyle="-", linewidth=2)
plt.plot(posterior_means_recursive, label="Posterior Mean (Recursive)", linestyle="--")
plt.fill_between(
    range(len(posterior_means_recursive)),
    posterior_means_recursive - 2 * np.sqrt(posterior_vars_recursive),
    posterior_means_recursive + 2 * np.sqrt(posterior_vars_recursive),
    color="lightblue",
    alpha=0.5,
    label="95% Credible Interval",
)
plt.title("Bayesian Regression Posterior Mean (Recursive Beta)")
plt.xlabel("Time")
plt.ylabel("Beta")
plt.legend()

# Posterior beta mean (Non-Recursive)
plt.subplot(5, 1, 3)
plt.plot(true_beta[window_size:], label="True Beta", color="black", linestyle="-", linewidth=2)
plt.plot(posterior_means_non_recursive, label="Posterior Mean (Non-Recursive)", linestyle="--")
plt.fill_between(
    range(len(posterior_means_non_recursive)),
    posterior_means_non_recursive - 2 * np.sqrt(posterior_vars_non_recursive),
    posterior_means_non_recursive + 2 * np.sqrt(posterior_vars_non_recursive),
    color="lightgreen",
    alpha=0.5,
    label="95% Credible Interval",
)
plt.title("Bayesian Regression Posterior Mean (Non-Recursive Beta)")
plt.xlabel("Time")
plt.ylabel("Beta")
plt.legend()

# Spread (Recursive)
plt.subplot(5, 1, 4)
plt.plot(spreads_recursive, label="Spread (Recursive)", linestyle="--", color="blue")
plt.fill_between(
    range(len(spreads_recursive)),
    spreads_recursive - spread_intervals_recursive,
    spreads_recursive + spread_intervals_recursive,
    color="lightblue",
    alpha=0.5,
    label="Spread Interval (Recursive)",
)
plt.title("Spread with Credible Interval (Recursive)")
plt.xlabel("Time")
plt.ylabel("Spread")
plt.legend()

# Spread (Non-Recursive)
plt.subplot(5, 1, 5)
plt.plot(spreads_non_recursive, label="Spread (Non-Recursive)", linestyle="--", color="green")
plt.fill_between(
    range(len(spreads_non_recursive)),
    spreads_non_recursive - spread_intervals_non_recursive,
    spreads_non_recursive + spread_intervals_non_recursive,
    color="lightgreen",
    alpha=0.5,
    label="Spread Interval (Non-Recursive)",
)
plt.title("Spread with Credible Interval (Non-Recursive)")
plt.xlabel("Time")
plt.ylabel("Spread")
plt.legend()

plt.show()