import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate data with non-smooth changing beta
n_data = 500

# Generate beta as a random walk with occasional abrupt changes
true_beta = np.zeros(n_data)
true_beta[0] = 1.5  # Initial beta value
for t in range(1, n_data):
    # Random walk update
    true_beta[t] = true_beta[t-1] + np.random.normal(0, 0.1)
    # Introduce abrupt changes with a 5% probability
    if np.random.rand() < 0.05:
        true_beta[t] += np.random.normal(0, 1.0)

# Generate random walk for X_t
x = np.cumsum(np.random.normal(0, 1, n_data))

# Generate Y_t with observation noise
noise_variance = 0.5 ** 2
y = true_beta * x + np.random.normal(0, np.sqrt(noise_variance), n_data)

# Sliding window size
window_size = 50

# Bayesian Prior
prior_mean = 0
prior_var = 10
alpha_0 = 2
beta_0 = 0.5

# Store results
ols_means = []
ols_variances = []
bayesian_means = []
bayesian_variances = []
true_spreads = []

for t in range(window_size, n_data):
    # Sliding window data
    x_window = x[t - window_size:t]
    y_window = y[t - window_size:t]
    spread_window = y_window - true_beta[t - window_size:t] * x_window

    # True spread
    true_spreads.append(spread_window[-1])  # Last spread in the window

    # OLS estimation
    beta_ols = np.sum(x_window * y_window) / np.sum(x_window ** 2)
    predicted_spread_ols = y_window - beta_ols * x_window
    ols_means.append(np.mean(predicted_spread_ols))
    ols_variances.append(noise_variance)  # Constant variance for OLS

    # Bayesian estimation
    xTx = np.sum(x_window ** 2)
    xTy = np.sum(x_window * y_window)

    posterior_var = 1 / (1 / prior_var + xTx / noise_variance)
    posterior_mean = posterior_var * (prior_mean / prior_var + xTy / noise_variance)

    predicted_spread_bayesian = y_window - posterior_mean * x_window
    bayesian_means.append(np.mean(predicted_spread_bayesian))
    bayesian_variances.append(noise_variance + (x_window[-1] ** 2) * posterior_var)

    # Update prior for Bayesian regression
    prior_mean = posterior_mean
    prior_var = posterior_var

# Convert results to numpy arrays
ols_means = np.array(ols_means)
ols_variances = np.array(ols_variances)
bayesian_means = np.array(bayesian_means)
bayesian_variances = np.array(bayesian_variances)
true_spreads = np.array(true_spreads)

# Step 2: Visualization
plt.figure(figsize=(15, 10))

# True beta
plt.subplot(2, 1, 1)
plt.plot(true_beta, label="True Beta", color="black", linestyle="-")
plt.title("True Beta (Non-Smooth Changes)")
plt.xlabel("Time")
plt.ylabel("Beta")
plt.legend()

# True spread
plt.subplot(2, 1, 2)
plt.plot(true_spreads, label="True Spread", color="black", linestyle="-")

# OLS results
plt.plot(ols_means, label="OLS Spread Mean", color="blue", linestyle="--")
plt.fill_between(
    range(len(ols_means)),
    ols_means - 2 * np.sqrt(ols_variances),
    ols_means + 2 * np.sqrt(ols_variances),
    color="blue",
    alpha=0.2,
    label="OLS Predictive 95% CI",
)

# Bayesian results
plt.plot(bayesian_means, label="Bayesian Spread Mean", color="orange", linestyle="--")
plt.fill_between(
    range(len(bayesian_means)),
    bayesian_means - 2 * np.sqrt(bayesian_variances),
    bayesian_means + 2 * np.sqrt(bayesian_variances),
    color="orange",
    alpha=0.3,
    label="Bayesian Predictive 95% CI",
)

# Plot details
plt.title("Comparison of OLS and Bayesian Regression for Non-Smooth Changing Beta")
plt.xlabel("Time")
plt.ylabel("Spread")
plt.legend()
plt.tight_layout()
plt.show()
