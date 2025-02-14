import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller

# Simulation parameters
np.random.seed(42)
n_data = 1000
window_size = 100

# Generate stationary AR(1) spread
true_phi = 0.7
spread_vol = 0.1
true_spread = np.zeros(n_data)
true_spread[0] = np.random.normal(0, spread_vol / np.sqrt(1 - true_phi ** 2))

for t in range(1, n_data):
    true_spread[t] = true_phi * true_spread[t - 1] + np.random.normal(0, spread_vol)

# Generate Y as random walk
Y = np.zeros(n_data)
Y[0] = 100
for t in range(1, n_data):
    Y[t] = Y[t - 1] + np.random.normal(0, 0.1)

# Generate X
true_beta = 1.2
X = true_beta * Y + true_spread + np.random.normal(0, 0.05, n_data)

# Arrays to store results
estimated_betas = np.zeros(n_data)
estimated_spreads = np.zeros(n_data)
spread_predictions = np.zeros(n_data)
spread_stds = np.zeros(n_data)

# Rolling window cointegration estimation
for t in range(window_size, n_data):
    # Get window data
    Y_window = Y[t - window_size:t]
    X_window = X[t - window_size:t]

    # Prepare regression data
    Y_reg = Y_window.reshape(-1, 1)
    X_reg = X_window.reshape(-1, 1)
    ones = np.ones_like(Y_reg)
    Y_with_const = np.hstack([ones, Y_reg])

    # OLS on levels for this window
    beta_hat = np.linalg.inv(Y_with_const.T @ Y_with_const) @ Y_with_const.T @ X_reg
    constant = float(beta_hat[0])
    beta = float(beta_hat[1])

    # Store beta and calculate spread
    estimated_betas[t] = beta
    estimated_spreads[t] = X[t] - (constant + beta * Y[t])

# Bayesian AR(1) model for spread prediction
mu_phi = 0.5
sigma_phi = 0.1
alpha_sigma = 2.0
beta_sigma = 0.1

# Scale spreads
valid_spreads = estimated_spreads[window_size:]
spread_scale = np.std(valid_spreads)
scaled_spreads = estimated_spreads / spread_scale

# Bayesian updating for spread prediction
for t in range(window_size + 1, n_data):
    spread_window = scaled_spreads[t - window_size:t]

    X_ar = spread_window[:-1].reshape(-1, 1)
    y_ar = spread_window[1:]

    XTX = float(X_ar.T @ X_ar)
    XTy = float(X_ar.T @ y_ar)
    yTy = float(y_ar.T @ y_ar)
    n = len(X_ar)

    post_var_phi = 1 / (1 / sigma_phi + XTX / beta_sigma)
    post_mean_phi = post_var_phi * (mu_phi / sigma_phi + XTy / beta_sigma)

    post_alpha = alpha_sigma + n / 2
    post_beta = beta_sigma + 0.5 * (yTy - 2 * post_mean_phi * XTy + (post_mean_phi ** 2) * XTX)

    spread_predictions[t] = float(post_mean_phi * scaled_spreads[t - 1]) * spread_scale
    pred_var = float(post_beta / post_alpha) * (1 + float(scaled_spreads[t - 1] ** 2) * post_var_phi) * (
                spread_scale ** 2)
    spread_stds[t] = np.sqrt(pred_var)

# Plotting
plot_start = window_size + 1
plt.figure(figsize=(15, 12))

# Plot 1: Rolling Beta Estimation
plt.subplot(3, 1, 1)
plt.axhline(y=true_beta, color='black', linestyle='-', label='True Beta')
plt.plot(range(plot_start, n_data), estimated_betas[plot_start:],
         label='Rolling Beta', color='blue', alpha=0.7)
plt.title('Rolling Beta Estimation')
plt.legend()
plt.grid(True)

# Plot 2: Spread Analysis
plt.subplot(3, 1, 2)
plt.plot(range(plot_start, n_data), true_spread[plot_start:],
         label='True Spread', color='black', alpha=0.7)
plt.plot(range(plot_start, n_data), estimated_spreads[plot_start:],
         label='Estimated Spread', color='blue', alpha=0.7)
plt.plot(range(plot_start, n_data), spread_predictions[plot_start:],
         label='Predicted Spread', color='red', alpha=0.7)
plt.fill_between(
    range(plot_start, n_data),
    spread_predictions[plot_start:] - 2 * spread_stds[plot_start:],
    spread_predictions[plot_start:] + 2 * spread_stds[plot_start:],
    color='red', alpha=0.2, label='95% Prediction Interval'
)
plt.title('Spread Analysis')
plt.legend()
plt.grid(True)

# Plot 3: ACF of Spreads
plt.subplot(3, 1, 3)
from statsmodels.tsa.stattools import acf

max_lags = 20
true_acf = acf(true_spread[plot_start:], nlags=max_lags)
est_acf = acf(estimated_spreads[plot_start:], nlags=max_lags)
lags = range(max_lags + 1)
plt.plot(lags, true_acf, label='True Spread ACF', marker='o')
plt.plot(lags, est_acf, label='Estimated Spread ACF', marker='o')
plt.title('Autocorrelation Function Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print statistics for post burn-in period
valid_period = slice(plot_start, None)
print("\nSpread Analysis Statistics:")
print(f"Mean Beta: {np.mean(estimated_betas[valid_period]):.4f}")
print(f"Beta Std: {np.std(estimated_betas[valid_period]):.4f}")
print(
    f"Correlation with True Spread: {np.corrcoef(true_spread[valid_period], estimated_spreads[valid_period])[0, 1]:.3f}")
print(f"True Spread Std: {np.std(true_spread[valid_period]):.4f}")
print(f"Estimated Spread Std: {np.std(estimated_spreads[valid_period]):.4f}")