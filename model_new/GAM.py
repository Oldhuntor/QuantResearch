import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrix
from sklearn.linear_model import Ridge
import matplotlib

print(matplotlib.get_backend())

# 1. Data Generating Process with Heteroskedastic Noise
np.random.seed(42)
T = 300  # Extended number of time points

time = np.linspace(0, 30, T)  # Extended time variable
X_t = np.random.normal(0, 1, T)  # Predictor

# True time-varying coefficients
beta_t = 1 + 0.5 * np.sin(0.5 * time)
mu_t = 0.2 * np.cos(0.3 * time)

# Generate more chaotic heteroskedastic noise
noise_variance = 0.1 * (1 + np.abs(20 * np.sin(0.3 * time) ** 2 + 15 * np.cos(0.15 * time)))
epsilon = np.random.normal(0, noise_variance, T)

# Generate response variable with heteroskedastic noise
Y_t = beta_t * X_t + mu_t + epsilon

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(time, Y_t, label='Observed $Y_t$', alpha=0.6)
plt.plot(time, beta_t * X_t + mu_t, label='True Signal', color='green')
plt.legend()
plt.xlabel('Time')
plt.ylabel('$Y_t$')
plt.title('Data Generation with Chaotic Heteroskedastic Noise')
plt.grid(True)
plt.show()

# 2. Separate Estimation for mu_t and beta_t
# Fit mu_t (Intercept)
basis_mu = dmatrix("bs(time, df=8, degree=3)", {"time": time}, return_type='dataframe')
model_mu = Ridge(alpha=0.1)
model_mu.fit(basis_mu, Y_t)
mu_hat = model_mu.predict(basis_mu)

# Compute residuals for beta_t
residuals_beta = Y_t - mu_hat

# Fit beta_t (Slope)
basis_beta = dmatrix("bs(time, df=6, degree=3)", {"time": time}, return_type='dataframe')
X_design = np.multiply(basis_beta.values, X_t[:, None])
model_beta = Ridge(alpha=1.0)
model_beta.fit(X_design, residuals_beta)

# Extract beta estimates
beta_hat = model_beta.predict(X_design)

# 3. One-Step Ahead Prediction with Noise Interval (Using Window Size)
window_size = 50
predictions_one_step = []
pred_noise_intervals = []
predicted_noise = []
pred_mu = []
pred_beta = []
one_step_residuals = []
one_step_noise = []

for t in range(window_size, T):
    past_time = time[t - window_size:t]
    past_Y = Y_t[t - window_size:t]
    past_X = X_t[t - window_size:t]

    # Estimate mu_t using past data
    past_basis_mu = dmatrix("bs(time, df=8, degree=3)", {"time": past_time}, return_type='dataframe')
    model_mu.fit(past_basis_mu, past_Y)
    pred_mu_t = model_mu.predict(basis_mu.iloc[[t]])[0]

    # Estimate beta_t using residuals
    past_residuals = past_Y - model_mu.predict(past_basis_mu)
    past_basis_beta = dmatrix("bs(time, df=6, degree=3)", {"time": past_time}, return_type='dataframe')
    past_X_design = np.multiply(past_basis_beta.values, past_X[:, None])
    model_beta.fit(past_X_design, past_residuals)
    pred_beta_t = model_beta.predict(np.multiply(basis_beta.iloc[[t]].values, X_t[t]))[0]

    # Final prediction
    pred = pred_mu_t + pred_beta_t
    predictions_one_step.append(pred)

    # Estimate noise variance
    pred_error = np.std(past_Y - model_mu.predict(past_basis_mu) - model_beta.predict(past_X_design))
    predicted_noise.append(pred_error)
    interval = 1.96 * pred_error
    pred_noise_intervals.append((pred_error - interval, pred_error + interval))

    # Store predicted components
    pred_mu.append(pred_mu_t)
    pred_beta.append(pred_beta_t)
    one_step_residuals.append(Y_t[t] - pred)
    one_step_noise.append(epsilon[t])

# Plot predicted mu_t
plt.figure(figsize=(10, 6))
plt.plot(time[window_size:], mu_t[window_size:], label='True $\\mu_t$', color='green')
plt.plot(time[window_size:], pred_mu, '--', label='Predicted $\\hat{\\mu}_t$', color='red')
plt.legend()
plt.xlabel('Time')
plt.ylabel('$\\mu_t$')
plt.title('One-Step Ahead Predictions for $\\mu_t$')
plt.grid(True)
plt.show()

# Plot predicted beta_t
plt.figure(figsize=(10, 6))
plt.plot(time[window_size:], beta_t[window_size:], label='True $\\beta_t$', color='blue')
plt.plot(time[window_size:], pred_beta, '--', label='Predicted $\\hat{\\beta}_t$', color='orange')
plt.legend()
plt.xlabel('Time')
plt.ylabel('$\\beta_t$')
plt.title('One-Step Ahead Predictions for $\\beta_t$')
plt.grid(True)
plt.show()

# Residual Analysis for One-Step Ahead Predictions
plt.figure(figsize=(10, 6))
plt.plot(time[window_size:], one_step_residuals, label='Residuals', color='purple')
plt.plot(time[window_size:], one_step_noise, '--', label='True Noise', color='green')
plt.fill_between(time[window_size:], [i[0] for i in pred_noise_intervals], [i[1] for i in pred_noise_intervals], color='red', alpha=0.2, label='95% CI')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Residuals / Noise')
plt.title('One-Step Ahead Residuals vs True Noise with 95% CI')
plt.grid(True)
plt.show()

print(f'Mean Squared Error: {np.mean(np.array(one_step_residuals) ** 2):.4f}')
