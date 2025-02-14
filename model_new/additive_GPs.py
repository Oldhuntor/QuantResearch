import numpy as np
import matplotlib.pyplot as plt
import GPy

# Joint Gaussian Process Model for Time-Varying Coefficients
def joint_gp_model(X_t, Y_t):
    T = len(X_t)
    time = np.linspace(0, 1, T).reshape(-1, 1)

    # Stack outputs [beta_t, mu_t, epsilon_t]
    Y_combined = np.hstack([Y_t * X_t[:, None], Y_t[:, None], np.zeros((T, 1))])

    # Define kernel using Coregionalize for multi-output
    kernel = GPy.kern.RBF(input_dim=1) ** GPy.kern.Coregionalize(input_dim=1, output_dim=3)

    # Prepare inputs
    X_time = np.tile(time, 3)[:, None]  # Time only
    output_index = np.repeat(np.arange(3), T)[:, None]  # Output index
    X_train = np.hstack([X_time, output_index])  # Combine time and output indices

    # Prepare outputs
    Y_train = Y_combined.flatten()[:, None]

    # Fit Gaussian Process Model
    model = GPy.models.GPRegression(X_time, Y_train, kernel=kernel)
    model.optimize()

    # Predictions and uncertainties
    pred_means, pred_vars = model.predict(X_time)

    # Extract estimates
    beta_t_est = pred_means[:T].flatten()
    mu_t_est = pred_means[T:2*T].flatten()
    noise_est = pred_means[2*T:].flatten()

    beta_std_est = np.sqrt(pred_vars[:T].flatten())
    mu_std_est = np.sqrt(pred_vars[T:2*T].flatten())
    noise_std_est = np.sqrt(pred_vars[2*T:].flatten())

    # Residuals
    Y_pred = beta_t_est * X_t + mu_t_est
    residuals = Y_t - Y_pred

    return beta_t_est, mu_t_est, beta_std_est, mu_std_est, residuals, noise_est, noise_std_est


# Example Usage for Joint GP Model
np.random.seed(42)
T = 300

time = np.linspace(0, 30, T)
X_t = np.random.normal(0, 1, T)

# True coefficients
beta_t = 1 + 0.5 * np.sin(0.5 * time)
mu_t = 0.2 * np.cos(0.3 * time)

# Noise and Response
noise_variance = 0.1 * (1 + np.abs(2 * np.sin(0.3 * time)**2 + 1.5 * np.cos(0.15 * time)))
epsilon = np.random.normal(0, noise_variance, T)
Y_t = beta_t * X_t + mu_t + epsilon

# Normalize data
X_t = (X_t - np.mean(X_t)) / np.std(X_t)
Y_t = (Y_t - np.mean(Y_t)) / np.std(Y_t)

# Run Joint Gaussian Process Model
beta_t_est, mu_t_est, beta_std_est, mu_std_est, residuals, noise_est, noise_std_est = joint_gp_model(X_t, Y_t)

# Plot Results
plt.figure(figsize=(10, 6))
plt.plot(time, beta_t, label=r'True $\beta_t$', color='blue')
plt.plot(time, beta_t_est, '--', label=r'Estimated $\beta_t$', color='orange')
plt.fill_between(time, beta_t_est - 1.96 * beta_std_est, beta_t_est + 1.96 * beta_std_est, color='orange', alpha=0.2)
plt.legend()
plt.xlabel('Time')
plt.ylabel(r'$\beta_t$')
plt.title(r'Estimated vs True $\beta_t$ with Uncertainty')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, mu_t, label=r'True $\mu_t$', color='green')
plt.plot(time, mu_t_est, '--', label=r'Estimated $\mu_t$', color='red')
plt.fill_between(time, mu_t_est - 1.96 * mu_std_est, mu_t_est + 1.96 * mu_std_est, color='red', alpha=0.2)
plt.legend()
plt.xlabel('Time')
plt.ylabel(r'$\mu_t$')
plt.title(r'Estimated vs True $\mu_t$ with Uncertainty')
plt.grid(True)
plt.show()

# Residuals and noise analysis
plt.figure(figsize=(10, 6))
plt.plot(time, residuals, label='Residuals', color='purple')
plt.plot(time, epsilon, '--', label='True Noise', color='gray')
plt.fill_between(time, noise_est - 1.96 * noise_std_est, noise_est + 1.96 * noise_std_est, color='gray', alpha=0.2)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Residuals and Noise')
plt.title('Residuals vs True Noise with Estimated Uncertainty')
plt.grid(True)
plt.show()

print("Joint Gaussian Process Model Analysis Complete!")
