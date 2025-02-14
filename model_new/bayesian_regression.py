import numpy as np
import matplotlib.pyplot as plt


def bayesian_rolling_window(X_t, Y_t, window_size=30):
    T = len(X_t)
    beta_t_est = np.zeros(T)
    mu_t_est = np.zeros(T)
    beta_var_est = np.zeros(T)
    mu_var_est = np.zeros(T)
    residual_var_est = np.zeros(T)
    Y_pred = np.zeros(T)
    Y_std_est = np.zeros(T)

    # Prior parameters
    beta_mean_prior = 0
    beta_var_prior = 1
    mu_mean_prior = 0
    mu_var_prior = 1
    sigma_prior = 1

    for t in range(window_size, T):
        # Get rolling window data
        X_window = np.float64(X_t[t - window_size:t])
        Y_window = np.float64(Y_t[t - window_size:t])

        # Posterior parameters for beta
        XTX = np.sum(X_window ** 2)
        XTY = np.sum(X_window * (Y_window - np.mean(Y_window)))
        beta_var_post = 1 / (1 / beta_var_prior + XTX / sigma_prior)
        beta_mean_post = beta_var_post * (beta_mean_prior / beta_var_prior + XTY / sigma_prior)

        # Posterior parameters for mu
        mu_var_post = 1 / (1 / mu_var_prior + window_size / sigma_prior)
        mu_mean_post = mu_var_post * (mu_mean_prior / mu_var_prior + np.sum(Y_window - beta_mean_post * X_window) / sigma_prior)

        # Estimate residual variance
        residuals_window = Y_window - (beta_mean_post * X_window + mu_mean_post)
        residual_var_est[t] = np.var(residuals_window)

        # Store estimates
        beta_t_est[t] = beta_mean_post
        mu_t_est[t] = mu_mean_post
        beta_var_est[t] = beta_var_post
        mu_var_est[t] = mu_var_post

        # Predict Y_t and its credible interval
        Y_pred[t] = beta_t_est[t] * X_t[t] + mu_t_est[t]
        Y_var_est = (X_t[t] ** 2) * (beta_var_est[t]) + (mu_var_est[t]) + (1 / sigma_prior)
        Y_std_est[t] = np.sqrt(Y_var_est)


        # Prior parameters
        beta_mean_prior = beta_mean_post
        beta_var_prior = beta_var_post
        mu_mean_prior = mu_mean_post
        mu_var_prior = mu_var_post

    residuals = Y_t - Y_pred

    data = {
        'y': {
            'mean': Y_pred,
            'upper': Y_pred + 1.96 * Y_std_est,
            'lower': Y_pred - 1.96 * Y_std_est
        },
        'beta': {
            'mean': beta_t_est,
            'upper':  beta_t_est + 1.96 * np.sqrt(beta_var_est),
            'lower': beta_t_est - 1.96 * np.sqrt(beta_var_est)
        },
        'mu': {
            'mean': mu_t_est,
            'upper': mu_t_est + 1.96 * np.sqrt(mu_var_est),
            'lower': mu_t_est - 1.96 * np.sqrt(mu_var_est)
        },
        'epsilon': {
            'mean': residuals,
            'upper': residuals + 1.96 * np.sqrt(residual_var_est),
            'lower': residuals - 1.96 * np.sqrt(residual_var_est)
        },
    }

    return data


if __name__ == '__main__':


    # 1. Data Generating Process with Heteroskedastic Noise
    np.random.seed(42)
    T = 300  # Number of time points

    time = np.linspace(0, 30, T)  # Time variable
    X_t = np.random.normal(0, 1, T)  # Predictor

    # True time-varying coefficients
    beta_t = 1 + 0.5 * np.sin(0.5 * time)
    mu_t = 0.2 * np.cos(0.3 * time)

    # Generate chaotic heteroskedastic noise
    noise_variance = 0.1 * (1 + np.abs(2 * np.sin(0.3 * time)**2 + 1.5 * np.cos(0.15 * time)))
    epsilon = np.random.normal(0, noise_variance, T)

    # Generate response variable
    Y_t = beta_t * X_t + mu_t + epsilon

    # Normalize data for regression
    X_t = (X_t - np.mean(X_t)) / np.std(X_t)
    Y_t = (Y_t - np.mean(Y_t)) / np.std(Y_t)

    # 2. Bayesian Conjugate Gaussian Regression with Rolling Window
    window_size = 30
    beta_t_est = np.zeros(T)
    mu_t_est = np.zeros(T)
    beta_var_est = np.zeros(T)
    mu_var_est = np.zeros(T)
    residual_var_est = np.zeros(T)

    # Prior parameters
    beta_mean_prior = 0
    beta_var_prior = 1
    mu_mean_prior = 0
    mu_var_prior = 1
    sigma_prior = 1

    for t in range(window_size, T):
        # Get rolling window data
        X_window = X_t[t - window_size:t]
        Y_window = Y_t[t - window_size:t]

        # Posterior parameters for beta
        XTX = np.sum(X_window ** 2)
        XTY = np.sum(X_window * (Y_window - np.mean(Y_window)))

        beta_var_post = 1 / (1 / beta_var_prior + XTX / sigma_prior)
        beta_mean_post = beta_var_post * (beta_mean_prior / beta_var_prior + XTY / sigma_prior)

        # Posterior parameters for mu
        mu_var_post = 1 / (1 / mu_var_prior + window_size / sigma_prior)
        mu_mean_post = mu_var_post * (mu_mean_prior / mu_var_prior + np.sum(Y_window - beta_mean_post * X_window) / sigma_prior)

        # Estimate residual variance
        residuals_window = Y_window - (beta_mean_post * X_window + mu_mean_post)
        residual_var_est[t] = np.var(residuals_window)

        # Store estimates
        beta_t_est[t] = beta_mean_post
        mu_t_est[t] = mu_mean_post
        beta_var_est[t] = beta_var_post
        mu_var_est[t] = mu_var_post

        beta_mean_prior = beta_mean_post
        beta_var_prior = beta_var_post
        mu_mean_prior = mu_mean_post
        mu_var_prior = mu_var_post

    # Create subplots
    fig, axs = plt.subplots(4, 1, figsize=(15, 10))

    # Plot beta_t with uncertainty
    axs[0].plot(time, beta_t, label=r'True $\beta_t$', color='blue')
    axs[0].plot(time, beta_t_est, '--', label=r'Estimated $\beta_t$', color='orange')
    axs[0].fill_between(time, beta_t_est - 1.96 * np.sqrt(beta_var_est),
                        beta_t_est + 1.96 * np.sqrt(beta_var_est), color='orange', alpha=0.2)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel(r'$\beta_t$')
    axs[0].set_title(r'Estimated vs True $\beta_t$ with Uncertainty')
    axs[0].legend()
    axs[0].grid(True)

    # Plot mu_t with uncertainty
    axs[1].plot(time, mu_t, label=r'True $\mu_t$', color='green')
    axs[1].plot(time, mu_t_est, '--', label=r'Estimated $\mu_t$', color='red')
    axs[1].fill_between(time, mu_t_est - 1.96 * np.sqrt(mu_var_est),
                        mu_t_est + 1.96 * np.sqrt(mu_var_est), color='red', alpha=0.2)
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel(r'$\mu_t$')
    axs[1].set_title(r'Estimated vs True $\mu_t$ with Uncertainty')
    axs[1].legend()
    axs[1].grid(True)

    # Plot Y_t vs estimated Y_t
    Y_t_est = beta_t_est * X_t + mu_t_est
    axs[2].plot(time, Y_t, label=r'True $Y_t$', color='blue')
    axs[2].plot(time, Y_t_est, '--', label=r'Estimated $\hat{Y_t}$', color='orange')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel(r'$Y_t$')
    axs[2].set_title(r'True vs Estimated $Y_t$')
    axs[2].legend()
    axs[2].grid(True)

    # Residual and noise analysis
    residuals = Y_t - Y_t_est
    epsilon = np.random.normal(0, 0.2, T)
    axs[3].plot(time, residuals, label='Residuals', color='purple')
    axs[3].plot(time, epsilon, '--', label='True Noise', color='gray')
    axs[3].fill_between(time, epsilon - 1.96 * np.sqrt(residual_var_est),
                        epsilon + 1.96 * np.sqrt(residual_var_est), color='gray', alpha=0.2)
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('Residuals and Noise')
    axs[3].set_title('Residuals vs True Noise with Estimated Uncertainty')
    axs[3].legend()
    axs[3].grid(True)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()