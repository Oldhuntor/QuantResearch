import numpy as np
import matplotlib.pyplot as plt
import patsy
from sklearn.linear_model import BayesianRidge

# Bayesian GAM with Splines

def bayesian_gam_with_splines(X_t, Y_t, df=10):
    T = len(X_t)
    time = np.linspace(0, 1, T)

    # Design matrices for splines
    design_matrix = patsy.dmatrix(f"bs(time, df={df}, degree=3)", {"time": time}, return_type='dataframe')

    # Joint Bayesian Ridge Regression for beta_t and mu_t
    X_joint = np.hstack([np.multiply(design_matrix.values, X_t[:, None]), design_matrix.values])
    model_joint = BayesianRidge()
    model_joint.fit(X_joint, Y_t)

    # Predict values with uncertainties
    Y_pred, Y_std = model_joint.predict(X_joint, return_std=True)

    # Separate beta_t and mu_t
    beta_t_est = design_matrix.values @ model_joint.coef_[:design_matrix.shape[1]]
    mu_t_est = design_matrix.values @ model_joint.coef_[design_matrix.shape[1]:]

    # Compute standard deviations for beta and mu
    coef_cov = np.linalg.inv(model_joint.alpha_ * np.eye(X_joint.shape[1]) + model_joint.lambda_ * X_joint.T @ X_joint)
    beta_std_est = np.sqrt(np.sum((design_matrix.values @ coef_cov[:design_matrix.shape[1], :design_matrix.shape[1]]) * design_matrix.values, axis=1))
    mu_std_est = np.sqrt(np.sum((design_matrix.values @ coef_cov[design_matrix.shape[1]:, design_matrix.shape[1]:]) * design_matrix.values, axis=1))

    # Posterior variance of noise
    residual_var_est = 1 / model_joint.alpha_  # Posterior noise variance

    residuals = Y_t - Y_pred

    data = {
        'y': {
            'mean': Y_pred,
            'upper': Y_pred + 1.96 * Y_std,
            'lower': Y_pred - 1.96 * Y_std
        },
        'beta': {
            'mean': beta_t_est,
            'upper': beta_t_est + 1.96 * beta_std_est,
            'lower': beta_t_est - 1.96 * beta_std_est
        },
        'mu': {
            'mean': mu_t_est,
            'upper': mu_t_est + 1.96 * mu_std_est,
            'lower': mu_t_est - 1.96 * mu_std_est
        },
        'epsilon': {
            'mean': residuals,
            'upper': residuals + 1.96 * np.sqrt(residual_var_est),
            'lower': residuals - 1.96 * np.sqrt(residual_var_est)
        },
    }

    return data



if __name__ == '__main__':

    # Example Usage for GAM with Splines
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

    # Run Bayesian GAM with Splines
    beta_t_est, mu_t_est, beta_std_est, mu_std_est, Y_pred, residual_var_est = bayesian_gam_with_splines(X_t, Y_t)

    fig, axes = plt.subplots(4, 1, figsize=(12, 14))  # 3 rows, 1 column

    # Subplot 1: Beta_t
    axes[0].plot(time, beta_t, label=r'True $\beta_t$', color='blue')
    axes[0].plot(time, beta_t_est, '--', label=r'Estimated $\beta_t$', color='orange')
    axes[0].fill_between(time, beta_t_est - 1.96 * beta_std_est,
                         beta_t_est + 1.96 * beta_std_est, color='orange', alpha=0.2)
    axes[0].legend()
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(r'$\beta_t$')
    axes[0].set_title(r'Estimated vs True $\beta_t$ with Uncertainty')
    axes[0].grid(True)

    # Subplot 2: Mu_t
    axes[1].plot(time, mu_t, label=r'True $\mu_t$', color='green')
    axes[1].plot(time, mu_t_est, '--', label=r'Estimated $\mu_t$', color='red')
    axes[1].fill_between(time, mu_t_est - 1.96 * mu_std_est,
                         mu_t_est + 1.96 * mu_std_est, color='red', alpha=0.2)
    axes[1].legend()
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel(r'$\mu_t$')
    axes[1].set_title(r'Estimated vs True $\mu_t$ with Uncertainty')
    axes[1].grid(True)

    # Subplot 3: Residuals and Noise
    residuals = Y_t - Y_pred
    axes[2].plot(time, residuals, label='Residuals', color='purple')
    axes[2].plot(time, epsilon, '--', label='True Noise', color='gray')
    axes[2].fill_between(time, -1.96 * np.sqrt(residual_var_est),
                         1.96 * np.sqrt(residual_var_est), color='gray', alpha=0.2)
    axes[2].legend()
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Residuals and Noise')
    axes[2].set_title('Residuals vs True Noise with Estimated Uncertainty')
    axes[2].grid(True)

    # Subplot 3: Residuals and Noise
    axes[3].plot(time, Y_t, label='observation', color='black')
    axes[3].plot(time, Y_pred, '--', label='Y_pred', color='blue')
    # axes[3].fill_between(time, -1.96 * np.sqrt(residual_var_est),
    #                      1.96 * np.sqrt(residual_var_est), color='gray', alpha=0.2)
    axes[3].legend()
    axes[3].set_xlabel('Time')
    axes[3].set_ylabel('Y and Y_pred')
    axes[3].set_title('Yt observation vs Y prediction with Estimated Uncertainty')
    axes[3].grid(True)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the figure
    plt.show()

    print("Bayesian GAM with Splines Analysis Complete!")
