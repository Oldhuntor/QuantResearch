import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az


def bsts_fit(x,y):
    with pm.Model() as model:
        # Priors for variances

        # sigma = pm.HalfCauchy('sigma', beta=1)
        # Random walk for log volatility
        log_sigma = pm.GaussianRandomWalk('log_sigma',
                                          sigma=0.1,
                                          shape=len(y),
                                          init_dist=pm.Normal.dist(mu=0, sigma=1))
        sigma = pm.Deterministic('sigma', pm.math.exp(log_sigma))

        sigma_beta = pm.HalfCauchy('sigma_beta', beta=1)
        sigma_mu = pm.HalfCauchy('sigma_mu', beta=1)

        # Gaussian Random Walks for beta and mu
        beta = pm.GaussianRandomWalk('beta', sigma=sigma_beta, init_dist=pm.Normal.dist(0, 10), shape=len(y))
        mu = pm.GaussianRandomWalk('mu', sigma=sigma_mu, init_dist=pm.Normal.dist(0, 10), shape=len(y))

        # Observation model
        Y_obs = pm.Normal('Y_obs', mu=beta * x + mu, sigma=sigma, observed=y)

        # ---- 3. MCMC Sampling ----
        trace = pm.sample(3000, tune=1000, chains=4, target_accept=0.9)
        ppc = pm.sample_posterior_predictive(trace, var_names=["Y_obs"], random_seed=42)

        # Extract posterior mean and 95% credible intervals
        beta_posterior = trace.posterior['beta'].mean(dim=("chain", "draw"))
        beta_lower = trace.posterior['beta'].quantile(0.025, dim=("chain", "draw"))
        beta_upper = trace.posterior['beta'].quantile(0.975, dim=("chain", "draw"))

        mu_posterior = trace.posterior['mu'].mean(dim=("chain", "draw"))
        mu_lower = trace.posterior['mu'].quantile(0.025, dim=("chain", "draw"))
        mu_upper = trace.posterior['mu'].quantile(0.975, dim=("chain", "draw"))

        # Extract posterior predictive samples
        y_pred_samples = ppc.posterior_predictive['Y_obs']

        # Calculate mean and 95% prediction interval
        y_pred_mean = y_pred_samples.mean(dim=('chain', 'draw')).values
        y_pred_lower = np.percentile(y_pred_samples.values, 2.5, axis=(0, 1))  # 2.5% quantile
        y_pred_upper = np.percentile(y_pred_samples.values, 97.5, axis=(0, 1))  # 97.5% quantile

        epsilon_samples = y - y_pred_samples.values  # Shape: (chains, draws, time)

        # ---- 3. Calculate Mean and Intervals ----
        # Mean residuals
        epsilon_mean = epsilon_samples.mean(axis=(0, 1))  # Average over chains and draws

        # 95% prediction intervals
        epsilon_lower = np.percentile(epsilon_samples, 2.5, axis=(0, 1))
        epsilon_upper = np.percentile(epsilon_samples, 97.5, axis=(0, 1))

        data = {
            'y': {
                'mean': y_pred_mean,
                'upper': y_pred_upper,
                'lower': y_pred_lower
            },
            'beta': {
                'mean': beta_posterior,
                'upper': beta_upper,
                'lower': beta_lower
            },
            'mu': {
                'mean': mu_posterior,
                'upper': mu_upper,
                'lower': mu_lower
            },
            'epsilon': {
                'mean': epsilon_mean,
                'upper': epsilon_upper,
                'lower': epsilon_lower
            },
        }

        return data


if __name__ == '__main__':
    # # ---- 1. Generate Data ----
    T = np.arange(1, 101)
    x = np.random.rand(100) * 10
    beta_true = np.sin(T / 10) + 2
    mu_true = np.cos(T / 20) * 5
    epsilon = np.random.normal(0, 2, 100)
    y = beta_true * x + mu_true + epsilon

    # ---- 2. PyMC Model ----
    trace, ppc = bsts_fit(x, y)

    # ---- 4. Posterior Analysis ----
    # Extract posterior mean and 95% credible intervals
    beta_posterior = trace.posterior['beta'].mean(dim=("chain", "draw"))
    beta_lower = trace.posterior['beta'].quantile(0.025, dim=("chain", "draw"))
    beta_upper = trace.posterior['beta'].quantile(0.975, dim=("chain", "draw"))

    mu_posterior = trace.posterior['mu'].mean(dim=("chain", "draw"))
    mu_lower = trace.posterior['mu'].quantile(0.025, dim=("chain", "draw"))
    mu_upper = trace.posterior['mu'].quantile(0.975, dim=("chain", "draw"))


    # az.plot_ppc(ppc, num_pp_samples=100)
    # plt.show()

    # Extract observed data
    y_obs = ppc.observed_data['Y_obs'].values

    # Extract posterior predictive samples
    y_pred_samples = ppc.posterior_predictive['Y_obs']

    # Calculate mean and 95% prediction interval
    y_pred_mean = y_pred_samples.mean(dim=('chain', 'draw')).values
    y_pred_lower = np.percentile(y_pred_samples.values, 2.5, axis=(0, 1))  # 2.5% quantile
    y_pred_upper = np.percentile(y_pred_samples.values, 97.5, axis=(0, 1)) # 97.5% quantile

    # ---- 2. Calculate Epsilon ----
    # Compute residuals for each sample
    epsilon_samples = y_obs - y_pred_samples.values  # Shape: (chains, draws, time)

    # ---- 3. Calculate Mean and Intervals ----
    # Mean residuals
    epsilon_mean = epsilon_samples.mean(axis=(0, 1))  # Average over chains and draws

    # 95% prediction intervals
    epsilon_lower = np.percentile(epsilon_samples, 2.5, axis=(0, 1))
    epsilon_upper = np.percentile(epsilon_samples, 97.5, axis=(0, 1))


    def infer_para(x):
        # New observation
        X_new = x  # New X_t

        # Extract last time step for beta and mu
        beta_last = trace.posterior['beta'].isel(beta_dim_0=-1).values  # Shape: (chains, draws)
        mu_last = trace.posterior['mu'].isel(mu_dim_0=-1).values  # Shape: (chains, draws)

        # Calculate predicted mean Y_pred for each posterior sample
        Y_pred = beta_last * X_new + mu_last
        # Flatten chains and draws
        Y_pred_flat = Y_pred.flatten()

        # Plot posterior predictive distribution for Y_pred
        az.plot_posterior(Y_pred_flat, hdi_prob=0.95)
        plt.title("Posterior Predictive Distribution for Y_pred (Exact Mean)")
        plt.grid(True)
        plt.show()

        return Y_pred_flat, beta_last, mu_last

    # ---- 5. Plot Results ----

    # Create subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 20))  # 4 rows, 1 column

    # ---- 1. Time-Series PPC ----
    axes[0].plot(T, y_obs, label='Observed Y', color='blue')  # Observed data
    axes[0].plot(T, y_pred_mean, label='Predicted Mean', color='orange')  # Predicted mean
    axes[0].fill_between(T, y_pred_lower, y_pred_upper, alpha=0.3, color='orange', label='95% PI')  # Prediction interval
    axes[0].set_title('Time Series Posterior Predictive Check')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Y')
    axes[0].legend()
    axes[0].grid(True)

    # ---- 2. Beta ----
    axes[1].plot(T, beta_true, label="True Beta", color='blue')
    axes[1].plot(T, beta_posterior, label="Estimated Beta", color='orange')
    axes[1].fill_between(T, beta_lower, beta_upper, alpha=0.3, color='orange', label='95% CI')
    axes[1].set_title("Beta Over Time")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Beta")
    axes[1].legend()
    axes[1].grid(True)

    # ---- 3. Mu ----
    axes[2].plot(T, mu_true, label="True Mu", color='blue')
    axes[2].plot(T, mu_posterior, label="Estimated Mu", color='orange')
    axes[2].fill_between(T, mu_lower, mu_upper, alpha=0.3, color='orange', label='95% CI')
    axes[2].set_title("Mu Over Time")
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("Mu")
    axes[2].legend()
    axes[2].grid(True)

    # ---- 4. Residual Errors (Epsilon) ----
    axes[3].plot(T, epsilon_mean, label="Mean Epsilon", color='green')  # Mean
    axes[3].plot(T, epsilon, label='True Epsilon', color='orange')
    axes[3].fill_between(T, epsilon_lower, epsilon_upper, alpha=0.3, color='green', label="95% PI")  # 95% Interval
    axes[3].set_title("Residual Errors (Epsilon) with 95% Prediction Interval")
    axes[3].set_xlabel("Time Step")
    axes[3].set_ylabel("Epsilon")
    axes[3].legend()
    axes[3].grid(True)

    # Adjust layout for readability
    plt.tight_layout()
    plt.show()
