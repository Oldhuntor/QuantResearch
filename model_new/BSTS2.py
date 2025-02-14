import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

def build_bsts_model(X, y, sigma_beta=0.1, sigma_mu=0.1, sigma_obs=0.1):
    """
    Build a Bayesian Structural Time Series model:
    Yt = Beta_t * Xt + mu_t + epsilon_t
    where Beta_t and mu_t follow random walks
    """
    T = len(y)  # Number of time points

    with pm.Model() as model:
        # Random walk for beta (time-varying coefficient)
        beta = pm.GaussianRandomWalk('beta',
                                     sigma=sigma_beta,
                                     shape=T,
                                     init_dist=pm.Normal.dist(mu=0, sigma=1))

        # Random walk for mu (time-varying level)
        mu = pm.GaussianRandomWalk('mu',
                                   sigma=sigma_mu,
                                   shape=T,
                                   init_dist=pm.Normal.dist(mu=0, sigma=1))

        # Random walk for log volatility
        log_sigma = pm.GaussianRandomWalk('log_sigma',
                                          sigma=0.1,
                                          shape=T,
                                          init_dist=pm.Normal.dist(mu=0, sigma=1))
        
        sigma = pm.Deterministic('sigma', pm.math.exp(log_sigma))

        # Likelihood
        mu_t = beta * X + mu
        y_obs = pm.Normal('y_obs', mu=mu_t, sigma=sigma, observed=y)

    return model


def fit_bsts_model(model, samples=1000, tune=1000, chains=4):
    """
    Fit the BSTS model using NUTS sampler
    """
    with model:
        trace = pm.sample(
            draws=samples,
            tune=tune,
            chains=chains,
            init='jitter+adapt_diag',
            return_inferencedata=True
        )
    return trace


def plot_results_with_components(trace, X, y, true_beta=None, true_mu=None, true_sigma=None):
    """
    Plot the results of the BSTS model including component comparisons
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 20})  # Global font size

    # Extract posterior means and credible intervals
    beta_post = trace.posterior['beta'].mean(dim=['chain', 'draw']).values
    mu_post = trace.posterior['mu'].mean(dim=['chain', 'draw']).values
    sigma_post = trace.posterior['sigma'].mean(dim=['chain', 'draw']).values

    # Calculate 95% credible intervals
    beta_lower = np.percentile(trace.posterior['beta'].values, 2.5, axis=(0, 1))
    beta_upper = np.percentile(trace.posterior['beta'].values, 97.5, axis=(0, 1))
    mu_lower = np.percentile(trace.posterior['mu'].values, 2.5, axis=(0, 1))
    mu_upper = np.percentile(trace.posterior['mu'].values, 97.5, axis=(0, 1))
    sigma_lower = np.percentile(trace.posterior['sigma'].values, 2.5, axis=(0, 1))
    sigma_upper = np.percentile(trace.posterior['sigma'].values, 97.5, axis=(0, 1))

    # Calculate fitted values
    y_fitted = beta_post * X + mu_post
    y_lower = y_fitted - 2 * sigma_post
    y_upper = y_fitted + 2 * sigma_post

    # Create plot
    fig, axes = plt.subplots(4, 1, figsize=(20, 15))

    # Plot Model Fit
    axes[0].plot(y, 'k.', label='Observed', alpha=0.5)
    axes[0].plot(y_fitted, 'b-', label='Fitted')
    axes[0].fill_between(range(len(y)), y_lower, y_upper,
                         color='b', alpha=0.2, label='95% PI')
    axes[0].set_title('Time Series Posterior Predictive Check')
    axes[0].legend()
    
    # Plot Beta
    axes[1].plot(beta_post, 'b-', label='Estimated Beta')
    axes[1].fill_between(range(len(beta_post)), beta_lower, beta_upper,
                         color='b', alpha=0.2, label='95% CI')
    if true_beta is not None:
        axes[1].plot(true_beta, 'r--', label='True Beta')
    axes[1].set_title('Beta Over Time')
    axes[1].legend()

    # Plot Mu
    axes[2].plot(mu_post, 'g-', label='Estimated Mu')
    axes[2].fill_between(range(len(mu_post)), mu_lower, mu_upper,
                         color='g', alpha=0.2, label='95% CI')
    if true_mu is not None:
        axes[2].plot(true_mu, 'r--', label='True Mu')
    axes[2].set_title('Mu Over Time')
    axes[2].legend()

    # Plot Sigma
    axes[3].plot(sigma_post, 'm-', label='Estimated noise level')
    axes[3].fill_between(range(len(sigma_post)), sigma_lower, sigma_upper,
                         color='m', alpha=0.2, label='95% CI')
    if true_sigma is not None:
        axes[3].plot(true_sigma, 'r--', label='True noise level')
    axes[3].set_title('Time-varying noise level')
    axes[3].legend()



    plt.tight_layout()
    return fig


# Example usage:
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    T = 100
    X = np.linspace(0, 10, T)

    # Generate true components
    true_beta = np.random.normal(0, 0.1, size=T).cumsum() + 1
    true_mu = np.random.normal(0, 0.1, size=T).cumsum()
    true_sigma = np.exp(np.random.normal(0, 0.1, size=T).cumsum())

    # Generate observations
    y = true_beta * X + true_mu + np.random.normal(0, true_sigma)

    # Build and fit model
    model = build_bsts_model(X, y)
    trace = fit_bsts_model(model)

    # Plot results with true components
    fig = plot_results_with_components(trace, X, y,
                                       true_beta=true_beta,
                                       true_mu=true_mu,
                                       true_sigma=true_sigma)
    plt.show()