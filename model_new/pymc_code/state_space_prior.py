import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import matplotlib
# matplotlib.use('TkAgg')

T = np.arange(1, 101)  # Time steps


with pm.Model() as prior_model:
    sigma_beta = pm.HalfCauchy('sigma_beta', beta=1)
    beta_prior = pm.GaussianRandomWalk('beta', sigma=sigma_beta, init_dist=pm.Normal.dist(0, 10), shape=len(T))

    sigma_mu = pm.HalfCauchy('sigma_mu', beta=1)
    mu_prior = pm.GaussianRandomWalk('mu', sigma=sigma_mu, init_dist=pm.Normal.dist(0, 10), shape=len(T))

    prior_samples = pm.sample_prior_predictive(1000)


# Extract prior samples from the dataset
mu_prior = prior_samples.prior['mu'].values  # Shape: (chain, draw, 100)
beta_prior = prior_samples.prior['beta'].values  # Shape: (chain, draw, 100)

# Compute summary statistics for the prior
mu_mean = mu_prior.mean(axis=(0, 1))  # Mean across chains and draws
mu_lower = np.percentile(mu_prior, 2.5, axis=(0, 1))  # 2.5% quantile
mu_upper = np.percentile(mu_prior, 97.5, axis=(0, 1))  # 97.5% quantile

beta_mean = beta_prior.mean(axis=(0, 1))
beta_lower = np.percentile(beta_prior, 2.5, axis=(0, 1))
beta_upper = np.percentile(beta_prior, 97.5, axis=(0, 1))



fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# 1. Mu Prior
axes[0].plot(T, mu_mean, label="Mean Prior for Mu", color='orange')
axes[0].fill_between(T, mu_lower, mu_upper, alpha=0.3, color='orange', label="95% CI")
axes[0].set_title('Prior Distribution for Mu (Intercept)')
axes[0].set_ylabel('Mu')
axes[0].legend()
axes[0].grid(True)

# 2. Beta Prior
axes[1].plot(T, beta_mean, label="Mean Prior for Beta", color='blue')
axes[1].fill_between(T, beta_lower, beta_upper, alpha=0.3, color='blue', label="95% CI")
axes[1].set_title('Prior Distribution for Beta (Slope)')
axes[1].set_ylabel('Beta')
axes[1].set_xlabel('Time Step')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('pymc_statespace_prior.png')
plt.show()
