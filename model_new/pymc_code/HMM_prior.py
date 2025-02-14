import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import matplotlib
# matplotlib.use('TkAgg')
# ---- 1. Data Generation ----
np.random.seed(0)
T = np.arange(1, 101)
x = np.random.rand(100) * 10
beta_true = np.sin(T / 10) + 2
mu_true = np.cos(T / 20) * 5
epsilon = np.random.normal(0, 2, 100)

# AR(1) Errors
ar_coef = 0.7
epsilon_ar = np.zeros(100)
epsilon_ar[0] = epsilon[0]
for t in range(1, 100):
    epsilon_ar[t] = ar_coef * epsilon_ar[t-1] + epsilon[t]

y = beta_true * x + mu_true + epsilon_ar

# ---- 2. PyMC Model with HMM Prior ----
n_states = 2  # Number of hidden states

with pm.Model(coords={"state": np.arange(n_states)}) as model:  # <-- Define 'state' dimension

    # Transition probabilities (Markov Chain)
    transition_probs = pm.Dirichlet('trans_probs', a=np.ones((n_states, n_states)))

    # Initial state probabilities
    init_probs = pm.Dirichlet('init_probs', a=np.ones(n_states))

    # Hidden state sequence (Discrete)
    z = pm.Categorical('z', p=init_probs, shape=len(T))  # Define the sequence of hidden states

    # Priors for each state using defined 'state' dimension
    beta = pm.Normal('beta', mu=[1, 3], sigma=1, dims='state')  # Define prior for beta
    mu = pm.Normal('mu', mu=[0, 5], sigma=2, dims='state')      # Define prior for mu

    # Observation model
    sigma = pm.HalfCauchy('sigma', beta=1)
    Y_obs = pm.Normal('Y_obs', mu=beta[z] * x + mu[z], sigma=sigma, observed=y)

    # ---- 3. Inference ----
    trace = pm.sample(1000, tune=1000, target_accept=0.9)
    ppc = pm.sample_posterior_predictive(trace, var_names=["Y_obs"], random_seed=42)


az.plot_trace(trace, var_names=['beta', 'mu', 'z'])
az.summary(trace, var_names=['beta', 'mu', 'z'])

z_post = trace.posterior['z'].mean(dim=('chain', 'draw'))
plt.figure(figsize=(12, 6))
plt.step(T, z_post, label='Estimated States')
plt.title('Hidden States Over Time')
plt.xlabel('Time Step')
plt.ylabel('State')
plt.grid(True)
plt.legend()
plt.show()

# plt.figure(figsize=(12, 6))
# plt.plot(T, y, label='Observed Y', color='blue')
# plt.plot(T, ppc['Y_obs'].mean(axis=0), label='Predicted Y', color='orange')
# plt.fill_between(T,
#                  np.percentile(ppc['Y_obs'], 2.5, axis=0),
#                  np.percentile(ppc['Y_obs'], 97.5, axis=0),
#                  alpha=0.3, color='orange', label='95% PI')
# plt.title('Observed vs Predicted Y with Prediction Interval')
# plt.xlabel('Time Step')
# plt.ylabel('Y')
# plt.legend()
# plt.grid(True)
# plt.show()
az.plot_ppc(ppc, num_pp_samples=100)
plt.show()