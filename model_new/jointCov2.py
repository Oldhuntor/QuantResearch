import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ZeroMean, ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt

class TimeVaryingGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(TimeVaryingGP, self).__init__(train_x, train_y, likelihood)

        # Separate kernels for each component
        self.beta_kernel = ScaleKernel(RBFKernel())
        # self.beta_kernel.base_kernel.register_prior(
        #     'lengthscale_prior',
        #     gpytorch.priors.GammaPrior(10.0, 20.0),
        #     'lengthscale'
        # )
        self.mu_kernel = ScaleKernel(RBFKernel())
        self.eps_kernel = ScaleKernel(RBFKernel())

        self.mean = ZeroMean()

    def forward(self, x):
        # Extract time and covariates
        t = x[:, 0]  # time index
        X = x[:, 1]  # covariate

        # Compute kernel matrices
        K_beta = self.beta_kernel(t)
        K_mu = self.mu_kernel(t)
        K_eps = self.eps_kernel(t)

        # Compute covariance matrix
        covar = X.unsqueeze(1) * K_beta * X.unsqueeze(0) + K_mu + K_eps

        mean = self.mean(x)
        return MultivariateNormal(mean, covar)


def train_model(X, y, n_iter=100):
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X).clone().detach().float()
        y = torch.from_numpy(y).clone().detach().float()
    else:
        X = X.clone().detach().float()
        y = y.clone().detach().float()

    # Initialize model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = TimeVaryingGP(X, y, likelihood)

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training loop
    model.train()
    likelihood.train()

    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()

    return model, likelihood


def predict_latent(model, X_train, y_train, X_new):
    model.eval()
    jitter = 1e-6

    with torch.no_grad():
        t_train = X_train[:, 0]
        x_train = X_train[:, 1]
        t_new = X_new[:, 0]
        x_new = X_new[:, 1]

        K_beta = model.beta_kernel(t_new, t_train).evaluate()
        K_mu = model.mu_kernel(t_new, t_train).evaluate()
        K_eps = model.eps_kernel(t_new, t_train).evaluate()

        K_total = x_train * model.beta_kernel(t_train).evaluate() * x_train.unsqueeze(-1) + \
                  model.mu_kernel(t_train).evaluate() + \
                  model.eps_kernel(t_train).evaluate() + \
                  model.likelihood.noise * torch.eye(len(t_train)) + \
                  jitter * torch.eye(len(t_train))

        K_new_beta = model.beta_kernel(t_new).evaluate() + jitter * torch.eye(len(t_new))
        K_new_mu = model.mu_kernel(t_new).evaluate() + jitter * torch.eye(len(t_new))
        K_new_eps = model.eps_kernel(t_new).evaluate() + jitter * torch.eye(len(t_new))

        # Compute posterior mean using Cholesky
        L = torch.linalg.cholesky(K_total)
        alpha = torch.linalg.solve_triangular(L, y_train.unsqueeze(1), upper=False)
        alpha = torch.linalg.solve_triangular(L.T, alpha, upper=True)

        K_stacked = torch.stack([
            x_new.unsqueeze(-1) * K_beta,
            K_mu,
            K_eps
        ])
        posterior_mean = K_stacked @ alpha

        # Compute posterior variance using Cholesky
        v_beta = torch.linalg.solve_triangular(L, (x_train * K_beta.T).T, upper=False)
        v_mu = torch.linalg.solve_triangular(L, K_mu.T, upper=False)
        v_eps = torch.linalg.solve_triangular(L, K_eps.T, upper=False)

        post_var_beta = K_new_beta - v_beta.T @ v_beta
        post_var_mu = K_new_mu - v_mu.T @ v_mu
        post_var_eps = K_new_eps - v_eps.T @ v_eps

        return {
            'mean': {
                'beta': posterior_mean[0].squeeze(),
                'mu': posterior_mean[1].squeeze(),
                'epsilon': posterior_mean[2].squeeze()
            },
            'variance': {
                'beta': post_var_beta.diag(),
                'mu': post_var_mu.diag(),
                'epsilon': post_var_eps.diag()
            }
        }


def predict(model, likelihood, X_new, X_train=None, y_train=None):
    model.eval()
    likelihood.eval()

    if X_train is None:
        X_train = model.train_inputs[0]
    if y_train is None:
        y_train = model.train_targets

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X_new))

    latent_values = predict_latent(model, X_train, y_train, X_new)
    return observed_pred.mean, observed_pred.variance, latent_values


def generate_synthetic_data(n_points=100):
    # Time points
    t = np.linspace(0, 10, n_points)

    # Generate true functions
    true_beta = 0.5 * np.sin(t / 2)
    true_mu = 0.3 * np.cos(t / 3)
    true_eps = np.random.normal(0, 0.1, n_points)

    # Generate covariate X
    X = np.random.normal(0, 1, n_points)

    # Generate Y
    Y = X * true_beta + true_mu + true_eps

    # Combine time and X into features
    features = np.column_stack((t, X))

    return features, Y, {'beta': true_beta, 'mu': true_mu, 'eps': true_eps}


def plot_results(features, Y, true_values, predictions, latent):
    t = features[:, 0]
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))

    # Plot Y
    axes[0].plot(t, Y, 'k.', label='Observed')
    axes[0].plot(t, predictions['mean'], 'r-', label='Predicted')
    axes[0].fill_between(t,
                         predictions['mean'] - 2 * np.sqrt(predictions['variance']),
                         predictions['mean'] + 2 * np.sqrt(predictions['variance']),
                         alpha=0.2)
    axes[0].set_title('Observations and Predictions')
    axes[0].legend()


    predictions = latent
    # Plot beta
    axes[1].plot(t, true_values['beta'], 'k-', label='True')
    axes[1].plot(t, predictions['mean']['beta'], 'r--', label='Predicted')
    axes[1].fill_between(t,
                         predictions['mean']['beta'] - 2 * np.sqrt(predictions['variance']['beta']),
                         predictions['mean']['beta'] + 2 * np.sqrt(predictions['variance']['beta']),
                         color='r', alpha=0.2)
    axes[1].set_title('Beta')
    axes[1].legend()

    # Plot mu
    axes[2].plot(t, true_values['mu'], 'k-', label='True')
    axes[2].plot(t, predictions['mean']['mu'], 'r--', label='Predicted')
    axes[2].fill_between(t,
                         predictions['mean']['mu'] - 2 * np.sqrt(predictions['variance']['mu']),
                         predictions['mean']['mu'] + 2 * np.sqrt(predictions['variance']['mu']),
                         color='r', alpha=0.2)
    axes[2].set_title('Mu')
    axes[2].legend()

    # Plot epsilon
    axes[3].plot(t, true_values['eps'], 'k-', label='True')
    axes[3].plot(t, predictions['mean']['epsilon'], 'r--', label='Predicted')
    axes[3].fill_between(t,
                         predictions['mean']['epsilon'] - 2 * np.sqrt(
                             predictions['variance']['epsilon']),
                         predictions['mean']['epsilon'] + 2 * np.sqrt(
                             predictions['variance']['epsilon']),
                         color='r', alpha=0.2)
    axes[3].set_title('Epsilon')
    axes[3].legend()

    plt.tight_layout()
    plt.show()


def fit_HGP(x, y):
    # Generate data
    n_points = len(x)
    t = np.linspace(0, 10, n_points)

    features = np.column_stack((t, x))

    # Convert to torch tensors
    X_torch = torch.from_numpy(features).float()
    Y_torch = torch.from_numpy(y).float()

    # Initialize and train model
    # likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # model = TimeVaryingGP(X_torch, Y_torch, likelihood)
    model, likelihood = train_model(X_torch, Y_torch)

    # Make predictions
    mean, variance, latent_values = predict(model, likelihood, X_torch)


    beta_t_est = latent_values['mean']['beta']
    mu_t_est = latent_values['mean']['mu']
    eps_t_est = latent_values['mean']['epsilon']

    beta_std_est = latent_values['variance']['beta']
    mu_std_est = latent_values['variance']['mu']
    eps_std_est = latent_values['variance']['epsilon']

    data = {
            'y': {
                'mean': mean,
                'upper': mean + 1.96 * np.sqrt(variance),
                'lower': mean - 1.96 * np.sqrt(variance)
            },
            'beta': {
                'mean': beta_t_est,
                'upper': beta_t_est + 1.96 *np.sqrt( beta_std_est),
                'lower': beta_t_est - 1.96 * np.sqrt(beta_std_est)
            },
            'mu': {
                'mean': mu_t_est,
                'upper': mu_t_est + 1.96 * np.sqrt(mu_std_est),
                'lower': mu_t_est - 1.96 * np.sqrt(mu_std_est)
            },
            'epsilon': {
                'mean': eps_t_est,
                'upper': eps_t_est + 1.96 * np.sqrt(eps_std_est),
                'lower': eps_t_est - 1.96 * np.sqrt(eps_std_est)
            },
    }

    return data




# Example usage
if __name__ == "__main__":
    # Generate data
    features, Y, true_values = generate_synthetic_data()

    # Convert to torch tensors
    X_torch = torch.from_numpy(features).float()
    Y_torch = torch.from_numpy(Y).float()

    # Initialize and train model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = TimeVaryingGP(X_torch, Y_torch, likelihood)
    model, likelihood = train_model(X_torch, Y_torch)

    # Make predictions
    mean, variance, latent_values = predict(model, likelihood, X_torch)

    # Plot results
    predictions = {
        'mean': mean.numpy(),
        'variance': variance.numpy(),
        'latent': {
            'beta': latent_values['mean']['beta'].numpy(),
            'mu': latent_values['mean']['mu'].numpy(),
            'epsilon': latent_values['mean']['epsilon'].numpy()
        }
    }

    plot_results(features, Y, true_values, predictions, latent_values)