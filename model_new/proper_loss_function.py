import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel


# Define the GP for beta_t
class BetaGP(ExactGP):
    def __init__(self, train_t, train_beta, likelihood):
        super(BetaGP, self).__init__(train_t, train_beta, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Define the GP for mu_t
class MuGP(ExactGP):
    def __init__(self, train_t, train_mu, likelihood):
        super(MuGP, self).__init__(train_t, train_mu, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Training data
torch.manual_seed(42)
n = 50  # Number of data points
train_t = torch.linspace(0, 10, n).reshape(-1, 1)  # Time points
X = torch.rand(n, 1)  # Input features
beta_true = 2.0 * torch.sin(train_t).squeeze()  # True beta_t
mu_true = 3.0 * torch.cos(train_t).squeeze()  # True mu_t
y_obs = beta_true * X.squeeze() + mu_true + 0.1 * torch.randn(n)  # Observations with noise

# Define likelihoods and models
likelihood_beta = GaussianLikelihood()
likelihood_mu = GaussianLikelihood()
model_beta = BetaGP(train_t, y_obs / X.squeeze(), likelihood_beta)
model_mu = MuGP(train_t, y_obs, likelihood_mu)

# Optimizer
optimizer = torch.optim.Adam(
    [{"params": model_beta.parameters()}, {"params": model_mu.parameters()}],
    lr=0.01
)

# Training loop
num_iter = 500
mll_beta = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_beta, model_beta)
mll_mu = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_mu, model_mu)

model_beta.train()
model_mu.train()
likelihood_beta.train()
likelihood_mu.train()

for i in range(num_iter):
    optimizer.zero_grad()

    # Predict beta_t and mu_t
    beta_pred = model_beta(train_t).mean  # Predict beta_t
    mu_pred = model_mu(train_t).mean  # Predict mu_t

    # Compute residuals
    residuals = y_obs - X.squeeze() * beta_pred - mu_pred

    # Residual-based loss
    loss = torch.mean(residuals ** 2)  # Mean squared error as the loss

    # Backpropagation
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"Iteration {i + 1}/{num_iter}, Loss: {loss.item():.4f}")
