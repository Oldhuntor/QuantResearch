import torch
import gpytorch
from matplotlib import pyplot as plt

# Generate synthetic data
torch.manual_seed(42)
N = 100
t = torch.linspace(0, 10, N).reshape(-1, 1)  # Time points
X = torch.sin(t) + 0.1 * torch.randn(N, 1)   # Predictor
true_beta = 2.0 + 0.5 * torch.sin(0.5 * t)   # True beta_t
mu = 1.0 + 0.3 * torch.cos(0.3 * t)         # True mu_t
epsilon = 0.1 * torch.randn(N, 1)           # Noise
Y = true_beta * X + mu + epsilon            # Response

# Prepare multi-task training data
train_x = torch.cat([t, t], dim=0)  # Duplicate time for two tasks
task_indices = torch.cat([torch.zeros(N), torch.ones(N)]).long()  # Task indices: 0=mu, 1=beta
train_y = torch.cat([Y.flatten(), X.flatten()])  # Combine outputs
# Prepare multi-task training data
train_x = t
train_y = torch.stack([Y.flatten(), X.flatten()], dim=-1)  # Shape [N, num_tasks]
task_indices = torch.arange(2).repeat_interleave(N)  # Not used directly, for clarity


# Define the Multi-output GP model
class MultiTaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        # Mean module
        self.mean_module = gpytorch.means.ConstantMean()

        # Data covariance module (input kernel)
        self.data_covar_module = gpytorch.kernels.RBFKernel()

        # Task covariance module
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)

        # Multitask kernel: combines data and task covariances
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            self.data_covar_module,
            num_tasks=2,
            rank=1
        )

    def forward(self, x):
        # Mean: shape [N, num_tasks]
        mean_x = self.mean_module(x).unsqueeze(-1).repeat(1, 2)

        # Covariance: combines input and task kernels
        covar = self.covar_module(x)

        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar)


# Define likelihood and model
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
model = MultiTaskGPModel(train_x, train_y, likelihood)

# Training
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(100):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)  # Shape [N, num_tasks]
    loss.backward()
    print(f"Iteration {i + 1}/100 - Loss: {loss.item():.4f}")
    optimizer.step()

# Evaluation
model.eval()
likelihood.eval()

with torch.no_grad():
    preds = likelihood(model(train_x))

# Extract predictions for mu and beta
mu_pred = preds.mean[:, 0]
beta_pred = preds.mean[:, 1]

# Plot results
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Mu_t predictions
axes[0].plot(t.numpy(), mu.numpy(), 'g-', label="True Mu")
axes[0].plot(t.numpy(), mu_pred.numpy(), 'b--', label="Predicted Mu")
axes[0].set_title("Mu_t: True vs Predicted")
axes[0].legend()

# Beta_t predictions
axes[1].plot(t.numpy(), true_beta.numpy(), 'g-', label="True Beta")
axes[1].plot(t.numpy(), beta_pred.numpy(), 'b--', label="Predicted Beta")
axes[1].set_title("Beta_t: True vs Predicted")
axes[1].legend()

plt.tight_layout()
plt.show()
