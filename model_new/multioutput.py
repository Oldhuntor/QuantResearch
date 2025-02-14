import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel, IndexKernel


class MultiOutputBetaGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_X, train_Y, likelihood):
        super(MultiOutputBetaGPModel, self).__init__(train_X, train_Y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        # Shared kernel for the input space
        self.base_kernel = ScaleKernel(RBFKernel(ard_num_dims=train_X.shape[1] - 1))  # Ignore task index
        self.task_kernel = IndexKernel(num_tasks=2, rank=1)

        # Combine input kernel and task kernel
        self.covar_module = self.base_kernel * self.task_kernel

    def forward(self, X):
        mean_X = self.mean_module(X)
        covar_X = self.covar_module(X)
        return gpytorch.distributions.MultivariateNormal(mean_X, covar_X)


# Simulated data
T = 100
x = torch.linspace(0, 5, T)  # Feature values
X = torch.stack([torch.ones(T), x], dim=0).T  # Shape [T, 2]
beta_true = torch.tensor([2.0, 0.5])  # True beta [beta0, beta1]
Y = X @ beta_true + torch.randn(T) * 0.1  # Observed Y with noise

# Create task indices and extended input
task_indices = torch.cat([torch.zeros(T), torch.ones(T)]).long()  # Task 0: beta0, Task 1: beta1
X_extended = torch.cat([X.repeat(2, 1), task_indices.unsqueeze(-1)], dim=1)  # Add task indices
Y_extended = torch.cat([Y, Y])  # Duplicate Y for both tasks

# Define likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = MultiOutputBetaGPModel(X_extended, Y_extended, likelihood)

# Training
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(50):
    optimizer.zero_grad()
    output = model(X_extended)
    loss = -mll(output, Y_extended)
    loss.backward()
    optimizer.step()

# Predictions
model.eval()
likelihood.eval()

with torch.no_grad():
    test_x = torch.linspace(0, 5, 20)  # Test inputs
    test_X = torch.stack([torch.ones(20), test_x], dim=0).T  # Test inputs [20, 2]

    # Add task indices for prediction
    task_indices = torch.cat([torch.zeros(20), torch.ones(20)]).long()
    test_X_extended = torch.cat([test_X.repeat(2, 1), task_indices.unsqueeze(-1)], dim=1)

    pred = model(test_X_extended)
    pred_mean = pred.mean.view(2, -1)  # Reshape to [2, num_test_points] for [beta0, beta1]

    print("Predicted Beta Coefficients:")
    print("Beta0:", pred_mean[0])
    print("Beta1:", pred_mean[1])
