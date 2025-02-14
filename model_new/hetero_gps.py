import torch
import gpytorch
from matplotlib import pyplot as plt

# Step 1: Generate Simulated Data
torch.manual_seed(42)
n_train = 100
n_test = 200

# Simulated input data
train_x = torch.linspace(0, 1, n_train)
test_x = torch.linspace(0, 1, n_test)

# True function and heteroscedastic noise
true_function = torch.sin(train_x * (2 * torch.pi))  # True underlying function
noise_variance = 0.05 + 0.2 * torch.sin(2 * torch.pi * train_x) ** 2  # Varying noise
train_y = true_function + torch.randn_like(train_x) * noise_variance.sqrt()  # Noisy observations


# Step 2: Define GP models
class HeteroscedasticGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, noise_model):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        self.noise_model = noise_model  # GP for noise variance

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predictive_noise_variance(self, x):
        return self.noise_model(x).mean.exp()


class NoiseVarianceGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Step 3: Set up models and likelihoods
noise_likelihood = gpytorch.likelihoods.GaussianLikelihood()
noise_model = NoiseVarianceGPModel(train_x, (train_y - true_function).pow(2), noise_likelihood)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = HeteroscedasticGPModel(train_x, train_y, likelihood, noise_model)

# Step 4: Train the models
model.train()
likelihood.train()
noise_model.train()
noise_likelihood.train()

# Optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    # {'params': noise_model.parameters()},
], lr=0.01)

# Marginal Log Likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
noise_mll = gpytorch.mlls.ExactMarginalLogLikelihood(noise_likelihood, noise_model)

training_iter = 300
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    noise_output = noise_model(train_x)
    loss = -mll(output, train_y) - noise_mll(noise_output, (train_y - true_function).pow(2))
    loss.backward()
    optimizer.step()
    if (i + 1) % 50 == 0:
        print(f"Iteration {i + 1}/{training_iter} - Loss: {loss.item():.4f}")

# Step 5: Evaluate the model
model.eval()
likelihood.eval()
noise_model.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))
    noise_variance_pred = model.predictive_noise_variance(test_x)

# Step 6: Visualize Results
with torch.no_grad():
    f, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot training data
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*', label='Training Data', alpha=0.6)

    # True function
    ax.plot(test_x.numpy(), torch.sin(test_x * (2 * torch.pi)).numpy(), 'b', label='True Function')

    # Predicted mean with confidence intervals
    lower, upper = observed_pred.confidence_region()
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'r', label='Predicted Mean')
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.3, label='Confidence Interval')

    # Predicted noise variance
    ax.plot(test_x.numpy(), noise_variance_pred.numpy(), 'g--', label='Predicted Noise Variance')

    ax.set_title('Heteroscedastic Gaussian Process Regression')
    ax.legend()
    plt.show()
