import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

# Define Latent GP Model
class LatentGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(LatentGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Generate synthetic data
def generate_data():
    torch.manual_seed(42)
    n = 50
    t = torch.linspace(0, 10, n)
    X = torch.sin(t) + torch.randn(n) * 0.1  # Input features
    beta_true = torch.sin(t / 2)  # True beta_t
    mu_true = torch.cos(t / 3)  # True mu_t
    Y = X * beta_true + mu_true + torch.randn(n) * 0.01  # Observed Y
    return t, X, beta_true, mu_true, Y

# Train a single GP model with regularization
def train_single_gp(train_x, train_y, regularization_weight=0.01):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = LatentGPModel(train_x, train_y, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(500):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        # Add L2 regularization on the predictions to avoid dominance
        regularization = regularization_weight * torch.norm(output.mean, p=2)
        total_loss = loss + regularization
        total_loss.backward()
        if i % 50 == 0:
            print(f"Iteration {i + 1}/500 - Loss: {total_loss.item()} (Reg: {regularization.item()})")
        optimizer.step()

    return model, likelihood

# Predict latent GPs
def predict_gp_model(model, likelihood, test_x):
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(test_x))
        mean = predictions.mean.detach()
        lower, upper = predictions.confidence_region()
        lower = lower.detach()
        upper = upper.detach()
    return mean, lower, upper

# Plot results
def plot_results(train_x, train_y, test_x, mean, lower, upper, true_values, title):
    plt.figure(figsize=(10, 6))
    plt.plot(train_x.numpy(), train_y.numpy(), 'k*', label='Train Data')
    # Ensure true_values matches test_x in size
    if len(true_values) != len(test_x):
        true_values = np.interp(test_x.numpy(), train_x.numpy(), true_values.numpy())
        true_values = torch.tensor(true_values)
    plt.plot(test_x.numpy(), true_values.numpy(), 'b', label='True Values')
    plt.plot(test_x.numpy(), mean.numpy(), 'r', label='GP Mean')
    plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.3, label='Confidence Interval')
    plt.title(title)
    plt.legend()
    plt.show()

# Main script
def main():
    t, X, beta_true, mu_true, Y = generate_data()
    t_normalized = (t - t.mean()) / t.std()  # Normalize time

    # Train GP for beta_t (Y / X as target)
    beta_train_y = Y / X
    print("Training GP for Beta_t...")
    beta_model, beta_likelihood = train_single_gp(t_normalized, beta_train_y, regularization_weight=0.01)

    # Predict beta_t and compute residuals
    t_pred = torch.linspace(0, 10, 100)
    t_pred_normalized = (t_pred - t.mean()) / t.std()

    beta_mean, beta_lower, beta_upper = predict_gp_model(beta_model, beta_likelihood, t_pred_normalized)

    with torch.no_grad():
        beta_train_pred, _, _ = predict_gp_model(beta_model, beta_likelihood, t_normalized)
        residual = Y - X * beta_train_pred

    # Train GP for mu_t (residual as target)
    print("Training GP for Mu_t...")
    mu_model, mu_likelihood = train_single_gp(t_normalized, residual, regularization_weight=0.01)
    mu_mean, mu_lower, mu_upper = predict_gp_model(mu_model, mu_likelihood, t_pred_normalized)

    # Predict Y
    Y_pred_mean = beta_mean * torch.sin(t_pred) + mu_mean
    Y_pred_lower = beta_lower * torch.sin(t_pred) + mu_lower
    Y_pred_upper = beta_upper * torch.sin(t_pred) + mu_upper

    # Plot results
    plot_results(t, beta_train_y, t_pred, beta_mean, beta_lower, beta_upper, beta_true, "Beta_t")
    plot_results(t, residual, t_pred, mu_mean, mu_lower, mu_upper, mu_true, "Mu_t")

    plt.figure(figsize=(10, 6))
    plt.plot(t.numpy(), Y.numpy(), 'k*', label='Observed Y')
    plt.plot(t_pred.numpy(), Y_pred_mean.numpy(), 'r', label='Predicted Y Mean')
    plt.fill_between(t_pred.numpy(), Y_pred_lower.numpy(), Y_pred_upper.numpy(), alpha=0.3, label='Confidence Interval')
    plt.title("Predictive Distribution of Y")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
