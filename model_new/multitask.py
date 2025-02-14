import torch
import gpytorch
import matplotlib.pyplot as plt

# ---- Model Definition ---- #
class LatentGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(LatentGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.LinearMean(input_size=1), num_tasks=2
        )

        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x).add_jitter(1e-4)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean, covar)

# ---- Generate Data ---- #
def generate_data(T=200, noise_std=0.1, seed=42):
    torch.manual_seed(seed)
    X = torch.linspace(0, 10, T).unsqueeze(-1)
    beta_true = 2 + 0.5 * torch.sin(X[:, 0])
    mu_true = 0.5 * torch.cos(X[:, 0])
    noise = noise_std * torch.randn(T)
    Y = beta_true * X[:, 0] + mu_true + noise
    return X, Y, beta_true, mu_true

# ---- Train Model ---- #
def train_model(train_x, train_y, likelihood, num_epochs=100, lr=0.1):
    model = LatentGPModel(train_x, train_y, likelihood)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()

    for i in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            print(f"Epoch {i+1}/{num_epochs}, Loss: {loss.item()}")
    return model

# ---- Inference ---- #
def infer_latent_gp(model, likelihood, test_x):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_x))
        mean = pred.mean
        lower, upper = pred.confidence_region()
    return mean, lower, upper

# ---- Plot Results ---- #
# ---- 绘制结果 ---- #
def plot_results(X, Y, pred_mean, pred_lower, pred_upper, true_beta, true_mu):
    x = X.squeeze()

    # 分别提取 beta 和 mu 的预测结果
    beta_mean = pred_mean[:, 0].squeeze()  # Shape (T,)
    beta_lower = pred_lower[:, 0].squeeze()  # Shape (T,)
    beta_upper = pred_upper[:, 0].squeeze()  # Shape (T,)
    mu_mean = pred_mean[:, 1].squeeze()    # Shape (T,)
    mu_lower = pred_lower[:, 1].squeeze()    # Shape (T,)
    mu_upper = pred_upper[:, 1].squeeze()    # Shape (T,)

    # # 绘制 Beta
    # plt.figure(figsize=(10, 6))
    # plt.plot(x.numpy(), beta_mean.numpy(), label="Beta Mean", color="blue")
    # plt.fill_between(x.numpy(), beta_lower.numpy(), beta_upper.numpy(), alpha=0.2, label="Beta CI", color="blue")
    # plt.plot(x.numpy(), true_beta.numpy(), "b--", label="True Beta")
    # plt.legend()
    # plt.title("Posterior for Beta")
    # plt.show()
    #
    # # 绘制 Mu
    # plt.figure(figsize=(10, 6))
    # plt.plot(x.numpy(), mu_mean.numpy(), label="Mu Mean", color="red")
    # plt.fill_between(x.numpy(), mu_lower.numpy(), mu_upper.numpy(), alpha=0.2, label="Mu CI", color="red")
    # plt.plot(x.numpy(), true_mu.numpy(), "r--", label="True Mu")
    # plt.legend()
    # plt.title("Posterior for Mu")
    # plt.show()

    # 绘制 Y 和预测
    plt.figure(figsize=(10, 6))
    plt.plot(x.numpy(), Y.numpy(), label="True Y", color="green")
    plt.plot(x.numpy(), (beta_mean * x + mu_mean).numpy(), label="Predicted Y", color="orange") # Correct Prediction
    plt.fill_between(x.numpy(), (beta_lower * x + mu_lower).numpy(), (beta_upper * x + mu_upper).numpy(), alpha=0.2, label="Prediction CI", color="orange")
    plt.legend()
    plt.title("Y and Prediction")
    plt.show()

# ---- Main Script ---- #
if __name__ == "__main__":
    X, Y, beta_true, mu_true = generate_data()
    train_x = X
    train_y = torch.stack([Y, X.squeeze()], dim=-1)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = train_model(train_x, train_y, likelihood, num_epochs=100, lr=0.1)

    pred_mean, pred_lower, pred_upper = infer_latent_gp(model, likelihood, train_x)
    beta_result, mu_result = (pred_mean[..., 0].squeeze(), pred_lower[..., 0].squeeze(), pred_upper[..., 0].squeeze()), (pred_mean[..., 1].squeeze(), pred_lower[..., 1].squeeze(), pred_upper[..., 1].squeeze())

    plot_results(train_x, Y, pred_mean.squeeze(), pred_lower.squeeze(), pred_upper.squeeze(), beta_result, mu_result, true_beta=beta_true, true_mu=mu_true)