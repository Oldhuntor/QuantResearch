import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
x_train = np.linspace(-3, 3, 100).astype(np.float32)
y_train = np.sin(x_train) + 0.5 * np.random.normal(size=len(x_train)).astype(np.float32)

# Convert to PyTorch tensors
x_train = torch.tensor(x_train).unsqueeze(1)  # (100, 1)
y_train = torch.tensor(y_train).unsqueeze(1)  # (100, 1)

# Define the Stochastic Regression Model
class StochasticRegression(nn.Module):
    def __init__(self):
        super(StochasticRegression, self).__init__()
        self.hidden1 = nn.Linear(1, 16)
        self.hidden2 = nn.Linear(16, 16)
        self.output = nn.Linear(16, 2)  # Output mean and log_sigma

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        params = self.output(x)  # [mu, log_sigma]
        mu = params[:, 0].unsqueeze(1)  # Mean
        log_sigma = params[:, 1].unsqueeze(1)  # Log standard deviation
        sigma = torch.exp(log_sigma)  # Convert log_sigma to sigma
        return mu, sigma

# Negative Log-Likelihood Loss Function
def nll_loss(mu, sigma, y):
    dist = torch.distributions.Normal(mu, sigma)
    return -dist.log_prob(y).mean()

# Initialize model, optimizer, and loss function
model = StochasticRegression()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training Loop
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    mu, sigma = model(x_train)
    loss = nll_loss(mu, sigma, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:  # Print loss every 50 epochs
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Predict on test data
x_test = torch.linspace(-4, 4, 200).unsqueeze(1)
model.eval()
with torch.no_grad():
    mu_pred, sigma_pred = model(x_test)

# Convert predictions to numpy for visualization
x_test_np = x_test.numpy()
mu_pred_np = mu_pred.numpy()
sigma_pred_np = sigma_pred.numpy()

# Plot training data and predictions
plt.figure(figsize=(10, 6))
plt.scatter(x_train.numpy(), y_train.numpy(), label="Training Data", alpha=0.7)
plt.plot(x_test_np, mu_pred_np, label="Predicted Mean", color="red")
plt.fill_between(
    x_test_np.flatten(),
    (mu_pred_np - 2 * sigma_pred_np).flatten(),
    (mu_pred_np + 2 * sigma_pred_np).flatten(),
    color="red",
    alpha=0.2,
    label="Uncertainty (±2σ)"
)
plt.title("Stochastic Regression with Uncertainty")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
