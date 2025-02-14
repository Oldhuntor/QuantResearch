import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Simulate the Data
np.random.seed(42)
n = 500  # Number of time steps
X = np.sin(2 * np.pi * np.linspace(0, 5, n)).reshape(-1, 1)  # Periodic input (sinusoidal)
true_beta = 0.5 * np.sin(np.linspace(0, 10, n))  # True beta_t (sinusoidal)
true_mu = 0.3 * np.cos(np.linspace(0, 10, n))  # True mu_t (cosine)
Y = X.flatten() * true_beta + true_mu + 0.1 * np.random.randn(n)  # Add noise

# Normalize input and output
X = (X - X.mean()) / X.std()
Y = (Y - Y.mean()) / Y.std()

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)


# Function to create windows of data
def create_windows(X, Y, window_size):
    X_windows = []
    Y_windows = []
    for i in range(len(X) - window_size + 1):
        X_windows.append(X[i:i + window_size])
        Y_windows.append(Y[i + window_size - 1])  # Predict the last value in the window
    return torch.stack(X_windows), torch.tensor(Y_windows, dtype=torch.float32)


window_size = 10  # Define the window size
X_windows, Y_windows = create_windows(X_tensor, Y_tensor, window_size)

# Ensure correct shape for LSTM input
X_windows = X_windows.view(-1, window_size, 1)  # Shape: (num_windows, window_size, 1)


# Step 2: Define the Model
class LatentRegressionModel(nn.Module):
    def __init__(self, latent_dim, window_size):
        super(LatentRegressionModel, self).__init__()

        # LSTM for temporal encoding
        self.lstm = nn.LSTM(input_size=1, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)

        # Encoder for beta_t
        self.beta_encoder = nn.Sequential(
            nn.Linear(128 + 1, 128),  # LSTM output + Y_t as inputs
            nn.ReLU(),
            nn.Linear(128, 2 * latent_dim)  # Mean and log-variance
        )

        # Encoder for mu_t
        self.mu_encoder = nn.Sequential(
            nn.Linear(128 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Mean and log-variance
        )

        # Reparameterization trick
        self.reparameterize = lambda mu, logvar: mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)

    def forward(self, X, Y):
        # LSTM encoding
        lstm_out, _ = self.lstm(X)  # X is already windowed
        lstm_out = lstm_out[:, -1, :]  # Take the last LSTM output for each window

        # Concatenate LSTM output and Y
        lstm_Y = torch.cat([lstm_out, Y.unsqueeze(1)], dim=1)

        # Encode beta_t
        beta_params = self.beta_encoder(lstm_Y)
        beta_mu, beta_logvar = torch.chunk(beta_params, 2, dim=1)
        beta_t = self.reparameterize(beta_mu, beta_logvar)

        # Encode mu_t
        mu_params = self.mu_encoder(lstm_Y)
        mu_mu, mu_logvar = torch.chunk(mu_params, 2, dim=1)
        mu_t = self.reparameterize(mu_mu, mu_logvar)

        # Reconstruct Y_t
        reconstructed_Y = (X[:, -1, 0] * beta_t).sum(dim=1) + mu_t.squeeze()

        return reconstructed_Y, beta_mu, beta_logvar, mu_mu, mu_logvar


# Initialize model
latent_dim = 1
model = LatentRegressionModel(latent_dim, window_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

# Xavier initialization
for layer in model.modules():
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)

# Step 3: Train the Model
epochs = 500
recon_losses = []
kl_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    reconstructed_Y, beta_mu, beta_logvar, mu_mu, mu_logvar = model(X_windows, Y_windows)

    # Reconstruction loss
    recon_loss = criterion(reconstructed_Y, Y_windows)

    # KL divergence
    kl_beta = -0.5 * torch.sum(1 + beta_logvar - beta_mu.pow(2) - beta_logvar.exp())
    kl_mu = -0.5 * torch.sum(1 + mu_logvar - mu_mu.pow(2) - mu_logvar.exp())

    # KL annealing
    kl_weight = min(0.05, epoch / epochs * 0.05)

    # Total loss
    loss = recon_loss + kl_weight * (kl_beta + kl_mu)
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()

    # Log losses
    recon_losses.append(recon_loss.item())
    kl_losses.append((kl_beta + kl_mu).item())

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, KL Weight: {kl_weight:.4f}")

# Step 4: Extract Latent Variables
model.eval()
with torch.no_grad():
    lstm_out, _ = model.lstm(X_windows)
    lstm_out = lstm_out[:, -1, :]

    # Concatenate LSTM output and Y
    lstm_Y = torch.cat([lstm_out, Y_windows.unsqueeze(1)], dim=1)

    beta_params = model.beta_encoder(lstm_Y)
    beta_mu, beta_logvar = torch.chunk(beta_params, 2, dim=1)
    beta_t = beta_mu.squeeze().numpy()
    beta_lower = (beta_mu - 1.96 * torch.exp(0.5 * beta_logvar)).squeeze().numpy()
    beta_upper = (beta_mu + 1.96 * torch.exp(0.5 * beta_logvar)).squeeze().numpy()

    mu_params = model.mu_encoder(lstm_Y)
    mu_mu, mu_logvar = torch.chunk(mu_params, 2, dim=1)
    mu_t = mu_mu.squeeze().numpy()
    mu_lower = (mu_mu - 1.96 * torch.exp(0.5 * mu_logvar)).squeeze().numpy()
    mu_upper = (mu_mu + 1.96 * torch.exp(0.5 * mu_logvar)).squeeze().numpy()

# Step 5: Visualize the Results
plt.figure(figsize=(12, 8))

# True vs. Predicted Beta_t with Interval
plt.subplot(2, 1, 1)
plt.plot(true_beta[window_size - 1:], label="True Beta_t", color="blue")
plt.plot(beta_t, label="Predicted Beta_t", linestyle="dashed", color="orange")
plt.fill_between(range(len(beta_t)), beta_lower, beta_upper, color="orange", alpha=0.2, label="95% CI")
plt.legend()
plt.title("Beta_t: True vs Predicted")
plt.xlabel("Time Step")
plt.ylabel("Beta_t")

# True vs. Predicted Mu_t with Interval
plt.subplot(2, 1, 2)
plt.plot(true_mu[window_size - 1:], label="True Mu_t", color="blue")
plt.plot(mu_t, label="Predicted Mu_t", linestyle="dashed", color="orange")
plt.fill_between(range(len(mu_t)), mu_lower, mu_upper, color="orange", alpha=0.2, label="95% CI")
plt.legend()
plt.title("Mu_t: True vs Predicted")
plt.xlabel("Time Step")
plt.ylabel("Mu_t")

plt.tight_layout()
plt.show()

# Step 6: Reconstruct Y_t
reconstructed_Y = (X_windows[:, -1, 0].numpy() * beta_t[:, None]).sum(axis=1) + mu_t

plt.figure(figsize=(10, 6))
plt.plot(Y[window_size - 1:], label="True Y_t", color="blue")
plt.plot(reconstructed_Y, label="Reconstructed Y_t", linestyle="dashed", color="orange")
plt.fill_between(range(len(reconstructed_Y)),
                 reconstructed_Y - 1.96 * 0.1,
                 reconstructed_Y + 1.96 * 0.1,
                 color="orange", alpha=0.2, label="95% CI")
plt.legend()
plt.title("Y_t: True vs Reconstructed")
plt.xlabel("Time Step")
plt.ylabel("Y_t")
plt.tight_layout()
plt.show()

# Step 7: Plot Loss Components
plt.figure(figsize=(8, 4))
plt.plot(recon_losses, label="Reconstruction Loss", color="blue")
plt.plot(kl_losses, label="KL Divergence Loss", color="orange")
plt.legend()
plt.title("Loss Components During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()
