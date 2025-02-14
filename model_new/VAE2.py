import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# 1. Data Generating Process with Heteroskedastic Noise
np.random.seed(42)
T = 300  # Number of time points

time = np.linspace(0, 30, T)  # Time variable
X_t = np.random.normal(0, 1, T)  # Predictor

# True time-varying coefficients
beta_t = 1 + 0.5 * np.sin(0.5 * time)
mu_t = 0.2 * np.cos(0.3 * time)

# Generate chaotic heteroskedastic noise
noise_variance = 0.1 * (1 + np.abs(2 * np.sin(0.3 * time)**2 + 1.5 * np.cos(0.15 * time)))
epsilon = np.random.normal(0, noise_variance, T)

# Generate response variable
Y_t = beta_t * X_t + mu_t + epsilon

# Normalize data for neural network
X_t = (X_t - np.mean(X_t)) / np.std(X_t)
Y_t = (Y_t - np.mean(Y_t)) / np.std(Y_t)

# Prepare data for PyTorch
X_t_tensor = torch.tensor(X_t, dtype=torch.float32).unsqueeze(1)
Y_t_tensor = torch.tensor(Y_t, dtype=torch.float32).unsqueeze(1)
data = TensorDataset(X_t_tensor, Y_t_tensor)
loader = DataLoader(data, batch_size=32, shuffle=True)

# 2. Define Variational Autoencoder (VAE) with Separate Encoders
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder for beta_t
        self.encoder_beta = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.mu_beta = nn.Linear(32, 1)
        self.log_var_beta = nn.Linear(32, 1)

        # Encoder for mu_t
        self.encoder_mu = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.mu_mu = nn.Linear(32, 1)
        self.log_var_mu = nn.Linear(32, 1)

    def encode_beta(self, x):
        h = self.encoder_beta(x)
        mu = self.mu_beta(h)
        log_var = self.log_var_beta(h)
        return mu, log_var

    def encode_mu(self, x):
        h = self.encoder_mu(x)
        mu = self.mu_mu(h)
        log_var = self.log_var_mu(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = 0.1 * torch.randn_like(std)  # Reduced noise
        return mu + eps * std

    def forward(self, x):
        mu_beta, log_var_beta = self.encode_beta(x)
        mu_mu, log_var_mu = self.encode_mu(x)

        z_beta = self.reparameterize(mu_beta, log_var_beta)
        z_mu = self.reparameterize(mu_mu, log_var_mu)

        # Reconstruct Y_t using decoded beta and mu
        recon_y = z_beta * x + z_mu

        return recon_y, mu_beta, log_var_beta, mu_mu, log_var_mu

# 3. Model Setup
input_dim = 1  # X_t only as input
latent_dim = 2  # For beta_t and mu_t
model = VAE(input_dim, latent_dim)
optimizer = Adam(model.parameters(), lr=0.001)

# 4. Loss Function (Reconstruction + KL Divergence)
def loss_function(recon_y, y, mu_beta, log_var_beta, mu_mu, log_var_mu):
    recon_loss = nn.MSELoss()(recon_y, y)  # Reconstruction loss for Y_t
    kl_beta = -0.5 * torch.sum(1 + log_var_beta - mu_beta.pow(2) - log_var_beta.exp())
    kl_mu = -0.5 * torch.sum(1 + log_var_mu - mu_mu.pow(2) - log_var_mu.exp())
    return recon_loss + 2.0 * (kl_beta + kl_mu) / y.size(0)  # Increased KL weight

# 5. Training the VAE
num_epochs = 100
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in loader:
        x, y = batch
        optimizer.zero_grad()
        recon_y, mu_beta, log_var_beta, mu_mu, log_var_mu = model(x)
        loss = loss_function(recon_y, y, mu_beta, log_var_beta, mu_mu, log_var_mu)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(loader):.4f}")

# 6. Evaluate Latent Space
model.eval()
with torch.no_grad():
    recon_y, mu_beta, log_var_beta, mu_mu, log_var_mu = model(X_t_tensor)

    # Extract latent variables
    beta_latent = mu_beta.numpy().flatten()
    mu_latent = mu_mu.numpy().flatten()

# Apply smoothing to predictions
window = 5
beta_latent_smooth = np.convolve(beta_latent, np.ones(window)/window, mode='same')
mu_latent_smooth = np.convolve(mu_latent, np.ones(window)/window, mode='same')

# Plot beta_t
plt.figure(figsize=(10, 6))
plt.plot(time, beta_t, label=r'True $\beta_t$', color='blue')
plt.plot(time, beta_latent_smooth, '--', label=r'Latent $\hat{\beta}_t$', color='orange')
plt.legend()
plt.xlabel('Time')
plt.ylabel(r'$\beta_t$')
plt.title(r'Latent Space for $\beta_t$')
plt.grid(True)
plt.show()

# Plot mu_t
plt.figure(figsize=(10, 6))
plt.plot(time, mu_t, label=r'True $\mu_t$', color='green')
plt.plot(time, mu_latent_smooth, '--', label=r'Latent $\hat{\mu}_t$', color='red')
plt.legend()
plt.xlabel('Time')
plt.ylabel(r'$\mu_t$')
plt.title(r'Latent Space for $\mu_t$')
plt.grid(True)
plt.show()

# Plot Y_t vs reconstructed Y_t
plt.figure(figsize=(10, 6))
plt.plot(time, Y_t, label=r'True $Y_t$', color='blue')
plt.plot(time, recon_y.numpy().flatten(), '--', label=r'Reconstructed $\hat{Y_t}$', color='orange')
plt.legend()
plt.xlabel('Time')
plt.ylabel(r'$Y_t$')
plt.title(r'True vs Reconstructed $Y_t$')
plt.grid(True)
plt.show()

print("VAE Training and Latent Space Evaluation Complete!")
