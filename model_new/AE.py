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

# 2. Define Autoencoder with Separate Encoders
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        # Encoder for beta_t
        self.encoder_beta = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Encoder for mu_t
        self.encoder_mu = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        z_beta = self.encoder_beta(x)
        z_mu = self.encoder_mu(x)

        # Reconstruct Y_t using decoded beta and mu
        recon_y = z_beta * x + z_mu

        return recon_y, z_beta, z_mu

# 3. Model Setup
input_dim = 1  # X_t only as input
model = Autoencoder(input_dim)
optimizer = Adam(model.parameters(), lr=0.001)

# 4. Loss Function (Reconstruction)
def loss_function(recon_y, y):
    recon_loss = nn.MSELoss()(recon_y, y)  # Reconstruction loss for Y_t
    return recon_loss

# 5. Training the Autoencoder
num_epochs = 100
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in loader:
        x, y = batch
        optimizer.zero_grad()
        recon_y, z_beta, z_mu = model(x)
        loss = loss_function(recon_y, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(loader):.4f}")

# 6. Evaluate Latent Space
model.eval()
with torch.no_grad():
    recon_y, z_beta, z_mu = model(X_t_tensor)

    # Extract latent variables
    beta_latent = z_beta.numpy().flatten()
    mu_latent = z_mu.numpy().flatten()

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

print("Autoencoder Training and Latent Space Evaluation Complete!")
