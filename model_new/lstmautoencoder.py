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
        X_windows.append(X[i:i+window_size])
        Y_windows.append(Y[i+window_size-1])  # Predict the last value in the window
    return torch.stack(X_windows), torch.tensor(Y_windows, dtype=torch.float32)

window_size = 10  # Define the window size
X_windows, Y_windows = create_windows(X_tensor, Y_tensor, window_size)

# Ensure correct shape for input
X_windows = X_windows.view(-1, window_size, 1)  # Shape: (num_windows, window_size, 1)

# Step 2: Define the Model
class AutoencoderLSTMModel(nn.Module):
    def __init__(self, latent_dim, window_size):
        super(AutoencoderLSTMModel, self).__init__()

        # LSTM Encoder
        self.lstm_encoder = nn.LSTM(input_size=1, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)

        # Latent representation for beta_t and mu_t
        self.beta_layer = nn.Linear(128, latent_dim)  # Map to beta_t latent space
        self.mu_layer = nn.Linear(128, latent_dim)    # Map to mu_t latent space

        # LSTM Decoder
        self.lstm_decoder = nn.LSTM(input_size=latent_dim, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        self.reconstruction_layer = nn.Linear(128, 1)

    def forward(self, X):
        # Encode with LSTM
        lstm_out, _ = self.lstm_encoder(X)
        encoded = lstm_out[:, -1, :]  # Take the last hidden state

        # Latent variables
        beta_t = self.beta_layer(encoded)
        mu_t = self.mu_layer(encoded)

        # Reconstruct Y_t using beta_t, mu_t, and X
        reconstructed_Y = torch.sum(X.squeeze(-1) * beta_t, dim=1) + mu_t.squeeze()

        return reconstructed_Y, beta_t, mu_t

# Initialize model
latent_dim = 1
model = AutoencoderLSTMModel(latent_dim, window_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Xavier initialization
for layer in model.modules():
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)

# Step 3: Train the Model
epochs = 500
recon_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    reconstructed_Y, beta_t, mu_t = model(X_windows)

    # Reconstruction loss
    recon_loss = criterion(reconstructed_Y.squeeze(), Y_windows)

    # Temporal regularization for beta_t and mu_t
    temporal_loss_beta = torch.mean((beta_t[1:] - beta_t[:-1])**2)
    temporal_loss_mu = torch.mean((mu_t[1:] - mu_t[:-1])**2)

    # Total loss
    loss = recon_loss + 0.01 * (temporal_loss_beta + temporal_loss_mu)
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    # Log losses
    recon_losses.append(recon_loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}")

# Step 4: Extract Latent Variables
model.eval()
with torch.no_grad():
    _, beta_t, mu_t = model(X_windows)
    beta_t = beta_t.squeeze().numpy()
    mu_t = mu_t.squeeze().numpy()

# Step 5: Visualize the Latent Representations
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(true_beta[window_size-1:], label="True Beta_t", color="blue")
plt.plot(beta_t, label="Predicted Beta_t", linestyle="dashed", color="orange")
plt.legend()
plt.title("Beta_t: True vs Predicted")
plt.xlabel("Time Step")
plt.ylabel("Beta_t")

plt.subplot(2, 1, 2)
plt.plot(true_mu[window_size-1:], label="True Mu_t", color="blue")
plt.plot(mu_t, label="Predicted Mu_t", linestyle="dashed", color="orange")
plt.legend()
plt.title("Mu_t: True vs Predicted")
plt.xlabel("Time Step")
plt.ylabel("Mu_t")

plt.tight_layout()
plt.show()

# Step 6: Reconstruct Y_t
reconstructed_Y = model(X_windows)[0].squeeze().detach().numpy()

plt.figure(figsize=(10, 6))
plt.plot(Y[window_size-1:], label="True Y_t", color="blue")
plt.plot(reconstructed_Y, label="Reconstructed Y_t", linestyle="dashed", color="orange")
plt.legend()
plt.title("Y_t: True vs Reconstructed")
plt.xlabel("Time Step")
plt.ylabel("Y_t")
plt.tight_layout()
plt.show()

# Step 7: Plot Loss Components
plt.figure(figsize=(8, 4))
plt.plot(recon_losses, label="Reconstruction Loss", color="blue")
plt.legend()
plt.title("Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()
