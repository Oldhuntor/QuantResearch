import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, encoding_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# Training function
def train_autoencoder(model, data, epochs=200, lr=1e-3, patience=20):
    # Split data into train and validation sets
    train_data, val_data = train_test_split(data, test_size=0.01, random_state=42)
    train_tensor = torch.FloatTensor(train_data)
    val_tensor = torch.FloatTensor(val_data)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # For early stopping
    best_val_loss = float('inf')
    patience_count = 0
    best_model = None

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        encoded, decoded = model(train_tensor)
        train_loss = torch.mean((decoded - train_tensor) ** 2)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            encoded_val, decoded_val = model(val_tensor)
            val_loss = torch.mean((decoded_val - val_tensor) ** 2)

        # Store losses
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            best_model = model.state_dict().copy()
        else:
            patience_count += 1

        if patience_count >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.yscale('log')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    return model


# Main workflow
def main():
    # Read and clean data (from previous code)
    data = pd.read_csv('/Users/hxh/PycharmProjects/Quantitative_Research/model_new/bayesian_clustering/transposed_stock_data.csv', header=None)
    data = data.iloc[1:, 1:]

    # Normalize data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    data_tensor = torch.FloatTensor(data_normalized)

    # Create and train model
    model = Autoencoder(input_dim=data.shape[1], encoding_dim=300)
    model = train_autoencoder(model, data_tensor, epochs=2000)

    # Get encoded representation
    encoded_data, decoded_data = model(data_tensor)

    encoded_data = encoded_data.detach().numpy()


    # # Plot results
    # plt.figure(figsize=(10, 6))
    # plt.scatter(encoded_data[:, 0], encoded_data[:, 1])
    #
    # # Add stock index numbers
    # for i in range(len(encoded_data)):
    #     plt.annotate(f'Stock {i}', (encoded_data[i, 0], encoded_data[i, 1]))
    #
    # plt.title('2D Encoded Representation of Stocks')
    # plt.grid(True)
    # plt.show()

    # Load and prepare data
    # Get the first time series
    index = 201
    first_series = data.iloc[index]
    first_series = np.float64(first_series.values)

    # Get reconstruction from model
    model.eval()
    with torch.no_grad():
        encoded, decoded = model(data_tensor)
        first_reconstruction_normalized = decoded[index].numpy()

    # Convert back to original scale
    first_reconstruction = scaler.inverse_transform(first_reconstruction_normalized.reshape(1, -1))[0]

    # Plot
    plt.figure(figsize=(12, 6))
    time_points = range(len(first_series))

    plt.plot(time_points, first_series, 'b', label='Original')
    plt.plot(time_points, first_reconstruction, 'r', label='Reconstructed')

    plt.title('Original vs Reconstructed Time Series')
    plt.xlabel('Time Point')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    index = 200
    first_series = data.iloc[index]
    first_series = np.float64(first_series.values)

    # Get reconstruction from model
    model.eval()
    with torch.no_grad():
        encoded, decoded = model(data_tensor)
        first_reconstruction_normalized = decoded[index].numpy()

    # Convert back to original scale
    first_reconstruction = scaler.inverse_transform(first_reconstruction_normalized.reshape(1, -1))[0]

    # Plot
    plt.figure(figsize=(12, 6))
    time_points = range(len(first_series))

    plt.plot(time_points, first_series, 'b', label='Original')
    plt.plot(time_points, first_reconstruction, 'r', label='Reconstructed')

    plt.title('Original vs Reconstructed Time Series')
    plt.xlabel('Time Point')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    index = 202
    first_series = data.iloc[index]
    first_series = np.float64(first_series.values)

    # Get reconstruction from model
    model.eval()
    with torch.no_grad():
        encoded, decoded = model(data_tensor)
        first_reconstruction_normalized = decoded[index].numpy()

    # Convert back to original scale
    first_reconstruction = scaler.inverse_transform(first_reconstruction_normalized.reshape(1, -1))[0]

    # Plot
    plt.figure(figsize=(12, 6))
    time_points = range(len(first_series))

    plt.plot(time_points, first_series, 'b', label='Original')
    plt.plot(time_points, first_reconstruction, 'r', label='Reconstructed')

    plt.title('Original vs Reconstructed Time Series')
    plt.xlabel('Time Point')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    # # Add error measurements
    # mse = np.mean((first_series - first_reconstruction) ** 2)
    # mae = np.mean(np.abs(first_series - first_reconstruction))
    # plt.text(0.02, 0.98, f'MSE: {mse:.6f}\nMAE: {mae:.6f}',
    #          transform=plt.gca().transAxes,
    #          verticalalignment='top',
    #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    #
    # plt.show()
    #
    # # Print the actual values
    # print("\nOriginal values:", first_series)
    # print("\nReconstructed values:", first_reconstruction)


if __name__ == "__main__":
    main()