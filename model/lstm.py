import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def generate_synthetic_data(total_length=1000, seq_length=20, step_size=1):
    # Generate one long sequence
    t = np.linspace(0, 8 * np.pi, total_length)
    # Create a single sine wave with noise
    signal = np.sin(t) + np.random.normal(0, 0.05, total_length)

    # Create rolling windows
    data = []
    for i in range(0, total_length - seq_length + 1, step_size):
        window = signal[i:i + seq_length]
        data.append(window.reshape(-1, 1))

    return np.array(data)


class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class LSTMAutoencoder(nn.Module):
    def __init__(self, hidden_size, num_layers=2):  # Increased layers
        super(LSTMAutoencoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=1,  # Changed to 1
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Encoder
        enc_output, (hidden, cell) = self.encoder(x)

        # Initialize decoder input as zeros
        decoder_input = torch.zeros_like(x)
        # Use encoder's last output as first decoder input
        decoder_input[:, 0, :] = x[:, 0, :]

        # Decoder (sequential processing)
        decoder_hidden = hidden
        decoder_cell = cell
        outputs = []

        for t in range(x.size(1)):
            # Use teacher forcing during training
            if self.training:
                current_input = x[:, t:t + 1, :]
            else:
                current_input = decoder_input[:, t:t + 1, :]

            lstm_out, (decoder_hidden, decoder_cell) = self.decoder(
                current_input, (decoder_hidden, decoder_cell)
            )
            output = self.output_layer(lstm_out)
            outputs.append(output)

            if not self.training:
                if t < x.size(1) - 1:
                    decoder_input[:, t + 1, :] = output.squeeze(1)

        outputs = torch.cat(outputs, dim=1)
        return outputs


def train_model(model, train_loader, n_epochs, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    train_losses = []

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_losses.append(loss.item())

        epoch_loss = np.mean(epoch_losses)
        train_losses.append(epoch_loss)

        # Learning rate scheduling
        scheduler.step(epoch_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {epoch_loss:.6f}')

    return train_losses


def main():
    # Modified parameters
    n_samples = 1000
    seq_length = 20
    batch_size = 64  # Increased batch size
    hidden_size = 64  # Increased hidden size
    n_epochs = 100  # Increased epochs

    # Generate data
    data = generate_synthetic_data(n_samples, seq_length)

    # Create dataset and dataloader
    dataset = TimeSeriesDataset(data)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = LSTMAutoencoder(hidden_size)

    # Train model
    losses = train_model(model, train_loader, n_epochs, learning_rate=0.001)

    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')  # Using log scale to better see loss progression
    plt.grid(True)
    plt.show()

    # Visualize reconstruction
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        reconstructed = model(sample_batch)

        # Plot original vs reconstructed for first sample
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(sample_batch[0].numpy())
        plt.title('Original')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(reconstructed[0].numpy())
        plt.title('Reconstructed')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()