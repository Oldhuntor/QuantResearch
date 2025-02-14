import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# Planar Flow Layer
class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.w = nn.Parameter(torch.randn(1, dim) * 0.1)
        self.b = nn.Parameter(torch.randn(1) * 0.1)
        self.u = nn.Parameter(torch.randn(1, dim) * 0.1)

    def forward(self, x):
        # f(x) = x + u * h(w^T x + b)
        activation = torch.tanh(torch.mm(x, self.w.t()) + self.b)
        z = x + self.u * activation

        # Computing log determinant of Jacobian
        psi = (1 - activation ** 2) * self.w  # d/dx tanh(wx + b)
        det_grad = 1 + torch.mm(psi, self.u.t())
        log_det = torch.log(torch.abs(det_grad) + 1e-6)

        return z, log_det


# Normalizing Flow Model
class NormalizingFlow(nn.Module):
    def __init__(self, dim, n_flows):
        super(NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(n_flows)])

    def forward(self, x):
        log_det_sum = 0

        for flow in self.flows:
            x, log_det = flow(x)
            log_det_sum += log_det

        return x, log_det_sum


# Generate 2D data from a mixture of Gaussians (target distribution)
def generate_target_data(n_samples):
    points = []
    for _ in range(n_samples):
        # Randomly choose which Gaussian to sample from
        if np.random.random() < 0.5:
            points.append(np.random.normal([2, 2], 0.5))
        else:
            points.append(np.random.normal([-2, -2], 0.5))
    return torch.FloatTensor(points)


# Training function
def train_flow(flow, n_epochs=10000, batch_size=100, lr=1e-3):
    optimizer = optim.Adam(flow.parameters(), lr=lr)

    for epoch in range(n_epochs):
        # Sample from base distribution (standard normal)
        x = torch.randn(batch_size, 2)

        # Forward pass through flow
        z, log_det = flow(x)

        # Compute loss (negative log likelihood)
        # Using a simple Gaussian mixture as target
        target_samples = generate_target_data(batch_size)
        loss = torch.mean((z - target_samples) ** 2) - torch.mean(log_det)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    return flow


# Visualization function
def visualize_flow(flow, n_samples=1000):
    # Generate samples from base distribution
    x = torch.randn(n_samples, 2)

    # Transform samples through flow
    z, _ = flow(x)
    z = z.detach().numpy()

    # Generate target samples
    target = generate_target_data(n_samples).numpy()

    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.scatter(x[:, 0].numpy(), x[:, 1].numpy(), alpha=0.5)
    plt.title('Base Distribution (Normal)')
    plt.axis('equal')

    plt.subplot(122)
    plt.scatter(z[:, 0], z[:, 1], alpha=0.5, label='Transformed')
    plt.scatter(target[:, 0], target[:, 1], alpha=0.5, label='Target')
    plt.title('Transformed vs Target Distribution')
    plt.legend()
    plt.axis('equal')

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize and train the flow
    dim = 2  # 2D distribution
    n_flows = 4  # Number of flow layers
    flow = NormalizingFlow(dim, n_flows)

    # Train the model
    flow = train_flow(flow)

    # Visualize results
    visualize_flow(flow)