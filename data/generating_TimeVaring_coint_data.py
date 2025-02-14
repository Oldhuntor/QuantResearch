import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_cointegration_data(
        T=200,
        seed=123,
        noise_std=0.1,
        change_points=(70, 140),
        beta_values=(1.0, 2.0, 0.5),
        mu_values=(0.0, -1.0, 1.0)
):
    """
    Generate a cointegrated time series X_t and Y_t with piecewise-constant beta_t and mu_t.

    Args:
        T: Length of the time series
        seed: Random seed for reproducibility
        noise_std: Std dev of Gaussian noise epsilon_t
        change_points: Indices where beta_t and mu_t jump
        beta_values: The piecewise constant values of beta
        mu_values: The piecewise constant values of mu

    Returns:
        time: torch.Tensor of shape (T,)
        X: torch.Tensor of shape (T,)
        Y: torch.Tensor of shape (T,)
        beta_t: torch.Tensor of shape (T,)
        mu_t: torch.Tensor of shape (T,)
    """
    torch.manual_seed(seed)

    # 1) Create time index
    time = torch.arange(T, dtype=torch.float32)

    # 2) Generate a Y_t process
    # Example: Let's use a random walk or AR(1) style:
    Y = torch.zeros(T)
    alpha = 0.9  # for AR(1)
    # Start Y_0 at 0, then evolve
    for t in range(1, T):
        Y[t] = alpha * Y[t - 1] + torch.randn(1) * 0.5

    # 3) Build piecewise constant beta_t and mu_t
    # We'll split T points into len(change_points)+1 segments
    beta_t = torch.zeros(T)
    mu_t = torch.zeros(T)

    # Example logic:
    # Segment 1: [0, change_points[0])
    # Segment 2: [change_points[0], change_points[1])
    # Segment 3: [change_points[1], T)
    # If you have more or fewer segments, adjust accordingly.
    segments = [0] + list(change_points) + [T]
    for i in range(len(segments) - 1):
        start_idx = segments[i]
        end_idx = segments[i + 1]
        beta_t[start_idx:end_idx] = beta_values[i]
        mu_t[start_idx:end_idx] = mu_values[i]

    # 4) Generate X_t = beta_t * Y_t + mu_t + noise
    noise = torch.randn(T) * noise_std
    X = beta_t * Y + mu_t + noise

    return time, X, Y, beta_t, mu_t, noise


def plot_cointegration_data(time, X, Y, beta_t, mu_t, title="Cointegration Series"):
    """
    Plot the generated cointegration data and piecewise parameters.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(time, X, label='X_t', color='blue')
    axs[0].plot(time, Y, label='Y_t', color='red', alpha=0.6)
    axs[0].set_ylabel("X_t & Y_t")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(time, beta_t, label='Beta_t', color='green')
    axs[1].set_ylabel("Beta_t")
    axs[1].grid(True)

    axs[2].plot(time, mu_t, label='Mu_t', color='orange')
    axs[2].set_ylabel("Mu_t")
    axs[2].set_xlabel("Time index t")
    axs[2].grid(True)

    axs[0].set_title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage:
    T = 200
    change_points = (70, 140)
    beta_values = (1.0, 2.0, 0.5)
    mu_values = (0.0, -1.0, 1.0)

    time, X, Y, beta_t, mu_t = generate_cointegration_data(
        T=T,
        seed=42,
        noise_std=0.05,
        change_points=change_points,
        beta_values=beta_values,
        mu_values=mu_values
    )

    plot_cointegration_data(time, X, Y, beta_t, mu_t, title="Cointegrated Series with Piecewise-Constant Beta & Mu")
