import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List


class CointegrationAdditivePGModel:
    class DataGenerator:
        @staticmethod
        def generate_cointegration_data(
                T: int = 200,
                seed: int = 42,
                noise_std: float = 0.1,
                change_points: Tuple[int, ...] = (70, 140),
                beta_values: Tuple[float, ...] = (1.0, 2.0, 0.5),
                mu_values: Tuple[float, ...] = (0.0, -1.0, 1.0),
                alpha_x: float = 0.9,
                alpha_y: float = 0.9
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Generate synthetic cointegrated time series with time-varying coefficients.
            """
            torch.manual_seed(seed)

            # Generate X as an AR(1) process
            X = torch.zeros(T, dtype=torch.float64)
            for t in range(1, T):
                X[t] = alpha_x * X[t - 1] + torch.randn(1, dtype=torch.float64) * 0.5

            # Create time-varying coefficients
            beta_t = torch.zeros(T, dtype=torch.float64)
            mu_t = torch.zeros(T, dtype=torch.float64)

            segments = [0] + list(change_points) + [T]
            for i in range(len(segments) - 1):
                start_idx = segments[i]
                end_idx = segments[i + 1]
                beta_t[start_idx:end_idx] = beta_values[i]
                mu_t[start_idx:end_idx] = mu_values[i]

            # Generate Y with time-varying coefficients
            noise = torch.randn(T, dtype=torch.float64) * noise_std
            Y = beta_t * X + mu_t + noise

            # Create time index
            time = torch.arange(T, dtype=torch.float64)

            return time, X, Y, beta_t, mu_t

    class GaussianProcessModel(gpytorch.models.ExactGP):
        def __init__(self, train_input, train_target, likelihood):
            """
            Additive Gaussian Process for Cointegration Modeling

            Args:
                train_input (torch.Tensor): Input tensor containing [train_X, train_Y]
                train_target (torch.Tensor): Target values (train_Y)
                likelihood (gpytorch.likelihoods): Noise likelihood
            """
            super().__init__(train_input, train_target, likelihood)

            # Store dimensions
            self.n_data = train_input.size(0)

            # Mean module for beta (coefficient for X)
            # self.beta_mean = gpytorch.means.LinearMean(input_size=1)
            self.beta_mean = gpytorch.means.ConstantMean()

            self.beta_covar = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

            # Mean module for gamma (coefficient for Y)
            # self.gamma_mean = gpytorch.means.ConstantMean()
            # self.gamma_covar = gpytorch.kernels.ScaleKernel(
            #     gpytorch.kernels.RBFKernel()
            # )

            # Mean module for mu (intercept)
            # self.mu_mean = gpytorch.means.LinearMeanGrad(input_size=2)
            # self.mu_mean = gpytorch.means.LinearMean(input_size=1)
            self.mu_mean = gpytorch.means.ConstantMean()
            self.mu_covar = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

        def forward(self, x):
            """
            Forward pass computing the mean and covariance

            Args:
                x (torch.Tensor): Input tensor containing [X, Y]

            Returns:
                gpytorch.distributions.MultivariateNormal
            """
            # Split input into X and Y components
            x_component = x[:, 0].unsqueeze(1)  # First column is X
            y_component = x[:, 1].unsqueeze(1)  # Second column is Y

            # Compute beta distribution (coefficient for X) based on input
            beta_mean = self.beta_mean(x)
            beta_covar = self.beta_covar(x)

            # beta_mean = beta_mean[:, 0]
            print(beta_mean)
            # Compute gamma distribution (coefficient for Y) based on input
            # gamma_mean = self.gamma_mean(x)
            # gamma_covar = self.gamma_covar(x)

            # Compute mu distribution (intercept) based on input
            mu_mean = self.mu_mean(x)
            mu_covar = self.mu_covar(x)

            # Combine means: beta * x + gamma * y + mu
            combined_mean = (
                    beta_mean.squeeze() * x_component.squeeze() +
                    # gamma_mean.squeeze() * y_component.squeeze() +
                    mu_mean.squeeze()
            )

            jitter = 1e-2

            # Combine covariances
            # combined_covar = beta_covar + mu_covar
            combined_covar = (x_component.t() @ beta_covar.evaluate() @ x_component) + mu_covar.evaluate() + \
                    torch.eye(x.size(0), dtype=x.dtype, device=x.device) * jitter

            return gpytorch.distributions.MultivariateNormal(combined_mean, combined_covar)



    @classmethod
    def train_model(
            cls,
            train_input: torch.Tensor,
            train_target: torch.Tensor,
            max_iter: int = 50
    ):
        """
        Train the Cointegration Model
        """
        # Initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = cls.GaussianProcessModel(train_input=train_input, train_target=train_target, likelihood=likelihood)

        # Training mode
        model.train()
        likelihood.train()

        # Optimizer
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            # {'params': likelihood.parameters()}
        ], lr=0.1)

        # Loss function
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        # Optimization loop
        for i in range(max_iter):
            optimizer.zero_grad()
            output = model(train_input)
            loss = -mll(output, train_target)
            loss.backward()
            optimizer.step()

        return model, likelihood

    @classmethod
    def predict(
            cls,
            model: gpytorch.models.ExactGP,
            likelihood: gpytorch.likelihoods.GaussianLikelihood,
            test_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions using the trained model
        """
        # Predictive mode
        model.eval()
        likelihood.eval()

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Predictive distribution
            pred_dist = model(test_input)
            y_pred = likelihood(model(test_input))
            # Extract mean and standard deviation
            pred_mean = y_pred.mean
            pred_var = y_pred.variance

            # Compute confidence intervals (2 standard deviations)
            lower = pred_mean - 2 * torch.sqrt(pred_var)
            upper = pred_mean + 2 * torch.sqrt(pred_var)

        return pred_mean, lower, upper

    @classmethod
    def extract_latent_functions(
            cls,
            model: gpytorch.models.ExactGP,
            time: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Extract estimated latent functions beta and mu with confidence intervals
        """
        # Evaluation mode
        model.eval()

        # Prepare time index
        time_index = time.unsqueeze(1)

        # Compute beta estimates and confidence
        with torch.no_grad():
            # Beta estimates
            beta_mean = model.beta_mean(time_index).detach()
            beta_covar = model.beta_covar(time_index)

            # Compute beta confidence intervals
            beta_std = torch.sqrt(beta_covar.diag())
            beta_lower = beta_mean - 2 * beta_std
            beta_upper = beta_mean + 2 * beta_std

            # Mu estimates
            mu_mean = model.mu_mean(time_index).detach()
            mu_covar = model.mu_covar(time_index)

            # Compute mu confidence intervals
            mu_std = torch.sqrt(mu_covar.diag())
            mu_lower = mu_mean - 2 * mu_std
            mu_upper = mu_mean + 2 * mu_std

        return (beta_mean, beta_lower, beta_upper), (mu_mean, mu_lower, mu_upper)

    # @classmethod
    # def infer_coefficients(cls, model, X_input, Y_input):
    #     """
    #     Infer beta_t and mu_t for given X_t and Y_t
    #
    #     Args:
    #         model: Trained GP model
    #         X_input: Input X values (torch.Tensor)
    #         Y_input: Input Y values (torch.Tensor)
    #
    #     Returns:
    #         Tuple containing:
    #         - (beta_mean, beta_lower, beta_upper): Beta coefficient estimates and confidence intervals
    #         - (mu_mean, mu_lower, mu_upper): Mu coefficient estimates and confidence intervals
    #     """
    #     model.eval()  # Set to evaluation mode
    #     with torch.no_grad():
    #         # Prepare input
    #         input_data = torch.stack([X_input, Y_input], dim=1)
    #         x_component = input_data[:, 0:1].float()
    #         # Get beta distribution
    #         beta_mean = model.beta_mean(x_component).detach()
    #         # beta_mean = beta_mean[:, 0]
    #         beta_covar = model.beta_covar(x_component)
    #         beta_std = torch.sqrt(beta_covar.diag()).detach()
    #         beta_lower = (beta_mean - 2 * beta_std).detach()
    #         beta_upper = (beta_mean + 2 * beta_std).detach()
    #
    #         # Get mu distribution
    #         mu_mean = model.mu_mean(x_component).detach()
    #         mu_covar = model.mu_covar(x_component)
    #         mu_std = torch.sqrt(mu_covar.diag()).detach()
    #         mu_lower = (mu_mean - 2 * mu_std).detach()
    #         mu_upper = (mu_mean + 2 * mu_std).detach()
    #
    #         return (beta_mean, beta_lower, beta_upper), (mu_mean, mu_lower, mu_upper)
    @classmethod
    def infer_coefficients(cls, model, X_input, Y_input, train_input, train_target):
        """
        Infer posterior mean for beta_t and mu_t using the explicit formula.

        Args:
            model: Trained GP model
            X_input: Test X values (torch.Tensor)
            Y_input: Test Y values (torch.Tensor) [used for validation only]
            train_input: Training X values (torch.Tensor)
            train_target: Training Y values (torch.Tensor)

        Returns:
            Tuple:
            - (beta_mean, beta_lower, beta_upper): Posterior Beta estimates and confidence intervals
            - (mu_mean, mu_lower, mu_upper): Posterior Mu estimates and confidence intervals
        """
        model.eval()  # Ensure evaluation mode

        # Prepare test and training inputs
        X_input = X_input.unsqueeze(-1).double()  # Test inputs (N, 1)
        train_input = train_input.unsqueeze(-1).double()  # Training inputs (M, 1)
        train_target = train_target.double()

        with torch.no_grad():
            # --- Posterior for Beta_t ---
            # 1. Prior mean
            beta_mean_prior_test = model.beta_mean(X_input)
            beta_mean_prior_train = model.beta_mean(train_input)

            # 2. Kernel computations
            beta_K_xx = model.beta_covar(train_input).evaluate()  # K(X, X)
            beta_K_x_star = model.beta_covar(X_input, train_input).evaluate()  # k(x*, X)

            # 3. Posterior mean
            beta_posterior_mean = (
                    beta_mean_prior_test
                    + beta_K_x_star @ torch.linalg.solve(
                beta_K_xx, train_target - beta_mean_prior_train
            )
            )

            # 4. Posterior variance
            beta_K_x_star_star = model.beta_covar(X_input).evaluate()  # k(x*, x*)
            beta_posterior_var = (
                    beta_K_x_star_star
                    - beta_K_x_star @ torch.linalg.solve(beta_K_xx, beta_K_x_star.T)
            )
            beta_std = torch.sqrt(beta_posterior_var.diag())

            beta_lower = beta_posterior_mean - 2 * beta_std
            beta_upper = beta_posterior_mean + 2 * beta_std

            # --- Posterior for Mu_t ---
            # 1. Prior mean
            mu_mean_prior_test = model.mu_mean(X_input)
            mu_mean_prior_train = model.mu_mean(train_input)

            # 2. Kernel computations
            mu_K_xx = model.mu_covar(train_input).evaluate()
            mu_K_x_star = model.mu_covar(X_input, train_input).evaluate()

            # 3. Posterior mean
            mu_posterior_mean = (
                    mu_mean_prior_test
                    + mu_K_x_star @ torch.linalg.solve(
                mu_K_xx, train_target - mu_mean_prior_train
            )
            )

            # 4. Posterior variance
            mu_K_x_star_star = model.mu_covar(X_input).evaluate()
            mu_posterior_var = (
                    mu_K_x_star_star
                    - mu_K_x_star @ torch.linalg.solve(mu_K_xx, mu_K_x_star.T)
            )
            mu_std = torch.sqrt(mu_posterior_var.diag())

            mu_lower = mu_posterior_mean - 2 * mu_std
            mu_upper = mu_posterior_mean + 2 * mu_std

        return (beta_posterior_mean, beta_lower, beta_upper), (mu_posterior_mean, mu_lower, mu_upper)


@staticmethod
def generate_smooth_cointegration_data(
        T: int = 200,
        seed: int = 42,
        noise_std: float = 1,
        alpha_x: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic cointegrated time series with smoothly varying coefficients.

    Args:
        T: Length of the time series
        seed: Random seed for reproducibility
        noise_std: Standard deviation of the noise
        alpha_x: AR(1) coefficient for X process

    Returns:
        time: Time index
        X: Cointegrating variable
        Y: Cointegrated series
        beta_t: True time-varying beta coefficient
        mu_t: True time-varying mu coefficient
    """
    torch.manual_seed(seed)

    # Generate time index
    time = torch.linspace(0, 4 * np.pi, T, dtype=torch.float64)

    # Generate X as an AR(1) process
    X = torch.zeros(T, dtype=torch.float64)
    for t in range(1, T):
        X[t] = alpha_x * X[t - 1] + torch.randn(1, dtype=torch.float64) * 0.5

    # Create smooth time-varying coefficients using sinusoidal functions
    beta_t = 2 + torch.sin(time / 2) + 0.5 * torch.sin(time / 4)  # Varies between 1 and 3
    mu_t = torch.cos(time / 3) - 0.5 * torch.cos(time / 6)  # Varies between -1.5 and 1.5

    # Generate Y with smooth time-varying coefficients
    noise = torch.randn(T, dtype=torch.float64) * noise_std
    Y = beta_t * X + mu_t + noise

    return time, X, Y, beta_t, mu_t, noise


def main():
    # Generate synthetic cointegrated data
    T = 200
    # time, X, Y, true_beta_t, true_mu_t = CointegrationAdditivePGModel.DataGenerator.generate_cointegration_data(T=T)
    time, X, Y, true_beta_t, true_mu_t, true_spread = generate_smooth_cointegration_data(T=T)

    # Prepare input for training
    train_input = torch.stack([X, Y], dim=1)
    train_target = Y  # Target is Y

    # Train Additive GP Model

    train_input = train_input.float()
    train_target = train_target.float()

    model, likelihood = CointegrationAdditivePGModel.train_model(train_input, train_target)

    # Make predictions
    pred_mean, pred_lower, pred_upper = CointegrationAdditivePGModel.predict(model, likelihood, train_input)

    # Extract latent functions
    beta_estimates, mu_estimates = CointegrationAdditivePGModel.infer_coefficients(model, X, Y, X, Y)

    # Visualize results
    plt.figure(figsize=(12, 16))

    # Plot 1: Original Time Series and Prediction
    plt.subplot(5, 1, 1)
    plt.title('Cointegrated Time Series: Original vs Prediction')
    plt.plot(time.numpy(), X.numpy(), label='X (Cointegrating Variable)')
    plt.plot(time.numpy(), Y.numpy(), label='True Y', alpha=0.7)
    plt.plot(time.numpy(), pred_mean.numpy(), label='Predicted Y', color='red')
    plt.fill_between(time.numpy(),
                     pred_lower.numpy(),
                     pred_upper.numpy(),
                     color='red', alpha=0.2)
    plt.legend()

    # Plot 2: Beta Coefficient
    plt.subplot(5, 1, 2)
    plt.title('Beta Coefficient')
    beta_mean, beta_lower, beta_upper = beta_estimates
    plt.plot(time.numpy(), beta_mean.numpy(), label='Estimated Beta', color='red')
    plt.fill_between(time.numpy(),
                     beta_lower.numpy(),
                     beta_upper.numpy(),
                     color='red', alpha=0.2)
    plt.plot(time.numpy(), true_beta_t.numpy(), label='True Beta', color='blue', linestyle='--')
    plt.legend()

    # Plot 3: Mu Coefficient
    plt.subplot(5, 1, 3)
    plt.title('Mu Coefficient')
    mu_mean, mu_lower, mu_upper = mu_estimates
    plt.plot(time.numpy(), mu_mean.numpy(), label='Estimated Mu', color='red')
    plt.fill_between(time.numpy(),
                     mu_lower.numpy(),
                     mu_upper.numpy(),
                     color='red', alpha=0.2)
    plt.plot(time.numpy(), true_mu_t.numpy(), label='True Mu', color='blue', linestyle='--')
    plt.legend()

    # Plot 4: Residuals
    plt.subplot(5, 1, 4)
    plt.title('Residuals')
    residuals = Y.numpy() - pred_mean.numpy()
    plt.plot(time.numpy(), residuals, label='Residuals', color='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.legend()


    plt.subplot(5, 1, 5)
    plt.title('spread estimation')
    spread_est = Y.numpy() - beta_mean.numpy() * X.numpy() - mu_mean.numpy()
    residuals = Y.numpy() - pred_mean.numpy()
    plt.plot(time.numpy(), spread_est, label='Spread_est', color='red')
    plt.plot(time.numpy(), residuals, label='Residuals', color='green')
    plt.plot(time.numpy(), true_spread, label='True Spread', color='blue')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()