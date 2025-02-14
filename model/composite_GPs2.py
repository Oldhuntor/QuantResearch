import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Tuple


class MultiplicativeGP(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_inputs, train_targets, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2,
                                       lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0))
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class AdditiveGP(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_inputs, train_targets, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2,
                                       lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0))
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class ErrorGP(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_inputs, train_targets, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2,
                                       lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0))
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class ThreeComponentGP:
    """
    A composite GP that models:
      Y_t = X_t * GP1(X_t,Y_t) + GP2(X_t,Y_t) + ErrorGP(X_t,Y_t)

    This code uses (X_norm, Y_norm) as inputs for each sub-GP,
    but the final combination explicitly multiplies GP1's output by X_norm in training.
    """

    def __init__(self, X: torch.Tensor, Y: torch.Tensor, window_size: int = 50):
        """
        Initialize Three-Component GP with a fixed sliding window size.
        We'll store the first 'window_size' points as the initial training set.
        """
        torch.set_default_dtype(torch.float64)
        self.window_size = window_size

        # Convert inputs to double tensors
        self.X_full = torch.tensor(X, dtype=torch.float64) if not torch.is_tensor(X) else X.double()
        self.Y_full = torch.tensor(Y, dtype=torch.float64) if not torch.is_tensor(Y) else Y.double()

        # Keep only the first 'window_size' points initially
        self.X = self.X_full[:window_size]
        self.Y = self.Y_full[:window_size]

        # Standardize
        self.X_mean, self.X_std = self.X.mean(), self.X.std()
        self.Y_mean, self.Y_std = self.Y.mean(), self.Y.std()

        X_normalized = (self.X - self.X_mean) / self.X_std
        Y_normalized = (self.Y - self.Y_mean) / self.Y_std

        self.Y_norm = Y_normalized
        self.X_norm = X_normalized

        # 2D input: [X_norm, Y_norm]
        # train_inputs = torch.stack([X_normalized, Y_normalized], dim=1)

        time_index = torch.arange(len(self.X), dtype=torch.float64)
        self.time_index = time_index
        beta_target = Y_normalized/X_normalized

        train_inputs = torch.stack([time_index, beta_target], dim=1)

        # Sub-GPs
        self.gp1 = MultiplicativeGP(train_inputs, beta_target)
        # self.gp2 = AdditiveGP(train_inputs, Y_normalized)
        # self.gp_error = ErrorGP(train_inputs, Y_normalized)

        # We'll store the training "sliding window" as we go
        self.train_inputs_gp1 = train_inputs.clone()
        self.train_targets_gp1 = beta_target.clone()
        # self.train_inputs_gp2 = train_inputs.clone()
        # self.train_targets_gp2 = Y_normalized.clone()
        # self.train_inputs_error = train_inputs.clone()
        # self.train_targets_error = Y_normalized.clone()

    def train_model(self, num_epochs: int = 100, learning_rate: float = 0.01, verbose: bool = True):
        """
        Full training on the current window.
        """
        self.gp1.train()
        # self.gp2.train()
        # self.gp_error.train()

        opt1 = torch.optim.Adam(self.gp1.parameters(), lr=learning_rate)
        # opt2 = torch.optim.Adam(self.gp2.parameters(), lr=learning_rate)
        # optE = torch.optim.Adam(self.gp_error.parameters(), lr=learning_rate)

        mll1 = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp1.likelihood, self.gp1)
        # mll2 = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp2.likelihood, self.gp2)
        # mllE = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_error.likelihood, self.gp_error)

        # train_inputs = self.train_inputs_gp1  # shape (M,2)
        for epoch in range(num_epochs):
            opt1.zero_grad()
            # opt2.zero_grad()
            # optE.zero_grad()

            with gpytorch.settings.cholesky_jitter(1e-3):
                output1 = self.gp1(self.train_inputs_gp1)  # GP1 => something
                # output2 = self.gp2(train_inputs)  # GP2 => something
                train_target_gp2 = self.Y_norm - output1 * self.X_norm




                self.gp2 = AdditiveGP(train_target_gp2, train_target_gp2)


                # According to Y = X * GP1 + GP2 + Error
                combined_mean = X_norm * output1.mean + output2.mean

                residuals = self.train_targets_gp1 - combined_mean
                self.gp_error.train_targets = residuals
                outputE = self.gp_error(self.train_inputs_error)

                loss1 = -mll1(output1, self.gp1.train_targets)
                loss2 = -mll2(output2, self.gp2.train_targets)
                lossE = -mllE(outputE, residuals)

                total_loss = loss1 + loss2 + lossE
                total_loss.backward()

            opt1.step()
            opt2.step()
            optE.step()

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Total Loss={total_loss.item():.4f}")


    def predict(self, X_new: torch.Tensor, Y_new: torch.Tensor, num_samples: int = 1000) -> Dict[
        str, Dict[str, torch.Tensor]]:
        """
        Make predictions with all components (gp1, gp2, error), returning a dict:
            'gp1': {...}, 'gp2': {...}, 'error': {...}, 'final': {...}
        The final combination is Y_pred = X_new * gp1 + gp2 + error
        """
        X_new = torch.tensor(X_new, dtype=torch.float64) if not torch.is_tensor(X_new) else X_new.double()
        Y_new = torch.tensor(Y_new, dtype=torch.float64) if not torch.is_tensor(Y_new) else Y_new.double()

        X_new_norm = (X_new - self.X_mean) / self.X_std
        Y_new_norm = (Y_new - self.Y_mean) / self.Y_std
        test_inputs = torch.stack([X_new_norm, Y_new_norm], dim=1)

        self.gp1.eval()
        self.gp2.eval()
        self.gp_error.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior1 = self.gp1(test_inputs)
            posterior2 = self.gp2(test_inputs)
            posterior_error = self.gp_error(test_inputs)

            samples1 = posterior1.sample(torch.Size([num_samples]))
            samples2 = posterior2.sample(torch.Size([num_samples]))
            samples_error = posterior_error.sample(torch.Size([num_samples]))

            # Un-normalize
            mean1 = posterior1.mean * self.Y_std
            mean2 = posterior2.mean * self.Y_std
            mean_error = posterior_error.mean * self.Y_std

            variance1 = posterior1.variance * (self.Y_std ** 2)
            variance2 = posterior2.variance * (self.Y_std ** 2)
            variance_error = posterior_error.variance * (self.Y_std ** 2)

            samples1 = samples1 * self.Y_std
            samples2 = samples2 * self.Y_std
            samples_error = samples_error * self.Y_std

            q_tensor = torch.tensor([0.025, 0.975], dtype=torch.float64)
            ci1 = torch.quantile(samples1, q_tensor, dim=0)
            ci2 = torch.quantile(samples2, q_tensor, dim=0)
            ci_error = torch.quantile(samples_error, q_tensor, dim=0)

            # Combine final
            X_scaled = X_new  # original scale
            final_samples = X_scaled.unsqueeze(0) * samples1 + samples2 + samples_error
            final_mean = X_scaled * mean1 + mean2 + mean_error
            final_variance = (X_scaled ** 2) * variance1 + variance2 + variance_error
            final_ci = torch.quantile(final_samples, q_tensor, dim=0)

            return {
                'gp1': {
                    'mean': mean1,
                    'variance': variance1,
                    'samples': samples1,
                    'credible_intervals': ci1
                },
                'gp2': {
                    'mean': mean2,
                    'variance': variance2,
                    'samples': samples2,
                    'credible_intervals': ci2
                },
                'error': {
                    'mean': mean_error,
                    'variance': variance_error,
                    'samples': samples_error,
                    'credible_intervals': ci_error
                },
                'final': {
                    'mean': final_mean,
                    'variance': final_variance,
                    'samples': final_samples,
                    'credible_intervals': final_ci
                }
            }

    def plot_all_components(self,
                            time_value: torch.Tensor,
                            X_new: torch.Tensor,
                            Y_new: torch.Tensor,
                            fig_size: Tuple[int, int] = (15, 12),
                            true_values: Optional[torch.Tensor] = None,
                            true_gp1: Optional[torch.Tensor] = None,
                            true_gp2: Optional[torch.Tensor] = None,
                            true_error: Optional[torch.Tensor] = None):
        """
        Plots gp1, gp2, error, and final with optional 'true' overlays.
        """
        predictions = self.predict(X_new, Y_new)

        fig = plt.figure(figsize=fig_size)
        gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 1])

        X_new = time_value
        # Subplot 1: Training Data
        ax0 = fig.add_subplot(gs[0, :])
        ax0.scatter(time_value, Y_new, c='blue', alpha=0.5, label='Training Data')

        if true_values is not None:
            ax0.plot(X_new.numpy(), true_values.numpy(), 'r-', label='True Values', alpha=0.7)
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
        ax0.set_title('Training Data')
        ax0.legend()
        ax0.grid(True)

        def plot_component(ax, pred_dict, title, color='b', ylabel=''):
            X_plot = X_new.numpy()
            mean = pred_dict['mean'].numpy()
            ci_lower = pred_dict['credible_intervals'][0].numpy()
            ci_upper = pred_dict['credible_intervals'][1].numpy()

            ax.plot(X_plot, mean, f'{color}--', label='Predicted Mean')
            ax.fill_between(X_plot, ci_lower, ci_upper, color=color, alpha=0.2, label='95% CI')
            ax.set_title(title)
            ax.set_xlabel('X')
            ax.set_ylabel(ylabel)
            ax.grid(True)

        # Subplot 2: Multiplicative Component (X * GP1)
        ax1 = fig.add_subplot(gs[1, 0])
        plot_component(ax1, predictions['gp1'], 'Multiplicative Component (X * GP1)', 'b', 'X * GP1')
        if true_gp1 is not None:
            ax1.plot(X_new.numpy(), true_gp1.numpy(), 'k-', label='True gp1')
        ax1.legend()

        # Subplot 3: Additive Component (GP2)
        ax2 = fig.add_subplot(gs[1, 1])
        plot_component(ax2, predictions['gp2'], 'Additive Component (GP2)', 'g', 'GP2')
        if true_gp2 is not None:
            ax2.plot(X_new.numpy(), true_gp2.numpy(), 'k-', label='True gp2')
        ax2.legend()

        # Subplot 4: Error Component
        ax3 = fig.add_subplot(gs[2, 0])
        plot_component(ax3, predictions['error'], 'Error Component', 'r', 'Error')
        if true_error is not None:
            ax3.plot(X_new.numpy(), true_error.numpy(), 'k--', label='True Error')
        ax3.legend()

        # Subplot 5: Final Combined
        ax4 = fig.add_subplot(gs[2, 1])
        plot_component(ax4, predictions['final'], 'Final Combined Prediction', 'b', 'Y')
        if true_values is not None:
            ax4.plot(X_new.numpy(), true_values.numpy(), 'r--', label='True Values', alpha=0.7)
        ax4.legend()

        plt.tight_layout()
        plt.show()
        return fig

    @staticmethod
    def generate_cointegration_data(
            T=200,
            seed=42,
            noise_std=0.1,
            change_points=(70, 140),
            beta_values=(1.0, 2.0, 0.5),
            mu_values=(0.0, -1.0, 1.0),
            alpha=0.9
    ):
        torch.manual_seed(seed)
        time = torch.arange(T, dtype=torch.float64)

        # Generate Y_t as an AR(1)
        X = torch.zeros(T, dtype=torch.float64)
        for t in range(1, T):
            X[t] = alpha * X[t - 1] + torch.randn(1, dtype=torch.float64) * 0.5

        beta_t = torch.zeros(T, dtype=torch.float64)
        mu_t = torch.zeros(T, dtype=torch.float64)

        segments = [0] + list(change_points) + [T]
        for i in range(len(segments) - 1):
            start_idx = segments[i]
            end_idx = segments[i + 1]
            beta_t[start_idx:end_idx] = beta_values[i]
            mu_t[start_idx:end_idx] = mu_values[i]

        noise = torch.randn(T, dtype=torch.float64) * noise_std
        Y = beta_t * X + mu_t + noise

        return time, X, Y, beta_t, mu_t


if __name__ == "__main__":

    data_points = 500
    window_size = 500
    change_points = (499,)
    time, X_coin, Y_coin, beta_true, mu_true = ThreeComponentGP.generate_cointegration_data(
        T=data_points,
        seed=42,
         change_points=change_points,
        beta_values=(1, 10),
        mu_values=(1, 10),
        alpha=1
    )

    # 2) Initialize the ThreeComponentGP with window_size=50
    model = ThreeComponentGP(X_coin, Y_coin, window_size=window_size)
    # Train the model on the initial window
    model.train_model(num_epochs=50, learning_rate=0.01)

    # 3) Simulate online updates for the rest of the points
    # N = len(X_coin)
    # for t in range(window_size, N):
    #     model.online_update(X_coin[t].item(), Y_coin[t].item())


    model.plot_all_components(
        X_new=X_coin,
        Y_new=Y_coin,
        time_value=time,
        fig_size=(20,10),
        true_values=Y_coin,
        true_gp1=beta_true,
        true_gp2=mu_true,
        true_error=None
    )
