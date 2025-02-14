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
    def __init__(self,
                 X: torch.Tensor,
                 Y: torch.Tensor,
                 window_size: int = 100):
        """
        Initialize Three-Component GP model with a fixed sliding window size.
        Xt = Yt * GP1(Xt,Yt) + GP2(Xt,Yt) + GP_error(Xt,Yt)

        Args:
            X: Input tensor (1D)
            Y: Target tensor (1D)
            window_size: fixed number of points to keep in memory
        """
        torch.set_default_dtype(torch.float64)
        self.window_size = window_size

        # Convert inputs to tensors
        self.X_full = torch.tensor(X, dtype=torch.float64) if not torch.is_tensor(X) else X.double()
        self.Y_full = torch.tensor(Y, dtype=torch.float64) if not torch.is_tensor(Y) else Y.double()

        # For demonstration, let's keep only the first 'window_size' points
        # as initial training window:
        self.X = self.X_full[:window_size]
        self.Y = self.Y_full[:window_size]

        # Standardize (based on the initial window)
        self.X_mean, self.X_std = self.X.mean(), self.X.std()
        self.Y_mean, self.Y_std = self.Y.mean(), self.Y.std()

        X_normalized = (self.X - self.X_mean) / self.X_std
        Y_normalized = (self.Y - self.Y_mean) / self.Y_std

        train_inputs = torch.stack([X_normalized, Y_normalized], dim=1)

        self.gp1 = MultiplicativeGP(train_inputs, Y_normalized)
        self.gp2 = AdditiveGP(train_inputs, Y_normalized)
        self.gp_error = ErrorGP(train_inputs, Y_normalized)

        # We'll store the training "sliding window" as we go
        self.train_inputs_gp1 = train_inputs.clone()  # shape: (window_size, 2)
        self.train_targets_gp1 = Y_normalized.clone()  # shape: (window_size)
        self.train_inputs_gp2 = train_inputs.clone()
        self.train_targets_gp2 = Y_normalized.clone()
        self.train_inputs_error = train_inputs.clone()
        self.train_targets_error = Y_normalized.clone()  # error GP initially same Y, later updated

        # We'll also store the inverse kernel matrices once computed
        self.K_inv_gp1 = None
        self.K_inv_gp2 = None
        self.K_inv_error = None

    def train_model(self, num_epochs: int = 100, learning_rate: float = 0.01, verbose: bool = True):
        """
        Naive full-batch training on the initial window.
        """
        self.gp1.train()
        self.gp2.train()
        self.gp_error.train()

        # Initialize optimizers
        optimizer1 = torch.optim.Adam(self.gp1.parameters(), lr=learning_rate)
        optimizer2 = torch.optim.Adam(self.gp2.parameters(), lr=learning_rate)
        optimizer_error = torch.optim.Adam(self.gp_error.parameters(), lr=learning_rate)

        mll1 = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp1.likelihood, self.gp1)
        mll2 = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp2.likelihood, self.gp2)
        mll_error = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_error.likelihood, self.gp_error)

        for i in range(num_epochs):
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer_error.zero_grad()

            with gpytorch.settings.cholesky_jitter(1e-3):
                # Forward pass
                output1 = self.gp1(self.train_inputs_gp1)
                output2 = self.gp2(self.train_inputs_gp2)

                X_norm = self.train_inputs_gp1[:, 0]
                combined_mean = X_norm * output1.mean + output2.mean

                # residual
                residuals = self.train_targets_gp1 - combined_mean
                self.gp_error.train_targets = residuals
                output_error = self.gp_error(self.train_inputs_error)

                loss1 = -mll1(output1, self.train_targets_gp1)
                loss2 = -mll2(output2, self.train_targets_gp2)
                loss_error = -mll_error(output_error, residuals)

                total_loss = loss1 + loss2 + loss_error
                total_loss.backward()

            optimizer1.step()
            optimizer2.step()
            optimizer_error.step()

            if verbose and (i + 1) % 10 == 0:
                print(f"Epoch {i + 1}/{num_epochs}, Total Loss={total_loss.item():.4f}")

        # After training, compute and store the inverses of the final kernel matrices
        self.K_inv_gp1 = self._compute_kernel_inverse(self.gp1, self.train_inputs_gp1)
        self.K_inv_gp2 = self._compute_kernel_inverse(self.gp2, self.train_inputs_gp2)
        self.K_inv_error = self._compute_kernel_inverse(self.gp_error, self.train_inputs_error)

    def _compute_kernel_inverse(self, gp_model, train_inputs):
        """
        Compute (K + noise*I)^{-1} for the trained GP model on the current window.
        """
        gp_model.eval()
        with torch.no_grad():
            # Compute full kernel matrix
            K = gp_model.covar_module(train_inputs).evaluate()  # shape (M, M)
            noise = gp_model.likelihood.noise  # scalar
            K = K + noise * torch.eye(K.size(0), dtype=K.dtype, device=K.device)

            K_inv = torch.inverse(K)  # naive approach, O(M^3)
        gp_model.train()  # revert to train mode
        return K_inv

    def online_update(self, x_new: float, y_new: float, is_fine_tuned: bool = False):
        """
        Perform a single sliding-window update using rank-1 downdate (remove oldest)
        and rank-1 update (add new point) for each sub-GP.

        x_new, y_new: new scalar observation
        """
        # 1. Identify the oldest point index
        oldest_idx = 0  # e.g. always remove index 0 from the window

        # 2. Remove oldest from X, Y
        x_old = self.X[oldest_idx]
        y_old = self.Y[oldest_idx]

        self.X = torch.cat([self.X[1:], torch.tensor([x_new], dtype=torch.float64)], dim=0)
        self.Y = torch.cat([self.Y[1:], torch.tensor([y_new], dtype=torch.float64)], dim=0)

        # 3. Update the standardization stats *optionally*
        # For a purely sliding window, you might re-compute mean/std from the new window:
        self.X_mean, self.X_std = self.X.mean(), self.X.std()
        self.Y_mean, self.Y_std = self.Y.mean(), self.Y.std()

        # Re-normalize the entire window
        X_norm = (self.X - self.X_mean) / self.X_std
        Y_norm = (self.Y - self.Y_mean) / self.Y_std

        train_inputs = torch.stack([X_norm, Y_norm], dim=1)

        # We need to do the same for gp1, gp2, gp_error.
        # BUT each GP might store a separate target (gp_error has the residual target).
        # For gp1, gp2: the "target" is Y_norm
        # For gp_error: the target is the residual, which depends on the combined mean from gp1, gp2.

        # 4. Apply rank-1 updates to K_inv for each sub-GP
        #    or re-compute from scratch for simplicity.
        #    For demonstration, let's do re-compute from scratch (naive).
        #    If you want true "Woodbury updates," see the rank-1 update snippet below.

        self.train_inputs_gp1 = train_inputs
        self.train_targets_gp1 = Y_norm

        self.train_inputs_gp2 = train_inputs
        self.train_targets_gp2 = Y_norm

        # We must compute the combined mean from gp1 + gp2 to get the new residual for gp_error.
        # But that requires gp1 and gp2 posteriors. Let's do a quick inference step:

        # Evaluate gp1, gp2 in eval mode on the new window
        self.gp1.eval()
        self.gp2.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post_gp1 = self.gp1(train_inputs)
            post_gp2 = self.gp2(train_inputs)
            # combined mean
            combined_mean = train_inputs[:, 0] * post_gp1.mean + post_gp2.mean
        residuals = self.train_targets_gp1 - combined_mean  # same shape as Y_norm

        self.train_inputs_error = train_inputs
        self.train_targets_error = residuals

        # For simplicity, let's just re-compute the kernel inverse from scratch:
        self.K_inv_gp1 = self._compute_kernel_inverse(self.gp1, self.train_inputs_gp1)
        self.K_inv_gp2 = self._compute_kernel_inverse(self.gp2, self.train_inputs_gp2)

        # Now re-compute error gp inverse
        self.gp_error.eval()
        with torch.no_grad():
            # Manually set the error GP's training data
            self.gp_error.set_train_data(inputs=self.train_inputs_error,
                                         targets=self.train_targets_error,
                                         strict=False)
        self.K_inv_error = self._compute_kernel_inverse(self.gp_error, self.train_inputs_error)
        self.gp_error.train()

        # For gp1:
        self.gp1.set_train_data(inputs=self.train_inputs_gp1, targets=self.train_targets_gp1, strict=False)

        # For gp2:
        self.gp2.set_train_data(inputs=self.train_inputs_gp2, targets=self.train_targets_gp2, strict=False)

        # For error gp:
        # self.gp_error.set_train_data(inputs=self.train_inputs_error, targets=self.train_targets_error, strict=False)

        # Optionally do a small gradient-based hyperparameter update here
        if is_fine_tuned:
            # if you want the GPs to adapt slightly:
            self._fine_tune_subGPs()

    def _fine_tune_subGPs(self, steps=5, lr=0.01):
        """
        Optionally do a short gradient-based update of hyperparameters after an online update.
        """
        self.gp1.train()
        self.gp2.train()
        self.gp_error.train()

        opt1 = torch.optim.Adam(self.gp1.parameters(), lr=lr)
        opt2 = torch.optim.Adam(self.gp2.parameters(), lr=lr)
        optE = torch.optim.Adam(self.gp_error.parameters(), lr=lr)

        mll1 = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp1.likelihood, self.gp1)
        mll2 = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp2.likelihood, self.gp2)
        mllE = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_error.likelihood, self.gp_error)

        for i in range(steps):
            opt1.zero_grad()
            opt2.zero_grad()
            optE.zero_grad()

            output1 = self.gp1(self.train_inputs_gp1)
            output2 = self.gp2(self.train_inputs_gp2)
            loss1 = -mll1(output1, self.train_targets_gp1)
            loss2 = -mll2(output2, self.train_targets_gp2)

            outputE = self.gp_error(self.train_inputs_error)
            lossE = -mllE(outputE, self.train_targets_error)

            total_loss = loss1 + loss2 + lossE
            total_loss.backward()
            opt1.step()
            opt2.step()
            optE.step()

    def predict(self, X_new: torch.Tensor, Y_new: torch.Tensor, num_samples: int = 1000) -> Dict[
        str, Dict[str, torch.Tensor]]:
        """
        Your original predict method, unchanged.
        """
        X_new = torch.tensor(X_new, dtype=torch.float64) if not torch.is_tensor(X_new) else X_new.double()
        Y_new = torch.tensor(Y_new, dtype=torch.float64) if not torch.is_tensor(Y_new) else Y_new.double()

        # Normalize inputs
        X_new_norm = (X_new - self.X_mean) / self.X_std
        Y_new_norm = (Y_new - self.Y_mean) / self.Y_std

        # Stack inputs
        test_inputs = torch.stack([X_new_norm, Y_new_norm], dim=1)

        self.gp1.eval()
        self.gp2.eval()
        self.gp_error.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Posterior from gp1, gp2, error
            posterior1 = self.gp1(test_inputs)
            posterior2 = self.gp2(test_inputs)
            posterior_error = self.gp_error(test_inputs)

            # Draw samples
            samples1 = posterior1.sample(torch.Size([num_samples]))
            samples2 = posterior2.sample(torch.Size([num_samples]))
            samples_error = posterior_error.sample(torch.Size([num_samples]))

            # Scale back
            mean1 = posterior1.mean * self.Y_std
            mean2 = posterior2.mean * self.Y_std
            mean_error = posterior_error.mean * self.Y_std

            variance1 = posterior1.variance * self.Y_std ** 2
            variance2 = posterior2.variance * self.Y_std ** 2
            variance_error = posterior_error.variance * self.Y_std ** 2

            samples1 = samples1 * self.Y_std
            samples2 = samples2 * self.Y_std
            samples_error = samples_error * self.Y_std

            q_tensor = torch.tensor([0.025, 0.975], dtype=torch.float64)
            ci1 = torch.quantile(samples1, q_tensor, dim=0)
            ci2 = torch.quantile(samples2, q_tensor, dim=0)
            ci_error = torch.quantile(samples_error, q_tensor, dim=0)

            X_scaled = X_new

            final_samples = X_scaled.unsqueeze(0) * samples1 + samples2 + samples_error
            final_mean = X_scaled * mean1 + mean2 + mean_error
            final_variance = (X_scaled ** 2 * variance1 + variance2 + variance_error)
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

    def plot_all_components(
            self,
            X_new: torch.Tensor,
            Y_new: torch.Tensor,
            fig_size: Tuple[int, int] = (25, 12),
            true_values: Optional[torch.Tensor] = None,
            time_values: Optional[torch.Tensor] = None,
            true_gp1: Optional[torch.Tensor] = None,
            true_gp2: Optional[torch.Tensor] = None,
            true_error: Optional[torch.Tensor] = None
    ):
        """
        Plot the three predicted GP components and the final combined prediction,
        and optionally overlay the 'true' gp1, gp2, and error if you have them.

        The predicted components come from self.predict(X_new, Y_new),
        which returns a dict with keys 'gp1', 'gp2', 'error', 'final'.
        """
        # 1) Get all predictions (including sub-components) at X_new, Y_new
        predictions = self.predict(X_new, Y_new)
        if time_values is not None:
            X_new = time_values
        fig = plt.figure(figsize=fig_size)
        gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 1])

        # 2) Top subplot: Training data and (optional) true final values
        ax0 = fig.add_subplot(gs[0, :])
        if time_values is None:
            ax0.scatter(self.X.numpy(), self.Y.numpy(), c='blue', alpha=0.5, label='Training Data')
        if true_values is not None:
            ax0.plot(X_new.numpy(), true_values.numpy(), 'r-', label='True Values', alpha=0.7)
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
        ax0.set_title('Training Data')
        ax0.legend()
        ax0.grid(True)

        # Helper function for each component subplot
        def plot_component(ax, pred_dict, title, color='b', ylabel=''):
            X_plot = X_new.numpy()
            mean = pred_dict['mean'].numpy()
            ci_lower = pred_dict['credible_intervals'][0].numpy()
            ci_upper = pred_dict['credible_intervals'][1].numpy()
            ax.plot(X_plot, mean, f'{color}--', label='Pred Mean')
            ax.fill_between(X_plot, ci_lower, ci_upper, color=color, alpha=0.2, label='95% CI')
            ax.set_xlabel('X')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True)

        # 3) Multiplicative component (gp1) subplot
        ax1 = fig.add_subplot(gs[1, 0])
        plot_component(ax1, predictions['gp1'], 'Multiplicative Component (Y * GP1)', 'b', 'GP1 * Y')
        # If you have the "true" gp1 array (e.g., Beta * Y), overlay it:
        if true_gp1 is not None:
            ax1.plot(X_new.numpy(), true_gp1.numpy(), 'k-', label='True gp1')
            ax1.legend()

        # 4) Additive component (gp2) subplot
        ax2 = fig.add_subplot(gs[1, 1])
        plot_component(ax2, predictions['gp2'], 'Additive Component (GP2)', 'g', 'GP2')
        if true_gp2 is not None:
            ax2.plot(X_new.numpy(), true_gp2.numpy(), 'k-', label='True gp2')
            ax2.legend()

        # 5) Error component subplot
        ax3 = fig.add_subplot(gs[2, 0])
        plot_component(ax3, predictions['error'], 'Error Component', 'r', 'Error')
        if true_error is not None:
            ax3.plot(X_new.numpy(), true_error.numpy(), 'k-', label='True Error')
            ax3.legend()

        # 6) Final combined prediction subplot
        ax4 = fig.add_subplot(gs[2, 1])
        plot_component(ax4, predictions['final'], 'Final Combined Prediction', 'b', 'Y')
        if true_values is not None:
            ax4.plot(X_new.numpy(), true_values.numpy(), 'r--', label='True Final')
            ax4.legend()

        plt.tight_layout()
        plt.show()
        return fig

    def predict_spread(self, X_eval: torch.Tensor, Y_eval: torch.Tensor,
                       return_components: bool = False):
        """
        Predict the 'spread' = X - (Beta_pred * Y + Mu_pred).
        This is effectively the residual if the GP's Beta_t, Mu_t were correct.

        Args:
            X_eval, Y_eval: Tensors for new input data
            return_components: If True, also return predicted Beta_t and Mu_t

        Returns:
            spread: predicted spread for each point (in original scale).
            (optionally) (beta_pred, mu_pred) if return_components is True.
        """
        # Convert & normalize
        X_eval = X_eval.double() if torch.is_tensor(X_eval) else torch.tensor(X_eval, dtype=torch.float64)
        Y_eval = Y_eval.double() if torch.is_tensor(Y_eval) else torch.tensor(Y_eval, dtype=torch.float64)

        X_norm = (X_eval - self.X_mean) / self.X_std
        Y_norm = (Y_eval - self.Y_mean) / self.Y_std
        test_inputs = torch.stack([X_norm, Y_norm], dim=1)

        self.gp1.eval()
        self.gp2.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior_beta = self.gp1(test_inputs)  # Beta predictions (normalized scale)
            posterior_mu = self.gp2(test_inputs)  # Mu predictions (normalized scale)

        # Un-normalize these predicted Beta, Mu (the GPs predicted normalized outputs)
        # Actually, "Beta(t)" was predicted in normalized Y space. So the true Beta(t) in original scale
        # is posterior_beta.mean * (Y_std / Y_std)? It's a subtlety. Usually we interpret "gp1" as Beta(t) directly,
        # but recall gp1 was trained on normalized Y. That might be okay if the code interprets gp1 as Beta(t).
        # We'll treat the gp1 output as "Beta_pred" directly in original scale by multiplying by Y_std, but remember
        # that in the final model we do: X ~ Y*(Beta_pred * Y_std) ???
        #
        # However, your code used "gp1.train_targets = Y_norm". That means the GP1 is effectively modeling Y_norm.
        # If we want Beta(t) directly, we might consider a direct approach. But let's follow the logic that
        # the final model is X = Y*gp1(...). If gp1 is "Beta(t)" in normalized space, the actual Beta(t) might be gp1(...) * (Y_std / Y_std?).
        #
        # For simplicity, let's interpret gp1(...) as Beta in "dimensionless" form.
        # So the "Beta_pred" in original space is just posterior_beta.mean (no further scale needed, because the code does Y_norm * Beta_pred to get part of the final).
        beta_pred = posterior_beta.mean
        mu_pred = posterior_mu.mean

        # Predicted X from the GPs is: X_pred = Y_eval * Beta_pred + Mu_pred + Error_pred
        # But we don't have error included in the "spread" logic— if we want the final best guess for X (no sampling):
        #   X_pred = Y_eval * beta_pred + mu_pred + predicted_error
        # For the "spread," we do: spread = X_eval - (beta_pred * Y_eval + mu_pred)

        # In practice, the GP's "Beta_pred" and "Mu_pred" are returned in "normalized Y" scale.
        # We'll treat them as dimensionless for the multiplicative part. So let's get them as is:
        spread_pred = X_eval - (beta_pred * Y_eval + mu_pred)

        if return_components:
            return spread_pred, beta_pred, mu_pred
        else:
            return spread_pred

    def plot_cointegration_spread(
            self, X_eval: torch.Tensor, Y_eval: torch.Tensor,
            time: torch.Tensor,
            true_beta: Optional[torch.Tensor] = None,
            true_mu: Optional[torch.Tensor] = None,
            title: str = "Cointegration Spread"
    ):
        """
        Plot the 'spread' = X - Beta(t)*Y - Mu(t).
        If true_beta, true_mu are provided, also plot the 'true spread' for comparison.
        """
        X_eval = X_eval.double()
        Y_eval = Y_eval.double()

        # Predicted spread from the GPs
        spread_pred, beta_pred, mu_pred = self.predict_spread(X_eval, Y_eval, return_components=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time.numpy(), spread_pred.numpy(), 'b-', label="Predicted Spread")

        if (true_beta is not None) and (true_mu is not None):
            # Compute true spread if we know the piecewise Beta, Mu
            spread_true = X_eval - (true_beta * Y_eval + true_mu)
            ax.plot(time.numpy(), spread_true.numpy(), 'r--', label="True Spread")

        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel("X (or Time Index)")
        ax.set_ylabel("Spread = X - Beta*Y - Mu")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

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
        """
        Generate a piecewise-constant cointegration dataset:
          X_t = Beta_t * Y_t + Mu_t + noise
        with discontinuous changes in Beta_t and Mu_t.

        Returns:
            time: (T,) tensor
            X: (T,) tensor
            Y: (T,) tensor
            beta_t: (T,) true piecewise Beta(t)
            mu_t:   (T,) true piecewise Mu(t)
        """
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


def plot_cointegration_data(time, X, Y, beta_t, mu_t, title="Cointegration Series"):
    """
    Plot the generated cointegration data and piecewise parameters (Beta_t, Mu_t).
    """
    fig, axs = plt.subplots(3, 1, figsize=(15, 5), sharex=True)
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
    # Generate sample data
    # data_points = 600
    # window_size = 100
    # torch.manual_seed(0)
    # X = torch.linspace(0, 6, data_points)

    # True components
    # true_gp1 = torch.sin(2 * np.pi * X)
    # true_gp2 = 0.5 * torch.cos(4 * np.pi * X)
    # true_error = torch.randn(data_points) * 0.1
    # Y = X * true_gp1 + true_gp2 + true_error

    # Create model with window_size=50, for example
    # X_train = X[:window_size]
    # Y_train = Y[:window_size]
    # X_train = X[-2*window_size:-window_size]
    # Y_train = Y[-2*window_size:-window_size]
    # model = ThreeComponentGP(X_train, Y_train, window_size=window_size)
    # model.train_model(num_epochs=100)
    # update_count = 0
    # Online updates: Suppose we get 10 new points beyond the original 100
    # X_stream = X[window_size: -window_size]
    # Y_stream = Y[window_size: -window_size]
    # for x_new, y_new in zip(X_stream, Y_stream):
    #     if update_count == 50:
    #         model.online_update(x_new.item(), y_new.item(),is_fine_tuned=True)
    #         update_count += 0
    #     else:
    #         model.online_update(x_new.item(), y_new.item())
    #         update_count += 1

    # Now let's do a final plot
    # Y_new = Y - true_error
    # model.plot_all_components(X, Y, true_values=Y, true_error=true_error, true_gp1=true_gp1, true_gp2=true_gp2)
    #
    #

    window_size = 50
    data_points = 400
    change_points = (170, 280)
    # 1) Generate piecewise-constant cointegration data
    time, X_coin, Y_coin, beta_true, mu_true = ThreeComponentGP.generate_cointegration_data(
        T=200,
        seed=42,
        noise_std=0.1,
        change_points=(70, 140),
        beta_values=(1.0, 2.0, 0.5),
        mu_values=(0.0, -1.0, 1.0),
        alpha=0.9
    )

    true_error = Y_coin - beta_true * X_coin - mu_true
    # 2) Initialize the composite GP with the (X_coin, Y_coin) data

    X_train = X_coin
    Y_train = Y_coin
    model = ThreeComponentGP(X_train, Y_train, window_size=window_size)
    model.train_model(num_epochs=100, learning_rate=0.01)

    update_count = 0
    # Online updates: Suppose we get 10 new points beyond the original 100
    X_stream = X_coin[window_size:]
    Y_stream = Y_coin[window_size:]
    for x_new, y_new in zip(X_stream, Y_stream):
        if update_count == 50:
            model.online_update(x_new.item(), y_new.item(), is_fine_tuned=True)
            update_count += 0
        else:
            model.online_update(x_new.item(), y_new.item())
            update_count += 1

    model.plot_all_components(X_coin, Y_coin, time_values=time, true_values=Y_coin, true_error=true_error,
                              true_gp1=beta_true, true_gp2=mu_true)
    plot_cointegration_data(time, X_coin, Y_coin, beta_true, mu_true)
