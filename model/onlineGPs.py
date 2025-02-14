import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, Dict, Tuple
from scipy.stats import norm


class OnlineGP:
    def __init__(self, kernel, noise_variance=1e-4, window_size=50):
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.window_size = window_size
        self.X = None
        self.y = None
        self.K_inv = None

    def compute_kernel(self, x1, x2=None):
        with torch.no_grad():
            if x2 is None:
                K = self.kernel(x1).evaluate()
            else:
                K = self.kernel(x1, x2).evaluate()
            return K.squeeze(-1)

    def initialize(self, X, y):
        self.X = X.clone()
        self.y = y.clone().reshape(-1)

        K = self.compute_kernel(X)
        K.diagonal().add_(self.noise_variance)
        self.K_inv = torch.linalg.inv(K)

    def predict(self, X_test):
        if X_test.dim() == 1:
            X_test = X_test.unsqueeze(0)

        k_star = self.compute_kernel(X_test, self.X)
        if k_star.dim() == 1:
            k_star = k_star.unsqueeze(0)

        mean = (k_star @ (self.K_inv @ self.y)).squeeze()

        k_star_star = self.compute_kernel(X_test)
        if k_star_star.dim() == 0:
            variance = k_star_star - (k_star @ self.K_inv @ k_star.t()).squeeze()
        else:
            variance = k_star_star.diag() - (k_star @ self.K_inv @ k_star.t()).diag()

        return mean, variance

    def update(self, x_new, y_new, compute_posterior=True):
        if x_new.dim() == 1:
            x_new = x_new.unsqueeze(0)

        # Compute necessary kernel values for new point
        k_new = self.compute_kernel(self.X, x_new)
        if k_new.dim() == 1:
            k_new = k_new.unsqueeze(1)

        k_new_new = self.compute_kernel(x_new, x_new).item()
        k_new_new += self.noise_variance

        # Woodbury matrix identity update to add new point
        v = k_new.t()  # Shape: (1, n)

        # Solve for the update matrix
        extended_K_inv = torch.zeros(len(self.X) + 1, len(self.X) + 1, device=self.K_inv.device)
        extended_K_inv[:len(self.X), :len(self.X)] = self.K_inv

        # Compute Schur complement for the new point
        schur_complement = k_new_new - v @ self.K_inv @ v.t()

        # Compute block update terms
        update_term = -self.K_inv @ v.t() / schur_complement

        # Update the inverse matrix
        extended_K_inv[:len(self.X), :len(self.X)] += update_term @ v
        extended_K_inv[-1, :len(self.X)] = update_term.squeeze()
        extended_K_inv[:len(self.X), -1] = update_term.squeeze()
        extended_K_inv[-1, -1] = 1 / schur_complement

        # Add new point
        self.X = torch.cat([self.X, x_new], dim=0)
        self.y = torch.cat([self.y, y_new.reshape(-1)])
        self.K_inv = extended_K_inv

        # Remove the oldest point if at window limit
        if len(self.X) > self.window_size:
            # Remove the first point (oldest)
            self.X = self.X[1:]
            self.y = self.y[1:]

            # Update K_inv using Schur complement
            self.K_inv = self.K_inv[1:, 1:]

        if compute_posterior:
            return self.predict(self.X)
        return None, None


class ThreeComponentOnlineGP:
    def __init__(self, gp1, gp2, gp_error, noise_variance=1e-4, window_size=50):
        self.gp1 = OnlineGP(gp1, noise_variance, window_size)
        self.gp2 = OnlineGP(gp2, noise_variance, window_size)
        self.gp_error = OnlineGP(gp_error, noise_variance, window_size)

        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None

    def initialize(self, X, Y):
        """Initialize all three GPs with training data"""
        self.X_mean, self.X_std = X.mean(), X.std()
        self.Y_mean, self.Y_std = Y.mean(), Y.std()

        X_norm = (X - self.X_mean) / self.X_std
        Y_norm = (Y - self.Y_mean) / self.Y_std

        train_inputs = torch.stack([X_norm, Y_norm], dim=1)

        self.gp1.initialize(train_inputs, Y_norm)
        self.gp2.initialize(train_inputs, Y_norm)

        mean1, _ = self.gp1.predict(train_inputs)
        mean2, _ = self.gp2.predict(train_inputs)
        residuals = Y_norm - (X_norm * mean1 + mean2)
        self.gp_error.initialize(train_inputs, residuals)

    def predict(self, X_new, Y_new):
        """Make predictions with all components"""
        X_norm = (X_new - self.X_mean) / self.X_std
        Y_norm = (Y_new - self.Y_mean) / self.Y_std

        if X_norm.dim() == 0:
            X_norm = X_norm.unsqueeze(0)
        if Y_norm.dim() == 0:
            Y_norm = Y_norm.unsqueeze(0)

        test_inputs = torch.stack([X_norm, Y_norm], dim=1)

        mean1, var1 = self.gp1.predict(test_inputs)
        mean2, var2 = self.gp2.predict(test_inputs)
        mean_error, var_error = self.gp_error.predict(test_inputs)

        mean1 = mean1 * self.Y_std
        mean2 = mean2 * self.Y_std
        mean_error = mean_error * self.Y_std

        var1 = var1 * self.Y_std ** 2
        var2 = var2 * self.Y_std ** 2
        var_error = var_error * self.Y_std ** 2

        final_mean = X_new * mean1 + mean2 + mean_error
        final_var = (X_new ** 2 * var1 + var2 + var_error)

        return {
            'gp1': {'mean': mean1, 'variance': var1},
            'gp2': {'mean': mean2, 'variance': var2},
            'error': {'mean': mean_error, 'variance': var_error},
            'final': {'mean': final_mean, 'variance': final_var}
        }

    def update(self, x_new, y_new):
        """Update all components with new observation"""
        x_norm = (x_new - self.X_mean) / self.X_std
        y_norm = (y_new - self.Y_mean) / self.Y_std

        if x_norm.dim() == 0:
            x_norm = x_norm.unsqueeze(0)
        if y_norm.dim() == 0:
            y_norm = y_norm.unsqueeze(0)

        new_input = torch.stack([x_norm, y_norm], dim=1)

        mean1, _ = self.gp1.predict(new_input)
        mean2, _ = self.gp2.predict(new_input)

        self.gp1.update(new_input, y_norm)
        self.gp2.update(new_input, y_norm)

        residual = y_norm - (x_norm * mean1 + mean2)
        self.gp_error.update(new_input, residual)

        return self.predict(x_new, y_new)

class MultiplicativeGP(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_inputs, train_targets, likelihood)

        # Mean and covariance for 2D input (X, Y)
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


class ThreeComponentGPRetrain:
    def __init__(self, window_size=50, n_epochs=50):
        self.window_size = window_size
        self.n_epochs = n_epochs
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        self.current_X = None
        self.current_y = None

    def initialize(self, X, Y):
        """Initialize with first window of data"""
        self.X_mean, self.X_std = X.mean(), X.std()
        self.Y_mean, self.Y_std = Y.mean(), Y.std()

        X_norm = (X - self.X_mean) / self.X_std
        Y_norm = (Y - self.Y_mean) / self.Y_std

        self.current_X = X[:self.window_size]
        self.current_y = Y[:self.window_size]

        train_inputs = torch.stack([X_norm[:self.window_size], Y_norm[:self.window_size]], dim=1)

        # Initialize new GP models
        self._init_models(train_inputs, Y_norm[:self.window_size])
        self._train_models()

    def _init_models(self, train_inputs, train_targets):
        """Initialize fresh GP models"""
        base_kernel1 = gpytorch.kernels.RBFKernel(ard_num_dims=2)
        base_kernel2 = gpytorch.kernels.RBFKernel(ard_num_dims=2)
        base_kernel_error = gpytorch.kernels.RBFKernel(ard_num_dims=2)

        self.kernel1 = gpytorch.kernels.ScaleKernel(base_kernel1)
        self.kernel2 = gpytorch.kernels.ScaleKernel(base_kernel2)
        self.kernel_error = gpytorch.kernels.ScaleKernel(base_kernel_error)

        self.gp1 = MultiplicativeGP(train_inputs, train_targets)
        self.gp2 = AdditiveGP(train_inputs, train_targets)

        # Compute initial residuals for error GP
        self.gp1.eval()
        self.gp2.eval()
        with torch.no_grad():
            X_norm = train_inputs[:, 0]
            mean1 = self.gp1(train_inputs).mean
            mean2 = self.gp2(train_inputs).mean
            residuals = train_targets - (X_norm * mean1 + mean2)

        self.gp_error = ErrorGP(train_inputs, residuals)

    def _train_models(self):
        """Train all three GPs"""
        self.gp1.train()
        self.gp2.train()
        self.gp_error.train()

        optimizer1 = torch.optim.Adam(self.gp1.parameters(), lr=0.1)
        optimizer2 = torch.optim.Adam(self.gp2.parameters(), lr=0.1)
        optimizer_error = torch.optim.Adam(self.gp_error.parameters(), lr=0.1)

        mll1 = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp1.likelihood, self.gp1)
        mll2 = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp2.likelihood, self.gp2)
        mll_error = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_error.likelihood, self.gp_error)

        for i in range(self.n_epochs):
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer_error.zero_grad()

            output1 = self.gp1(self.gp1.train_inputs[0])
            output2 = self.gp2(self.gp2.train_inputs[0])

            loss1 = -mll1(output1, self.gp1.train_targets)
            loss2 = -mll2(output2, self.gp2.train_targets)
            loss_error = -mll_error(self.gp_error(self.gp_error.train_inputs[0]),
                                    self.gp_error.train_targets)

            total_loss = loss1 + loss2 + loss_error
            total_loss.backward()

            optimizer1.step()
            optimizer2.step()
            optimizer_error.step()

    def predict(self, X_new, Y_new):
        """Make predictions"""
        X_norm = (X_new - self.X_mean) / self.X_std
        Y_norm = (Y_new - self.Y_mean) / self.Y_std

        if X_norm.dim() == 0:
            X_norm = X_norm.unsqueeze(0)
        if Y_norm.dim() == 0:
            Y_norm = Y_norm.unsqueeze(0)

        test_inputs = torch.stack([X_norm, Y_norm], dim=1)

        self.gp1.eval()
        self.gp2.eval()
        self.gp_error.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output1 = self.gp1(test_inputs)
            output2 = self.gp2(test_inputs)
            output_error = self.gp_error(test_inputs)

            mean1 = output1.mean * self.Y_std
            mean2 = output2.mean * self.Y_std
            mean_error = output_error.mean * self.Y_std

            var1 = output1.variance * self.Y_std ** 2
            var2 = output2.variance * self.Y_std ** 2
            var_error = output_error.variance * self.Y_std ** 2

            final_mean = X_new * mean1 + mean2 + mean_error
            final_var = (X_new ** 2 * var1 + var2 + var_error)

        return {
            'gp1': {'mean': mean1, 'variance': var1},
            'gp2': {'mean': mean2, 'variance': var2},
            'error': {'mean': mean_error, 'variance': var_error},
            'final': {'mean': final_mean, 'variance': final_var}
        }

    def update(self, x_new, y_new):
        """Update by retraining on sliding window"""
        # Add new point to window
        self.current_X = torch.cat([self.current_X[1:], x_new.unsqueeze(0)])
        self.current_y = torch.cat([self.current_y[1:], y_new.unsqueeze(0)])

        # Normalize data
        X_norm = (self.current_X - self.X_mean) / self.X_std
        Y_norm = (self.current_y - self.Y_mean) / self.Y_std

        # Create training inputs
        train_inputs = torch.stack([X_norm, Y_norm], dim=1)

        # Reinitialize and retrain models
        self._init_models(train_inputs, Y_norm)
        self._train_models()

        return self.predict(x_new, y_new)


def generate_sequence(n_points=100):
    """Generate synthetic sequence with known components"""
    X = torch.linspace(0, 10, n_points)

    true_gp1 = torch.sin(0.5 * np.pi * X)
    true_gp2 = 0.5 * torch.cos(np.pi * X)
    true_error = torch.randn_like(X) * 0.1

    Y = X * true_gp1 + true_gp2 + true_error

    return X, Y, {'gp1': true_gp1, 'gp2': true_gp2, 'error': true_error}


def one_step_ahead_prediction(model, X, Y, train_window=50):
    """Make one-step-ahead predictions"""
    predictions = []
    true_values = []

    # Initialize with first train_window points
    print("Initializing model...")
    model.initialize(X[:train_window], Y[:train_window])

    # Make predictions for remaining points
    print(f"Making predictions for {len(X) - train_window - 1} points...")
    for i in range(train_window, len(X) - 1):
        x_next = X[i + 1]
        y_next = Y[i + 1]

        # Get prediction
        try:
            preds = model.predict(x_next, y_next)
            # Ensure prediction is in the correct format
            if not isinstance(preds, dict):
                # If prediction is just a value, format it properly
                preds = {
                    'final': {
                        'mean': preds if torch.is_tensor(preds) else torch.tensor(preds),
                        'variance': torch.tensor(0.0)  # Default variance if not available
                    },
                    'gp1': {'mean': torch.tensor(0.0), 'variance': torch.tensor(0.0)},
                    'gp2': {'mean': torch.tensor(0.0), 'variance': torch.tensor(0.0)},
                    'error': {'mean': torch.tensor(0.0), 'variance': torch.tensor(0.0)}
                }
        except Exception as e:
            print(f"Error during prediction at step {i}: {e}")
            print(f"x_next shape: {x_next.shape}, value: {x_next}")
            print(f"y_next shape: {y_next.shape}, value: {y_next}")
            raise

        predictions.append(preds)

        # Update model
        try:
            model.update(x_next, y_next)
        except Exception as e:
            print(f"Error during update at step {i}: {e}")
            raise

        true_values.append(y_next.item())

        if (i - train_window + 1) % 50 == 0:
            print(f"Processed {i - train_window + 1} points...")

    return predictions, true_values


def plot_individual_comparisons(X, Y, full_model_preds, retrain_preds, woodbury_preds, window_size, true_components):
    """Plot individual comparisons for each method with true values"""
    valid_idx = slice(window_size + 1, None)
    x_valid = X[valid_idx].numpy()

    fig, axs = plt.subplots(3, 2, figsize=(30, 18))

    # Full Dataset Method
    axs[0, 0].plot(X.numpy(), Y.numpy(), 'k-', label='True Y', alpha=0.5)
    axs[0, 0].plot(X.numpy(), full_model_preds['final']['mean'].numpy(), 'b-', label='Full Dataset')
    mean = full_model_preds['final']['mean'].numpy()
    std = torch.sqrt(full_model_preds['final']['variance']).numpy()
    axs[0, 0].fill_between(X.numpy(), mean - 2 * std, mean + 2 * std, color='b', alpha=0.2)
    axs[0, 0].set_title('Full Dataset: Y Prediction')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Error component comparison
    axs[0, 1].plot(X.numpy(), true_components['error'].numpy(), 'k-', label='True Error', alpha=0.5)
    axs[0, 1].plot(X.numpy(), full_model_preds['error']['mean'].numpy(), 'b-', label='Predicted Error')
    mean = full_model_preds['error']['mean'].numpy()
    std = torch.sqrt(full_model_preds['error']['variance']).numpy()
    axs[0, 1].fill_between(X.numpy(), mean - 2 * std, mean + 2 * std, color='b', alpha=0.2)
    axs[0, 1].set_title('Full Dataset: Error Component')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Rolling Retrain Method
    retrain_mean = np.array([p['final']['mean'].item() if torch.is_tensor(p['final']['mean'])
                             else p['final']['mean'] for p in retrain_preds])
    retrain_std = np.array([np.sqrt(p['final']['variance'].item() if torch.is_tensor(p['final']['variance'])
                                    else p['final']['variance']) for p in retrain_preds])
    retrain_error = np.array([p['error']['mean'].item() if torch.is_tensor(p['error']['mean'])
                              else p['error']['mean'] for p in retrain_preds])
    retrain_error_std = np.array([np.sqrt(p['error']['variance'].item() if torch.is_tensor(p['error']['variance'])
                                          else p['error']['variance']) for p in retrain_preds])

    axs[1, 0].plot(X.numpy(), Y.numpy(), 'k-', label='True Y', alpha=0.5)
    axs[1, 0].plot(x_valid, retrain_mean, 'r-', label='Rolling Retrain')
    axs[1, 0].fill_between(x_valid, retrain_mean - 2 * retrain_std, retrain_mean + 2 * retrain_std, color='r',
                           alpha=0.2)
    axs[1, 0].set_title('Rolling Retrain: Y Prediction')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(X.numpy(), true_components['error'].numpy(), 'k-', label='True Error', alpha=0.5)
    axs[1, 1].plot(x_valid, retrain_error, 'r-', label='Predicted Error')
    axs[1, 1].fill_between(x_valid, retrain_error - 2 * retrain_error_std, retrain_error + 2 * retrain_error_std,
                           color='r', alpha=0.2)
    axs[1, 1].set_title('Rolling Retrain: Error Component')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Woodbury Method
    woodbury_mean = np.array([p['final']['mean'].item() if torch.is_tensor(p['final']['mean'])
                              else p['final']['mean'] for p in woodbury_preds])
    woodbury_std = np.array([np.sqrt(p['final']['variance'].item() if torch.is_tensor(p['final']['variance'])
                                     else p['final']['variance']) for p in woodbury_preds])
    woodbury_error = np.array([p['error']['mean'].item() if torch.is_tensor(p['error']['mean'])
                               else p['error']['mean'] for p in woodbury_preds])
    woodbury_error_std = np.array([np.sqrt(p['error']['variance'].item() if torch.is_tensor(p['error']['variance'])
                                           else p['error']['variance']) for p in woodbury_preds])

    axs[2, 0].plot(X.numpy(), Y.numpy(), 'k-', label='True Y', alpha=0.5)
    axs[2, 0].plot(x_valid, woodbury_mean, 'g-', label='Rolling Woodbury')
    axs[2, 0].fill_between(x_valid, woodbury_mean - 2 * woodbury_std, woodbury_mean + 2 * woodbury_std, color='g',
                           alpha=0.2)
    axs[2, 0].set_title('Rolling Woodbury: Y Prediction')
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    axs[2, 1].plot(X.numpy(), true_components['error'].numpy(), 'k-', label='True Error', alpha=0.5)
    axs[2, 1].plot(x_valid, woodbury_error, 'g-', label='Predicted Error')
    axs[2, 1].fill_between(x_valid, woodbury_error - 2 * woodbury_error_std, woodbury_error + 2 * woodbury_error_std,
                           color='g', alpha=0.2)
    axs[2, 1].set_title('Rolling Woodbury: Error Component')
    axs[2, 1].legend()
    axs[2, 1].grid(True)

    plt.tight_layout()
    plt.show()

def calculate_and_print_mses(X, Y, full_model_preds, retrain_preds, woodbury_preds, window_size, true_components):
    valid_idx = slice(window_size+1, None)
    x_valid = X[valid_idx].numpy()
    y_valid = Y[valid_idx].numpy()
    true_error_valid = true_components['error'][valid_idx].numpy()

    # Full dataset MSEs
    full_y_mse = np.mean((full_model_preds['final']['mean'][valid_idx].numpy() - y_valid) ** 2)
    full_error_mse = np.mean((full_model_preds['error']['mean'][valid_idx].numpy() - true_error_valid) ** 2)

    # Rolling retrain MSEs
    retrain_mean = np.array([p['final']['mean'].item() if torch.is_tensor(p['final']['mean'])
                            else p['final']['mean'] for p in retrain_preds])
    retrain_error = np.array([p['error']['mean'].item() if torch.is_tensor(p['error']['mean'])
                             else p['error']['mean'] for p in retrain_preds])
    retrain_y_mse = np.mean((retrain_mean - y_valid) ** 2)
    retrain_error_mse = np.mean((retrain_error - true_error_valid) ** 2)

    # Woodbury MSEs
    woodbury_mean = np.array([p['final']['mean'].item() if torch.is_tensor(p['final']['mean'])
                             else p['final']['mean'] for p in woodbury_preds])
    woodbury_error = np.array([p['error']['mean'].item() if torch.is_tensor(p['error']['mean'])
                              else p['error']['mean'] for p in woodbury_preds])
    woodbury_y_mse = np.mean((woodbury_mean - y_valid) ** 2)
    woodbury_error_mse = np.mean((woodbury_error - true_error_valid) ** 2)

    print("\nMean Squared Errors:")
    print("-" * 50)
    print("Full Dataset:")
    print(f"  Y prediction MSE: {full_y_mse:.6f}")
    print(f"  Error component MSE: {full_error_mse:.6f}")
    print("\nRolling Retrain:")
    print(f"  Y prediction MSE: {retrain_y_mse:.6f}")
    print(f"  Error component MSE: {retrain_error_mse:.6f}")
    print("\nRolling Woodbury:")
    print(f"  Y prediction MSE: {woodbury_y_mse:.6f}")
    print(f"  Error component MSE: {woodbury_error_mse:.6f}")


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(0)
    np.random.seed(0)

    # Generate data
    def generate_sequence(n_points=100):
        X = torch.linspace(0, 10, n_points)
        true_gp1 = torch.sin(0.5 * np.pi * X)
        true_gp2 = 0.5 * torch.cos(np.pi * X)
        true_error = torch.randn_like(X) * 0.1
        Y = X * true_gp1 + true_gp2 + true_error
        return X, Y, {'gp1': true_gp1, 'gp2': true_gp2, 'error': true_error}

    def generate_cointegration_data(
            T=200,
            noise_std=0.1,
            change_points=(70, 140),
            beta_values=(1.0, 2.0, 0.5),
            mu_values=(0.0, -1.0, 1.0),
            alpha=0.9
    ):
        time = torch.arange(T, dtype=torch.float64)

        # Generate Y_t as an AR(1)
        X = torch.zeros(T)
        for t in range(1, T):
            X[t] = alpha * X[t - 1] + torch.randn(1) * 0.5

        beta_t = torch.zeros(T)
        mu_t = torch.zeros(T)

        segments = [0] + list(change_points) + [T]
        for i in range(len(segments) - 1):
            start_idx = segments[i]
            end_idx = segments[i + 1]
            beta_t[start_idx:end_idx] = beta_values[i]
            mu_t[start_idx:end_idx] = mu_values[i]

        noise = torch.randn(T) * noise_std
        Y = beta_t * X + mu_t + noise
        return time, X, Y, {'gp1': beta_t, 'gp2': mu_t, 'error': noise}


    # Generate data
    print("Generating data...")
    # time, X, Y,  true_components = generate_cointegration_data(
    #     T=300,  # longer series
    #     change_points=(100, 200),  # different change points
    #     beta_values=(0.5, 1.5, 1.0),  # different betas
    #     noise_std=0.2,  # more noise
    #     alpha=1
    # )
    window_size = 50

    X, Y, true_components = generate_sequence(300)

    # Initialize and train models
    print("\nTraining models...")

    # Full dataset model
    base_kernel1 = gpytorch.kernels.RBFKernel(ard_num_dims=2)
    base_kernel2 = gpytorch.kernels.RBFKernel(ard_num_dims=2)
    base_kernel_error = gpytorch.kernels.RBFKernel(ard_num_dims=2)
    kernel1 = gpytorch.kernels.ScaleKernel(base_kernel1)
    kernel2 = gpytorch.kernels.ScaleKernel(base_kernel2)
    kernel_error = gpytorch.kernels.ScaleKernel(base_kernel_error)

    # Get predictions from each model
    # 1. Full dataset
    print("Getting full dataset predictions...")
    model_full = ThreeComponentGPRetrain(window_size=len(X))  # Use full length
    model_full.initialize(X, Y)
    full_predictions = model_full.predict(X, Y)

    # 2. Rolling retrain
    print("Getting rolling retrain predictions...")
    model_retrain = ThreeComponentGPRetrain(window_size=window_size)
    retrain_preds, _ = one_step_ahead_prediction(model_retrain, X, Y, train_window=window_size)

    # 3. Rolling Woodbury
    print("Getting Woodbury predictions...")
    model_woodbury = ThreeComponentOnlineGP(kernel1, kernel2, kernel_error, window_size=window_size)
    woodbury_preds, _ = one_step_ahead_prediction(model_woodbury, X, Y, train_window=window_size)

    # Plot comparisons
    print("\nPlotting results...")

    plot_individual_comparisons(X, Y, full_predictions, retrain_preds, woodbury_preds,
                                    window_size, true_components)

    calculate_and_print_mses(X, Y, full_predictions, retrain_preds, woodbury_preds, window_size, true_components)
