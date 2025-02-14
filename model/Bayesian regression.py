import torch
import matplotlib.pyplot as plt


def generate_synthetic_data(N=50, seed=123):
    torch.manual_seed(seed)
    x = torch.linspace(0, 5, N).unsqueeze(1)  # shape (N,1)
    slope_true = 2.0
    intercept_true = -1.0
    sigma_true = 0.5

    noise = torch.randn(N,1)*sigma_true
    y = intercept_true + slope_true*x + noise
    return x, y, slope_true, intercept_true, sigma_true


def normal_inv_gamma_batch(X, y, beta_0, Lambda_0, alpha_0, delta_0):
    """
    Batch (non-recursive) posterior update for Normal–Inverse-Gamma prior in linear regression.
    Returns posterior params: (beta_N, Lambda_N, alpha_N, delta_N)
    """
    N, d = X.shape
    alpha_N = alpha_0 + 0.5 * N
    Lambda_N = Lambda_0 + X.T @ X

    beta_N = torch.linalg.inv(Lambda_N) @ (Lambda_0 @ beta_0 + X.T @ y)

    term1 = (y.T @ y).item()
    term2 = (beta_0.T @ Lambda_0 @ beta_0).item()
    term3 = (beta_N.T @ Lambda_N @ beta_N).item()
    delta_N = delta_0 + 0.5 * (term1 + term2 - term3)

    return beta_N, Lambda_N, alpha_N, delta_N


def normal_inv_gamma_recursive(X, y, beta_0, Lambda_0, alpha_0, delta_0):
    """
    Recursive (sequential) posterior updates for each observation.
    Returns final posterior params + traces.
    """
    N, d = X.shape
    beta_curr = beta_0.clone()
    Lambda_curr = Lambda_0.clone()
    alpha_curr = alpha_0
    delta_curr = delta_0

    for i in range(N):
        x_i = X[i].unsqueeze(0)  # shape (1,d)
        y_i = y[i].unsqueeze(0)  # shape (1,1)

        alpha_curr = alpha_curr + 0.5
        Lambda_new = Lambda_curr + x_i.T @ x_i
        beta_new = torch.linalg.inv(Lambda_new) @ (Lambda_curr @ beta_curr + x_i.T @ y_i)

        termA = (beta_curr.T @ Lambda_curr @ beta_curr).item()
        termB = (beta_new.T @ Lambda_new @ beta_new).item()
        termY = (y_i.T @ y_i).item()
        delta_curr = delta_curr + 0.5 * (termY + termA - termB)

        # Update
        beta_curr = beta_new
        Lambda_curr = Lambda_new

    return beta_curr, Lambda_curr, alpha_curr, delta_curr


def posterior_mean_sigma2(beta_N, Lambda_N, alpha_N, delta_N):
    """
    Posterior mean of beta, and mean of sigma^2.
    """
    beta_mean = beta_N.squeeze()
    sigma2_mean = delta_N / (alpha_N - 1.0) if alpha_N > 1 else float('nan')
    return beta_mean, sigma2_mean


def posterior_predictive_interval(x_star,
                                  beta_N, Lambda_N, alpha_N, delta_N,
                                  alpha=0.05):
    """
    For a single test point x_star (1,d), compute the posterior predictive mean and
    credible interval for y*, given Normal–InverseGamma posterior.

    Posterior predictive variance =
      delta_N/(alpha_N - 1) * (1 + x_star * (Lambda_N^-1) * x_star^T) * (alpha_N / (alpha_N - 1)) (there are different forms).

    But typically, the posterior predictive for y* has a t-distribution with (2*alpha_N) dof.
    For simplicity, let's approximate with Normal using mean = x_star^T beta_N,
    var = (delta_N / alpha_N) * [1 + x_star (Lambda_N^-1) x_star^T].
    """
    d = x_star.shape[1]
    # Posterior mean of beta
    beta_post = beta_N
    # Posterior predictive mean
    y_mean = (x_star @ beta_post).item()

    # Posterior predictive variance (rough approximate form):
    # var(y*) = delta_N / alpha_N * [1 + x_star (Lambda_N^-1) x_star^T].
    inv_LambdaN = torch.linalg.inv(Lambda_N)
    mid = (x_star @ inv_LambdaN @ x_star.transpose(0, 1)).item()
    var_y = (delta_N / alpha_N) * (1 + mid)

    # Approx. Normal-based credible interval
    std_y = torch.sqrt(torch.tensor(var_y))
    # For alpha=0.05, z=1.96 if normal. For a t-dist w. dof=2 alpha_N,
    # we can do a quick approximation. Let's do z=1.96 for demonstration.
    z = 1.96
    lower = y_mean - z * std_y
    upper = y_mean + z * std_y
    return y_mean, lower, upper


# ============================= DEMO =============================
if __name__ == "__main__":
    # 1) Synthetic data
    X, y, beta_true, intercept, sigma_true = generate_synthetic_data(N=50)
    N, d = X.shape

    print(f"True beta={beta_true}, True sigma^2={(sigma_true ** 2):.3f}")

    # 2) Priors
    beta_0 = torch.zeros(d, 1)
    Lambda_0 = 0.01 * torch.eye(d)
    alpha_0 = 2.0
    delta_0 = 1.0

    # 3) Batch Posterior
    betaN_batch, LambdaN_batch, alphaN_batch, deltaN_batch = normal_inv_gamma_batch(
        X, y, beta_0, Lambda_0, alpha_0, delta_0
    )
    beta_batch_mean, sigma2_batch_mean = posterior_mean_sigma2(
        betaN_batch, LambdaN_batch, alphaN_batch, deltaN_batch
    )

    # 4) Recursive Posterior
    betaN_seq, LambdaN_seq, alphaN_seq, deltaN_seq = normal_inv_gamma_recursive(
        X, y, beta_0, Lambda_0, alpha_0, delta_0
    )
    beta_seq_mean, sigma2_seq_mean = posterior_mean_sigma2(
        betaN_seq, LambdaN_seq, alphaN_seq, deltaN_seq
    )

    print("\n--- FINAL POSTERIORS ---")
    print(f"Batch: beta={beta_batch_mean.item():.4f}, sigma^2={sigma2_batch_mean:.4f}")
    print(f"Recursive: beta={beta_seq_mean.item():.4f}, sigma^2={sigma2_seq_mean:.4f}")

    # 5) Compare "error term" credible intervals for each data point
    # We'll interpret "error" as (observed y_i - predicted mean), but the credible interval
    # for the error can be viewed as the posterior predictive interval of y_i minus the actual y_i.
    error_intervals_batch = []
    error_intervals_seq = []
    actual_errors = []

    for i in range(N):
        x_i = X[i].unsqueeze(0)  # shape (1,d)
        y_i = y[i].item()
        # Posterior predictive for the batch approach
        ymean_b, lb_b, ub_b = posterior_predictive_interval(x_i,
                                                            betaN_batch, LambdaN_batch, alphaN_batch, deltaN_batch)
        # The "error" is y_i - ymean_b.
        # The credible interval for the error is [ (y_i-lb_b), (y_i-ub_b) ],
        # or equivalently the posterior predictive interval around 0 if we shift by y_i.
        emean_b = y_i - ymean_b
        e_lower_b = y_i - ub_b
        e_upper_b = y_i - lb_b
        error_intervals_batch.append((emean_b, e_lower_b, e_upper_b))

        # Posterior predictive for the final recursive approach
        ymean_s, lb_s, ub_s = posterior_predictive_interval(x_i,
                                                            betaN_seq, LambdaN_seq, alphaN_seq, deltaN_seq)
        emean_s = y_i - ymean_s
        e_lower_s = y_i - ub_s
        e_upper_s = y_i - lb_s
        error_intervals_seq.append((emean_s, e_lower_s, e_upper_s))

        actual_errors.append(y_i - (beta_true * x_i).item())  # the "true" error if beta_true known

    # Let's plot the final error estimates for each point side-by-side
    error_intervals_batch = torch.tensor(error_intervals_batch)  # shape (N,3)
    error_intervals_seq = torch.tensor(error_intervals_seq)  # shape (N,3)
    mean_batch = error_intervals_batch[:, 0]
    lower_batch = error_intervals_batch[:, 1]
    upper_batch = error_intervals_batch[:, 2]

    mean_seq = error_intervals_seq[:, 0]
    lower_seq = error_intervals_seq[:, 1]
    upper_seq = error_intervals_seq[:, 2]

    idx = torch.arange(N)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.errorbar(idx.numpy(), mean_batch.numpy(),
                 yerr=[(mean_batch - lower_batch).numpy(), (upper_batch - mean_batch).numpy()],
                 fmt='o', color='blue', ecolor='lightblue', label='Batch Error CI', alpha=0.8)
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Batch: Error Mean & 95% CI per data point")
    plt.xlabel("Data Index")
    plt.ylabel("Error = y_i - E[y_i]")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.errorbar(idx.numpy(), mean_seq.numpy(),
                 yerr=[(mean_seq - lower_seq).numpy(), (upper_seq - mean_seq).numpy()],
                 fmt='o', color='red', ecolor='salmon', label='Recursive Error CI', alpha=0.8)
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Recursive: Error Mean & 95% CI per data point")
    plt.xlabel("Data Index")
    plt.ylabel("Error = y_i - E[y_i]")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # The two methods should yield nearly identical final credible intervals
    # for the error once all data is processed,
    # though minor numerical differences may appear.
