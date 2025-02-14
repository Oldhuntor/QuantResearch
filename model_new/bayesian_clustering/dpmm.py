import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def generate_sample_time_series(n_series=100, length=50):
    """Generate synthetic time series data with different patterns"""
    patterns = {
        'increasing': lambda t: 0.05 * t + np.sin(t / 10),
        'decreasing': lambda t: -0.05 * t + np.sin(t / 10),
        'cyclical': lambda t: 2 * np.sin(t / 5),
    }

    t = np.arange(length)
    data = []
    true_labels = []

    for i in range(n_series):
        pattern_type = np.random.choice(list(patterns.keys()))
        series = patterns[pattern_type](t)
        series += np.random.normal(0, 0.2, length)
        data.append(series)
        true_labels.append(pattern_type)

    return np.array(data), true_labels


class TimeSeriesDPMM:
    def __init__(self, alpha=1.0, min_covar=1e-6):
        self.alpha = alpha
        self.min_covar = min_covar  # Minimum covariance value for stability
        self.clusters = []
        self.assignments = []

    def _compute_covariance(self, data):
        """Compute stable covariance matrix with regularization"""
        n_samples = len(data)
        if n_samples < 2:
            return np.eye(data.shape[1]) * self.min_covar

        # Add regularization to ensure positive definiteness
        cov = np.cov(data.T) + np.eye(data.shape[1]) * self.min_covar
        # Ensure symmetry
        cov = (cov + cov.T) / 2
        return cov

    def _safe_log_sum(self, log_probs):
        """Numerically stable log-sum-exp"""
        max_log_prob = np.max(log_probs)
        if np.isinf(max_log_prob):
            return np.zeros_like(log_probs)
        exp_probs = np.exp(log_probs - max_log_prob)
        return exp_probs / (exp_probs.sum() + 1e-10)

    def fit(self, X, n_iterations=50):
        n_series = len(X)
        self.assignments = np.arange(n_series)

        for iteration in range(n_iterations):
            for i in range(n_series):
                current_assignment = self.assignments[i]
                log_probs = []

                for j in range(max(self.assignments) + 2):
                    # Calculate cluster membership excluding current point
                    mask = self.assignments == j
                    mask[i] = False  # Exclude current point
                    cluster_members = X[mask]
                    nj = len(cluster_members)

                    # CRP probability
                    crp_prob = nj / (n_series - 1 + self.alpha) if j <= max(self.assignments) else \
                        self.alpha / (n_series - 1 + self.alpha)

                    # Likelihood calculation
                    try:
                        if nj > 0:
                            mean = np.mean(cluster_members, axis=0)
                            cov = self._compute_covariance(cluster_members)
                        else:
                            # Prior for new cluster
                            mean = np.zeros(X.shape[1])
                            cov = np.eye(X.shape[1])

                        likelihood = multivariate_normal.logpdf(X[i], mean=mean, cov=cov)
                        log_prob = np.log(max(crp_prob, 1e-300)) + likelihood
                    except (ValueError, np.linalg.LinAlgError):
                        log_prob = -np.inf

                    log_probs.append(log_prob)

                # Safe probability normalization
                probs = self._safe_log_sum(np.array(log_probs))
                if np.any(np.isnan(probs)):
                    continue  # Skip this update if we get NaN probabilities

                # Sample new assignment
                try:
                    self.assignments[i] = np.random.choice(len(probs), p=probs)
                except ValueError:
                    # If probabilities don't sum to 1 within numerical precision
                    self.assignments[i] = current_assignment

        return self


# Generate and prepare data
n_series = 100
length = 50
X, true_labels = generate_sample_time_series(n_series, length)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit the model
dpmm = TimeSeriesDPMM(alpha=1.0)
dpmm.fit(X_scaled)

# Visualization
plt.figure(figsize=(15, 10))

# Plot original patterns
plt.subplot(2, 1, 1)
for i, label in enumerate(np.unique(true_labels)):
    mask = np.array(true_labels) == label
    for series in X[mask]:
        plt.plot(series, alpha=0.3, label=label if label not in plt.gca().get_legend_handles_labels()[1] else "")
plt.title('Original Time Series Patterns')
plt.legend()
plt.grid(True, alpha=0.15)

# Plot DPMM clusters
plt.subplot(2, 1, 2)
unique_clusters = np.unique(dpmm.assignments)
for cluster in unique_clusters:
    mask = dpmm.assignments == cluster
    for series in X[mask]:
        plt.plot(series, alpha=0.3, label=f'Cluster {cluster}'
        if f'Cluster {cluster}' not in plt.gca().get_legend_handles_labels()[1] else "")
plt.title(f'DPMM Clustered Patterns (Found {len(unique_clusters)} clusters)')
plt.legend()
plt.grid(True, alpha=0.15)

plt.tight_layout()
plt.show()

# Print clustering statistics
print("\nClustering Statistics:")
for cluster in unique_clusters:
    mask = dpmm.assignments == cluster
    print(f"Cluster {cluster}: {np.sum(mask)} time series")