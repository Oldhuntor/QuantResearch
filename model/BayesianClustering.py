import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import matplotlib.pyplot as plt


class BayesianTimeSeriesClustering:
    """
    Implements Bayesian clustering for time series data using a mixture of Gaussians
    with Gibbs sampling for inference.
    """

    def __init__(self, n_clusters=2, window_size=10, n_iterations=100):
        self.n_clusters = n_clusters
        self.window_size = window_size
        self.n_iterations = n_iterations
        self.means = None
        self.covs = None
        self.weights = None

    def _create_windows(self, X):
        """Convert time series into overlapping windows."""
        n_samples = len(X)
        windows = np.array([X[i:i + self.window_size]
                            for i in range(n_samples - self.window_size + 1)])
        return windows

    def _initialize_parameters(self, X):
        """Initialize cluster parameters using random assignments."""
        n_samples = len(X)

        # Random cluster assignments
        assignments = np.random.randint(0, self.n_clusters, size=n_samples)

        # Initialize means, covariances, and weights
        self.means = np.zeros((self.n_clusters, self.window_size))
        self.covs = np.zeros((self.n_clusters, self.window_size, self.window_size))

        for k in range(self.n_clusters):
            if np.sum(assignments == k) > 0:
                cluster_data = X[assignments == k]
                self.means[k] = np.mean(cluster_data, axis=0)
                self.covs[k] = np.cov(cluster_data.T) + np.eye(self.window_size) * 1e-6

        self.weights = np.ones(self.n_clusters) / self.n_clusters

        return assignments

    def _compute_log_likelihood(self, X):
        """Compute log likelihood for each time series under each cluster."""
        n_samples = len(X)
        log_likes = np.zeros((n_samples, self.n_clusters))

        for k in range(self.n_clusters):
            try:
                mvn = multivariate_normal(self.means[k], self.covs[k])
                log_likes[:, k] = mvn.logpdf(X) + np.log(self.weights[k])
            except:
                log_likes[:, k] = -np.inf

        return log_likes

    def fit(self, X):
        """
        Fit the Bayesian clustering model using Gibbs sampling.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_timestamps)
            Time series data to cluster
        """
        # Create windows from time series
        windows = self._create_windows(X)
        n_samples = len(windows)

        # Initialize parameters
        assignments = self._initialize_parameters(windows)

        # Gibbs sampling
        for iteration in range(self.n_iterations):
            # Sample assignments
            log_likes = self._compute_log_likelihood(windows)
            probs = np.exp(log_likes - logsumexp(log_likes, axis=1)[:, np.newaxis])
            assignments = np.array([np.random.choice(self.n_clusters, p=p)
                                    for p in probs])

            # Update parameters
            for k in range(self.n_clusters):
                if np.sum(assignments == k) > 0:
                    cluster_data = windows[assignments == k]
                    self.means[k] = np.mean(cluster_data, axis=0)
                    self.covs[k] = np.cov(cluster_data.T) + np.eye(self.window_size) * 1e-6
                    self.weights[k] = np.mean(assignments == k)

        self.assignments_ = assignments
        return self

    def predict(self, X):
        """Predict cluster assignments for new time series data."""
        windows = self._create_windows(X)
        log_likes = self._compute_log_likelihood(windows)
        return np.argmax(log_likes, axis=1)


# Generate example time series data
def generate_example_timeseries(n_series=100, length=200, n_patterns=2):
    X = np.zeros((n_series, length))
    true_clusters = np.zeros(n_series)
    t = np.linspace(0, 20, length)

    samples_per_pattern = n_series // n_patterns

    for i in range(n_patterns):
        start_idx = i * samples_per_pattern
        end_idx = (i + 1) * samples_per_pattern

        if i == 0:
            # Pattern 1: Sine wave with noise
            freq = 1.0
            for j in range(start_idx, end_idx):
                X[j] = np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.1, length)
                true_clusters[j] = 0
        else:
            # Pattern 2: Square wave with noise
            freq = 0.5
            for j in range(start_idx, end_idx):
                X[j] = np.sign(np.sin(2 * np.pi * freq * t)) + np.random.normal(0, 0.1, length)
                true_clusters[j] = 1

    return X, true_clusters


# Example usage
n_series = 100
length = 200
X, true_clusters = generate_example_timeseries(n_series=n_series, length=length)

# Fit clustering model
model = BayesianTimeSeriesClustering(n_clusters=2, window_size=20, n_iterations=50)
model.fit(X)
cluster_assignments = model.predict(X)

# Visualize results
plt.figure(figsize=(15, 10))

# Plot example time series from each cluster
plt.subplot(2, 1, 1)
for i in range(min(3, n_series)):
    plt.plot(X[i] + i * 3, label=f'Cluster {cluster_assignments[i]}')
plt.title('Example Time Series with Cluster Assignments')
plt.legend()

# Plot cluster means
plt.subplot(2, 1, 2)
for k in range(model.n_clusters):
    plt.plot(model.means[k], label=f'Cluster {k} Mean')
plt.title('Cluster Means')
plt.legend()

plt.tight_layout()
plt.show()