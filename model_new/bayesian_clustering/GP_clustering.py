import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.cluster import KMeans

def gp_time_series_clustering(X, n_clusters=3):
    """
    Perform Gaussian Process-based clustering on time series data.

    Parameters:
    X : numpy array
        Input time series data of shape (n_samples, n_features).
    n_clusters : int
        Number of clusters for k-means clustering.

    Returns:
    labels : numpy array
        Cluster labels assigned to each time series.
    """
    # Fit Gaussian Processes and extract hyperparameters
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    hyperparams = []

    for i in range(X.shape[0]):
        # Fit GP to each time series
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        t = np.arange(X.shape[1]).reshape(-1, 1)  # Time steps
        gp.fit(t, X[i, :])

        # Extract learned hyperparameters
        params = gp.kernel_.theta  # Log-transformed hyperparameters
        hyperparams.append(params)

    hyperparams = np.array(hyperparams)

    # Cluster based on GP hyperparameters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(hyperparams)
    labels = kmeans.labels_

    # Visualize clusters
    plt.figure(figsize=(10, 6))
    for i in range(n_clusters):
        plt.plot(X[labels == i].T, alpha=0.3)
    plt.title("Gaussian Process Clustering")
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.show()

    # Print cluster assignments
    print(f"Cluster assignments: {labels}")
    return labels
