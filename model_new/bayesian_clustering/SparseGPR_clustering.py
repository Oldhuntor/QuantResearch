import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import gpytorch
import torch


class SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        # Initialize variational distribution and variational strategy
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        # Define kernel: RBF kernel with constant mean
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def sparse_gpr_hyperparameter_clustering(data, n_clusters=3, num_inducing=10, verbose=True):
    """
    Cluster time series based on Sparse Gaussian Process Regression hyperparameters.

    Parameters:
    data : pd.DataFrame
        DataFrame where rows represent symbols and columns represent time points.
        The first column should contain stock symbols.
    n_clusters : int
        Number of clusters to group symbols into.
    num_inducing : int
        Number of inducing points for sparse GP.
    verbose : bool
        If True, prints progress information during execution.

    Returns:
    dict
        Dictionary where keys are cluster labels and values are lists of symbols.
    """
    # Extract symbols and time series data
    symbols = data.iloc[:, 0].values
    time_points = torch.tensor(np.arange(data.shape[1] - 1), dtype=torch.float32)  # Time indices
    time_series = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float32)  # Time series values

    # Initialize list to store hyperparameters
    hyperparameters = []

    if verbose:
        print("Starting Sparse Gaussian Process Regression for each stock...\n")

    for idx, ts in enumerate(time_series):
        ts = ts.unsqueeze(-1)  # Add a dimension for compatibility (n_samples, 1)

        # Select inducing points (uniformly spaced in time)
        inducing_points = time_points[::len(time_points) // num_inducing][:num_inducing].unsqueeze(-1)

        # Initialize Sparse GP model
        model = SparseGPModel(inducing_points)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # Define training components
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(time_points))

        # Training loop (simple with fixed iterations for now)
        training_iter = 100  # You can adjust this
        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(time_points.unsqueeze(-1))
            loss = -mll(output, ts).sum()
            loss.backward()
            optimizer.step()

        # Extract learned hyperparameters
        kernel_params = model.covar_module.base_kernel.lengthscale.item()
        outputscale = model.covar_module.outputscale.item()
        noise = likelihood.noise.item()

        hyperparameters.append([outputscale, kernel_params, noise])

        # Verbose progress update
        if verbose:
            print(f"Processed {idx + 1}/{len(time_series)}: Symbol = {symbols[idx]}, "
                  f"Output Scale = {outputscale:.3f}, Length Scale = {kernel_params:.3f}, noise = {noise:.3f}")

    if verbose:
        print("\nFinished Sparse Gaussian Process Regression.")
        print("Starting clustering on hyperparameters...\n")

    # Perform clustering on hyperparameters
    hyperparameters = np.array(hyperparameters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(hyperparameters)

    # Organize symbols into clusters
    clusters = {}
    for cluster_id in range(n_clusters):
        clusters[f'Cluster{cluster_id+1}'] = symbols[labels == cluster_id].tolist()

        # Verbose cluster summary
        if verbose:
            print(f"Cluster {cluster_id + 1}: {len(clusters[f'Cluster{cluster_id+1}'])} symbols")

    if verbose:
        print("\nClustering complete. Summary:")
        for cluster, symbols in clusters.items():
            print(f"{cluster}: {symbols}")

    return clusters


# Example usage:
# Assuming `df` is the DataFrame loaded from your data
# clusters = sparse_gpr_hyperparameter_clustering(df, n_clusters=3, num_inducing=10, verbose=True)
# print(clusters)


# Example usage:
# Assuming `df` is the DataFrame loaded from your data
# clusters = gpr_hyperparameter_clustering(df, n_clusters=3, verbose=True)
# print(clusters)


df = pd.read_csv('/Users/hxh/PycharmProjects/Quantitative_Research/model_new/bayesian_clustering/transposed_stock_data.csv')
clusters = sparse_gpr_hyperparameter_clustering(df, n_clusters=10, verbose=True)
print(clusters)