import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import gpytorch
import torch
import multiprocessing as mp
from functools import partial
from tqdm import tqdm


class SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def process_single_series(args):
    """
    Process a single time series with Sparse GPR.
    This function will run in a separate process.
    """
    time_series, time_points, num_inducing, symbol = args

    # Convert to tensor and reshape
    ts = torch.tensor(time_series, dtype=torch.float32).unsqueeze(-1)
    time_points_tensor = torch.tensor(time_points, dtype=torch.float32)

    # Select inducing points
    inducing_points = time_points_tensor[::len(time_points) // num_inducing][:num_inducing].unsqueeze(-1)

    # Initialize model
    model = SparseGPModel(inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # Training setup
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(time_points))

    # Training loop
    training_iter = 100
    for _ in range(training_iter):
        optimizer.zero_grad()
        output = model(time_points_tensor.unsqueeze(-1))
        loss = -mll(output, ts).sum()
        loss.backward()
        optimizer.step()

    # Extract hyperparameters
    kernel_params = model.covar_module.base_kernel.lengthscale.item()
    outputscale = model.covar_module.outputscale.item()
    noise = likelihood.noise.item()

    return symbol, outputscale, kernel_params, noise


def sparse_gpr_hyperparameter_clustering(data, n_clusters=3, num_inducing=10, n_processes=None, verbose=True):
    """
    Parallel implementation of Sparse GPR clustering.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame where rows represent symbols and columns represent time points.
        The first column should contain stock symbols.
    n_clusters : int
        Number of clusters to group symbols into.
    num_inducing : int
        Number of inducing points for sparse GP.
    n_processes : int, optional
        Number of processes to use. If None, uses cpu_count().
    verbose : bool
        If True, shows progress bar during execution.

    Returns:
    --------
    dict
        Dictionary where keys are cluster labels and values are lists of symbols.
    """
    # Extract data
    symbols = data.iloc[:, 0].values
    time_points = np.arange(data.shape[1] - 1)
    time_series = data.iloc[:, 1:].values

    # Setup multiprocessing
    if n_processes is None:
        n_processes = mp.cpu_count()

    # Prepare arguments for parallel processing
    process_args = [
        (ts, time_points, num_inducing, symbol)
        for ts, symbol in zip(time_series, symbols)
    ]

    if verbose:
        print(f"Starting parallel Sparse GPR with {n_processes} processes...")

    # Run parallel processing with progress bar
    with mp.Pool(processes=n_processes) as pool:
        if verbose:
            results = list(tqdm(
                pool.imap(process_single_series, process_args),
                total=len(process_args),
                desc="Processing time series"
            ))
        else:
            results = pool.map(process_single_series, process_args)

    print(results)
    # Organize results
    symbols = [r[0] for r in results]
    hyperparameters = np.array([[r[1], r[2]] for r in results])

    if verbose:
        print("\nStarting clustering on hyperparameters...")

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(hyperparameters)

    # Organize symbols into clusters
    clusters = {}
    for cluster_id in range(n_clusters):
        cluster_symbols = np.array(symbols)[labels == cluster_id].tolist()
        clusters[f'Cluster{cluster_id + 1}'] = cluster_symbols

        if verbose:
            print(f"Cluster {cluster_id + 1}: {len(cluster_symbols)} symbols")

    if verbose:
        print("\nClustering complete. Summary:")
        for cluster, cluster_symbols in clusters.items():
            print(f"{cluster}: {cluster_symbols}")

    return clusters


# Example usage
if __name__ == "__main__":
    # Set the number of processes to use (optional)
    n_processes = mp.cpu_count() - 2  # Leave one CPU free

    sample_data = pd.read_csv("/Users/hxh/PycharmProjects/Quantitative_Research/model_new/bayesian_clustering/transposed_stock_data.csv")
    # Run clustering
    clusters = sparse_gpr_hyperparameter_clustering(
        sample_data,
        n_clusters=30,
        num_inducing=10,
        n_processes=n_processes,
        verbose=True
    )