"""
Essential dimensionality metrics for testing 2D manifold hypothesis.
Three key metrics: TwoNN, Correlation Dimension, Local PCA 2D Energy
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import pdist


def twonn_dimension(X):
    """
    TwoNN intrinsic dimension (Facco et al. 2017).
    Uses empirical CDF and regression through origin.

    Args:
        X: np.ndarray of shape (n_samples, n_features)

    Returns:
        float: Estimated intrinsic dimension
    """
    N = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=3).fit(X)
    dists, _ = nbrs.kneighbors(X)

    r1 = np.maximum(dists[:, 1], 1e-15)
    r2 = dists[:, 2]
    mu = r2 / r1

    # Sort and compute empirical CDF
    mu_sorted = np.sort(mu)
    Femp = (np.arange(1, N+1) - 0.5) / N

    # Use middle 80% for regression
    start = int(0.1 * N)
    end = int(0.9 * N)

    x = np.log(mu_sorted[start:end]).reshape(-1, 1)
    y = -np.log(1 - Femp[start:end]).reshape(-1, 1)

    lr = LinearRegression(fit_intercept=False)
    lr.fit(x, y)

    return lr.coef_[0][0]


def correlation_dimension(X, n_samples=1000, local_k=30):
    """
    Correlation dimension from log-log scaling of LOCAL neighborhoods.

    Args:
        X: np.ndarray of shape (n_samples, n_features)
        n_samples: int, number of samples to use for efficiency
        local_k: int, defines local scale (use distances up to k-th nearest neighbor)

    Returns:
        float: Estimated local correlation dimension
    """
    if X.shape[0] > n_samples:
        idx = np.random.choice(X.shape[0], n_samples, replace=False)
        X = X[idx]

    # For local correlation dimension, find k-NN distances to set scale
    nbrs = NearestNeighbors(n_neighbors=min(local_k + 1, X.shape[0])).fit(X)
    knn_distances, _ = nbrs.kneighbors(X)

    # Use median of 30th neighbor distances as r_max (local scale)
    r_max_local = np.median(knn_distances[:, min(30, knn_distances.shape[1]-1)])

    # All pairwise distances for correlation counting
    distances = pdist(X)
    distances = distances[distances > 0]

    if len(distances) == 0:
        return np.nan

    # Set r_min as typical nearest neighbor distance, r_max as 30th neighbor distance
    r_min = np.percentile(knn_distances[:, 1], 10)  # 10th percentile of 1st NN distances
    r_max = r_max_local

    # Create log-spaced radii in local range
    radii = np.logspace(np.log10(r_min), np.log10(r_max), 15)

    counts = [(distances < r).sum() / len(distances) for r in radii]

    # Fit middle portion
    valid = np.array(counts) > 0
    if valid.sum() > 5:
        log_r = np.log(radii[valid])
        log_c = np.log(np.array(counts)[valid])

        # Use broader middle region for local fit
        mid_start = len(log_r) // 5
        mid_end = 4 * len(log_r) // 5

        if mid_end > mid_start + 2:
            slope, _ = np.polyfit(log_r[mid_start:mid_end], log_c[mid_start:mid_end], 1)
            return slope

    return np.nan


def local_pca_2d_energy(X, k=20):
    """
    Fraction of variance explained by first 2 PCs in local neighborhoods.
    Close to 1.0 indicates 2D manifold.

    Args:
        X: np.ndarray of shape (n_samples, n_features)
        k: int, number of neighbors for local PCA

    Returns:
        float: Mean fraction of variance explained by 2D subspace
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    _, indices = nbrs.kneighbors(X)

    energy_ratios = []
    for neighbors in indices:
        local_cov = np.cov(X[neighbors].T)
        eigenvalues = np.linalg.eigvalsh(local_cov)[::-1]

        if len(eigenvalues) >= 2:
            energy_2d = eigenvalues[:2].sum()
            total_energy = eigenvalues.sum() + 1e-12
            energy_ratios.append(energy_2d / total_energy)

    return np.mean(energy_ratios) if energy_ratios else 0


def test_for_2d_manifold(X):
    """
    Test if data lies on 2D manifold using all three metrics.

    Args:
        X: np.ndarray of shape (n_samples, n_features)

    Returns:
        dict: Results with 'twonn', 'correlation', 'pca_2d_energy' keys
        bool: Whether data is consistent with 2D manifold
    """
    results = {
        'twonn': twonn_dimension(X),
        'correlation': correlation_dimension(X),
        'pca_2d_energy': local_pca_2d_energy(X)
    }

    # Check if consistent with 2D
    is_2d = (
        1.5 < results['twonn'] < 3.5 and
        1.5 < results['correlation'] < 2.5 and
        results['pca_2d_energy'] > 0.9
    )

    return results, is_2d


# Additional utility metrics for backwards compatibility
def participation_ratio(X):
    """
    Compute participation ratio (effective dimensionality).
    PR = (sum(lambda_i))^2 / sum(lambda_i^2)

    Args:
        X: np.ndarray of shape (n_samples, n_features)

    Returns:
        float: Participation ratio
    """
    # Center the data
    centered = X - X.mean(axis=0)

    # Compute covariance eigenvalues
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Keep positive eigenvalues

    if len(eigenvalues) == 0:
        return 0

    # Participation ratio
    pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)

    return pr


def mle_dimension(X, k_max=20):
    """
    Maximum likelihood estimation of intrinsic dimension.
    Averages over different k values for robustness.

    Args:
        X: np.ndarray of shape (n_samples, n_features)
        k_max: int, maximum number of neighbors to consider

    Returns:
        float: Estimated dimension
    """
    n_samples = X.shape[0]
    k_max = min(k_max, n_samples - 1)

    # Find k_max nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k_max+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Remove self (first neighbor)
    distances = distances[:, 1:]

    dimensions = []
    for k in range(2, k_max):
        # MLE estimate for this k
        r_k = distances[:, k-1]
        valid = r_k > 0

        if valid.sum() > 0:
            log_ratios = []
            for j in range(1, k):
                r_j = distances[valid, j-1]
                log_ratios.append(np.log(r_k[valid] / r_j))

            mk = np.mean(log_ratios)
            if mk > 0:
                d_k = 1 / mk
                dimensions.append(d_k)

    return np.mean(dimensions) if dimensions else float('inf')