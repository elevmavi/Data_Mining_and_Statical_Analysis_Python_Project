import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.spatial.distance import squareform, pdist


def compute_distances(X):
    """
    Compute pairwise distances between data points using various distance metrics.

    Parameters:
    X (numpy.ndarray): Input data matrix of shape (n_samples, n_features).

    Returns:
    tuple: A tuple of distance matrices corresponding to different distance metrics.
    """
    # Calculate pairwise distances using various methods
    D1 = pdist(X, 'euclidean')  # Compute pairwise Euclidean distances
    D2 = pdist(X, 'seuclidean', V=np.std(X,
                                         axis=0))  # Compute pairwise standardized Euclidean distances, where V: standard deviation for each feature (column)
    D3 = pdist(X, 'cityblock')  # Manhattan (cityblock) distance
    D4 = pdist(X, 'minkowski',
               p=3)  # Minkowski distance with p=3 (larger diffs compared to (e.g., p=2 for Euclidean distance and p=1 for Manhattan distance)
    D5 = pdist(X, 'chebyshev')  # Chebyshev distance
    D6 = pdist(X, lambda u, v: distance.mahalanobis(u, v, np.linalg.inv(
        np.cov(X, rowvar=False))))  # Mahalanobis distance, where VI: inverse covariance matrix d(u,v)
    D7 = pdist(X, 'cosine')  # Cosine distance

    return D1, D2, D3, D4, D5, D6, D7


def compute_and_plot_distances(X, i, j):
    """
    Compute and visualize pairwise distances between specified observations.

    Parameters:
    X (numpy.ndarray): Input data matrix of shape (n_samples, n_features).
    i (int): Index of the first observation.
    j (int): Index of the second observation.
    """
    # Compute distances between observations i and j
    D1, D2, D3, D4, D5, D6, D7 = compute_distances(X)

    # Select distances for observations i and j, convert the condensed distance matrix to a square distance matrix
    distances = [squareform(D1)[i, j], squareform(D2)[i, j], squareform(D3)[i, j],
                 squareform(D4)[i, j], squareform(D5)[i, j], squareform(D6)[i, j],
                 squareform(D7)[i, j]]

    # Find the method yielding the maximum and minimum distances
    methods = ['euclidean', 'seuclidean', 'cityblock', 'minkowski', 'chebyshev', 'mahalanobis', 'cosine']
    max_distance_method = methods[np.argmax(distances)]
    min_distance_method = methods[np.argmin(distances)]

    # Plot distances as a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(distances)), distances, tick_label=methods)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Distances between observations {i} and {j}")
    plt.xlabel("Distance Metric")
    plt.ylabel("Distance Value")
    plt.show()

    # Print maximum and minimum distances and their corresponding methods
    print(
        f"Maximum distance between observations {i} and {j} ({max(distances):.2f}) using method: {max_distance_method}")
    print(
        f"Minimum distance between observations {i} and {j} ({min(distances):.2f}) using method: {min_distance_method}")


# Create a random matrix with 100 rows and 5 columns
X = np.random.randn(100, 5)

# Compute and plot distances for observations 24 and 75
i = 24
j = 75
compute_and_plot_distances(X, i, j)

# Compute and plot distances for observations 1 and 100
i = 0
j = 99
compute_and_plot_distances(X, i, j)
