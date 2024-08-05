from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from src.shared.data_processing import etl_mat_file_fill_na

xV = etl_mat_file_fill_na('resources/data/xV.mat', 'xV')

# Reshape to (600, 1)
first_feature = xV.iloc[:, 0].values.reshape(-1, 1)

# Calculate pairwise Euclidean distances for the first feature
distances_first_feature = pdist(first_feature, metric='euclidean')

# Perform hierarchical clustering with single linkage
Z_single_euclidean = linkage(distances_first_feature, method='single')

# Extract cluster labels for different numbers of clusters (2 to 10)
for n_clusters in range(2, 11):
    # Use fcluster to obtain cluster labels
    cluster_labels = fcluster(Z_single_euclidean, t=n_clusters, criterion='maxclust')

    # Print cluster labels for the current number of clusters
    print(f"Cluster labels for 1st column Single Linkage {n_clusters} clusters (Euclidean Distance):")
    print(cluster_labels)

# Calculate pairwise Euclidean distances for all features of xV
distances_xV = pdist(xV, metric='euclidean')

# Perform hierarchical clustering with average linkage
Z = linkage(distances_xV, method='average')

# Extract cluster labels for different numbers of clusters (2 to 10)
for n_clusters in range(2, 11):
    # Use fcluster to obtain cluster labels
    cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    # Print cluster labels for the current number of clusters
    print(f"Cluster labels for all data, with Average Linkage {n_clusters} clusters (Euclidean Distance):")
    print(cluster_labels)
