import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

from src.shared.data_processing import etl_text_file, etl_mat_file_fill_na

# Load iris data
iris_data = etl_text_file('resources/data/iris.txt', ',')

# Extract features and target labels from irisData
data = iris_data.iloc[:, :4].values  # Features (sepal length, sepal width, petal length, petal width)
species = iris_data.iloc[:, 4].values  # Target labels (species)

# Calculate pairwise distances using Euclidean distance
euclidean_distances = pdist(iris_data, metric='euclidean')

# # Perform hierarchical clustering using single linkage
Z_single_euclidean = linkage(euclidean_distances, method='single')

# Plot dendrogram for Simple Linkage
plt.figure(figsize=(12, 8))
dendrogram(Z_single_euclidean, labels=species)
plt.title('Hierarchical Clustering Dendrogram (Simple Linkage)')
plt.xlabel('Species')
plt.ylabel('Distance')
plt.show()

k = 3  # Number of clusters
# Assign observations to clusters for Simple Linkage
max_cluster_single = fcluster(Z_single_euclidean, k, criterion='maxclust')

# Crosstab to compare predicted clusters with true species labels for irisData
print("Clustering Result for Simple Linkage:")
print(pd.crosstab(max_cluster_single, species, rownames=['Cluster'], colnames=['Species']))

# Perform hierarchical clustering using average linkage
Z_average_euclidean = linkage(euclidean_distances, method='average')

# Plot dendrogram for Average Linkage
plt.figure(figsize=(12, 8))
dendrogram(Z_average_euclidean, labels=species)
plt.title('Hierarchical Clustering Dendrogram (Average Linkage)')
plt.xlabel('Species')
plt.ylabel('Distance')
plt.show()

# Assign observations to clusters for Average Linkage
max_cluster_average = fcluster(Z_average_euclidean, k, criterion='maxclust')

# Crosstab to compare predicted clusters with true species labels for irisData
print("Clustering Result for Average Linkage:")
print(pd.crosstab(max_cluster_average, species, rownames=['Cluster'], colnames=['Species']))

# Perform hierarchical clustering using average linkage
Z_complete_euclidean = linkage(euclidean_distances, method='complete')

# Plot dendrogram for Average Linkage
plt.figure(figsize=(12, 8))
dendrogram(Z_complete_euclidean, labels=species)
plt.title('Hierarchical Clustering Dendrogram (Complete Linkage)')
plt.xlabel('Species')
plt.ylabel('Distance')
plt.show()

# Assign observations to clusters for Average Linkage
clusters_complete = fcluster(Z_complete_euclidean, k, criterion='maxclust')

# Crosstab to compare predicted clusters with true species labels for irisData
print("Clustering Result for Complete Linkage:")
print(pd.crosstab(clusters_complete, species, rownames=['Cluster'], colnames=['Species']))

# Define combinations of features (columns) for clustering
feature_combinations = [
    [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],  # Pairwise combinations
    [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3]  # All combinations
]

# Define distance metrics
distance_metrics = ['euclidean', 'cityblock', 'chebyshev']

# Iterate over all combinations of features and distance metrics
for features in feature_combinations:
    data_subset = iris_data.iloc[:, features]

    for metric in distance_metrics:
        # Calculate pairwise distances using current metric
        distances_subset = pdist(data_subset, metric=metric)

        # Perform hierarchical clustering using single linkage and current metric
        Z_subset = linkage(distances_subset, method='single')

        # Plot dendrogram for the current clustering
        plt.figure(figsize=(12, 8))
        dendrogram(Z_subset, labels=species)
        plt.title(f'Hierarchical Clustering Dendrogram (Simple Linkage, Features {features}, Metric {metric})')
        plt.xlabel('Species')
        plt.ylabel('Distance')
        plt.show()

        # Assign observations to clusters using single linkage for the current clustering
        k = 3  # Number of clusters
        clusters_subset = fcluster(Z_subset, k, criterion='maxclust')

        # Compare predicted clusters with true species labels using pd.crosstab
        print(f"Clustering Result for Simple Linkage, Features {features}, Metric {metric}:")
        print(pd.crosstab(clusters_subset, species, rownames=['Cluster'], colnames=['Species']))


# Extract the features of interest
features_1 = iris_data.iloc[:, [2, 3]]  # Features [3, 4]
features_2 = iris_data.iloc[:, [0, 1, 2, 3]]  # Features [1, 2, 3, 4]

# Define the feature names for better clarity
feature_names_1 = ['petal length', 'petal width']
feature_names_2 = ['sepal length', 'sepal width', 'petal length', 'petal width']

# Define the number of clusters to evaluate (from 4 to 10 clusters)
num_clusters_range = range(4, 11)

# Perform hierarchical clustering with single linkage and Euclidean distance for each feature set
for features, feature_names in zip([features_1, features_2], [feature_names_1, feature_names_2]):
    print(f"Clustering results for features: {', '.join(feature_names)}")
    for num_clusters in num_clusters_range:
        # Calculate pairwise distances using Euclidean distance
        distances = linkage(features, method='single', metric='euclidean')

        # Assign observations to clusters
        clusters = fcluster(distances, num_clusters, criterion='maxclust')

        # Compute and print the cluster sizes
        cluster_sizes = [np.sum(clusters == i) for i in range(1, num_clusters + 1)]
        print(f"Number of clusters: {num_clusters}, Cluster sizes: {cluster_sizes}")

        # Plot the dendrogram for visualization (optional)
        plt.figure(figsize=(10, 6))
        plt.title(f"Dendrogram for {num_clusters} Clusters using Features: {', '.join(feature_names)}")
        dendrogram(distances, labels=iris_data.iloc[:, 4].values, leaf_rotation=90)
        plt.xlabel('Species')
        plt.ylabel('Distance')
        plt.show()
