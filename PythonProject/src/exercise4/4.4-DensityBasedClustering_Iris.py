import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from src.shared.data_processing import etl_text_file


def plot_cluster_result(X, labels, epsilon, min_samples):
    # Create a scatter plot of the data points colored by the cluster labels
    unique_labels = np.unique(labels)

    plt.figure()
    for label in unique_labels:
        if label == -1:
            # Plot noise points as black
            plt.scatter(X[labels == label, 0], X[labels == label, 1], c='k', marker='.', label='Noise')
        else:
            # Plot points for each cluster
            plt.scatter(X[labels == label, 0], X[labels == label, 1], label=f'Cluster {label}')

    plt.title(f'DBSCAN Clustering (ε = {epsilon}, MinPts = {min_samples})')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.legend()
    plt.show()


# Load iris data
iris_data = etl_text_file('resources/data/iris.txt', ',')
X = iris_data.iloc[:, [2, 3]].values  # Using petal length and petal width for clustering

# Initialize DBSCAN model
epsilon = 0.2  # Set the maximum distance between two samples to be considered in the same neighborhood
min_samples = 4  # Minimum number of samples required to form a dense region (core point)

dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)

# Perform DBSCAN clustering
clusters = dbscan.fit_predict(X)

# Scatter plot of the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.8)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('DBSCAN Clustering')
plt.colorbar(label='Cluster')
plt.show()

plot_cluster_result(X, clusters, epsilon, min_samples)

# Z-score normalization
scaler = StandardScaler()
X_zscore = scaler.fit_transform(X)

dbscan_zscore = DBSCAN(eps=epsilon, min_samples=min_samples)
cluster_labels_zscore = dbscan_zscore.fit_predict(X_zscore)

# Plotting Results with Z-score Normalized Data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels_zscore, cmap='viridis')
plt.title(f'DBSCAN Clustering (Z-score Normalized)')
plt.xlabel('Petal Length (Z-score)')
plt.ylabel('Petal Width (Z-score)')
plt.show()
