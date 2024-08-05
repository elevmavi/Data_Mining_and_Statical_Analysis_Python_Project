from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from src.shared.data_processing import etl_mat_file_fill_na

xV = etl_mat_file_fill_na('resources/data/xV.mat', 'xV')
feature1 = xV.iloc[:, [0, 1]].values
feature2 = xV.iloc[:, :].values
feature3 = xV.iloc[:, [296, 305]].values

# Define the number of clusters
k = 3

# Apply k-means algorithm
kmeans = KMeans(n_clusters=k)
IDX1 = kmeans.fit_predict(feature1)
C1 = kmeans.cluster_centers_

# Plot the data points and centroids in the feature space
plt.figure()
for cluster in range(k):
    plt.scatter(feature1[IDX1 == cluster, 0], feature1[IDX1 == cluster, 1], label=f'Cluster {cluster + 1}', s=12)
plt.scatter(C1[:, 0], C1[:, 1], marker='x', color='k', label='Centroids', s=200, linewidth=2)
plt.title('Clustering Result with Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()

# Apply k-means algorithm
IDX2 = kmeans.fit_predict(feature2)
C2 = kmeans.cluster_centers_

# Plot the data points and centroids in the feature space
plt.figure()
for cluster in range(k):
    plt.scatter(feature1[IDX2 == cluster, 0], feature1[IDX2 == cluster, 1], label=f'Cluster {cluster + 1}', s=12)
plt.scatter(C2[:, 0], C2[:, 1], marker='x', color='k', label='Centroids', s=200, linewidth=2)
plt.title('Clustering Result with Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()

# Apply k-means algorithm
IDX3 = kmeans.fit_predict(feature3)
C3 = kmeans.cluster_centers_

# Plot the data points and centroids in the feature space
plt.figure()
for cluster in range(k):
    plt.scatter(feature1[IDX3 == cluster, 0], feature1[IDX3 == cluster, 1], label=f'Cluster {cluster + 1}', s=12)
plt.scatter(C3[:, 0], C3[:, 1], marker='x', color='k', label='Centroids', s=200, linewidth=2)
plt.title('Clustering Result with Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()
