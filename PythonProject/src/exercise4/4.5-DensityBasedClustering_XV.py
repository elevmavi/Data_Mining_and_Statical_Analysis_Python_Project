from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

from src.shared.data_processing import etl_mat_file_fill_na

xV = etl_mat_file_fill_na('resources/data/xV.mat', 'xV')
feature1 = xV.iloc[:, [0, 1]].values

# Initialize DBSCAN model
epsilon = 0.3  # Set the maximum distance between two samples to be considered in the same neighborhood
min_samples = 50  # Minimum number of samples required to form a dense region (core point)

dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)

# Perform DBSCAN clustering
clusters = dbscan.fit_predict(feature1)

# Scatter plot of the clusters
plt.figure(figsize=(8, 6))
plt.scatter(feature1[:, 0], feature1[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.8)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('DBSCAN Clustering')
plt.colorbar(label='Cluster')
plt.show()

# Initialize DBSCAN model
epsilon = 0.2  # Set the maximum distance between two samples to be considered in the same neighborhood
min_samples = 4  # Minimum number of samples required to form a dense region (core point)

dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)

# Perform DBSCAN clustering
clusters = dbscan.fit_predict(feature1)

# Scatter plot of the clusters
plt.figure(figsize=(8, 6))
plt.scatter(feature1[:, 0], feature1[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.8)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('DBSCAN Clustering')
plt.colorbar(label='Cluster')
plt.show()


# Initialize DBSCAN model
epsilon = 0.7  # Set the maximum distance between two samples to be considered in the same neighborhood
min_samples = 10  # Minimum number of samples required to form a dense region (core point)

dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)

# Perform DBSCAN clustering
clusters = dbscan.fit_predict(feature1)

# Scatter plot of the clusters
plt.figure(figsize=(8, 6))
plt.scatter(feature1[:, 0], feature1[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.8)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('DBSCAN Clustering')
plt.colorbar(label='Cluster')
plt.show()
