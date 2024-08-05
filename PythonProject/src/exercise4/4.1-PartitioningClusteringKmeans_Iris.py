from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from src.shared.data_processing import etl_text_file

# Load iris data
iris_data = etl_text_file('resources/data/iris.txt', ',')

feature1 = iris_data.iloc[:, [2, 3]].values  # Use the last two dimensions (petal length and petal width)
feature2 = iris_data.iloc[:, 0:4].values

k = 3  # Number of clusters

# Apply K-means algorithm
kmeans = KMeans(n_clusters=k, random_state=0)
IDX1 = kmeans.fit_predict(feature1)
C = kmeans.cluster_centers_

# Plotting the data and clusters
plt.figure(figsize=(12, 6))

for cluster in range(k):
    plt.scatter(feature1[IDX1 == cluster, 0], feature1[IDX1 == cluster, 1], label=f'Cluster {cluster + 1}', s=12)
plt.scatter(C[:, 0], C[:, 1], marker='x', color='k', label='Centroids', s=200, linewidth=2)
plt.title('Clustering Result with Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()

IDX2 = kmeans.fit_predict(feature2)
D = kmeans.cluster_centers_

# Plotting the data and clusters
plt.figure(figsize=(12, 6))

for cluster in range(k):
    plt.scatter(feature1[IDX2 == cluster, 0], feature1[IDX2 == cluster, 1], label=f'Cluster {cluster + 1}', s=12)
plt.scatter(D[:, 0], D[:, 1], marker='x', color='k', label='Centroids', s=200, linewidth=2)
plt.title('Clustering Result with Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()
