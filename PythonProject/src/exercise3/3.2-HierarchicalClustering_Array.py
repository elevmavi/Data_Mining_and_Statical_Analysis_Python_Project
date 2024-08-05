import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

A = np.array([
    [1, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 1],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0]
])

B = ['recipe', 'physics', 'travel', 'hotel', 'travel', 'recipe']

# Compute pairwise cosine distances
cosine_distances = pdist(A, metric='cosine')

# Perform hierarchical clustering
Z_single_cosine = linkage(cosine_distances, method='single')  # Single linkage clustering

# Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z_single_cosine, labels=B, color_threshold=0.5)
plt.title('Hierarchical Single Linkage Clustering Dendrogram (Cosine Distance)')
plt.xlabel('Documents')
plt.ylabel('Distance')
plt.xticks(rotation=45)
plt.show()

# Compute pairwise Jaccard distances
jaccard_distances = pdist(A, metric="jaccard")
# Perform hierarchical clustering using single linkage
Z_single_jaccard = linkage(jaccard_distances, method='single')  # Single linkage clustering

# Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z_single_jaccard, labels=B, color_threshold=0.5)
plt.title('Hierarchical Single Linkage Clustering Dendrogram (Jaccard Distance)')
plt.xlabel('Documents')
plt.ylabel('Distance')
plt.xticks(rotation=45)
plt.show()

# Perform hierarchical clustering
Z_avg_cosine = linkage(cosine_distances, method='average')  # Average linkage clustering

# Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z_avg_cosine, labels=B, color_threshold=0.5)
plt.title('Hierarchical Average Linkage Clustering Dendrogram (Cosine Distance)')
plt.xlabel('Documents')
plt.ylabel('Distance')
plt.xticks(rotation=45)
plt.show()

Z_avg_jaccard = linkage(jaccard_distances, method='average')  # Average linkage clustering

# Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z_avg_jaccard, labels=B, color_threshold=0.5)
plt.title('Hierarchical Average Linkage  Clustering Dendrogram (Jaccard Distance)')
plt.xlabel('Documents')
plt.ylabel('Distance')
plt.xticks(rotation=45)
plt.show()
