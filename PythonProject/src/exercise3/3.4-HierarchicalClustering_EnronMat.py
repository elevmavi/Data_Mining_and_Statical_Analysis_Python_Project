from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

from src.shared.data_processing import etl_mat_file_fill_na

data = etl_mat_file_fill_na('resources/data/enron100.mat', 'en2')

# Compute pairwise Jaccard distances for the first 2 rows (emails) using columns 2 and 3 (word count and frequency)
first_two_rows = data.iloc[0:2, 1:3]
first_thousand_rows = data.iloc[0:1000, 1:3]
all_data = data.iloc[:, 1:3]

# Calculate pairwise Jaccard similarity using the word frequency data
jaccard_similarity_values = pdist(first_two_rows, metric='jaccard')

# Perform hierarchical clustering with single linkage using Jaccard similarity
Z_single_jaccard = linkage(jaccard_similarity_values, method='single')

# Visualize the dendrogram for Jaccard similarity with single linkage
plt.figure(figsize=(12, 8))
dendrogram(Z_single_jaccard)
plt.title('Hierarchical Clustering Dendrogram (Jaccard Similarity)')
plt.xlabel('Emails')
plt.ylabel('Distance (Jaccard)')
plt.show()

# Calculate pairwise cosine similarity using the word frequency data
cosine_similarity_values = pdist(first_thousand_rows, metric='cosine')

# Perform hierarchical clustering with single linkage using cosine similarity
Z_single_cosine = linkage(cosine_similarity_values, method='single')

# Visualize the dendrogram for Cosine similarity with single linkage
plt.figure(figsize=(12, 8))
dendrogram(Z_single_cosine)
plt.title('Hierarchical Clustering Dendrogram (Single Linkage, Cosine Similarity)')
plt.xlabel('Words')
plt.ylabel('Cosine Similarity')
plt.show()

# Calculate pairwise cosine similarity using the word frequency data
cosine_similarity = pdist(all_data, metric='cosine')

# Perform hierarchical clustering with average linkage using cosine similarity
Z_average_cosine = linkage(cosine_similarity, method='average')

plt.figure(figsize=(12, 8))
dendrogram(Z_average_cosine)
plt.title('Hierarchical Clustering Dendrogram (Average Linkage, Cosine Similarity)')
plt.xlabel('Words')
plt.ylabel('Cosine Similarity')
plt.show()
