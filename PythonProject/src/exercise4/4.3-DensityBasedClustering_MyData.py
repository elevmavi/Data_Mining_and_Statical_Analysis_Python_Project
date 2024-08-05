from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

from src.shared.data_processing import etl_mat_file_fill_na

X = etl_mat_file_fill_na('resources/data/mydata.mat', 'X')
column1 = X.iloc[:, 0]
column2 = X.iloc[:, 1]

# Run DBSCAN clustering algorithm
epsilon = 0.5
MinPts = 15
dbscan = DBSCAN(eps=epsilon, min_samples=MinPts)
IDX = dbscan.fit_predict(X)

# Plotting results
plt.figure(1)
plt.scatter(column1, column2)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.figure(2)
plt.scatter(column1, column2, c=IDX, cmap='viridis')
plt.title(f'DBSCAN Clustering (epsilon = {epsilon}, MinPts = {MinPts})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster ID')

plt.show()
