import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.shared.data_processing import etl_mat_file

# Load data
data = etl_mat_file('resources/data/xV600x470.mat', 'xV1')
initidx = data.iloc[:, 0].values
xV = data.iloc[:, 1:].values
fcols = xV.shape[1]

# Remove columns with NaN values
xV = xV[:, ~np.any(np.isnan(xV), axis=0)]

# Calculate Fisher score
class_means = np.mean(xV[initidx == 1], axis=0)
overall_mean = np.mean(xV, axis=0)
class_variance = np.var(xV[initidx == 1], axis=0)
overall_variance = np.var(xV, axis=0)
fisher_score = ((class_means - overall_mean) ** 2) / (class_variance + overall_variance)

# Sort features based on Fisher score
sorted_indices = np.argsort(fisher_score)[::-1]

# Plot heatmap
rho = np.corrcoef(xV[:, :50], rowvar=False)
sns.heatmap(rho)
plt.show()

# Find and print highly correlated features
threshold = 0.5
highly_correlated = np.where(rho >= threshold)
correlated_pairs = []
for i in range(len(highly_correlated[0])):
    if highly_correlated[0][i] < highly_correlated[1][i]:
        correlated_pairs.append((highly_correlated[0][i], highly_correlated[1][i], rho[highly_correlated[0][i], highly_correlated[1][i]]))

print("Highly correlated features:")
print(correlated_pairs)

# Print features with high Fisher score
num_features_to_keep = 50
selected_features = sorted_indices[:num_features_to_keep]
print("Selected features based on Fisher score:")
print(selected_features)