import numpy as np
import matplotlib.pyplot as plt

from src.shared.data_processing import etl_text_file


def linear_transform(data):
    # Compute min and max values for each column
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)

    # Linear transformation (min-max scaling) for each column
    normalized_data = (data - min_vals) / (max_vals - min_vals)

    return normalized_data


def zscore_transform(data):
    # Compute mean and standard deviation for each column
    mean_vals = np.mean(data, axis=0)
    std_vals = np.std(data, axis=0)

    # Z-score transformation for each column
    normalized_data = (data - mean_vals) / std_vals

    return normalized_data


# Load iris dataset
iris = etl_text_file('resources/data/iris.txt', ',').iloc[:, 0:4]
irisV = iris.copy()  # Create a copy of the original iris dataset

# Perform linear normalization
iris_linear_normalized = linear_transform(iris)

# Plotting the linearly normalized data
plt.figure(figsize=(8, 6))
plt.plot(iris_linear_normalized)
plt.title('Linear Normalization (Min-Max Scaling)')
plt.xlabel('Index')
plt.ylabel('Normalized Value')
plt.grid(True)
plt.show()

# Perform z-score normalization
iris_zscore_normalized = zscore_transform(irisV)

# Plotting the z-score normalized data
plt.figure(figsize=(8, 6))
plt.plot(iris_zscore_normalized)
plt.title('Z-score Normalization')
plt.xlabel('Index')
plt.ylabel('Normalized Value')
plt.grid(True)
plt.show()

# Given data (from MATLAB)
data = np.array([
    [-0.3999, -0.2625, -1.0106],
    [0.6900, 0.2573, 0.6145],
    [0.8156, -1.0565, 0.5077],
    [0.7119, -0.2625, -0.0708],
    [0.4376, -0.8051, 0.5913],
    [0.6686, 0.5287, -0.6436],
    [1.1908, 0.2193, 0.3803],
    [0.4376, -0.9219, -1.0091],
    [-0.0198, -0.2625, -0.0195],
    [-0.1567, -0.0592, -0.0482]
])
dataV = data.copy()

# Perform linear normalization
data_linear_normalized = linear_transform(data)

# Plotting the linearly normalized data
plt.figure(figsize=(8, 6))
plt.plot(data_linear_normalized)
plt.title('Linear Normalization (Min-Max Scaling)')
plt.xlabel('Index')
plt.ylabel('Normalized Value')
plt.grid(True)
plt.show()

# Perform z-score normalization
data_zscore_normalized = zscore_transform(dataV)

# Plotting the z-score normalized data
plt.figure(figsize=(8, 6))
plt.plot(data_zscore_normalized)
plt.title('Z-score Normalization')
plt.xlabel('Index')
plt.ylabel('Normalized Value')
plt.grid(True)
plt.show()
