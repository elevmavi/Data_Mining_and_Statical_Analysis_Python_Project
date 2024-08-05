import numpy as np
import matplotlib.pyplot as plt

# Given dataV (including NaN values)
dataV = np.array([
    [-0.3999, np.nan, -1.0106],
    [0.6900, 0.2573, 0.6145],
    [0.8156, -1.0565, 0.5077],
    [0.7119, np.nan, np.nan],
    [np.nan, -0.8051, 0.5913],
    [0.6686, 0.5287, -0.6436],
    [1.1908, 0.2193, 0.3803],
    [np.nan, -0.9219, -1.0091],
    [-0.0198, np.nan, -0.0195],
    [-0.1567, -0.0592, -0.0482]
])

# Calculate mean along columns (ignoring NaN values)
mean_values = np.nanmean(dataV, axis=0)
print("Mean values along columns (excluding NaN):", mean_values)

# # Remove rows containing NaN values
# data = dataV[~np.any(np.isnan(dataV), axis=1)]
#
# # Plotting the cleaned data
# plt.figure(figsize=(8, 6))
# plt.plot(data)
# plt.title('Removing Rows with NaNs')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.grid(True)
# plt.show()

# # Remove columns containing NaN values
# data = dataV[:, ~np.any(np.isnan(dataV), axis=0)]
#
# # Plotting the data after removing columns with NaNs
# plt.figure(figsize=(8, 6))
# plt.plot(data)
# plt.title('Removing Columns with NaNs')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.grid(True)
# plt.show()

# # Replace NaN values with 0
# data = np.nan_to_num(dataV, nan=0.0)
#
# # Plotting the data after replacing NaNs with 0
# plt.figure(figsize=(8, 6))
# plt.plot(data)
# plt.title('Replacing NaNs with 0')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.grid(True)
# plt.show()

# Find NaN values in dataV
notNaN = ~np.isnan(dataV)

# Replace NaN values with 0
dataV[np.isnan(dataV)] = 0

# Calculate the total number of non-NaN values in each column
totalNo = np.sum(notNaN, axis=0)

# Calculate the sum of values in each column
columnTot = np.sum(dataV, axis=0)

# Calculate the mean value of each column (excluding NaN values)
colMean = np.divide(columnTot, totalNo, where=totalNo != 0)

# Replace NaN values with the column mean values
for i in range(colMean.shape[0]):
    dataV[np.isnan(dataV[:, i]), i] = colMean[i]

# Plotting the data after replacing NaNs with column means
plt.figure(figsize=(8, 6))
plt.plot(dataV)
plt.title('Replacing NaNs with Column Means')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()
