import matplotlib.pyplot as plt
import numpy as np

from src.shared.data_processing import etl_text_file

# Load iris dataset
iris = etl_text_file('resources/data/iris.txt', ',').values
irisV = iris.copy()  # Create a copy of the original iris dataset

# Introduce NaN values into the iris dataset
ro, co = iris.shape
p1 = 60  # Percentage of NaN values to introduce
p = int(p1 * ro / 100)

# Create random indices to introduce NaN values
r1 = np.random.permutation(ro)

# Replace random rows with NaN values in each column
for i in range(4):
    irisV[r1[:p], i] = np.nan

# Remove rows containing NaN values
data = irisV[~np.any(np.isnan(iris), axis=1)]

# Plotting the cleaned data
plt.figure(figsize=(8, 6))
plt.plot(data[:, 0:4])
plt.title('Removing Rows with NaNs')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Remove columns containing NaN values
data = irisV[:, ~np.any(np.isnan(iris), axis=0)]

# Plotting the data after removing columns with NaNs
plt.figure(figsize=(8, 6))
plt.plot(data[:, 0:4])
plt.title('Removing Columns with NaNs')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Replace NaN values with 0
data = np.nan_to_num(irisV, nan=0.0)

# Plotting the data after replacing NaNs with 0
plt.figure(figsize=(8, 6))
plt.plot(data[:, 0:4])
plt.title('Replacing NaNs with 0')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Find NaN values in irisV
notNaN = ~np.isnan(irisV)

# Replace NaN values with 0
irisV[np.isnan(irisV)] = 0

# Calculate the total number of non-NaN values in each column
totalNo = np.sum(notNaN, axis=0)

# Calculate the sum of values in each column
columnTot = np.sum(irisV, axis=0)

# Calculate the mean value of each column (excluding NaN values)
colMean = np.divide(columnTot, totalNo, where=totalNo != 0)

# Replace NaN values with the column mean values
for i in range(colMean.shape[0]):
    irisV[np.isnan(irisV[:, i]), i] = colMean[i]

# Plotting the data after replacing NaNs with column means
plt.figure(figsize=(8, 6))
plt.plot(irisV[:, 0:4])
plt.title('Replacing NaNs with Column Means')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()
