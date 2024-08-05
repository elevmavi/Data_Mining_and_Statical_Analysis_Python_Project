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

# Plot the irisV dataset (scatter plot excluding NaN values)
plt.figure(figsize=(8, 6))
plt.scatter(irisV[:, 0], irisV[:, 1], c=irisV[:, 4], cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Scatter Plot of Iris Dataset (with NaN Values)')
plt.colorbar(label='Species')
plt.grid(True)
plt.show()
