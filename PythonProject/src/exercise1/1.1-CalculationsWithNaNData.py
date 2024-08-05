import numpy as np

# Create a 3x3 magic square with float data type
a = np.array([[8, 1, 6],
                    [3, 5, 7],
                    [4, 9, 2]], dtype=float)  # Specify dtype=float for floating-point values

# Calculate the sum of all elements in the matrix
sum_a = np.sum(a, axis=0)
print("Sum of all elements in a:", sum_a)

# Transpose the matrix and then calculate the sum (equivalent to sum(a')')
sum_a_transposed = np.sum(a.T, axis=0)
print("Sum of all elements in a transposed:", sum_a_transposed)

# Set element at (2,2) to NaN
a[1, 1] = np.nan

# Calculate the sum of all elements in the modified matrix (ignoring NaN values)
sum_a_modified = np.sum(a, axis=0)  # Use np.nansum to ignore NaN values
print("Sum of all elements in a (with NaN):", sum_a_modified)
