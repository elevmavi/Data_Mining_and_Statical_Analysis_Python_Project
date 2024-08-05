import numpy as np


def delNaNsRows(X):
    """
    Remove rows containing NaN values from a numpy array.

    Parameters:
    X (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Array with NaN-containing rows removed.
    """
    return X[~np.any(np.isnan(X), axis=1)]


# Create a sample array with NaN values
X = np.array([[1, 2, 3],
              [4, np.nan, 6],
              [7, 8, 9]])

# Remove rows with NaN values
X_cleaned = delNaNsRows(X)
print("Array after removing rows with NaN values:")
print(X_cleaned)

# Calculate correlation coefficients after removing NaN-containing rows
C = np.corrcoef(delNaNsRows(X))
print("Correlation coefficients after removing NaN-containing rows:")
print(C)
