import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Given data
t = np.arange(1900, 1991, 10)
p = np.array([75.995, 91.972, 105.711, 123.203, 131.669,
              150.697, 179.323, 203.212, 226.505, 249.633])

# Interpolation at a specific point (1975) using linear interpolation
p_interp_1975 = np.interp(1975, t, p)
print("Interpolated value at 1975:", p_interp_1975)

# Given data
t = np.arange(1900, 1991, 10)
p = np.array([75.995, 91.972, 105.711, 123.203, 131.669,
              150.697, 179.323, 203.212, 226.505, 249.633])

# Define the range of x values for interpolation
x = np.arange(1900, 2001)  # from 1900 to 2000 inclusive

# Perform nearest neighbor interpolation
indices = np.floor((x - t[0]) / 10).astype(int)  # Compute indices for nearest neighbor lookup
indices = np.clip(indices, 0, len(t) - 1)  # Clip indices to valid range
y = p[indices]  # Perform nearest neighbor interpolation

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(t, p, 'o', label='Original Data')  # Plot original data points
plt.plot(x, y, label='Interpolated Data (Nearest)', linestyle='--')  # Plot interpolated data
plt.title('Nearest Neighbor Interpolation')
plt.xlabel('Year')
plt.ylabel('Population (in millions)')
plt.legend()
plt.grid(True)
plt.show()

# Given data
tab = np.array([
    [1950, 150.697],
    [1960, 179.323],
    [1970, 203.212],
    [1980, 226.505],
    [1990, 249.633]
])

# Create a cubic spline interpolation function
interp_func = interp1d(tab[:, 0], tab[:, 1], kind='cubic')

# Interpolation at a specific point (1975) using cubic spline
p_1975 = interp_func(1975)
print("Interpolated value at 1975 (Cubic Spline):", p_1975)

# Define a finer range of x values for interpolation
x_range = np.arange(1950, 1991)

# Compute interpolated values using cubic spline
y_interp = interp_func(x_range)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(tab[:, 0], tab[:, 1], 'o', label='Original Data')  # Plot original data points
plt.plot(x_range, y_interp, label='Interpolated Data (Cubic Spline)', linestyle='--')  # Plot interpolated data
plt.title('Population Interpolation using Cubic Spline')
plt.xlabel('Year')
plt.ylabel('Population (in millions)')
plt.legend()
plt.grid(True)
plt.show()
