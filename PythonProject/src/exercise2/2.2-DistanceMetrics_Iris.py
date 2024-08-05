from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

from src.shared.data_processing import etl_text_file

irisData = etl_text_file('resources/data/iris.txt', ',')

X1 = irisData.iloc[:, 0:4].values

# Select the first observation (first row) as the reference point
reference_observation1 = X1[0]

# Compute Euclidean distances from the reference observation to all other observations
euclidean_d_to_reference1 = cdist([reference_observation1], X1, metric='euclidean')[0]

# Plotting the distances
# plt.figure(figsize=(12, 6))
# plt.plot(euclidean_d_to_reference, marker='o', linestyle='-', color='b', label='Distance from 1st Observation')
# plt.title('Euclidean Distance from First Observation to Other Observations.txt')
# plt.xlabel('Observation Index')
# plt.ylabel('Distance to First Observation')
# plt.grid(True)
# plt.legend()
# plt.show()

cityblock_d_to_reference = cdist([reference_observation1], X1, metric='cityblock')[0]
#
# plt.figure(figsize=(12, 6))
# plt.plot(cityblock_d_to_reference, marker='o', linestyle='-', color='g', label='Distance from 1st Observation')
# plt.title('Cityblock Distance from First Observation to Other Observations.txt')
# plt.xlabel('Observation Index')
# plt.ylabel('Distance to First Observation')
# plt.grid(True)
# plt.legend()
# plt.show()

# Plotting the first dataset (A1) as a line plot
# plt.figure(figsize=(10, 6))
# plt.plot(range(len(euclidean_d_to_reference1)), euclidean_d_to_reference1, marker='o', linestyle='-', color='b', alpha=0.5, label='Euclidean distance')
#
# # Plotting the second dataset (A2) as a line plot
# plt.plot(range(len(cityblock_d_to_reference)), cityblock_d_to_reference, marker='s', linestyle='-', color='g', alpha=0.5, label='Cityblock distance')
#
# # Adding labels, title, and legend
# plt.xlabel('Distance Metric')
# plt.ylabel('Distance Value')
# plt.title('Euclidean vs Cityblock Distance Metrics')
# plt.legend()  # Show legend with labels for each dataset
# plt.grid(True)  # Enable grid for better visualization
# plt.show()

X2 = irisData.iloc[:, 0:2].values
reference_observation2 = X2[0]
euclidean_d_to_reference2 = cdist([reference_observation2], X2, metric='euclidean')[0]
plt.figure(figsize=(10, 6))
plt.plot(range(len(euclidean_d_to_reference1)), euclidean_d_to_reference1, marker='o', linestyle='-', color='b',
         alpha=0.5, label='Euclidean distance 0-4')

# Plotting the second dataset (A2) as a line plot
plt.plot(range(len(euclidean_d_to_reference2)), euclidean_d_to_reference2, marker='s', linestyle='-', color='g',
         alpha=0.5, label='Euclidean distance 0-2')

# Adding labels, title, and legend
plt.xlabel('Distance Metric')
plt.ylabel('Distance Value')
plt.title('Euclidean vs Cityblock Distance Metrics')
plt.legend()  # Show legend with labels for each dataset
plt.grid(True)  # Enable grid for better visualization
plt.show()
