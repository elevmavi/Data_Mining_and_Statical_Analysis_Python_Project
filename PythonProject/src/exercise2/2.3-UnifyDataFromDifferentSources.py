import pandas
from matplotlib import pyplot as plt

from src.shared.data_processing import etl_text_file, etl_mat_file_fill_na, etl_excel_file, etl_accdb_file

# Load files
xV1 = etl_mat_file_fill_na('resources/data/xV1.mat', 'xV1')
xV2 = etl_text_file('resources/data/xV2.txt', '\t')
xV3 = etl_excel_file('resources/data/xV3.xls', 0)
xV4 = etl_accdb_file('resources/data/xV4DB.accdb', "xV4")
xV = pandas.concat([xV1, xV2, xV3, xV4])
print("Concatenated data frame has NaN values: {}".format(xV.isna().any().any()))

# Create a new figure
plt.figure(1)
# scatter diagram with two first columns
plt.scatter(xV.iloc[:, 0], xV.iloc[:, 1], alpha=0.6, marker='o')
plt.title(f'xV Columns 0 vs 1')

# Create a new figure
plt.figure(2)
columns_xV = [xV.iloc[:, i] for i in range(13)]
# scatter diagrams foreach column pair
for i in range(12):
    plt.subplot(3, 4, i + 1)  # i+1 because subplot index starts from 1
    plt.scatter(columns_xV[i], columns_xV[i + 1], alpha=0.2, marker='o', s=10)
    plt.title(f'xV Columns {i + 1} vs {i + 2}')  # Set subplot title
    plt.xlabel(f'Column {i + 1}')
    plt.ylabel(f'Column {i + 2}')

plt.tight_layout()  # Adjust subplot layout to prevent overlap
plt.show()  # Display the figure

# Create xVa by concatenating first 50 rows of each xV1, xV2, xV3, xV4
xVa = pandas.concat([col.iloc[:50] for col in [xV1, xV2, xV3, xV4]], axis=0)
# Create xVb by concatenating rows 51-100 of each xV1, xV2, xV3, xV4
xVb = pandas.concat([col.iloc[50:100] for col in [xV1, xV2, xV3, xV4]], axis=0)
# Create xVc by concatenating rows 101-150 of each xV1, xV2, xV3, xV4
xVc = pandas.concat([col.iloc[100:150] for col in [xV1, xV2, xV3, xV4]], axis=0)
# Concatenate xVa, xVb, xVc vertically to get xVd
xVd = pandas.concat((xVa, xVb, xVc), axis=0)

# Initialize a new figure (figure 3 in this case)
plt.figure(3, figsize=(12, 12))  # Adjust figsize as needed

# Iterate over the range 1 to 100 (Python uses 0-based indexing)
for i in range(100):
    plt.subplot(10, 10, i + 1)  # Create a subplot in a 10x10 grid (1-based indexing)
    # Extract data for the current subplot from xVd
    x = pandas.concat((xVd.iloc[0:200, i], xVd.iloc[200:400, i], xVd.iloc[400:600, i]))
    y = pandas.concat((xVd.iloc[0:200, i + 1], xVd.iloc[200:400, i + 1], xVd.iloc[400:600, i + 1]))
    # Plot the concatenated data as points ('.') in the current subplot
    plt.plot(x, y, '.', markersize=5)  # Adjust markersize as needed

    # Set subplot title (optional)
    plt.title(f'xVd Plot {i + 1}')

# Adjust layout to prevent overlap of subplots
plt.tight_layout()
# Show the plot
plt.show()

# Create a new figure
plt.figure(4)
columns_xV2 = [xV2.iloc[:, i] for i in range(13)]
# scatter diagrams foreach column pair
for i in range(12):
    plt.subplot(3, 4, i + 1)  # i+1 because subplot index starts from 1
    plt.scatter(columns_xV2[i], columns_xV2[i + 1], alpha=0.6, marker='o', s=10)
    plt.title(f'xV2 Columns {i + 1} vs {i + 2}')  # Set subplot title
    plt.xlabel(f'Column {i + 1}')
    plt.ylabel(f'Column {i + 2}')

plt.tight_layout()  # Adjust subplot layout to prevent overlap
plt.show()  # Display the figure

# Create a new figure
plt.figure(5, figsize=(12, 12))  # Adjust figsize as needed

# Iterate over the range 1 to 100 (Python uses 0-based indexing)
for i in range(100):
    plt.subplot(10, 10, i + 1)  # Create a subplot in a 10x10 grid (1-based indexing)
    # Extract data for the current subplot from xV2
    x = pandas.concat([xV2.iloc[0:50, i], xV2.iloc[50:100, i], xV2.iloc[100:150, i]])
    y = pandas.concat([xV2.iloc[0:50, i + 1], xV2.iloc[50:100, i + 1], xV2.iloc[100:150, i + 1]])

    # Plot the concatenated data as points ('.') in the current subplot
    plt.plot(x, y, '.', markersize=5)  # Adjust markersize as needed

    # Set subplot title (optional)
    plt.title(f'xV2 Plot {i + 1}')

# Adjust layout to prevent overlap of subplots
plt.tight_layout()

# Show the plot
plt.show()
