import matplotlib.pyplot as plt

from src.shared.data_processing import etl_text_file

# Load iris dataset
iris = etl_text_file('resources/data/iris.txt', ',')

# Define iris species based on the fifth column (species code)
setosa = iris[iris[4] == 1].iloc[:, :4]  # data for setosa
versicolor = iris[iris[4] == 2].iloc[:, :4]  # data for versicolor
virginica = iris[iris[4] == 3].iloc[:, :4]  # data for virginica

# Total number of observations
obsv_n = iris.shape[0]

# Characteristics of iris (features)
characteristics = ['sepal length', 'sepal width', 'petal length', 'petal width']

# Pairs of characteristics for scatter plots
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# Plotting pairwise scatter plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, pair in enumerate(pairs):
    x, y = pair
    ax = axes[idx // 3, idx % 3]
    ax.plot(setosa[x], setosa[y], '.', label='setosa')
    ax.plot(versicolor[x], versicolor[y], '.', label='versicolor')
    ax.plot(virginica[x], virginica[y], '.', label='virginica')
    ax.set_xlabel(characteristics[x])
    ax.set_ylabel(characteristics[y])
    ax.legend()

plt.tight_layout()
plt.show()
