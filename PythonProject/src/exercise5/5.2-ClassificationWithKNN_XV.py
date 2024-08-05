import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.shared.data_processing import etl_mat_file_fill_na

xV = etl_mat_file_fill_na('resources/data/xV.mat', 'xV')

# Split data into features (X) and target (y)
X = xV.iloc[:, [0, 1]]  # Features: all columns except the last one
Y = xV.iloc[:, -1]  # Target: last column

# Convert continuous target to binary class labels (example threshold)
threshold = np.median(Y)  # Example threshold for binary classification
# Map continuous values to binary classes (0 or 1)
y_binary = np.where(Y <= threshold, 0, 1)

# Split data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# Define k values
k_values = [1, 3, 5, 20]

# Loop over different k values
for k in k_values:
    # Initialize kNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the classifier
    knn.fit(X_train, y_train)

    # Predict on the test set
    y_pred = knn.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"Performance for k = {k}:")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{confusion}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print("\n")

# Plotting the classification performance
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test.iloc[:, 0], y=X_test.iloc[:, 1], hue=y_pred)
plt.title("Predicted Classes")
plt.show()

# Extract all features (excluding the last column which is assumed to be the target)
X_all = xV.iloc[:, :-1]

# Split data into training and testing sets
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_binary, test_size=0.3, random_state=42)

# Loop over different k values
for k in k_values:
    # Initialize kNN classifier
    knn_all = KNeighborsClassifier(n_neighbors=k)

    # Train the classifier
    knn_all.fit(X_train_all, y_train_all)

    # Predict on the test set
    y_pred_all = knn_all.predict(X_test_all)

    # Evaluate performance
    accuracy_all = accuracy_score(y_test_all, y_pred_all)
    confusion_all = confusion_matrix(y_test_all, y_pred_all)
    precision_all = precision_score(y_test_all, y_pred_all)
    recall_all = recall_score(y_test_all, y_pred_all)

    print(f"Performance for k = {k} with all features:")
    print(f"Accuracy: {accuracy_all}")
    print(f"Confusion Matrix:\n{confusion_all}")
    print(f"Precision: {precision_all}")
    print(f"Recall: {recall_all}")
    print("\n")

# Plotting the classification performance
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test.iloc[:, 0], y=X_test.iloc[:, 1], hue=y_pred)
plt.title("Predicted Classes")
plt.show()
