import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from src.shared.data_processing import etl_text_file


def calculate_performance_metrics(conf_matrix):
    """
    Calculate performance metrics from a confusion matrix for a multiclass classification task.

    Parameters:
    - conf_matrix (ndarray): Confusion matrix of shape (n_classes, n_classes), where
                             conf_matrix[i, j] represents the count of true samples of class i
                             predicted as class j by the classifier.

    Returns:
    - metrics (dict): Dictionary containing performance metrics for each class and overall.

    Performance Metrics:
    - TP: True Positives
    - FP: False Positives
    - FN: False Negatives
    - TN: True Negatives
    - Accuracy: Overall accuracy of the classifier
    - Error Rate: Overall error rate of the classifier
    - Sensitivity (Recall): True Positive Rate (TPR) or Recall
    - Specificity: True Negative Rate (TNR)
    - Precision: Positive Predictive Value (PPV)
    - NPV (Negative Predictive Value)
    - FPR (False Positive Rate)
    - FNR (False Negative Rate)
    - LR+ (Positive Likelihood Ratio)
    - LR- (Negative Likelihood Ratio)
    - ERA (Error Rate Average)
    - F1 Score: Harmonic mean of Precision and Recall (F1 Score)

    """
    num_classes = conf_matrix.shape[0]
    metrics = {}

    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FP = np.sum(conf_matrix[:, i]) - TP
        FN = np.sum(conf_matrix[i, :]) - TP
        TN = np.sum(conf_matrix) - (TP + FP + FN)

        accuracy = (TP + TN) / np.sum(conf_matrix)
        error_rate = (FP + FN) / np.sum(conf_matrix)
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
        lrp = sensitivity / fpr if fpr > 0 else np.inf
        lrn = fnr / specificity if specificity > 0 else np.inf
        era = (fpr + fnr) / 2
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        metrics[f'Class {i + 1}'] = {
            'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
            'Accuracy': accuracy, 'Error Rate': error_rate,
            'Sensitivity (Recall)': sensitivity, 'Specificity': specificity,
            'Precision': precision, 'NPV (Negative Predictive Value)': npv,
            'FPR (False Positive Rate)': fpr, 'FNR (False Negative Rate)': fnr,
            'LR+ (Positive Likelihood Ratio)': lrp, 'LR- (Negative Likelihood Ratio)': lrn,
            'ERA (Error Rate Average)': era, 'F1 Score': f1_score
        }

    return metrics


# Load iris data
iris_data = etl_text_file('resources/data/iris.txt', ',')

# Step 1: Extracting the required dimensions from the DataFrame
X = iris_data.iloc[:, [2, 3]].values  # Selecting the 3rd and 4th columns (0-based indexing)
Y = iris_data.iloc[:, 4]  # Selecting the 4th column (0-based indexing)

# Step 2: Splitting the dataset into training (X1) and test (X2) sets
X1 = X[np.r_[0:40, 50:90, 100:140]]  # Training set
X2 = X[np.r_[40:50, 90:100, 140:150]]  # Test set

# Step 3: Selecting the corresponding classes for training and test sets
c1 = iris_data.iloc[np.r_[0:40, 50:90, 100:140], 4]  # Classes for training set
c2 = iris_data.iloc[np.r_[40:50, 90:100, 140:150], 4]  # Classes for test set

# Step 4: Applying the k-Nearest Neighbors (kNN) algorithm
k = 3  # Number of nearest neighbors to consider
knn_different_classifier = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

# Fitting the classifier with the training data and corresponding classes
knn_different_classifier.fit(X1, c1)

# Predicting classes for the test data
predicted_classes = knn_different_classifier.predict(X2)

plt.figure()
# Plotting kNN classified classes from IDX
plt.plot(predicted_classes, 'ro', markersize=12, markerfacecolor='none', label='kNN class')
# Plotting original classes from c2
plt.plot(c2.values, 'bx', markersize=12, label='Original Class')

# Adding legend and labels
plt.legend(loc='upper left')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.title('kNN Classification vs. Original Classes')

# Display the plot
plt.show()

conf_matrix = confusion_matrix(c2, predicted_classes)

# Display the confusion matrix as a heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(c2))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

# Calculate performance metrics
metrics = calculate_performance_metrics(conf_matrix)

# Display the performance metrics
for class_label, metrics_dict in metrics.items():
    print(f'Class {class_label}:')
    for metric_name, value in metrics_dict.items():
        if isinstance(value, float):
            print(f'{metric_name}: {value:.4f}')
        else:
            print(f'{metric_name}: {value}')  # Print non-numeric values as-is
    print()

for i in range(3):
    # Get the number of rows and columns in X
    ro, co = X.shape

    # Define the percentage of rows to use as training set
    p1 = 70
    p = round(p1 * ro / 100)  # Number of rows for training set

    # Randomly shuffle the indices of all rows in X
    r1 = np.random.permutation(ro)

    # Split the shuffled indices into training and test set indices
    train_indices = r1[:p]
    test_indices = r1[p:]

    # Create training and test sets based on the selected indices
    X_train = X[train_indices]
    X_test = X[test_indices]

    # Get corresponding classes (labels) for training and test sets
    c1_train = iris_data.iloc[train_indices, 4]  # Last column (class labels) for training set
    c2_test = iris_data.iloc[test_indices, 4]  # Last column (class labels) for test set

    # Fit the classifier with the training data and corresponding classes
    knn_different_classifier.fit(X_train, c1_train)

    # Predict classes for the test data
    all_data_predicted_classes = knn_different_classifier.predict(X_test)

    plt.figure()
    # Plotting kNN classified classes from IDX
    plt.plot(all_data_predicted_classes, 'ro', markersize=12, markerfacecolor='none', label='kNN class')
    # Plotting original classes from c2
    plt.plot(c2_test.values, 'bx', markersize=12, label='Original Class')

    # Adding legend and labels
    plt.legend(loc='upper left')
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.title('kNN Classification vs. Original Classes')

    # Display the plot
    plt.show()

    all_data_conf_matrix = confusion_matrix(c2_test, all_data_predicted_classes)

    # Display the confusion matrix as a heatmap
    all_data_disp = ConfusionMatrixDisplay(confusion_matrix=all_data_conf_matrix, display_labels=np.unique(c2_test))
    all_data_disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.show()

    # Calculate performance metrics
    all_data_metrics = calculate_performance_metrics(all_data_conf_matrix)

    # Display the performance metrics
    for class_label, metrics_dict in all_data_metrics.items():
        print(f'Class {class_label}:')
        for metric_name, value in metrics_dict.items():
            if isinstance(value, float):
                print(f'{metric_name}: {value:.4f}')
            else:
                print(f'{metric_name}: {value}')  # Print non-numeric values as-is
        print()

# Define the number of folds for cross-validation
num_folds = 10

# Perform cross-validation with 10-fold stratified splits
# Using 'accuracy' scoring to compute classification accuracy
# Note: 'cv' parameter specifies the number of folds (k=10)
cv_scores = cross_val_score(knn_different_classifier, X, Y, cv=num_folds, scoring='accuracy')

# Calculate the mean error rate (1 - mean of cross-validation scores)
mean_error_rate = 1 - np.mean(cv_scores)

# Print the mean error rate across 10 folds
print(f"Mean Error Rate across {num_folds} folds: {mean_error_rate:.4f}")

# Define list of dimensions to consider (all, combinations)
dimensions_to_try = [
    [0],
    [1],
    [2],
    [3],
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
    [1, 3],
    [2, 3],
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, 3],
    [1, 2, 3],
    [0, 1, 2, 3]
]

# Define list of different numbers of neighbors (k) to try
neighbors_to_try = [1, 2, 5, 15]

# Dictionary to store results (mean error rates) for different configurations
results = {}

# Iterate over each combination of dimensions and number of neighbors
for dimensions in dimensions_to_try:
    X_subset = iris_data.iloc[:, dimensions]  # Subset of X based on selected dimensions

    for k in neighbors_to_try:
        knn_different_classifier = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

        # Step 1: Extracting the required dimensions from the DataFrame
        X = iris_data.iloc[:, dimensions].values  # Selecting the columns based on dimensions array (0-based indexing)
        Y = iris_data.iloc[:, 4]  # Selecting the 4th columns (0-based indexing)

        # Get the number of rows and columns in X
        ro, co = X.shape

        # Define the percentage of rows to use as training set
        p1 = 70
        p = round(p1 * ro / 100)  # Number of rows for training set

        # Randomly shuffle the indices of all rows in X
        r1 = np.random.permutation(ro)

        # Split the shuffled indices into training and test set indices
        train_indices = r1[:p]
        test_indices = r1[p:]

        # Create training and test sets based on the selected indices
        X_train = X[train_indices]
        X_test = X[test_indices]

        # Get corresponding classes (labels) for training and test sets
        c1_train = iris_data.iloc[train_indices, 4]  # Last column (class labels) for training set
        c2_test = iris_data.iloc[test_indices, 4]  # Last column (class labels) for test set

        # Fit the classifier with the training data and corresponding classes
        knn_different_classifier.fit(X_train, c1_train)

        # Predict classes for the test data
        all_data_predicted_classes = knn_different_classifier.predict(X_test)

        plt.figure()
        # Plotting kNN classified classes from IDX
        plt.plot(all_data_predicted_classes, 'ro', markersize=12, markerfacecolor='none', label='kNN class')
        # Plotting original classes from c2
        plt.plot(c2_test.values, 'bx', markersize=12, label='Original Class')

        # Adding legend and labels
        plt.legend(loc='upper left')
        plt.xlabel('Sample Index')
        plt.ylabel('Class')
        plt.title(f'kNN Classification vs. Original Classes, Dimensions: {dimensions} and k={k}')

        # Display the plot
        plt.show()

        all_data_conf_matrix = confusion_matrix(c2_test, all_data_predicted_classes)

        # Display the confusion matrix as a heatmap
        all_data_disp = ConfusionMatrixDisplay(confusion_matrix=all_data_conf_matrix, display_labels=np.unique(c2_test))
        all_data_disp.plot(cmap='Blues', values_format='d')
        plt.title(f'Confusion Matrix, Dimensions: {dimensions} and k={k}')
        plt.show()

        # Calculate performance metrics
        all_data_metrics = calculate_performance_metrics(all_data_conf_matrix)

        # Display the performance metrics
        for class_label, metrics_dict in all_data_metrics.items():
            print(f'Class {class_label}:')
            for metric_name, value in metrics_dict.items():
                if isinstance(value, float):
                    print(f'{metric_name}: {value:.4f}')
                else:
                    print(f'{metric_name}: {value}')  # Print non-numeric values as-is
            print()

        # Perform 10-fold cross-validation and compute mean error rate
        cv_scores = cross_val_score(knn_different_classifier, X_subset, Y, cv=num_folds, scoring='accuracy')
        mean_error_rate = 1 - np.mean(cv_scores)

        # Print the mean error rate across 10 folds
        print(f"Mean Error Rate across {num_folds} folds: {mean_error_rate:.4f}")

# Print results
for config, error_rate in results.items():
    print(f"Configuration: {config} - Mean Error Rate: {error_rate:.4f}")
