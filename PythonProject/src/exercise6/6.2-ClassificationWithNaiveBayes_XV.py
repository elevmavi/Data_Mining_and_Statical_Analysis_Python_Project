import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from src.shared.data_processing import etl_mat_file_fill_na


def evaluate_classifier(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a Gaussian Naive Bayes classifier using given training and test data.

    Parameters:
    X_train (numpy.ndarray): Training feature matrix of shape (n_samples_train, n_features).
    X_test (numpy.ndarray): Test feature matrix of shape (n_samples_test, n_features).
    y_train (numpy.ndarray): Training target vector of shape (n_samples_train,).
    y_test (numpy.ndarray): Test target vector of shape (n_samples_test,).

    Returns:
    dict: Dictionary containing evaluation metrics and predicted labels.
        - 'y_pred' (numpy.ndarray): Predicted labels for the test data.
        - 'confusion_matrix' (numpy.ndarray): Confusion matrix of shape (n_classes, n_classes).
        - 'accuracy' (float): Classification accuracy on the test data.
        - 'error_rate' (float): Classification error rate on the test data.
        - 'sensitivity' (float): Macro-averaged recall (sensitivity) on the test data.
        - 'specificity' (float): Macro-averaged specificity on the test data.
    """

    # Initialize Naive Bayes classifier
    nb_classifier = GaussianNB()
    # Train the classifier on the training data
    nb_classifier.fit(X_train, y_train)
    # Predict labels for the test data
    y_pred = nb_classifier.predict(X_test)
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Compute error rate
    error_rate = 1 - accuracy
    # Compute sensitivity (macro average recall)
    sensitivity = recall_score(y_test, y_pred, average='macro')
    # Compute specificity for each class
    num_classes = len(np.unique(y))
    specificity = []
    for i in range(num_classes):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp)
        specificity.append(spec)
    # Compute macro-averaged specificity
    macro_specificity = np.mean(specificity)

    return {
        'nb_classifier': nb_classifier,
        'y_pred': y_pred,
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'error_rate': error_rate,
        'sensitivity': sensitivity,
        'specificity': macro_specificity
    }


data = etl_mat_file_fill_na('resources/data/xV2.mat', 'xV')

X = data.iloc[:, [0, 1]].values  # Select the first two columns as features
y = data.iloc[:, 5].values  # Select the sixth column as target class

# Split data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

results = evaluate_classifier(X_train, X_test, y_train, y_test)
print("Accuracy:", results['accuracy'])
print("Confusion Matrix:")
print(results['confusion_matrix'])
print("Error rates:", results['error_rate'])
print("Sensitivities:", results['sensitivity'])
print("Specificities:", results['specificity'])

# Posterior probability
posterior_probs = results['nb_classifier'].predict_proba(X_test)
print("Posterior probabilities:")
print(posterior_probs[:5])

# Plot predicted vs. actual classes for the test set
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=results['y_pred'], cmap='viridis', label='Predicted')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x', label='Actual')
plt.title('Naive Bayes Classifier: Predicted vs. Actual Classes')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Example of repeating the process 10 times with random subsampling
num_iterations = 10
accuracies = []
conf_matrices = []
error_rates = []
sensitivities = []
specificities = []

for _ in range(num_iterations):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)
    results = evaluate_classifier(X_train, X_test, y_train, y_test)
    accuracies.append(results['accuracy'])
    conf_matrices.append(results['confusion_matrix'])
    error_rates.append(results['error_rate'])
    sensitivities.append(results['sensitivity'])
    specificities.append(results['specificity'])
    # Posterior probability
    posterior_probs = results['nb_classifier'].predict_proba(X_test)
    print("Posterior probabilities:")
    print(posterior_probs[:5])

    # Plot predicted vs. actual classes for the test set
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=results['y_pred'], cmap='viridis', label='Predicted')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x', label='Actual')
    plt.title('Naive Bayes Classifier: Predicted vs. Actual Classes')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Compute mean accuracy and confusion matrix across iterations
mean_accuracy = np.mean(accuracies, axis=0)
mean_conf_matrix = np.mean(conf_matrices, axis=0)
mean_error_rates = np.mean(error_rates, axis=0)
mean_sensitivities = np.mean(sensitivities, axis=0)
mean_specificity = np.mean(specificities, axis=0)

print("Mean Accuracy:", mean_accuracy)
print("Mean Confusion Matrix:")
print(mean_conf_matrix)
print("Mean Error rates:", mean_error_rates)
print("Mean Sensitivities:", mean_sensitivities)
print("Mean Specificities:", mean_specificity)

X = data.iloc[:, :5].values  # Select the first two columns as features

accuracies = []
conf_matrices = []
error_rates = []
sensitivities = []
specificities = []
posterior_probs = []

for _ in range(num_iterations):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)
    results = evaluate_classifier(X_train, X_test, y_train, y_test)
    accuracies.append(results['accuracy'])
    conf_matrices.append(results['confusion_matrix'])
    error_rates.append(results['error_rate'])
    sensitivities.append(results['sensitivity'])
    specificities.append(results['specificity'])
    # Posterior probability
    posterior_probs = results['nb_classifier'].predict_proba(X_test)
    print("Posterior probabilities:")
    print(posterior_probs[:5])

    # Plot predicted vs. actual classes for the test set
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=results['y_pred'], cmap='viridis', label='Predicted')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x', label='Actual')
    plt.title('Naive Bayes Classifier: Predicted vs. Actual Classes')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Compute mean accuracy and confusion matrix across iterations
mean_accuracy = np.mean(accuracies, axis=0)
mean_conf_matrix = np.mean(conf_matrices, axis=0)
mean_error_rates = np.mean(error_rates, axis=0)
mean_sensitivities = np.mean(sensitivities, axis=0)
mean_specificity = np.mean(specificities, axis=0)
mean_posterior_probs = np.mean(posterior_probs, axis=0)

print("Mean Accuracy:", mean_accuracy)
print("Mean Confusion Matrix:")
print(mean_conf_matrix)
print("Mean Error rates:", mean_error_rates)
print("Mean Sensitivities:", mean_sensitivities)
print("Mean Specificities:", mean_specificity)
print("Mean Posterior Probs:", mean_specificity)
