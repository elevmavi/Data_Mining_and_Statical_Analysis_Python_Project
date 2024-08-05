import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, precision_score, confusion_matrix, accuracy_score, f1_score, auc, \
    roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from ucimlrepo import fetch_ucirepo

# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# Convert features (X) and targets (y) to pandas DataFrames
X_df = pd.DataFrame(heart_disease.data.features, columns=heart_disease.feature_names)
y_df = heart_disease.data.targets

# Convert continuous target to binary class labels (example threshold)
threshold = np.median(y_df)  # Example threshold for binary classification
# Map continuous values to binary classes (0 or 1)
y_binary = np.where(y_df <= threshold, 0, 1)

# Handle missing values in the target vector y
imputer = SimpleImputer(strategy='median')  # Use median imputation (you can also use 'mean' or 'constant' strategies)
X_imputed = pd.DataFrame(imputer.fit_transform(X_df), columns=X_df.columns)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_binary, test_size=0.2, random_state=42)

# List of k values to test
k_values = [1, 3, 5, 7, 9]

# Loop over different k values
for k in k_values:
    # Initialize kNN classifier
    knn_all = KNeighborsClassifier(n_neighbors=k)

    # Train the classifier
    knn_all.fit(X_train, y_train.ravel())

    # Predict on the test set
    y_pred = knn_all.predict(X_train)

    # Evaluate performance
    # Calculate accuracy
    accuracy = accuracy_score(y_train, y_pred)
    # Calculate confusion matrix
    confusion = confusion_matrix(y_train, y_pred)
    # Calculate precision, recall, and F1 score
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)

    # Calculate ROC curve and AUC-ROC
    fpr, tpr, thresholds_roc = roc_curve(y_train, y_pred)
    auc_roc = roc_auc_score(y_train, y_pred)

    # Calculate Precision-Recall curve and AUC-PR
    precision_pr, recall_pr, thresholds_pr = precision_recall_curve(y_train, y_pred)
    auc_pr = auc(recall_pr, precision_pr)

    print(f"Performance for k = {k} with all features:")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{confusion}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"AUC-ROC: {auc_roc}")
    print(f"AUC-PR: {auc_pr}")
    print("\n")
