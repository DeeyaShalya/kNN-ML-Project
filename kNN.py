import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('E:\PROJECT-AI-ML\German Breast Cancer Study Group\German Breast Cancer Study Group.csv')

# Display the first few rows of the dataset
print("Dataset Overview:")
print(data.head())

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Display statistical summary
print("\nStatistical Summary:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Load the dataset
data = pd.read_csv('E:\PROJECT-AI-ML\German Breast Cancer Study Group\German Breast Cancer Study Group.csv')

# Display the first few rows of the dataset
print("Dataset Overview:")
print(tabulate(data.head(), headers='keys', tablefmt='psql'))

# Display dataset information
print("\nDataset Information:")
data_info = data.info()
print(data_info)

# Display statistical summary
print("\nStatistical Summary:")
print(tabulate(data.describe(), headers='keys', tablefmt='psql'))

# Check for missing values
print("\nMissing Values:")
missing_values = data.isnull().sum()
print(tabulate(missing_values.reset_index(), headers=['Column', 'Missing Values'], tablefmt='psql'))

# Visualize the distribution of the target variable
sns.countplot(x='status', data=data)
plt.title('Distribution of Target Variable')
plt.show()

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Define features (X) and target (y)
X = data.drop(columns=['status']).values
y = data['status'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("\nTraining Set Shape:")
print(tabulate([["X_train", X_train.shape], ["y_train", y_train.shape]], headers=['Set', 'Shape'], tablefmt='psql'))
print("\nTesting Set Shape:")
print(tabulate([["X_test", X_test.shape], ["y_test", y_test.shape]], headers=['Set', 'Shape'], tablefmt='psql'))

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit KNN Classifier to the Training Set
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict the Test-Set Results
predictions = knn.predict(X_test)

# Check the Accuracy Score
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

# Rebuild KNN Classification Model Using Different Values of k
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, predictions))

plt.plot(k_values, accuracies, marker='o')
plt.title('KNN Accuracy for Different Values of k')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Metrics
class_report = classification_report(y_test, predictions)
print("\nClassification Report:")
print(class_report)

# ROC - AUC
y_prob = knn.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# k-Fold Cross Validation
kf = KFold(n_splits=10, random_state=42, shuffle=True)
cv_scores = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
print(f'k-Fold Cross Validation Accuracy: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}')

# Subset of features
features = ['age', 'size', 'grade', 'nodes', 'pgr']

# Define features (X) and target (y)
X = data[features].values
y = data['status'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("\nTraining Set Shape (Condition 1):")
print(tabulate([["X_train", X_train.shape], ["y_train", y_train.shape]], headers=['Set', 'Shape'], tablefmt='psql'))
print("\nTesting Set Shape (Condition 1):")
print(tabulate([["X_test", X_test.shape], ["y_test", y_test.shape]], headers=['Set', 'Shape'], tablefmt='psql'))

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit KNN Classifier to the Training Set
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict the Test-Set Results
predictions = knn.predict(X_test)

# Check the Accuracy Score
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy (Condition 1): {accuracy:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Condition 1)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Metrics
class_report = classification_report(y_test, predictions)
print("\nClassification Report (Condition 1):")
print(class_report)

# Subset of features
features = ['age', 'size', 'grade', 'nodes', 'pgr']

# Define features (X) and target (y)
X = data[features].values
y = data['status'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("\nTraining Set Shape (Condition 1):")
print(tabulate([["X_train", X_train.shape], ["y_train", y_train.shape]], headers=['Set', 'Shape'], tablefmt='psql'))
print("\nTesting Set Shape (Condition 1):")
print(tabulate([["X_test", X_test.shape], ["y_test", y_test.shape]], headers=['Set', 'Shape'], tablefmt='psql'))

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit KNN Classifier to the Training Set
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict the Test-Set Results
predictions = knn.predict(X_test)

# Check the Accuracy Score
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy (Condition 1): {accuracy:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Condition 1)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Metrics
class_report = classification_report(y_test, predictions)
print("\nClassification Report (Condition 1):")
print(class_report)

# ROC - AUC
y_prob = knn.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve (Condition 1)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Rebuild KNN Classification Model Using Different Values of k
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, predictions))

plt.plot(k_values, accuracies, marker='o')
plt.title('KNN Accuracy for Different Values of k (Condition 1)')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

# Define features (X) and target (y)
X = data.drop(columns=['status']).values
y = data['status'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("\nTraining Set Shape (Condition 2):")
print(tabulate([["X_train", X_train.shape], ["y_train", y_train.shape]], headers=['Set', 'Shape'], tablefmt='psql'))
print("\nTesting Set Shape (Condition 2):")
print(tabulate([["X_test", X_test.shape], ["y_test", y_test.shape]], headers=['Set', 'Shape'], tablefmt='psql'))

# Feature Scaling using Min-Max Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit KNN Classifier to the Training Set
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict the Test-Set Results
predictions = knn.predict(X_test)

# Check the Accuracy Score
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy (Condition 2): {accuracy:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Condition 2)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Metrics
class_report = classification_report(y_test, predictions)
print("\nClassification Report (Condition 2):")
print(class_report)

# ROC - AUC
y_prob = knn.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve (Condition 2)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Rebuild KNN Classification Model Using Different Values of k
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, predictions))

plt.plot(k_values, accuracies, marker='o')
plt.title('KNN Accuracy for Different Values of k (Condition 2)')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

# Define features (X) and target (y)
X = data.drop(columns=['status']).values
y = data['status'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("\nTraining Set Shape (Condition 3):")
print(tabulate([["X_train", X_train.shape], ["y_train", y_train.shape]], headers=['Set', 'Shape'], tablefmt='psql'))
print("\nTesting Set Shape (Condition 3):")
print(tabulate([["X_test", X_test.shape], ["y_test", y_test.shape]], headers=['Set', 'Shape'], tablefmt='psql'))

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit KNN Classifier to the Training Set with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict the Test-Set Results
predictions = knn.predict(X_test)

# Check the Accuracy Score
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy (Condition 3): {accuracy:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Condition 3)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Metrics
class_report = classification_report(y_test, predictions)
print("\nClassification Report (Condition 3):")
print(class_report)

# ROC - AUC
y_prob = knn.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve (Condition 3)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Rebuild KNN Classification Model Using Different Values of k
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, predictions))

plt.plot(k_values, accuracies, marker='o')
plt.title('KNN Accuracy for Different Values of k (Condition 3)')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

# Define features (X) and target (y)
X = data.drop(columns=['status']).values
y = data['status'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("\nTraining Set Shape (Condition 4):")
print(tabulate([["X_train", X_train.shape], ["y_train", y_train.shape]], headers=['Set', 'Shape'], tablefmt='psql'))
print("\nTesting Set Shape (Condition 4):")
print(tabulate([["X_test", X_test.shape], ["y_test", y_test.shape]], headers=['Set', 'Shape'], tablefmt='psql'))

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit KNN Classifier to the Training Set with k=10
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

# Predict the Test-Set Results
predictions = knn.predict(X_test)

# Check the Accuracy Score
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy (Condition 4): {accuracy:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Condition 4)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Metrics
class_report = classification_report(y_test, predictions)
print("\nClassification Report (Condition 4):")
print(class_report)

# ROC - AUC
y_prob = knn.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve (Condition 4)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Rebuild KNN Classification Model Using Different Values of k
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, predictions))

plt.plot(k_values, accuracies, marker='o')
plt.title('KNN Accuracy for Different Values of k (Condition 4)')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()