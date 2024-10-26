import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Update the print statement
clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
# Assuming clean_data is already loaded and split into features (X) and labels (y)
X = clean_data[:, :-1]  # All columns except the last one
y = clean_data[:, -1]   # The last column

# Create and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# Get the best split information
best_feature = clf.tree_.feature[0]
best_split = clf.tree_.threshold[0]
best_gain = clf.tree_.impurity[0] - (
    clf.tree_.impurity[1] * clf.tree_.n_node_samples[1] +
    clf.tree_.impurity[2] * clf.tree_.n_node_samples[2]
) / clf.tree_.n_node_samples[0]

print(f"Best feature: {best_feature}")
print(f"Best split point: {best_split}")
print(f"Information gain: {best_gain}")

# Predict and calculate accuracy
y_pred = clf.predict(X)
print(f"Predictions size: {len(y_pred)}")
print(f"Predictions: {y_pred}")
print(f"True labels: {y}")
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy}")
