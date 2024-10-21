import numpy as np
import matplotlib.pyplot as plt

'''
The FIND_SPLIT function chooses the attribute and value that result in the highest information gain.

For datasets with continuous attributes, the decision-tree learning algorithm searches for the split point
(defined by an attribute and a value) that gives the highest information gain. For example, with two
attributes (A0 and A1) ranging from 0 to 10, the algorithm might determine that splitting the dataset
according to "A1 > 4" provides the most information.

An efficient method for finding good split points:
1. Sort the values of the attribute
2. Consider only split points between two examples in sorted order
3. Keep track of running totals of examples for each class on both sides of the split point

Evaluating Information Gain:
- Let the training dataset Sall have K different labels
- Define two subsets (Sleft and Sright) based on the splitting rule (e.g., "A1 > 4")
- Compute the distribution of each label for the dataset and subsets
- Calculate information gain using entropy:

    Gain(Sall, Sleft, Sright) = H(Sall) - Remainder(Sleft, Sright)

    Where:
    H(dataset) = -sum(pk * log2(pk)) for k = 1 to K
    Remainder(Sleft, Sright) = (|Sleft| * H(Sleft) + |Sright| * H(Sright)) / |Sall|
    
    |S| represents the number of samples in subset S
    pk is the proportion of samples with label k in the dataset

Implementation Notes:
- Use only Numpy, Matplotlib, and standard Python libraries
- Other libraries like scikit-learn are NOT allowed
- Implement the tree using dictionaries to store nodes:
  {'attribute': attr, 'value': val, 'left': left_node, 'right': right_node, 'leaf': is_leaf}
  where 'left' and 'right' are also nodes, and 'leaf' is a boolean indicating if the node is terminal

'''

#Import data using np.loadtxt
clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
#noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')

def calculate_information_gain(data, left_indices, right_indices):
    n_samples, n_features = data.shape
    n_classes = len(np.unique(data[:, -1]))
    
    # Calculate dataset entropy
    class_counts = np.bincount(data[:, -1].astype(int), minlength=n_classes)
    class_probs = class_counts / n_samples
    entropy_dataset = -np.sum(class_probs[class_probs > 0] * np.log2(class_probs[class_probs > 0]))

    # Calculate entropy for left and right subsets
    left_class_counts = np.bincount(data[left_indices, -1].astype(int), minlength=n_classes)
    right_class_counts = np.bincount(data[right_indices, -1].astype(int), minlength=n_classes)
    
    left_probs = left_class_counts / len(left_indices)
    right_probs = right_class_counts / len(right_indices)
    
    entropy_left = -np.sum(left_probs[left_probs > 0] * np.log2(left_probs[left_probs > 0]))
    entropy_right = -np.sum(right_probs[right_probs > 0] * np.log2(right_probs[right_probs > 0]))

    # Calculate the remainder
    remainder = (len(left_indices) * entropy_left + len(right_indices) * entropy_right) / n_samples

    # Calculate the information gain
    information_gain = entropy_dataset - remainder

    return information_gain

def find_split(data):
    n_samples, n_features = data.shape
    best_gain = -float('inf')
    best_split = None
    best_feature = None

    for feature in range(n_features - 1):  # Exclude the last column (labels)
        # Sort the data by the current feature
        sorted_indices = data[:, feature].argsort()
        sorted_data = data[sorted_indices]

        # Consider only split points between two examples in sorted order
        for i in range(1, n_samples):
            if sorted_data[i, feature] != sorted_data[i-1, feature]:
                split_point = (sorted_data[i-1, feature] + sorted_data[i, feature]) / 2

                # Calculate the information gain
                left_indices = sorted_indices[:i]
                right_indices = sorted_indices[i:]
                gain = calculate_information_gain(data, left_indices, right_indices)

                # Update best split if necessary
                if gain > best_gain:
                    best_gain = gain
                    best_split = split_point
                    best_feature = feature

    return best_gain, best_split, best_feature

# Update the print statement
print(find_split(clean_data))
