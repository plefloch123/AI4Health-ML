import numpy as np

#Function to calculate information entropy - H(x)
def entropy(dataset):
    """
    Calculate the entropy of a dataset.
    args:
        dataset: a numpy array where each row is a data instance
    returns:
        entropy: the entropy of the dataset
    """
    labels = dataset[:, -1] #last column of dataset
    classes, counts = np.unique(labels, return_counts=True) #sorts y, and count number of each item in y
    probabilities = counts / counts.sum() #probability of each class
    return -np.sum(probabilities * np.log2(probabilities)) #entropy formula


# Function to calculate information gain -IG
def information_gain(dataset, split_attribute, split_value):
    """
    Calculate the information gain achieved by splitting the dataset based on the given attribute and value.
    args:
        dataset: a numpy array where each row is a data instance
        split_attribute: the index of the attribute to split on
        split_value: the value of the attribute to split on
    returns:
        gain: the information gain achieved by the split
    """
    # Split the dataset based on the given split attribute and split value
    left_split = dataset[dataset[:, split_attribute] <= split_value]
    right_split = dataset[dataset[:, split_attribute] > split_value]
    
    # Compute the probabilities of each split
    p_left = len(left_split) / len(dataset)
    p_right = len(right_split) / len(dataset)
    
    # Calculate entropy for the full dataset and each split subset
    gain = entropy(dataset) - (p_left * entropy(left_split) + p_right * entropy(right_split))
    return gain


# Function to find the best split based on information gain
def find_split(data):
    """
    Find the best split for a given dataset using information gain.
    args:
        data: a numpy array where each row is a data instance
    returns:
        best_feature: the index of the best feature to split on
        best_split: the value of the best feature to split on
        best_gain: the information gain achieved by the best split
    """
    n_samples, n_features = data.shape
    best_gain = -float('inf')
    best_feature = None
    best_split = None

    # Iterate over each feature (exclude the last column which is labels)
    for feature in range(n_features - 1):
        # Sort data based on current feature
        sorted_indices = data[:, feature].argsort()
        sorted_data = data[sorted_indices]

        # Loop over possible split points (between consecutive unique values)
        for i in range(1, n_samples):
            if sorted_data[i, feature] != sorted_data[i - 1, feature]:
                # Midpoint between consecutive values as a potential split
                split_value = (sorted_data[i - 1, feature] + sorted_data[i, feature]) / 2

                # Calculate the information gain for the split
                gain = information_gain(data, feature, split_value)

                # Update the best split if this gain is higher
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_split = split_value

    return best_feature, best_split, best_gain


# Function to train a decision tree
def decision_tree_learning(data, depth=0):
    """
    Train a decision tree on the given data.

    args:
        data: a numpy array where each row is a data instance
        depth: the current depth of the tree

    returns:
        node: a dictionary representing the decision tree
    """
    # Base case: If all samples have the same label, return a leaf node
    labels = data[:, -1]  # Last column is assumed to be labels
    if len(np.unique(labels)) == 1:
        # All labels are the same, create a leaf node
        return {'leaf': True, 'label': labels[0], 'depth': depth}

    # Recursive case: Find the best split
    best_feature, best_split, best_gain = find_split(data)
    
    # If no gain, return a leaf node with the most common label
    if best_gain == 0:
        most_common_label = np.bincount(labels.astype(int)).argmax()
        return {'leaf': True, 'label': most_common_label, 'depth': depth}
    
    # Partition data into left and right based on the best split
    left_data = data[data[:, best_feature] <= best_split]
    right_data = data[data[:, best_feature] > best_split]
    
    # Create the current decision node
    node = {
        'leaf': False,
        'attribute': best_feature,
        'value': best_split,
        'left': decision_tree_learning(left_data, depth + 1),
        'right': decision_tree_learning(right_data, depth + 1),
        'depth': depth
    }
    
    return node


def get_tree_depth(node):
    """
    Recursively calculate the depth of the tree.
    :param node: Dictionary containing tree structure
    :return: Integer representing the depth of the tree
    """
    if node["leaf"]:
        return 1
    left_depth = get_tree_depth(node["left"])
    right_depth = get_tree_depth(node["right"])
    return 1 + max(left_depth, right_depth)


# Function to plot the decision tree
def k_fold_split(data, n_folds=10, seed=None):
    """Splits the data into k folds with an optional random seed
    
    args:
        data: a numpy array where each row is a data instance
        n_folds: the number of folds for cross-validation
        seed: an optional random seed for reproducibility
    
    returns:
        folds: a list of numpy arrays containing the indices for each fold
    """
    indices = np.arange(len(data))
    
    if seed is not None:
        np.random.seed(seed)  # Set the random seed for reproducibility
    
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    return folds


# Function to evaluate the decision tree using k-fold cross-validation
def cross_validate_and_evaluate(data, n_folds=10, seed=None):
    """
    Perform k-fold cross-validation on the given dataset and evaluate the decision tree.
    
    args:
        data: a numpy array where each row is a data instance
        n_folds: the number of folds for cross-validation
        seed: an optional random seed for reproducibility
    
    returns:
        all_true_labels: a numpy array of true labels from all folds
        all_predictions: a numpy array of predicted labels from all folds
    """

    folds = k_fold_split(data, n_folds, seed)
    all_predictions = []
    all_true_labels = []

    for i in range(n_folds):
        # Prepare training and testing sets
        test_indices = folds[i]
        train_indices = np.concatenate(folds[:i] + folds[i+1:])  # All other folds
        train_data, test_data = data[train_indices], data[test_indices]
        
        # Train the decision tree
        tree = decision_tree_learning(train_data)  # Your decision tree training function
        
        # Get true labels from the test data
        true_labels = test_data[:, -1]  # Assuming the last column is the label
        predictions = classify(test_data, tree)  # Your classification function
        
        all_predictions.extend(predictions)
        all_true_labels.extend(true_labels)

    return np.array(all_true_labels), np.array(all_predictions)


# Function to predict the class label for a single data instance using the decision tree
def predict(row, tree):
    """
    Predict the class label for a single data instance using the decision tree.
    
    :param row: The data instance to classify (as a numpy array)
    :param tree: The decision tree (as a dictionary)
    :return: The predicted class label
    """
    if tree['leaf']:  # Check if it's a leaf node
        return tree['label']  # Return the label at the leaf node
    
    # Get the attribute and value for the current node
    attribute = tree['attribute']
    value = tree['value']
    
    # Decide which branch to follow based on the row's attribute value
    if row[attribute] <= value:
        return predict(row, tree['left'])  # Go left
    else:
        return predict(row, tree['right'])  # Go right


# Function to classify a dataset using the trained decision tree
def classify(data, tree):
    """
    Classify a dataset using the trained decision tree.
    args:
        data: a numpy array where each row is a data instance
        tree: a dictionary representing the decision tree
    returns:
        predictions: a list of predicted class labels
    """
    predictions = []
    for row in data:
        predictions.append(predict(row, tree))  # Implement the predict function based on your tree structure
    return predictions



def confusion_matrix(true_labels, predictions, num_classes):
    """Compute confusion matrix as a 2D numpy array.
    
    args:
        true_labels: a list of true labels
        predictions: a list of predicted labels
        num_classes: number of classes in the classification problem

    returns:
        matrix: a 2D numpy array representing the confusion matrix
    """

    # Convert the labels and predictions to numpy arrays into integers
    true_labels = np.array(true_labels, dtype=int)
    predictions = np.array(predictions, dtype=int)
    num_classes = int(num_classes)

    # Create the confusion matrix
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Check if the true and predicted values are in the valid range
    for true, pred in zip(true_labels, predictions):
        if true < 1 or true > num_classes or pred < 1 or pred > num_classes:
            raise ValueError(f"Label or prediction out of bounds: true={true}, pred={pred}")
        
        # Subtract 1 from each value to get the 0-based index since the preds and true 
        # are from 1-4 and we need 0-3 for matrix index
        matrix[true-1, pred-1] += 1

    return matrix


def calculate_metrics(conf_matrix):
    """Calculate accuracy, precision, recall, and F1-score from confusion matrix.
    
    args:
        conf_matrix: a 2D numpy array representing the confusion matrix
        
    returns:
        accuracy: the accuracy of the model
        precision: a numpy array of precision values for each class
        recall: a numpy array of recall values for each class
        f1: a numpy array of F1-score values for each class
    """
    num_classes = conf_matrix.shape[0]
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for i in range(num_classes):
        precision[i] = conf_matrix[i, i] / np.sum(conf_matrix[:, i]) if np.sum(conf_matrix[:, i]) > 0 else 0
        recall[i] = conf_matrix[i, i] / np.sum(conf_matrix[i, :]) if np.sum(conf_matrix[i, :]) > 0 else 0
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    return accuracy, precision, recall, f1


def prune_tree(node, validation_data):
    """Recursively prune the decision tree.
    
    args:
        node: a dictionary representing the decision tree node
        validation_data: a numpy array of validation data instances

    returns:
        node: the pruned decision tree node
    """
    # Base case: If the node is a leaf, return it as is
    if node.get('leaf', False):
        return node
    
    # Recursively prune left and right subtrees if they exist
    if 'left' in node:
        node['left'] = prune_tree(node['left'], validation_data)
    if 'right' in node:
        node['right'] = prune_tree(node['right'], validation_data)
    
    # Calculate error without pruning
    error_without_pruning = compute_validation_error(validation_data, node)
    
    # Create a leaf node with most common class
    predictions = classify(validation_data, node)
    # Convert predictions list to numpy array before using astype
    predictions = np.array(predictions)
    most_common_label = np.bincount(predictions.astype(int)).argmax()
    leaf_node = {'leaf': True, 'label': most_common_label}
    
    # Calculate error with pruning
    error_with_pruning = compute_validation_error(validation_data, leaf_node)
    
    # If pruning improves or maintains accuracy, return leaf node
    if error_with_pruning <= error_without_pruning:
        return leaf_node
    
    # Otherwise keep the subtree
    return node


def compute_validation_error(validation_data, node):
    """Compute the validation error for the given node.
    
    args:
        validation_data: a numpy array of validation data instances
        node: a dictionary representing the decision tree node
        
    returns:
        error: the validation error of the decision tree on the validation data
    """
    predictions = classify(validation_data, node)
    true_labels = validation_data[:, -1].astype(int)
    error = np.mean(predictions != true_labels)
    return error


def nested_cross_validation(data, num_outer_folds=10, num_inner_folds=9, seed=None):
    """Perform nested cross-validation on the given dataset.

    args:
        data: a numpy array where each row is a data instance
        num_outer_folds: the number of outer folds for cross-validation
        num_inner_folds: the number of inner folds for cross-validation
        seed: an optional random seed for reproducibility
    
    returns:
        mean_accuracy: the mean accuracy across all outer and inner folds
        mean_precision: the mean precision across all outer and inner folds (per class)
        mean_recall: the mean recall across all outer and inner folds (per class)
        mean_f1: the mean F1-score across all outer and inner folds (per class)
        cum_conf_matrix: the cumulative confusion matrix across all outer folds
        avg_depth_before_pruning: the average depth of the tree before pruning
        avg_depth_after_pruning: the average depth of the tree after pruning
    """
    # Split data into 10 outer folds
    outer_folds = k_fold_split(data, n_folds=num_outer_folds, seed=seed)
    
    # Lists to store per-class metrics across all outer and inner folds
    all_test_accuracies = []
    all_test_precisions = []
    all_test_recalls = []
    all_test_f1_scores = []

    # Lists to store depths before and after pruning
    depths_before_pruning = []
    depths_after_pruning = []

    # Determine the number of classes in the dataset
    num_classes = len(np.unique(data[:, -1]))

    # Initialize a cumulative confusion matrix with zeros
    cum_conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # Iterate over each outer fold as the test set
    for outer_idx in range(num_outer_folds):
        test_idx = outer_folds[outer_idx]  # Outer fold test data indices
        train_val_folds = [outer_folds[i] for i in range(num_outer_folds) if i != outer_idx]  # Remaining folds for train/validate
        
        test_data = data[test_idx]
        train_val_data = data[np.concatenate(train_val_folds)]

        # Split train/validate data into 9 inner folds for inner cross-validation
        inner_folds = k_fold_split(train_val_data, n_folds=num_inner_folds, seed=seed)
        
        # Perform inner cross-validation (train and validate on 9 splits)
        for inner_idx in range(num_inner_folds):
            # Select one inner fold for validation and the rest for training
            inner_val_idx = inner_folds[inner_idx]
            inner_train_idx = np.concatenate([inner_folds[j] for j in range(num_inner_folds) if j != inner_idx])
            
            inner_train_data = train_val_data[inner_train_idx]
            inner_val_data = train_val_data[inner_val_idx]

            # Train the decision tree on the inner training data
            tree = decision_tree_learning(inner_train_data)

            # Calculate depth before pruning
            depth_before = get_tree_depth(tree)
            depths_before_pruning.append(depth_before)

            # Prune the tree based on inner validation data
            tree = prune_tree(tree, inner_val_data)

            # Calculate depth after pruning
            depth_after = get_tree_depth(tree)
            depths_after_pruning.append(depth_after)

            # Evaluate on the outer test set
            predictions = classify(test_data, tree)
            true_labels = test_data[:, -1].astype(int)

            # Calculate metrics for each class
            conf_matrix = confusion_matrix(true_labels, predictions, num_classes=num_classes)
            accuracy, precision, recall, f1 = calculate_metrics(conf_matrix)

            # Append metrics per class for each fold
            all_test_accuracies.append(accuracy)
            all_test_precisions.append(precision)
            all_test_recalls.append(recall)
            all_test_f1_scores.append(f1)

            # Add the confusion matrix for this fold to the cumulative confusion matrix
            cum_conf_matrix += conf_matrix

    # Calculate the average depths before and after pruning
    avg_depth_before_pruning = np.mean(depths_before_pruning)
    avg_depth_after_pruning = np.mean(depths_after_pruning)

    # Calculate the final averaged results across all 90 evaluations per class
    mean_accuracy = np.mean(all_test_accuracies)
    mean_precision = np.mean(all_test_precisions, axis=0)
    mean_recall = np.mean(all_test_recalls, axis=0)
    mean_f1 = np.mean(all_test_f1_scores, axis=0)

    return (mean_accuracy, mean_precision, mean_recall, mean_f1, cum_conf_matrix,
            avg_depth_before_pruning, avg_depth_after_pruning)