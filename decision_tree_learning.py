def decision_tree_learning(dataset, depth=5):
    labels = dataset[:, -1]  # Last column is the label
    
    # Base case: if all labels are the same, return a leaf node
    if len(np.unique(labels)) == 1:
        return {'label': labels[0], 'depth': depth}
    
    # Otherwise, find the best split
    attribute, value = find_split(dataset)
    if attribute is None:
        # If no split is possible, return a leaf with the majority label
        unique, counts = np.unique(labels, return_counts=True)
        majority_label = unique[np.argmax(counts)]
        return {'label': majority_label, 'depth': depth}
    
    # Split the dataset
    left_split = dataset[dataset[:, attribute] <= value]
    right_split = dataset[dataset[:, attribute] > value]
    
    # Create a new decision node as a dictionary
    tree = {
        'attribute': attribute,
        'value': value,
        'depth': depth,
        'left': decision_tree_learning(left_split, depth + 1),
        'right': decision_tree_learning(right_split, depth + 1)
    }
    
    return tree