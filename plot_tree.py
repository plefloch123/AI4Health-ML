import matplotlib.pyplot as plt

def plot_tree(tree, depth=0, pos=None, parent_pos=None, ax=None, x_offset=0.5, y_offset=1.0):
    if pos is None:
        pos = {id(tree): (0.5, 1.0)}  # The root is centered at (0.5, 1.0)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_axis_off()  # Hide axis
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Draw connection between parent node and current node
    if parent_pos is not None:
        ax.plot([parent_pos[0], pos[id(tree)][0]], [parent_pos[1], pos[id(tree)][1]], 'k-', lw=2)  # Draw line
    
    # Check if it is a leaf node
    if 'label' in tree:
        node_text = f"Leaf\nLabel: {tree['label']}\nDepth: {tree['depth']}"
        ax.text(pos[id(tree)][0], pos[id(tree)][1], node_text, ha='center', va='center', 
                bbox=dict(facecolor='lightgreen', edgecolor='black'))
    else:
        # Not a leaf node, plot attribute and split value
        node_text = f"Node\nAttr: {tree['attribute']}\nVal: {tree['value']}\nDepth: {tree['depth']}"
        ax.text(pos[id(tree)][0], pos[id(tree)][1], node_text, ha='center', va='center', 
                bbox=dict(facecolor='lightblue', edgecolor='black'))

    # Plot the left and right children
    left_child = tree.get('left')
    right_child = tree.get('right')

    if left_child is not None:
        # Set position for the left child
        pos[id(left_child)] = (pos[id(tree)][0] - x_offset / (depth + 1), pos[id(tree)][1] - y_offset)
        plot_tree(left_child, depth + 1, pos, pos[id(tree)], ax, x_offset, y_offset)

    if right_child is not None:
        # Set position for the right child
        pos[id(right_child)] = (pos[id(tree)][0] + x_offset / (depth + 1), pos[id(tree)][1] - y_offset)
        plot_tree(right_child, depth + 1, pos, pos[id(tree)], ax, x_offset, y_offset)

    # If it's the root call, show the plot
    if parent_pos is None:
        plt.show()