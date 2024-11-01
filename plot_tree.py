import numpy as np
import matplotlib.pyplot as plt
from utils import decision_tree_learning

def plot_tree(node, x=0.5, y=1, level=1, spacing=0.3, ax=None, width_multiplier=50, height_multiplier=2.5):
    """
    Recursively plots the decision tree with improved spacing.
    :param node: Dictionary containing tree structure
    :param x: X-coordinate of the current node
    :param y: Y-coordinate of the current node
    :param level: Depth level of the node in the tree
    :param spacing: Horizontal spacing for child nodes
    :param ax: Matplotlib Axes object
    :param width_multiplier: Controls horizontal scaling of the plot
    :param height_multiplier: Controls vertical scaling of the plot
    """
    # Initialize the figure and axes at the root level
    if ax is None:
        max_depth = get_tree_depth(node)  # Calculate the depth of the tree
        fig, ax = plt.subplots(figsize=(width_multiplier, height_multiplier * max_depth))
        ax.axis("off")

    # Check if the node is a leaf
    if node["leaf"]:
        ax.text(x, y, f"Leaf: {node['label']}", 
                ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", 
                                                    edgecolor="black", facecolor="lightblue"))
    else:
        # Plot current node with split info
        ax.text(x, y, f"{node['attribute']} <= {node['value']:.2f}",
                ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", 
                                                    edgecolor="black", facecolor="lightgreen"))
        
        # Adjust horizontal spacing based on the level
        adjusted_spacing = spacing / (level + 1)
        
        # Calculate child node positions
        left_x = x - adjusted_spacing
        right_x = x + adjusted_spacing
        child_y = y - 0.15
        
        # Draw the branches
        ax.plot([x, left_x], [y, child_y], "k-", lw=1)
        ax.plot([x, right_x], [y, child_y], "k-", lw=1)
        
        # Recursively plot left and right children
        plot_tree(node["left"], x=left_x, y=child_y, level=level + 1, spacing=spacing, ax=ax, 
                  width_multiplier=width_multiplier, height_multiplier=height_multiplier)
        plot_tree(node["right"], x=right_x, y=child_y, level=level + 1, spacing=spacing, ax=ax, 
                  width_multiplier=width_multiplier, height_multiplier=height_multiplier)

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

# Example usage with the generated decision tree
if __name__ == "__main__":
    # Load the clean and noisy datasets
    clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
    noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')

    # Plot the decision tree for the clean dataset
    tree = decision_tree_learning(clean_data)
    plot_tree(tree)
    plt.show()  # Show plot after plotting clean dataset

    # Plot the decision tree for the noisy dataset
    tree = decision_tree_learning(noisy_data)
    plot_tree(tree)
    plt.show()  # Show plot after plotting noisy dataset
