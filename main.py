import numpy as np
import matplotlib.pyplot as plt
from utils import confusion_matrix, cross_validate_and_evaluate, calculate_metrics, nested_cross_validation
from plot_tree import plot_tree  # Import function for plotting the tree
from plot_confusion_matrix import plot_confusion_matrix  # Import function for plotting the confusion matrix

def main(dataset_type, prune=False, seed=42, plot_conf_matrix_option=False):
    """
    Main function to evaluate the decision tree on a chosen dataset (clean or noisy).

    Args:
        dataset_type (str): 'clean' or 'noisy' to specify the dataset.
        prune (bool): Whether to prune the decision tree.
        seed (int): Random seed for reproducibility.
        plot_conf_matrix_option (bool): Whether to plot the confusion matrix.
    """
    # Set dataset path based on user choice
    dataset_path = f"wifi_db/{dataset_type}_dataset.txt"
    
    # Load dataset
    data = np.loadtxt(dataset_path)

    # Evaluate and optionally prune the dataset
    evaluate_and_plot(data, dataset_type.capitalize(), seed, prune, plot_conf_matrix_option)

def evaluate_and_plot(data, dataset_name, seed, prune, plot_conf_matrix_option):
    """
    Evaluate a dataset, compute and plot metrics, optionally prune the model, display metrics, and plot the tree/confusion matrix.

    Args:
        data (np.ndarray): The dataset to evaluate.
        dataset_name (str): Name of the dataset (e.g., "Clean" or "Noisy").
        seed (int): Random seed for reproducibility.
        prune (bool): Whether to apply pruning.
        plot_conf_matrix_option (bool): Whether to plot the confusion matrix.
    """
    # Evaluate initial metrics
    true_labels, predictions = cross_validate_and_evaluate(data, seed=seed)
    num_classes = np.max(np.unique(true_labels))
    conf_matrix = confusion_matrix(true_labels, predictions, num_classes)
    accuracy, precision, recall, f1 = calculate_metrics(conf_matrix)

    # Display initial metrics
    print(f"\n{dataset_name} Data Evaluation (Before Pruning):")
    print_metrics(conf_matrix, accuracy, precision, recall, f1)
    
    # Optionally plot the confusion matrix
    if plot_conf_matrix_option:
        plot_confusion_matrix(conf_matrix, title=f"{dataset_name} Confusion Matrix (Before Pruning)")

    # If pruning is requested, evaluate pruned metrics
    if prune:
        pruned_metrics = nested_cross_validation(data, seed=seed)
        
        # Display pruned metrics
        print(f"\n{dataset_name} Data Evaluation (After Pruning):")
        print_metrics(pruned_metrics[4], pruned_metrics[0], pruned_metrics[1], pruned_metrics[2], pruned_metrics[3])
        
        # Optionally plot the pruned confusion matrix
        if plot_conf_matrix_option:
            plot_confusion_matrix(pruned_metrics[4], title=f"{dataset_name} Pruned Confusion Matrix")

        # Print tree depth information and optionally plot the pruned tree
        print("Average Depth Before Pruning:", pruned_metrics[5])
        print("Average Depth After Pruning:", pruned_metrics[6])

def print_metrics(conf_matrix, accuracy, precision, recall, f1):
    """
    Print metrics including the confusion matrix, accuracy, precision, recall, and F1 score.

    Args:
        conf_matrix (np.ndarray): Confusion matrix.
        accuracy (float): Accuracy of the model.
        precision (float): Precision of the model.
        recall (float): Recall of the model.
        f1 (float): F1 score of the model.
    """
    print("Confusion Matrix:\n", conf_matrix)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

if __name__ == "__main__":
    # Prompt user for dataset type (clean or noisy)
    dataset_type = input("Enter dataset type ('clean' or 'noisy'): ").strip().lower()
    while dataset_type not in ["clean", "noisy"]:
        dataset_type = input("Invalid input. Please enter 'clean' or 'noisy': ").strip().lower()

    # Prompt user for pruning choice
    prune_choice = input("Do you want to prune the tree? (True/False): ").strip().capitalize()
    prune = True if prune_choice == "True" else False

    # Prompt user for random seed
    seed = int(input("Enter the random seed (default 42): ").strip() or 42)

    # Prompt user for confusion matrix plotting choice
    plot_conf_matrix_choice = input("Do you want to plot the confusion matrix? (True/False): ").strip().capitalize()
    plot_conf_matrix_option = True if plot_conf_matrix_choice == "True" else False

    # Run the main function with user-specified arguments
    main(dataset_type, prune, seed, plot_conf_matrix_option)
