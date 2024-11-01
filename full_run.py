import numpy as np
from utils import confusion_matrix, cross_validate_and_evaluate, calculate_metrics, nested_cross_validation

def main():
    """
    Main function to evaluate the decision tree on clean and noisy datasets.

    It loads the clean and noisy datasets, performs cross-validation, and evaluates the decision tree model.
    After, it prunes both trees and evaluates the pruned trees on the clean and noisy datasets.
    """
    
    #Import data using np.loadtxt
    clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
    noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')

    # Example usage with a specific seed
    seed = 42  # Choose your seed for reproducibility

    # Assuming true_labels_clean and predictions_clean are already obtained
    true_labels_clean, predictions_clean = cross_validate_and_evaluate(clean_data, seed=seed)
    num_classes = np.max(np.unique(true_labels_clean))  # This should be 4

    # Create confusion matrix
    conf_matrix_clean = confusion_matrix(true_labels_clean, predictions_clean, num_classes)
    accuracy_clean, precision_clean, recall_clean, f1_clean = calculate_metrics(conf_matrix_clean)

    # Output results for clean data
    print("Clean Data Evaluation:")
    print("Confusion Matrix:\n", conf_matrix_clean)
    print("Accuracy:", accuracy_clean)
    print("Precision:", precision_clean)
    print("Recall:", recall_clean)
    print("F1-Score:", f1_clean)

    true_labels_noisy, predictions_noisy = cross_validate_and_evaluate(noisy_data, seed=seed)
    conf_matrix_noisy = confusion_matrix(true_labels_noisy, predictions_noisy, num_classes)
    accuracy_noisy, precision_noisy, recall_noisy, f1_noisy = calculate_metrics(conf_matrix_noisy)

    # Output results for noisy data
    print("\nNoisy Data Evaluation:")
    print("Confusion Matrix:\n", conf_matrix_noisy)
    print("Accuracy:", accuracy_noisy)
    print("Precision:", precision_noisy)
    print("Recall:", recall_noisy)
    print("F1-Score:", f1_noisy)

    ####### Prunning the tree #######
    ## Clean data
    print("\nStaring to Prune the tree for clean data:")
    pruned_metrics_clean = nested_cross_validation(clean_data, seed=seed)
    print("Clean Data Metrics After Pruning:")
    print("Confusion Matrix:\n", pruned_metrics_clean[4])
    print("Accuracy:", pruned_metrics_clean[0])
    print("Precision:", pruned_metrics_clean[1])
    print("Recall:", pruned_metrics_clean[2])
    print("F1 Score:", pruned_metrics_clean[3])
    print("Average Depth Before Pruning:", pruned_metrics_clean[5])
    print("Average Depth After Pruning:", pruned_metrics_clean[6])

    ## Noisy data
    print("\nStaring to Prune the tree for noisy data:")
    pruned_metrics_noisy = nested_cross_validation(noisy_data, seed=seed)
    print("\nNoisy Data Metrics After Pruning:")
    print("Confusion Matrix:\n", pruned_metrics_noisy[4])
    print("Accuracy:", pruned_metrics_noisy[0])
    print("Precision:", pruned_metrics_noisy[1])
    print("Recall:", pruned_metrics_noisy[2])
    print("F1 Score:", pruned_metrics_noisy[3])
    print("Average Depth Before Pruning:", pruned_metrics_noisy[5])
    print("Average Depth After Pruning:", pruned_metrics_noisy[6])

if __name__ == "__main__":
    main()