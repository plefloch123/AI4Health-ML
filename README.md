# Decision Tree Coursework

This project implements a decision tree model for classification on clean and noisy datasets. The codebase includes functions for training and evaluating the decision tree, visualizing the results through confusion matrices and tree plots, and pruning the trees for optimized performance. Each module is designed to support specific tasks, and we provide multiple ways to run evaluations and visualizations.

## Repository Structure

Here's a breakdown of each file in the repository:

- **`utils.py`**: Contains utility functions used throughout the project (All the functions to create the trees, prune the trees and calulcate the confusion matrices are here)

- **`full_run.py`**: Runs the entire pipeline on both the clean and noisy datasets. This script:
  1. Loads the clean and noisy datasets.
  2. Trains and evaluates the decision trees.
  3. Outputs confusion matrices for both datasets.
  4. Prunes the decision trees.
  5. Evaluates the pruned trees and outputs the confusion matrices.

- **`plot_confusion_matrix.py`**: Plots a confusion matrix for a both dataset.

- **`plot_tree.py`**: Visualizes the decision tree structure for a given dataset.

- **`main.py`**: The primary script that will ask the user what he wants to decide such has:
  1. Load the specified datasets (clean or noisy).
  2. Do the prunning (True or False).
  3. The seed number.
  4. plt the confusion matrices (True or False)

- **`coursework.ipynb`**: A Jupyter notebook detailing the development process for the coursework. It includes explanations, code for each function, and final results with visualizations.

## Usage Guide

### Running the Code

1. **Running Full Evaluation and Pruning**:
   - To run the entire pipeline on both datasets (clean and noisy) and display metrics and confusion matrices, use `full_run.py`:

     ```bash
     python full_run.py
     ```

3. **Plotting the Confusion Matrix**:
   - Use `plot_confusion_matrix.py` to visualize the confusion matrix for both dataset.

     ```bash
     python plot_confusion_matrix.py
     ```

4. **Visualizing the Decision Tree**:
   - Use `plot_tree.py` to display the structure of the decision trees trained on a clean and noisey dataset.

     ```bash
     python plot_tree.py
     ```

5. **Running the Main Script with Arguments**:
   - The `main.py` script allows for flexible evaluation. It gives option to choose the seed, the dataset, to plot or not the confusion matrix and to prune the tree or not

     ```bash
     python main.py
     ```

6. **Viewing the Coursework Notebook**:
   - To review the full development process, open `coursework.ipynb` in Jupyter Notebook
