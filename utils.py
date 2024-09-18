import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path):
    """
    Load and preprocess data from a CSV file.

    Parameters:
    - file_path: Path to the CSV file containing the dataset.

    Returns:
    - X: Features matrix (numpy array).
    - y: Target vector (numpy array, reshaped as a column vector).
    """
    # Load data from the specified CSV file
    data = pd.read_csv(file_path)
    
    # Separate features (all columns except the last) and target (last column)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Convert target labels to binary numeric values
    # Assuming 'goodware' is labeled as 0 and 'malware' as 1
    y = np.where(y == 'goodware', 0, 1)  # Adjust this according to your dataset
    
    # Ensure y is a column vector
    return X, y.reshape(-1, 1) 

import matplotlib.pyplot as plt
import os

def save_elapsed_time(elapsed_time, mode, folder_path, filename=None):
    """
    Save the elapsed time as an image in the specified folder.

    Parameters:
    - elapsed_time: The time elapsed during the process
    - folder_path: Path to the folder where the image will be saved
    - filename: Name of the image file to save. If None, the filename will be based on the elapsed time.
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Set default filename if not provided
    if filename is None:
        filename = f'{elapsed_time:.2f}_seconds.png'
    
    # Full path to save the file
    file_path = os.path.join(folder_path, filename)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis('off')  # Turn off the axis

    # Add text to the figure
    text = f'Elapsed Time for {mode}: {elapsed_time:.2f} seconds'
    ax.text(0.5, 0.5, text, fontsize=12, ha='center', va='center', weight='bold')

    # Save the plot as an image
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()



def hiddnLs_train_val_loss(results_df, folder_path, filename='train_val_loss_table.png'):
    """
    Save the results DataFrame as an image with the lowest validation loss bold,
    in a folder named 'pre-training'.

    Parameters:
    - results_df: DataFrame containing the results for each hidden size pair
    - filename: Name of the image file to save
    """
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Full path to save the file
    file_path = os.path.join(folder_path, filename)
    
    # Find the index of the row with the lowest validation loss
    min_loss_idx = results_df['Validation Loss'].idxmin()
    
    fig, ax = plt.subplots(figsize=(12, len(results_df) * 0.4))  # Adjust figure size
    ax.axis('off')

    # Create a table and add it to the plot
    table = ax.table(cellText=results_df.values,
                     colLabels=results_df.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])

    # Iterate through the table cells to apply formatting
    for (i, j), cell in table._cells.items():
        if i == 0:  # Skip the header row
            continue
        
        # Check if the cell is in the row with the minimum validation loss
        if i == min_loss_idx + 1:  # +1 because header row is at index 0
            cell.set_fontsize(12)  # Adjust font size if needed
            cell.set_text_props(weight='bold')

    # Save the plot as an image
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()



def plot_validation_loss(epochs, losses, folder_path):
    """
    Plot validation loss over epochs and save the plot as an image.

    Parameters:
    - epochs: List or numpy array of epoch numbers.
    - losses: List or numpy array of validation losses recorded during training.
    - folder_path: Directory path to save the plot image.
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the file path for the plot image
    filename = os.path.join(folder_path, 'validation_loss.png')

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, color='blue', lw=2, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Over Epochs')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_accuracy(epochs, accuracies, folder_path):
    """
    Plot accuracy over epochs and save the plot as an image.

    Parameters:
    - epochs: List or numpy array of epoch numbers.
    - accuracies: List or numpy array of accuracy scores recorded during training.
    - folder_path: Directory path to save the plot image.
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the file path for the plot image
    filename = os.path.join(folder_path, 'accuracy.png')

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, color='green', lw=2, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_precision(epochs, precisions, folder_path):
    """
    Plot precision over epochs and save the plot as an image.

    Parameters:
    - epochs: List or numpy array of epoch numbers.
    - precisions: List or numpy array of precision scores recorded during training.
    - folder_path: Directory path to save the plot image.
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the file path for the plot image
    filename = os.path.join(folder_path, 'precision.png')

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, precisions, color='orange', lw=2, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Precision Over Epochs')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_recall(epochs, recalls, folder_path):
    """
    Plot recall over epochs and save the plot as an image.

    Parameters:
    - epochs: List or numpy array of epoch numbers.
    - recalls: List or numpy array of recall scores recorded during training.
    - folder_path: Directory path to save the plot image.
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the file path for the plot image
    filename = os.path.join(folder_path, 'recall.png')

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, recalls, color='red', lw=2, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Recall Over Epochs')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_f1_score(epochs, f1_scores, folder_path):
    """
    Plot F1 score over epochs and save the plot as an image.

    Parameters:
    - epochs: List or numpy array of epoch numbers.
    - f1_scores: List or numpy array of F1 scores recorded during training.
    - folder_path: Directory path to save the plot image.
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the file path for the plot image
    filename = os.path.join(folder_path, 'f1_score.png')

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, f1_scores, color='purple', lw=2, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Over Epochs')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()



def plot_confusion_matrix(cm, class_names, folder_path):
    """
    Plot a confusion matrix as a heatmap and save it as an image.

    Parameters:
    - cm: Confusion matrix (numpy array).
    - class_names: List of class names (strings) for labels.
    - folder_path: Directory path to save the confusion matrix image.
    """
    plt.figure(figsize=(8, 6))
    
    # Create a heatmap with the confusion matrix data
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
    
    # Label the axes and add a title
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # Define the filename and save the plot
    filename = os.path.join(folder_path, 'confusion_matrix.png')
    plt.savefig(filename)
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, folder_path):
    """
    Plot the ROC curve and save it as an image.

    Parameters:
    - fpr: False positive rates (list or numpy array).
    - tpr: True positive rates (list or numpy array).
    - roc_auc: Area under the ROC curve (float).
    - folder_path: Directory path to save the ROC curve image.
    """
    plt.figure(figsize=(8, 6))
    
    # Plot the ROC curve with the specified false positive rate, true positive rate, and AUC
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    
    # Add a diagonal line for random guessing
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    # Set axis limits
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Label the axes and add a title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    
    # Add a legend and save the plot
    plt.legend(loc='lower right')
    
    # Define the filename and save the plot
    filename = os.path.join(folder_path, 'roc_curve.png')
    plt.savefig(filename)
    plt.close()

def generate_metrics_report(test_loss, accuracy, precision, recall, f1, folder_path):
    """
    Generate a metrics report including textual metrics.

    Parameters:
    - test_loss: Loss on the test set.
    - accuracy: Accuracy score on the test set.
    - precision: Precision score on the test set.
    - recall: Recall score on the test set.
    - f1: F1 score on the test set.
    - folder_path: Directory path to save the metrics report image.
    """
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Define the file path for the metrics report image
    filename = os.path.join(folder_path, 'metrics_report.png')

    # Create a figure
    plt.figure(figsize=(8, 6))
    
    # Display Metrics
    metrics_text = (
        f'Test Loss: {test_loss:.4f}\n'
        f'Accuracy: {accuracy:.4f}\n'
        f'Precision: {precision:.4f}\n'
        f'Recall: {recall:.4f}\n'
        f'F1 Score: {f1:.4f}'
    )
    plt.text(0.5, 0.5, metrics_text, fontsize=14, va='center', ha='center')
    plt.axis('off')
    plt.title('Metrics Report')
    
    # Save the metrics report image
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
