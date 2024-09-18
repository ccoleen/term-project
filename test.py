import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from neural_network import NeuralNetwork
from utils import plot_confusion_matrix, plot_roc_curve, load_data, generate_metrics_report

def test_model(model_file, test_file):
    """
    Test the neural network model using the provided test dataset.
    
    Parameters:
    - model_file: Path to the file where the trained model is saved
    - test_file: Path to the CSV file containing the test data
    
    This function:
    1. Loads the test dataset
    2. Creates a neural network instance with the same architecture
    3. Loads the saved model parameters
    4. Tests the neural network and saves performance metrics
    5. Plots the confusion matrix and ROC curve
    """
    # Load the test dataset
    X_test, y_test = load_data(test_file)
    print(f'Test data loaded from {test_file}')

    # Create a NeuralNetwork instance with the same architecture as during training
    # Adjust the hidden layer sizes based on your training configuration
    hidden_size1 = 20  # Set this to the size of the first hidden layer used during training
    hidden_size2 = 10  # Set this to the size of the second hidden layer used during training
    nn_loaded = NeuralNetwork(input_size=X_test.shape[1], hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=1, learning_rate=0.001)
    
    # Load the saved model parameters
    nn_loaded.load_model(model_file)
    print(f'Model loaded from {model_file}')

    # Test the neural network on the test dataset
    test_loss, test_accuracy, test_precision, test_recall, test_f1, fpr, tpr, roc_auc = nn_loaded.test(X_test, y_test)
    
    # Print test results
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall: {test_recall:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')
    
    # Compute confusion matrix
    predictions = (nn_loaded.forward(X_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, predictions)
    
    # Generate the confusion matrix
    print('Plotting confusion matrix...')
    plot_confusion_matrix(cm, ['Goodware', 'Malware'], "testing")
    
    # Generate the ROC curve
    print('Plotting ROC curve...')
    plot_roc_curve(fpr, tpr, roc_auc, "testing")

    # Generate the evaluation report
    print('Generating evaluation report...')
    predictions = (nn_loaded.forward(X_test) > 0.5).astype(int)
    generate_metrics_report(test_loss, test_accuracy, test_precision, test_recall, test_f1, "testing")

    print('See reports at >testing<')

if __name__ == "__main__":
    # Define the file paths
    model_file = 'detect_malwares.pkl' 
    test_file = 'data/test_data.csv'
    
    # Call the test function
    test_model(model_file, test_file)
