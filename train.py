from neural_network import NeuralNetwork
from sklearn.metrics import confusion_matrix
from utils import *
import numpy as np
import pandas as pd
import time

def cross_validate(hidden_sizes1, hidden_sizes2, X_train, y_train, X_val, y_val, epochs=15000, batch_size=32):
    """
    Perform cross-validation to find the best hidden layer sizes for the neural network with two hidden layers.

    Parameters:
    - hidden_sizes1: List of possible sizes for the first hidden layer
    - hidden_sizes2: List of possible sizes for the second hidden layer
    - X_train: Training features
    - y_train: Training labels
    - X_val: Validation features
    - y_val: Validation labels
    - epochs: Number of epochs to train each model
    - batch_size: Size of the mini-batches for training

    Returns:
    - best_hidden_size1: Best size for the first hidden layer
    - best_hidden_size2: Best size for the second hidden layer
    """
    best_hidden_size1 = None
    best_hidden_size2 = None
    best_avg_val_loss = float('inf')

    results= []

    # Iterate over all combinations of hidden sizes
    for hidden_size1 in hidden_sizes1:
        for hidden_size2 in hidden_sizes2:
            print(f'Testing hidden sizes: ({hidden_size1}, {hidden_size2})')

            # Initialize the neural network with the current hidden sizes
            nn = NeuralNetwork(input_size=X_train.shape[1], hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=1, learning_rate=0.001)
            
            # Train the neural network on the training data and validate on the validation data
            nn.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, training=False)

            # Evaluate the performance on the validation set
            train_predictions = nn.forward(X_train)
            train_loss = nn.calculate_loss(train_predictions, y_train)

            # Evaluate the performance on the validation set
            val_predictions = nn.forward(X_val)
            val_loss = nn.calculate_loss(val_predictions, y_val)
            
            print(f'Validation loss for hidden sizes ({hidden_size1}, {hidden_size2}): {val_loss}')

            # Store results
            results.append({
                'Hidden Layer 1 Size': int(hidden_size1),
                'Hidden Layer 2 Size': int(hidden_size2),
                'Train Loss': train_loss,
                'Validation Loss': val_loss
            })

            # Update the best hidden sizes based on validation loss
            if val_loss < best_avg_val_loss:
                best_avg_val_loss = val_loss
                best_hidden_size1 = hidden_size1
                best_hidden_size2 = hidden_size2
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    return best_hidden_size1, best_hidden_size2, results_df

def main():
    """
    Main function to load data, perform cross-validation, train the final model, and save it.
    """
    # Paths to data files
    train_file = 'data/train_data.csv'
    val_file = 'data/val_data.csv'
    test_file = 'data/test_data.csv'

    # Load datasets
    X_train, y_train = load_data(train_file)
    X_val, y_val = load_data(val_file)

    # Define ranges of hidden layer sizes to test
    hidden_sizes1 = [5, 6, 7, 8, 9, 10, 11, 12, 13]
    hidden_sizes2 = [1, 2, 3, 4, 5]

    # Start time for cross-validation
    start_time = time.time()

    # Perform cross-validation to find the best hidden layer sizes
    best_hidden_size1, best_hidden_size2, results_df = cross_validate(hidden_sizes1, hidden_sizes2, X_train, y_train, X_val, y_val)

    # End time for cross-validation
    end_time = time.time()

    # Calculate and save elapsed time
    elapsed_time = end_time - start_time
    save_elapsed_time(elapsed_time, "Finding Best Hidden Layer Sizes", "pre-training")

    print(f'Best hidden sizes: ({best_hidden_size1}, {best_hidden_size2})')
    print(f'Elapsed time for cross-validation: {elapsed_time:.2f} seconds')
    hiddnLs_train_val_loss(results_df, "pre-training")

    # Create and train the final neural network with the best hidden layer sizes
    nn = NeuralNetwork(input_size=X_train.shape[1], hidden_size1=best_hidden_size1, hidden_size2=best_hidden_size2, output_size=1, learning_rate=0.001)

    # Start time for training
    start_time = time.time()

    nn.train(X_train, y_train, X_val, y_val, epochs=15000, batch_size=32, training=True)

    # End time for training
    end_time = time.time()

    # Calculate and save elapsed time
    elapsed_time = end_time - start_time
    save_elapsed_time(elapsed_time, "Training", "training")

    print(f'Elapsed time for training: {elapsed_time:.2f} seconds')

    # Save the trained model to a file
    model_file = 'detect_malwares.pkl'
    nn.save_model(model_file)
    print(f'Model saved to {model_file}')

if __name__ == "__main__":
    main()
