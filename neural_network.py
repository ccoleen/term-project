import numpy as np
import pickle
from utils import plot_validation_loss, plot_accuracy, plot_precision, plot_recall, plot_f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

def initialize_biases(size):
    """
    Initialize biases with -1, 0, or 1, where half of the biases are -1,
    half are 1, and if the size is odd, one bias is 0.

    Parameters:
    - size: Tuple representing the shape of the biases (e.g., (1, hidden_size1)).

    Returns:
    - biases: Initialized biases (numpy array).
    """
    total_elements = np.prod(size)

    # Calculate the number of -1s, 1s, and 0s
    num_neg_ones = total_elements // 2
    num_ones = num_neg_ones
    
    # If the total number of elements is odd, add one 0
    if total_elements % 2 != 0:
        num_zeros = 1
    else:
        num_zeros = 0
    
    # Create an array with the required number of -1s, 1s, and 0s
    biases = np.concatenate([np.full(num_neg_ones, -1.0),
                              np.full(num_ones, 1.0),
                              np.full(num_zeros, 0.0)])
    
    # Shuffle the biases to randomize their positions
    np.random.shuffle(biases)

    # Reshape to the specified size
    return biases.reshape(size)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate):
        """
        Initialize the neural network with given parameters for two hidden layers.
        
        Parameters:
        - input_size: Number of input features
        - hidden_size1: Number of neurons in the first hidden layer
        - hidden_size2: Number of neurons in the second hidden layer
        - output_size: Number of output neurons (usually 1 for binary classification)
        - learning_rate: Learning rate for the gradient descent algorithm
        """
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights with Xavier initialization for better convergence
        self.weights_input_hidden1 = np.random.randn(self.input_size, self.hidden_size1) * np.sqrt(1. / self.input_size)
        self.weights_hidden1_hidden2 = np.random.randn(self.hidden_size1, self.hidden_size2) * np.sqrt(1. / self.hidden_size1)
        self.weights_hidden2_output = np.random.randn(self.hidden_size2, self.output_size) * np.sqrt(1. / self.hidden_size2)

        # Initialize biases with -1, 0, or 1
        self.bias_hidden1 = initialize_biases((1, self.hidden_size1))
        self.bias_hidden2 = initialize_biases((1, self.hidden_size2))
        self.bias_output = initialize_biases((1, self.output_size))

    def sigmoid(self, x):
        """
        Compute the sigmoid activation function.
        
        Parameters:
        - x: Input values
        
        Returns:
        - Sigmoid of the input values
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Compute the derivative of the sigmoid function.
        
        Parameters:
        - x: Input values
        
        Returns:
        - Derivative of the sigmoid function
        """
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, X):
        """
        Perform the forward pass through the network.
        
        Parameters:
        - X: Input data
        
        Returns:
        - Final output of the network
        """
        # Compute first hidden layer
        self.hidden1_input = np.dot(X, self.weights_input_hidden1) + self.bias_hidden1
        self.hidden1_output = self.sigmoid(self.hidden1_input)
        
        # Compute second hidden layer
        self.hidden2_input = np.dot(self.hidden1_output, self.weights_hidden1_hidden2) + self.bias_hidden2
        self.hidden2_output = self.sigmoid(self.hidden2_input)
        
        # Compute final output
        self.final_input = np.dot(self.hidden2_output, self.weights_hidden2_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        
        return self.final_output

    def backward(self, X, y):
        """
        Perform the backward pass to update weights and biases.
        
        Parameters:
        - X: Input data
        - y: True labels
        """
        # Compute the error at the output layer
        output_error = self.final_output - y
        output_delta = output_error * self.sigmoid_derivative(self.final_input)

        # Compute gradients for the second hidden layer
        hidden2_error = np.dot(output_delta, self.weights_hidden2_output.T)
        hidden2_delta = hidden2_error * self.sigmoid_derivative(self.hidden2_input)

        # Compute gradients for the first hidden layer
        hidden1_error = np.dot(hidden2_delta, self.weights_hidden1_hidden2.T)
        hidden1_delta = hidden1_error * self.sigmoid_derivative(self.hidden1_input)

        # Update weights and biases using gradient descent
        self.weights_hidden2_output -= self.learning_rate * np.dot(self.hidden2_output.T, output_delta)
        self.bias_output -= self.learning_rate * np.sum(output_delta, axis=0, keepdims=True)

        self.weights_hidden1_hidden2 -= self.learning_rate * np.dot(self.hidden1_output.T, hidden2_delta)
        self.bias_hidden2 -= self.learning_rate * np.sum(hidden2_delta, axis=0, keepdims=True)

        self.weights_input_hidden1 -= self.learning_rate * np.dot(X.T, hidden1_delta)
        self.bias_hidden1 -= self.learning_rate * np.sum(hidden1_delta, axis=0, keepdims=True)

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, training):
        """
        Train the neural network using mini-batch gradient descent.
        
        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - X_val: Validation features
        - y_val: Validation labels
        - epochs: Number of training epochs
        - batch_size: Size of each mini-batch
        """
        num_batches = X_train.shape[0] // batch_size

        val_losses = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        epochs_list = []
        
        previous_val_loss = float('inf')  # Initialize with a large number

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            # Process mini-batches
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                
                # Forward pass on the current batch
                self.forward(X_batch)
                
                # Backward pass to update weights and biases
                self.backward(X_batch, y_batch)

            # Validate the model after each epoch
            val_predictions = self.forward(X_val)
            val_loss = self.calculate_loss(val_predictions, y_val)

            if training:
                # Record validation loss over epoch
                val_losses.append(val_loss)
                epochs_list.append(epoch+1)

                # Calculate and record other metrics
                bin_predictions = (val_predictions > 0.5).astype(int)
                accuracies.append(accuracy_score(y_val, bin_predictions))
                precisions.append(precision_score(y_val, bin_predictions))
                recalls.append(recall_score(y_val, bin_predictions))
                f1_scores.append(f1_score(y_val, bin_predictions))

            # Early stopping if validation loss increases
            if val_loss >= previous_val_loss:
                print(f'Early stopping on epoch {epoch + 1} due to increase in validation loss.')
                break

            previous_val_loss = val_loss

            if epoch % 100 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss}')

        if training:    
            plot_validation_loss(epochs_list, val_losses, "training")
            plot_accuracy(epochs_list, accuracies, "training")
            plot_precision(epochs_list, precisions, "training")
            plot_recall(epochs_list, recalls, "training")
            plot_f1_score(epochs_list, f1_scores, "training")

    def calculate_loss(self, predictions, targets):
        """
        Compute the binary cross-entropy loss.
        
        Parameters:
        - predictions: Predicted values
        - targets: True labels
        
        Returns:
        - Binary cross-entropy loss
        """
        epsilon = 1e-9
        predictions = np.clip(predictions, epsilon, 1 - epsilon)  # Avoid log(0) issues
        return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

    def test(self, X_test, y_test):
        """
        Evaluate the model on the test set.
        
        Parameters:
        - X_test: Test features
        - y_test: Test labels
        
        Returns:
        - test_loss: Loss on the test set
        - accuracy: Accuracy score on the test set
        - precision: Precision score on the test set
        - recall: Recall score on the test set
        - f1: F1 score on the test set
        - fpr: False positive rate for ROC curve
        - tpr: True positive rate for ROC curve
        - roc_auc: Area under the ROC curve
        """
        test_predictions = self.forward(X_test)
        test_loss = self.calculate_loss(test_predictions, y_test)
        
        # Binarize predictions
        bin_predictions = (test_predictions > 0.5).astype(int)
        
        # Calculate metrics with zero_division parameter
        accuracy = accuracy_score(y_test, bin_predictions)
        precision = precision_score(y_test, bin_predictions, zero_division=0)
        recall = recall_score(y_test, bin_predictions, zero_division=0)
        f1 = f1_score(y_test, bin_predictions, zero_division=0)

        # Compute ROC Curve and AUC
        fpr, tpr, _ = roc_curve(y_test, test_predictions)
        roc_auc = auc(fpr, tpr)

        return test_loss, accuracy, precision, recall, f1, fpr, tpr, roc_auc
        
    def save_model(self, filename):
        """
        Save the model parameters and layer sizes to a file.
        
        Parameters:
        - filename: Name of the file to save the model
        """
        with open(filename, 'wb') as file:
            pickle.dump({
                'input_size': self.input_size,
                'hidden_size1': self.hidden_size1,
                'hidden_size2': self.hidden_size2,
                'output_size': self.output_size,
                'weights_input_hidden1': self.weights_input_hidden1,
                'weights_hidden1_hidden2': self.weights_hidden1_hidden2,
                'weights_hidden2_output': self.weights_hidden2_output,
                'bias_hidden1': self.bias_hidden1,
                'bias_hidden2': self.bias_hidden2,
                'bias_output': self.bias_output
            }, file)

    def load_model(self, filename):
        """
        Load model parameters and layer sizes from a file.
        
        Parameters:
        - filename: Name of the file to load the model from
        """
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            self.input_size = data['input_size']
            self.hidden_size1 = data['hidden_size1']
            self.hidden_size2 = data['hidden_size2']
            self.output_size = data['output_size']
            self.weights_input_hidden1 = data['weights_input_hidden1']
            self.weights_hidden1_hidden2 = data['weights_hidden1_hidden2']
            self.weights_hidden2_output = data['weights_hidden2_output']
            self.bias_hidden1 = data['bias_hidden1']
            self.bias_hidden2 = data['bias_hidden2']
            self.bias_output = data['bias_output']
