# Malware Detection using Neural Networks

This project aims to detect malware in mobile applications using a neural network model. It involves data preprocessing, model training, validation, testing, and generating various metrics to evaluate the model's performance.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Files Overview](#files-overview)
- [Results](#results)

## Introduction

Malware detection is critical in today's digital landscape. This project utilizes a neural network to classify mobile applications as either 'goodware' or 'malware'. The architecture includes two hidden layers, and the model is evaluated using confusion matrices and ROC curves.

## Dataset

The dataset used in this project is `Detecting Malwares Among Mobile Apps.csv`. It contains features extracted from mobile applications along with their labels (either 'goodware' or 'malware').

### Data Split

The dataset is split into three parts:
- **Training Set**: Used to train the model.
- **Validation Set**: Used to tune the hyperparameters.
- **Test Set**: Used to evaluate the model's performance.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
## Usage

1. **Data Preparation**: Run train_test_split.py to read, clean, and split the dataset into training, validation, and test sets.
   ```bash
   python train_test_split.py
2. **Model Training**: Run train.py to perform cross-validation to find the best hidden layer sizes and train the model.
   ```bash
   python train.py
3. **Model Testing**: Run test.py to evaluate the trained model on the test dataset and generate performance metrics.

## Files Overview
- **train_test_split.py**: Contains functions for reading, cleaning, and splitting the dataset.
- **train.py**: Includes the model training process.
- **test.py**: Tests the trained model and generates metrics.
- **neural_network.py**: Implements the neural network architecture.
- **utils.py**: Contains utility functions for loading data, plotting metrics, and saving results.

## Results

After running the test, various metrics are generated including:

- Confusion Matrix
- ROC Curve
- Accuracy, Precision, Recall, F1 Score
- Validation Loss Plots

All plots and metrics are saved in respective folders for review.

