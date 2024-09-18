import pandas as pd
import os
from sklearn.model_selection import train_test_split

def read_data(file_path):
    """Read the CSV data from the given file path."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Drop rows with any NaN values from the DataFrame."""
    return df.dropna()

def split_data(X, y, test_size=0.3, val_size=0.3, random_state=42):
    """
        Split the data into training, validation, and test sets.
        
        Parameters:
        - X: Features
        - y: Target
        - test_size: Proportion of the data to include in the test split
        - val_size: Proportion of the training+validation data to include in the validation split
        - random_state: Seed for the random number generator
        
        Returns:
        - X_train: Training features
        - X_val: Validation features
        - X_test: Test features
        - y_train: Training target
        - y_val: Validation target
        - y_test: Test target
    """
    # Split into training+validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Split the training+validation set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size / (1 - test_size),  # proportion of the remaining data
        random_state=random_state,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def reset_index_and_check(df_features, df_labels):
    """Reset the index for features and labels and check for NaN values."""
    df_features.reset_index(drop=True, inplace=True)
    df_labels.reset_index(drop=True, inplace=True)
    assert df_features.isnull().sum().sum() == 0
    assert df_labels.isnull().sum().sum() == 0

def save_data(X_train, y_train, X_val, y_val, X_test, y_test, folder_path):
    """Save the training, validation, and test data to CSV files."""
    # Combine features and labels
    train_data = pd.concat([X_train, y_train], axis=1)
    val_data = pd.concat([X_val, y_val], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Save DataFrames to CSV
    train_data.to_csv(os.path.join(folder_path, 'train_data.csv'), index=False, sep=',')
    val_data.to_csv(os.path.join(folder_path, 'val_data.csv'), index=False, sep=',')
    test_data.to_csv(os.path.join(folder_path, 'test_data.csv'), index=False, sep=',')

    print("Data saved successfully in the 'data' folder.")

def main():
    # File path
    file_path = 'Detecting Malwares Among Mobile Apps.csv'
    folder_path = 'data'

    # Read and clean data
    df = read_data(file_path)

    # Drop the 'Label' column
    data_excluding_label = df.drop(columns=['Label'])

    # Get the number of rows and columns
    num_rows, num_columns = data_excluding_label.shape

    print(f"Number of rows: {num_rows}")
    print(f"Number of columns (excluding 'Label'): {num_columns}")


    df_clean = clean_data(df)

    # Separate features and target variable
    X = df_clean.drop('Label', axis=1)
    y = df_clean['Label']

    # Split data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Reset index and ensure no NaNs
    reset_index_and_check(X_train, y_train)
    reset_index_and_check(X_val, y_val)
    reset_index_and_check(X_test, y_test)

    # Save data to CSV
    save_data(X_train, y_train, X_val, y_val, X_test, y_test, folder_path)

if __name__ == "__main__":
    main()
