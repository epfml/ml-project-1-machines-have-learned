# Import useful libraries
import numpy as np
from helpers import *

# Define the path to the dataset
data_path = "data/dataset_to_release_2"

def clean_data(x_train, x_test):
    # Create a copy of each array to avoid modifying the original data
    x_train_cleaned = x_train.copy()
    x_test_cleaned = x_test.copy()

    # Calculate the fraction of NaN values for each column
    nan_fraction_train = np.isnan(x_train_cleaned).mean(axis=0)

    # Create a mask for columns with less than 50% NaN values
    mask_null = nan_fraction_train < 0.8

    # Use the mask to filter out columns with 50% or more NaN values
    x_train_cleaned = x_train_cleaned[:, mask_null]
    x_test_cleaned = x_test_cleaned[:, mask_null]

    # Identify non-constant columns, columns that have a standard deviation not equal to zero
    mask_variance = (np.nanstd(x_train_cleaned, axis=0) != 0.0)

    # Retain only the non-constant columns in the x_train_copied dataset
    x_train_cleaned = x_train_cleaned[:, mask_variance]
    x_test_cleaned = x_test_cleaned[:, mask_variance]

    # For each element in x arrays, if the element is NaN, replace it with the median
    x_train_cleaned = np.where(np.isnan(x_train_cleaned), np.nanmedian(x_train_cleaned, axis=0), x_train_cleaned)
    x_test_cleaned = np.where(np.isnan(x_test_cleaned), np.nanmedian(x_test_cleaned, axis=0), x_test_cleaned)

    return x_train_cleaned, x_test_cleaned

def standardize(x):
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

# Calculate the least squares solution.
def least_squares(y, tx):
    A = tx.T.dot(tx)
    B = tx.T.dot(y)
    w, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    mse = compute_loss(y, tx, w)
    
    return (w, mse)

# Calculate the mse loss
def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    mse = 1 / 2 * np.mean(e**2)
    return mse

def main():
    
    # Load data from the specified dataset path
    print("Data loading started...")
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)
    
    # Clean the data
    print("Data cleaning started...")
    x_train_cleaned, x_test_cleaned = clean_data(x_train, x_test)
    
    # Standardize the training and testing datasets
    x_train_cleaned, mean_train_cleaned, std_train_cleaned = standardize(x_train_cleaned)
    x_test_cleaned, mean_test_cleaned, std_test_cleaned = standardize(x_test_cleaned)
    
    # Train our model
    print("Training started...")
    w, loss = least_squares(y_train, x_train_cleaned)

    # Test our model
    print("Prediction started...")
    threshold = 0.265
    predictions = x_test_cleaned @ w
    preds_submission = np.where(predictions >= threshold, 1, -1)
    
    # Create the submission .csv file
    print("Generating output.csv file...")
    create_csv_submission(test_ids, preds_submission, "output")
    
    print("Finished. Please open output.csv file")

if __name__ == "__main__":
    main()