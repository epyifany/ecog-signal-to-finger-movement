import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import joblib

def load_mat_file(file_path):
    """
    Load a MATLAB file and extract train/test data and labels.
    """
    data = loadmat(file_path)
    train_data = data['train_data']
    train_dg = data['train_dg']
    test_data = data['test_data']
    return train_data, train_dg, test_data

def normalize_data(data):
    """
    Normalize data to have zero mean and unit variance for each channel.
    """
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def shift_labels(train_data, train_dg, lag):
    """
    Shift finger flexion data to account for a specified lag in time steps.
    """
    if lag > 0:
        return train_data[:-lag], train_dg[lag:]
    elif lag < 0:
        return train_data[-lag:], train_dg[:lag]
    else:
        return train_data, train_dg

def preprocess_subject(file_path, output_base_path, lag_range):
    """
    Preprocess data for a single subject for a range of lags and save the preprocessed data.
    """
    print(f"Loading data from {file_path}...")
    train_data, train_dg, test_data = load_mat_file(file_path)
    
    print("Normalizing train and test data...")
    train_data = normalize_data(train_data)
    test_data = normalize_data(test_data)
    
    for lag in lag_range:
        print(f"Shifting labels by {lag} time steps...")
        shifted_train_data, shifted_train_dg = shift_labels(train_data, train_dg, lag=lag)
        
        print("Splitting training data into training and validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(
            shifted_train_data, shifted_train_dg, test_size=0.2, random_state=42
        )
        
        output_path = f"{output_base_path}_lag{lag}.pkl"
        print(f"Saving preprocessed data to {output_path}...")
        preprocessed_data = {
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train,
            "y_val": y_val,
            "test_data": test_data,
        }
        joblib.dump(preprocessed_data, output_path)
        print(f"Preprocessing for lag {lag} complete. Data saved to {output_path}.")

# File paths for the MATLAB files and output files
subject_files = {
    'sub1_comp.mat': 'sub1_preprocessed',
    'sub2_comp.mat': 'sub2_preprocessed',
    'sub3_comp.mat': 'sub3_preprocessed'
}

# Define the fine-tuned lag range (30ms to 50ms in steps of 2ms)
lag_range = range(30, 51, 2)

# Process each subject file for each lag
for file_path, output_base_path in subject_files.items():
    try:
        preprocess_subject(file_path, output_base_path, lag_range)
    except FileNotFoundError:
        print(f"File {file_path} not found. Skipping.")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

print("Preprocessing complete for all subjects and lags.")
