import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import pickle

def load_mat_file(file_path):
    """
    Load a MATLAB .mat file and extract ECoG signals and finger flexion data.
    """
    mat_data = loadmat(file_path)

    # Extract variables (update keys if needed based on your .mat file)
    train_data = mat_data['train_data']  # ECoG signals
    train_dg = mat_data['train_dg']      # Finger positions

    print(f"Loaded train_data shape: {train_data.shape}")
    print(f"Loaded train_dg shape: {train_dg.shape}")

    return train_data, train_dg

def create_sliding_windows(X, y, window_size, stride):
    """
    Create sliding windows for time-series data.
    """
    windows_X, windows_y = [], []
    for i in range(0, X.shape[0] - window_size, stride):
        windows_X.append(X[i:i + window_size, :])  # Shape: (window_size, num_channels)
        windows_y.append(y[i:i + window_size, :])  # Shape: (window_size, num_fingers)

    windows_X = np.array(windows_X)
    windows_y = np.array(windows_y)

    print(f"Sliding windows created: X_windows shape: {windows_X.shape}, y_windows shape: {windows_y.shape}")
    return windows_X, windows_y

def preprocess_and_save(train_data, train_dg, output_file, window_size=1000, stride=250):
    """
    Normalize and preprocess the data, then save it as a pickle file.
    """
    # Normalize ECoG signals
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)

    # Create sliding windows
    X, y = create_sliding_windows(train_data, train_dg, window_size, stride)

    # Save to pickle file
    with open(output_file, "wb") as f:
        pickle.dump({'X_train': X, 'y_train': y}, f)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    # Input .mat file
    mat_file = "sub1_comp.mat"  # Update this path as needed
    output_file = "sub1_preprocessed.pkl"

    # Load the .mat file
    train_data, train_dg = load_mat_file(mat_file)

    # Preprocess data and save
    preprocess_and_save(train_data, train_dg, output_file)
