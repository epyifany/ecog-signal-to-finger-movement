import pickle

def load_preprocessed_data(file_path):
    """
    Load the preprocessed data and check shapes.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    X_train = data['X_train']
    y_train = data['y_train']

    print(f"Loaded preprocessed data: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    return X_train, y_train

if __name__ == "__main__":
    # Path to preprocessed data
    preprocessed_file = "sub1_preprocessed.pkl"  # Update this path as needed

    # Load and debug preprocessed data
    X_train, y_train = load_preprocessed_data(preprocessed_file)

    # Expected shape checks
    window_size, num_channels = X_train.shape[1], X_train.shape[2]
    num_fingers = y_train.shape[2]
    print(f"Window size: {window_size}, Number of channels: {num_channels}")
    print(f"Number of fingers (outputs): {num_fingers}")
