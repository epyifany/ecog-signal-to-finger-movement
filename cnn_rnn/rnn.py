import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------
# Data Loading and Preprocessing
# ----------------------------------------------

def load_preprocessed_data(file_path):
    """
    Load the preprocessed data.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    X, y = data['X_train'], data['y_train']
    return X, y

def preprocess_data(X, y):
    """
    Normalize the data using StandardScaler and shift labels to account for temporal delay.
    """
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_X.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
    y = scaler_y.fit_transform(y.reshape(-1, y.shape[2])).reshape(y.shape)

    # Shift labels for temporal alignment
    y = np.roll(y, shift=-37, axis=1)  # Adjust this based on the delay
    return X, y

def split_data(X, y, test_size=0.2):
    """
    Split data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# ----------------------------------------------
# RNN Model Definition
# ----------------------------------------------

def build_rnn_model(input_shape, rnn_type="LSTM"):
    """
    Define an RNN model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    if rnn_type == "LSTM":
        model.add(LSTM(128, return_sequences=True))
    elif rnn_type == "GRU":
        model.add(GRU(128, return_sequences=True))
    else:
        raise ValueError("Unsupported RNN type. Use 'LSTM' or 'GRU'.")
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(5, activation='linear')))  # One output per finger
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
    return model

# ----------------------------------------------
# Model Training and Evaluation
# ----------------------------------------------

def calculate_metrics(y_true, y_pred):
    """
    Calculate MSE, RMSE, and average correlation coefficient.
    """
    mse = mean_squared_error(y_true.reshape(-1, 5), y_pred.reshape(-1, 5))
    rmse = np.sqrt(mse)
    corr_coeffs = [
        np.corrcoef(y_true[:, :, i].flatten(), y_pred[:, :, i].flatten())[0, 1]
        if np.var(y_true[:, :, i]) > 0 else 0
        for i in range(y_true.shape[2])
    ]
    avg_corr = np.mean(corr_coeffs)
    return mse, rmse, avg_corr, corr_coeffs

def train_and_evaluate_model(X_train, y_train, X_test, y_test, input_shape, rnn_type="LSTM"):
    """
    Train the RNN model and evaluate performance.
    """
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    model = build_rnn_model(input_shape, rnn_type)
    print(f"Model output shape: {model.output_shape}")

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse, rmse, avg_corr, corr_coeffs = calculate_metrics(y_test, y_pred)
    print(f"Test MSE: {mse:.4f}, RMSE: {rmse:.4f}, Avg Corr: {avg_corr:.4f}")
    print(f"Per-channel Correlations: {corr_coeffs}")

    results = {
        "Metric": ["MSE", "RMSE", "Avg Corr"] + [f"Corr Channel {i+1}" for i in range(len(corr_coeffs))],
        "Value": [mse, rmse, avg_corr] + corr_coeffs
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"rnn_{rnn_type.lower()}_evaluation_results.csv", index=False)
    print(f"Model evaluation results saved to rnn_{rnn_type.lower()}_evaluation_results.csv.")

    # Plot training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Save the model
    model.save(f"rnn_{rnn_type.lower()}_finger_flexion_model.h5")
    print(f"Model saved as rnn_{rnn_type.lower()}_finger_flexion_model.h5")

# ----------------------------------------------
# Main Execution
# ----------------------------------------------

if __name__ == "__main__":
    preprocessed_file = "sub1_preprocessed.pkl"  # Update this path as needed
    X, y = load_preprocessed_data(preprocessed_file)

    X, y = preprocess_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X, y)

    input_shape = (X_train.shape[1], X_train.shape[2])
    train_and_evaluate_model(X_train, y_train, X_test, y_test, input_shape, rnn_type="LSTM")
