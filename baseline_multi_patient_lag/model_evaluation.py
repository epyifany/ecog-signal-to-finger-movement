import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import os
import glob

# ----------------------------------------------
# Model Evaluation Methods
# ----------------------------------------------

def calculate_metrics(y_true, y_pred):
    """
    Calculate key regression metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    corr_coeffs = [
        np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1] if np.var(y_true[:, i]) > 0 else 0
        for i in range(y_true.shape[1])
    ]
    avg_corr = np.mean(corr_coeffs)
    return mse, rmse, avg_corr

def evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Train and evaluate a model, returning performance metrics.
    """
    print(f"Evaluating model: {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return calculate_metrics(y_val, y_pred)

# ----------------------------------------------
# Feature Preprocessing
# ----------------------------------------------

def preprocess_features(X):
    """
    Preprocess features by removing low-variance features and scaling.
    """
    # Remove low-variance features
    feature_variance = np.var(X, axis=0)
    low_variance_features = np.where(feature_variance < 1e-5)[0]
    if len(low_variance_features) > 0:
        print(f"Removing {len(low_variance_features)} low-variance features...")
        X = np.delete(X, low_variance_features, axis=1)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

# ----------------------------------------------
# Model Definitions
# ----------------------------------------------

def define_models():
    """
    Define baseline models for evaluation.
    """
    return {
        "Ridge Regression": MultiOutputRegressor(Ridge(alpha=1.0)),
        "Hist Gradient Boosting": MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=100, random_state=42)),
        "XGBoost": MultiOutputRegressor(XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42)),
        "LightGBM": MultiOutputRegressor(LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42)),
        "CatBoost": MultiOutputRegressor(CatBoostRegressor(iterations=200, learning_rate=0.05, depth=7, verbose=0, random_state=42))
    }

# ----------------------------------------------
# Process and Evaluate Data
# ----------------------------------------------

def evaluate_preprocessed_files(preprocessed_files, output_path):
    """
    Evaluate models on all preprocessed data files (accounting for patients and lags) and save results.
    """
    results = []

    # Define models to evaluate
    models = define_models()

    for file_path in preprocessed_files:
        print(f"Loading preprocessed data from {file_path}...")
        data = joblib.load(file_path)
        X_train, X_val = data['X_train'], data['X_val']
        y_train, y_val = data['y_train'], data['y_val']

        # Extract patient ID and lag value from filename
        patient_id = file_path.split("_preprocessed")[0]
        lag = int(file_path.split("_lag")[-1].split(".")[0])

        # Preprocess features
        print("Preprocessing features...")
        X_train = preprocess_features(X_train)
        X_val = preprocess_features(X_val)

        for model_name, model in models.items():
            mse, rmse, avg_corr = evaluate_model(model, X_train, y_train, X_val, y_val)
            results.append({
                "Patient": patient_id,
                "Lag": lag,
                "Model": model_name,
                "MSE": mse,
                "RMSE": rmse,
                "Avg Corr": avg_corr,
            })

    # Save results to a CSV
    results_df = pd.DataFrame(results)
    results_df.sort_values(by=["Patient", "Lag", "Avg Corr"], ascending=[True, True, False], inplace=True)
    results_df.to_csv(output_path, index=False)
    print(f"Model evaluation results saved to {output_path}.")
    print(results_df)

# ----------------------------------------------
# Main Execution
# ----------------------------------------------

if __name__ == "__main__":
    # Path to preprocessed files and output results
    preprocessed_files = glob.glob("sub*_preprocessed_lag*.pkl")  # Match all preprocessed .pkl files
    output_path = "model_evaluation_results.csv"

    # Evaluate models on the preprocessed files
    evaluate_preprocessed_files(preprocessed_files, output_path)
