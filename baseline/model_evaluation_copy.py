import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
from joblib import Parallel, delayed
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

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

def evaluate_model(model, X, y, cv_folds=3, use_cv=True):
    """
    Perform model evaluation with cross-validation or train-test split.
    """
    print(f"Evaluating {model.__class__.__name__}...")
    metrics_list = []

    if use_cv:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        metrics_list = Parallel(n_jobs=-1)(
            delayed(_fit_and_evaluate)(
                model, X[train_idx], y[train_idx], X[val_idx], y[val_idx]
            )
            for train_idx, val_idx in kf.split(X)
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        metrics_list.append(_fit_and_evaluate(model, X_train, y_train, X_val, y_val))

    avg_metrics = np.mean(metrics_list, axis=0)
    print(f"Overall Metrics -> MSE: {avg_metrics[0]:.4f}, RMSE: {avg_metrics[1]:.4f}, Avg Corr: {avg_metrics[2]:.4f}")
    return avg_metrics

def _fit_and_evaluate(model, X_train, y_train, X_val, y_val):
    """
    Fit the model and calculate metrics for a single train-validation split.
    """
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
# Main Execution
# ----------------------------------------------

def evaluate_full_dataset(input_path, output_path, models, cv_folds=3, use_cv=True):
    """
    Evaluate multiple models on the full dataset and save results.
    """
    # Load data
    print(f"Loading data from {input_path}...")
    data = joblib.load(input_path)
    X, y = data['X_train'], data['y_train']

    # Preprocess features
    X = preprocess_features(X)

    # Evaluate each model
    results = []
    for model_name, model in models.items():
        metrics = evaluate_model(model, X, y, cv_folds=cv_folds, use_cv=use_cv)
        results.append({
            "Model": model_name,
            "MSE": metrics[0],
            "RMSE": metrics[1],
            "Avg Corr": metrics[2]
        })

        # Plot feature importance for supported models
        if hasattr(model, "feature_importances_") or hasattr(model, "estimators_"):
            print(f"Feature importance for {model_name}:")
            plot_feature_importance(model, model_name)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Model evaluation results saved to {output_path}.")



def plot_feature_importance(model, model_name):
    """
    Plot feature importance for supported tree-based models.
    """
    try:
        if hasattr(model, "feature_importances_"):  # For models like LightGBM, XGBoost, CatBoost
            importances = model.feature_importances_
        elif hasattr(model, "estimators_"):  # For ensemble models like Random Forest
            importances = model.estimators_[0].feature_importances_
        else:
            print(f"Feature importance is not available for {model_name}.")
            return

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances)
        plt.title(f"Feature Importance - {model_name}")
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
        plt.show()
    except Exception as e:
        print(f"Error plotting feature importance for {model_name}: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input data (preprocessed).")
    parser.add_argument("--output", required=True, help="Path to save evaluation results.")
    parser.add_argument("--cv_folds", type=int, default=3, help="Number of folds for cross-validation.")
    parser.add_argument("--use_cv", action="store_true", help="Use cross-validation for evaluation.")
    args = parser.parse_args()

    # Define models to evaluate
    models = {
        "Ridge Regression": MultiOutputRegressor(Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=7, n_jobs=-1, random_state=42),
        "Hist Gradient Boosting": MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=100, random_state=42)),
        "XGBoost": MultiOutputRegressor(XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42)),
        "LightGBM": MultiOutputRegressor(LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, min_split_gain=0.1, random_state=42)),
        "CatBoost": MultiOutputRegressor(CatBoostRegressor(iterations=200, learning_rate=0.05, depth=7, verbose=0, random_state=42))
    }

    # Run evaluation
    evaluate_full_dataset(args.input, args.output, models=models, cv_folds=args.cv_folds, use_cv=args.use_cv)
