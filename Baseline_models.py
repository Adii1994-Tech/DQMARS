"""
Baseline Models (RF and SVC) for DQMARS Framework
Description: Trains and evaluates baseline classifiers (Random Forest and SVC)
             for QoS profile assignment.
             Includes placeholders for DQMARS two phases:
             1. Resource slice allocation
             2. QoS profile assignment
Metrics reported: Accuracy, Inference Time
"""

import argparse
import json
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# -------------------------------
# Phase 1: Resource Slice Allocation
# -------------------------------
def allocate_resource_slice(X):
    """
    Placeholder for resource slice allocation.
    Assigns all flows to a single slice (dummy allocation).
    """
    slice_ids = np.zeros(len(X))
    return slice_ids

# -------------------------------
# Phase 2: QoS Profile Assignment (Baseline Models)
# -------------------------------
def assign_qos_profile(X, y, model_type, params):
    """
    Train and evaluate the baseline model
    Args:
        X : np.ndarray, features
        y : np.ndarray, target labels
        model_type : str, "RF" or "SVC"
        params : dict, model hyperparameters
    Returns:
        model : trained classifier
        accuracy : float
        inference_time_s : float
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize model
    if model_type == "RF":
        model = RandomForestClassifier(**params)
    elif model_type == "SVC":
        model = SVC(**params)
    else:
        raise ValueError("Unsupported model type. Choose 'RF' or 'SVC'.")

    # Train model
    model.fit(X_train, y_train)

    # Measure inference time
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time_s = time.time() - start_time

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, inference_time_s

# -------------------------------
# Main function
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train baseline models on DQMARS datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset CSV file")
    parser.add_argument("--config", type=str, default="./config/baseline_models.json", help="Path to JSON config file")
    parser.add_argument("--model", type=str, required=True, choices=["RF", "SVC"], help="Choose model: RF or SVC")
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.dataset)
    feature_cols = [col for col in df.columns if col not in ['Channel']]
    X = df[feature_cols].values
    y = df['Channel'].values - 1  # Ensure labels start from 0

    # Allocate resource slices (Phase 1)
    slices = allocate_resource_slice(X)
    print(f"Resource slices allocated: unique slices = {np.unique(slices)}")

    # Load model hyperparameters
    with open(args.config) as f:
        config = json.load(f)
    model_params = config[args.model]

    # Train and evaluate (Phase 2)
    model, accuracy, inference_time = assign_qos_profile(X, y, args.model, model_params)

    print(f"\n=== {args.model} Performance Summary ===")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Total inference time: {inference_time:.4f} s")

if __name__ == "__main__":
    main()
