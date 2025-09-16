"""
LightNet (Full) for DQMARS Framework
Description: Trains and evaluates the LightNet classifier for QoS profile assignment.
             Includes placeholders for DQMARS two phases:
             1. Resource slice allocation
             2. QoS profile assignment
Metrics reported: Accuracy, Feature Importance, CPU usage, FLOPs, Parameters
"""

import argparse
import json
import pandas as pd
import numpy as np
import psutil
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier

# -------------------------------
# Phase 1: Resource Slice Allocation
# -------------------------------
def allocate_resource_slice(X):
    """
    Placeholder function for resource slice allocation.
    Each traffic type is mapped to a dedicated slice.
    Args:
        X : np.ndarray, features of flows
    Returns:
        slice_ids : np.ndarray, assigned slice for each flow
    """
    slice_ids = np.zeros(len(X))  # Example: assign all flows to slice 0
    return slice_ids

# -------------------------------
# Phase 2: QoS Profile Assignment (LightNet)
# -------------------------------
def assign_qos_profile(X, y, params):
    """
    Train LightNet (TabNet) on QoS profile assignment
    Args:
        X : np.ndarray, features
        y : np.ndarray, target labels
        params : dict, TabNet hyperparameters
    Returns:
        model : trained TabNetClassifier
        X_test, y_test, y_pred : test data and predictions
        metrics : dict, contains accuracy, feature importance, CPU usage, FLOPs, parameters
    """
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize TabNet
    model = TabNetClassifier(
        n_d=params.get("n_d", 8),
        n_a=params.get("n_a", 8),
        n_steps=params.get("n_steps", 3),
        gamma=params.get("gamma", 1.3),
        n_independent=params.get("n_independent", 2),
        n_shared=params.get("n_shared", 2),
        cat_emb_dim=params.get("cat_emb_dim", 1)
    )

    # Train model
    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=params.get("max_epochs", 50),
        patience=params.get("patience", 10),
        batch_size=params.get("batch_size", 256),
        virtual_batch_size=params.get("virtual_batch_size", 64),
        verbose=1
    )

    # Measure CPU usage during inference
    start_time = time.time()
    cpu_usage_samples = []
    for i in range(len(X_test)):
        psutil.cpu_percent(interval=None)  # reset
        _ = model.predict(X_test[i].reshape(1, -1))
        cpu_usage_samples.append(psutil.cpu_percent(interval=0.01))
    avg_cpu_usage = np.mean(cpu_usage_samples)
    inference_time = time.time() - start_time

    # Predict on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    feature_importance = model.feature_importances_

    # Compute FLOPs and parameters
    flops = model.compute_flops(X_test)
    params_count = model.compute_params()

    # Summary metrics
    metrics = {
        "accuracy": accuracy,
        "feature_importance": feature_importance,
        "avg_cpu_usage": avg_cpu_usage,
        "FLOPs": flops,
        "params": params_count,
        "inference_time_s": inference_time
    }

    return model, X_test, y_test, y_pred, metrics

# -------------------------------
# Main function
# -------------------------------
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train LightNet on DQMARS datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset CSV file")
    parser.add_argument("--config", type=str, default="./config/lightnet.json", help="Path to JSON config file")
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.dataset)

    # Extract features and target
    feature_cols = [col for col in df.columns if col not in ['Channel']]
    X = df[feature_cols].values
    y = df['Channel'].values - 1  # Ensure labels start from 0

    # Allocate resource slices (Phase 1)
    slices = allocate_resource_slice(X)
    print(f"Resource slices allocated: unique slices = {np.unique(slices)}")

    # Load TabNet hyperparameters
    with open(args.config) as f:
        params = json.load(f)

    # Assign QoS profiles (Phase 2)
    model, X_test, y_test, y_pred, metrics = assign_qos_profile(X, y, params)

    # Print summary
    print("\n=== LightNet Performance Summary ===")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Feature importance: {metrics['feature_importance']}")
    print(f"Average CPU usage during inference: {metrics['avg_cpu_usage']:.2f}%")
    print(f"FLOPs: {metrics['FLOPs']/1e3:.2f} K")
    print(f"Parameters: {metrics['params']/1e3:.2f} K")
    print(f"Total inference time: {metrics['inference_time_s']:.4f} s")

if __name__ == "__main__":
    main()

