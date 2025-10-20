#!/usr/bin/env python3
"""
Custom XGBoost training script for SageMaker PyTorch Framework Estimator.

This script demonstrates running XGBoost within a PyTorch container,
leveraging the modern Python ecosystem while training with XGBoost library.

Uses sagemaker_training.environment for clean access to SageMaker environment.
"""

import os
import pandas as pd
import xgboost as xgb
import json
from sagemaker_training import environment


def load_data(data_path):
    """
    Load data from SageMaker channel path.

    Handles CSV files with target in first column and no headers.
    Supports multiple CSV files in the channel directory.

    Args:
        data_path (str): Path to data directory (e.g., /opt/ml/input/data/train)

    Returns:
        tuple: (X, y) where X is features DataFrame and y is target Series
    """
    # Get all CSV files in the directory
    files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.csv')]

    if not files:
        raise ValueError(f"No CSV files found in {data_path}")

    print(f"Loading {len(files)} file(s) from {data_path}")

    # Load and concatenate all CSV files
    dfs = []
    for file in files:
        df = pd.read_csv(file, header=None)
        dfs.append(df)
        print(f"  Loaded {file}: {df.shape}")

    # Combine all dataframes
    data = pd.concat(dfs, axis=0, ignore_index=True)

    # First column is target, rest are features (XGBoost format)
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]

    print(f"Total samples: {len(X)}, Features: {X.shape[1]}")

    return X, y


def train():
    """
    Train XGBoost model using sagemaker_training.environment for configuration.
    """
    # Access SageMaker environment - provides all paths and hyperparameters
    env = environment.Environment()

    print("=" * 60)
    print("SAGEMAKER ENVIRONMENT")
    print("=" * 60)
    print(env)
    print("=" * 60)

    # Access paths directly from environment
    model_dir = env.model_dir                           # /opt/ml/model
    train_dir = env.channel_input_dirs['train']         # /opt/ml/input/data/train
    validation_dir = env.channel_input_dirs['validation']  # /opt/ml/input/data/validation

    # Access hyperparameters as dictionary (much cleaner than argparse!)
    max_depth = int(env.hyperparameters.get('max-depth', 5))
    eta = float(env.hyperparameters.get('eta', 0.2))
    gamma = float(env.hyperparameters.get('gamma', 4))
    min_child_weight = float(env.hyperparameters.get('min-child-weight', 6))
    subsample = float(env.hyperparameters.get('subsample', 0.8))
    objective = env.hyperparameters.get('objective', 'binary:logistic')
    num_round = int(env.hyperparameters.get('num-round', 100))
    eval_metric = env.hyperparameters.get('eval-metric', 'auc')

    # Get system info (useful for distributed training)
    num_gpus = env.num_gpus
    num_cpus = env.num_cpus
    hosts = env.hosts
    current_host = env.current_host

    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Model directory: {model_dir}")
    print(f"Train data: {train_dir}")
    print(f"Validation data: {validation_dir}")
    print(f"\nSystem Info:")
    print(f"  GPUs: {num_gpus}, CPUs: {num_cpus}")
    print(f"  Hosts: {hosts}, Current: {current_host}")
    print("\nHyperparameters:")
    print(f"  max_depth: {max_depth}")
    print(f"  eta: {eta}")
    print(f"  gamma: {gamma}")
    print(f"  min_child_weight: {min_child_weight}")
    print(f"  subsample: {subsample}")
    print(f"  objective: {objective}")
    print(f"  num_round: {num_round}")
    print(f"  eval_metric: {eval_metric}")
    print("=" * 60)

    # Load training and validation data
    print("\n📥 Loading data...")
    X_train, y_train = load_data(train_dir)
    X_val, y_val = load_data(validation_dir)

    # Create DMatrix for XGBoost
    print("\n🔨 Creating DMatrix...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Set up parameters
    params = {
        'max_depth': max_depth,
        'eta': eta,
        'gamma': gamma,
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'objective': objective,
        'eval_metric': eval_metric,
        'seed': 42  # For reproducibility
    }

    print("\n🚀 Starting training...")
    print(f"Training for {num_round} rounds with early stopping")

    # Train model with evaluation
    watchlist = [(dtrain, 'train'), (dval, 'validation')]

    # Store evaluation results for analysis
    evals_result = {}

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_round,
        evals=watchlist,
        evals_result=evals_result,
        early_stopping_rounds=10,
        verbose_eval=10  # Print every 10 rounds
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    # Get final metrics
    train_metric = evals_result['train'][eval_metric][-1]
    val_metric = evals_result['validation'][eval_metric][-1]

    print(f"Final train {eval_metric}: {train_metric:.4f}")
    print(f"Final validation {eval_metric}: {val_metric:.4f}")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score:.4f}")

    # Print in format that can be captured by SageMaker metrics
    print(f"\nvalidation-{eval_metric}:{val_metric:.4f}")
    print(f"train-{eval_metric}:{train_metric:.4f}")

    # Save model
    print("\n💾 Saving model...")
    model_path = os.path.join(model_dir, 'xgboost-model.bin')
    model.save_model(model_path)
    print(f"✅ Model saved to {model_path}")

    # Save model metadata
    metadata = {
        'best_iteration': int(model.best_iteration),
        'best_score': float(model.best_score),
        'num_features': X_train.shape[1],
        'hyperparameters': {
            'max_depth': max_depth,
            'eta': eta,
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'objective': objective,
            'num_round': num_round,
            'eval_metric': eval_metric
        },
        'final_metrics': {
            'train': {eval_metric: train_metric},
            'validation': {eval_metric: val_metric}
        },
        'system_info': {
            'num_gpus': num_gpus,
            'num_cpus': num_cpus,
            'hosts': hosts,
            'current_host': current_host
        }
    }

    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to {metadata_path}")

    print("\n" + "=" * 60)
    print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == '__main__':
    train()
