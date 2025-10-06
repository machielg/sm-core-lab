#!/usr/bin/env python3
"""
Custom inference handler for XGBoost models in PyTorch container.

Uses sagemaker_inference toolkit for CSV encoding/decoding, following
the same simple pattern as lab2-boyl with scikit-learn.
"""

import os
import numpy as np
import xgboost as xgb
from sagemaker_inference import decoder, encoder


def model_fn(model_dir):
    """
    Load the XGBoost model from the model directory.

    Args:
        model_dir (str): Path to the directory containing the model artifact

    Returns:
        xgboost.Booster: Loaded XGBoost model
    """
    model_path = os.path.join(model_dir, 'xgboost-model')
    model = xgb.Booster()
    model.load_model(model_path)
    return model


def input_fn(request_body, content_type):
    """
    Decode input data using SageMaker inference toolkit.

    Same pattern as lab2-boyl, just wraps result in XGBoost DMatrix.
    Handles structured arrays from CSV decoding by converting to regular 2D array.
    """
    data = decoder.decode(request_body, content_type)

    # Handle structured arrays (record arrays with named fields)
    if data.dtype.names:
        data = np.column_stack([data[name] for name in data.dtype.names])

    return xgb.DMatrix(data.astype('float32'))


def predict_fn(input_data, model):
    """
    Make predictions with the loaded model.

    Args:
        input_data (xgboost.DMatrix): Preprocessed input data
        model (xgboost.Booster): Loaded XGBoost model

    Returns:
        numpy.ndarray: Model predictions
    """
    return model.predict(input_data)


def output_fn(prediction, accept):
    """
    Encode predictions using SageMaker inference toolkit.

    Same pattern as lab2-boyl.
    """
    return encoder.encode(prediction, accept)
