import os
import sys
from os.path import join
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Mirrored in the calling notebook processing_inputs and processing_outputs
OUTPUT_DIR = "/opt/ml/processing/output"
INPUT_DIR = "/opt/ml/processing/input/data"

def process():
    _debug()

    churn_file = Path(INPUT_DIR, 'churn.txt')
    df = pd.read_csv(churn_file)

    # Phone number is unique - will not add value to classifier
    df = df.drop("Phone", axis=1)

    # Cast Area Code to non-numeric
    df["Area Code"] = df["Area Code"].astype(object)

    # Remove one feature from highly corelated pairs
    df = df.drop(["Day Charge", "Eve Charge", "Night Charge", "Intl Charge"], axis=1)

    # One-hot encode catagorical features into numeric features
    model_data = pd.get_dummies(df)
    model_data = pd.concat(
        [
            model_data["Churn?_True."],
            model_data.drop(["Churn?_False.", "Churn?_True."], axis=1),
        ],
        axis=1,
    )
    model_data = model_data.astype(float)

    # Split data into train and validation datasets
    train_data, validation_data = train_test_split(model_data, test_size=0.33, random_state=42)

    # Further split the validation dataset into test and validation datasets.
    validation_data, test_data = train_test_split(validation_data, test_size=0.33, random_state=42)

    # Remove and store the target column for the test data. This is used for calculating performance metrics after training, on unseen data.
    test_target_column = test_data["Churn?_True."]
    test_data.drop(["Churn?_True."], axis=1, inplace=True)

    # Store all datasets locally
    train_data.to_csv(join(OUTPUT_DIR, "train.csv"), header=False, index=False)
    validation_data.to_csv(join(OUTPUT_DIR, "validation.csv"), header=False, index=False)
    test_data.to_csv(join(OUTPUT_DIR, "test.csv"), header=False, index=False)


def _debug():
    # Method 1: Print all env vars nicely formatted
    print("=== ALL ENVIRONMENT VARIABLES ===")
    for key, value in os.environ.items():
        print(f"{key}: {value}")

    # Method 3: Separate script name from arguments
    print("\n=== PROGRAM EXECUTION ===")
    print(f"Script name: {sys.argv[0]}")
    print(f"Arguments: {sys.argv[1:]}")


if __name__ == '__main__':
    process()