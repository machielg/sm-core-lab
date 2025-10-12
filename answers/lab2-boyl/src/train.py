import os
import shutil

import joblib
import pandas as pd
from sagemaker_training import environment
from sklearn.ensemble import RandomForestClassifier


def train():
    # No argparse needed!
    env = environment.Environment()
    print(env)

    # Access all paths directly
    model_dir = env.model_dir              # /opt/ml/model
    train_dir = env.channel_input_dirs['train']  # /opt/ml/input/data/train
    # output_dir = env.output_data_dir       # /opt/ml/output/data
    target_column = env.hyperparameters.get('target_column', 'target')

    # Access hyperparameters as a dictionary
    learning_rate = float(env.hyperparameters.get('learning_rate', 0.01))
    batch_size = int(env.hyperparameters.get('batch_size', 32))
    # epochs = int(env.hyperparameters.get('epochs', 10))

    # Get system info
    num_gpus = env.num_gpus
    num_cpus = env.num_cpus
    hosts = env.hosts
    current_host = env.current_host

    print(f"{num_gpus=}, {num_cpus=}, {hosts=}, {current_host=}")

    # Now do your training...
    print(f"Training with learning_rate={learning_rate}, batch_size={batch_size}")

    # Your ML code here
    train_data = pd.read_csv(os.path.join(train_dir, 'train.csv'))
    x_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(x_train, y_train)

    # Save model
    joblib.dump(model, os.path.join(model_dir, 'model.joblib'))

    # Manual packaging
    code_dir = os.path.join(env.model_dir, 'code')
    os.makedirs(code_dir, exist_ok=True)
    shutil.copy('inference.py', os.path.join(code_dir, 'inference.py'))

if __name__ == '__main__':
    train()
