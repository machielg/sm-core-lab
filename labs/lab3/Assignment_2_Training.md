# 🎓 Assignment 2: Training with Framework Estimators

## 🎯 Your Mission
Use the SageMaker SDK's Estimator classes to train models with a simpler, more Pythonic API than the raw TrainingJob approach from Lab 1.

## 🤔 Understanding Estimators

### What's Different from Lab 1?
**Key Insight:** In Lab 1, you manually configured:
- `AlgorithmSpecification` shapes
- `Channel` configurations
- `ResourceConfig` shapes
- `HyperParameters` dictionaries

**With Estimators:** All of this is abstracted into clean, Pythonic interfaces.

**Research Questions:**
1. What's the difference between `Estimator`, `Framework`, and algorithm-specific estimators (like `XGBoost`)?
2. How does the SDK handle S3 paths and channels automatically?
3. When would you use the generic `Estimator` vs. framework-specific ones?

## 🛠️ What You Need to Build

### 1. Choose Your Estimator Type

**Available Options:**
```python
from sagemaker.estimator import Estimator
from sagemaker.xgboost import XGBoost
from sagemaker.pytorch import PyTorch
```


**Framework Estimator**
```python
from sagemaker.pytorch import PyTorch

xgb_estimator = PyTorch(
    entry_point='train.py',          # Your training script
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    framework_version='2.x',
    hyperparameters={
        'max-depth': 5,
        'eta': 0.2,
        'objective': 'binary:logistic',
        'num-round': 100
    }
)
```

**Your Task:** Find out why Pytorch is a subclass of Framework

**Think About:**
- How would you add custom evaluation metrics?

### 2. Create a Custom Training Script

**Your Challenge:** Write `train.py` that follows SageMaker's training script conventions.

**Script Structure:**
```python
# train.py
import argparse
import os
import pandas as pd
import xgboost as xgb
import pickle

def parse_args():
    """
    Parse hyperparameters passed by SageMaker
    """
    parser = argparse.ArgumentParser()

    # Hyperparameters (passed from estimator)
    parser.add_argument('--max-depth', type=int, default=5)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--num-round', type=int, default=100)
    parser.add_argument('--eval-metric', type=str, default='auc')

    # SageMaker-specific environment variables (automatically set)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))

    return parser.parse_args()

def load_data(data_path):
    """
    Load data from SageMaker channel path

    Your Task: Implement data loading logic
    Research: How does SageMaker mount S3 data in the training container?
    """
    files = [os.path.join(data_path, file) for file in os.listdir(data_path)]
    # TODO: Load and concatenate all CSV files
    # TODO: Separate features and target
    # Return: X, y
    pass

def train(args):
    """
    Train XGBoost model

    Your Task: Implement training logic
    """
    # Load training and validation data
    X_train, y_train = load_data(args.train)
    X_val, y_val = load_data(args.validation)

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Set up parameters
    params = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'objective': args.objective,
        'eval_metric': args.eval_metric
    }

    # Train model with evaluation
    watchlist = [(dtrain, 'train'), (dval, 'validation')]
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=watchlist,
        early_stopping_rounds=10
    )

    # Save model to model_dir (SageMaker will upload to S3)
    model_path = os.path.join(args.model_dir, 'xgboost-model')
    model.save_model(model_path)

    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    args = parse_args()
    train(args)
```

**Research Questions:**
1. What are the SageMaker environment variables (`SM_MODEL_DIR`, `SM_CHANNEL_TRAIN`, etc.)?
2. How does SageMaker know where to save your model?
3. What happens to print statements in your training script?

### 3. Train Your Model with the Estimator

**Your Task:** Configure inputs and start training.

**Training Pattern:**
```python
from sagemaker.inputs import TrainingInput

# Create training inputs (much simpler than Lab 1!)
train_input = TrainingInput(
    s3_data=s3_train_data,      # S3 URI from your processing job
    content_type='text/csv'
)

validation_input = TrainingInput(
    s3_data=s3_validation_data,
    content_type='text/csv'
)

# Start training (asynchronous)
xgb_estimator.fit({
    'train': train_input,
    'validation': validation_input
})
```

**Compare to Lab 1:**
- No manual `Channel` configuration
- No manual `AlgorithmSpecification`
- No manual image URI retrieval
- Just `.fit()` with data!

**Research:**
1. How does the estimator know what container image to use?
2. What's the difference between `.fit()` and `.fit(..., wait=False)`?
3. How do you access training job logs in real-time?

### 4. Retrieve and Analyze Training Results

**Your Challenge:** Access model artifacts and training metrics.

**Getting Model Location:**
```python
# Where is your trained model?
model_data = xgb_estimator.model_data
print(f"Model artifacts: {model_data}")

# Get the training job name
training_job_name = xgb_estimator.latest_training_job.name
print(f"Training job: {training_job_name}")
```

**Analyzing Training Metrics:**
```python
# Access training job description
training_info = xgb_estimator.latest_training_job.describe()

# TODO: Extract and visualize metrics from training logs
# Hint: Look at the CloudWatch logs or use TrainingJobAnalytics
```

**Advanced Analysis:**
```python
from sagemaker import TrainingJobAnalytics

# Get training metrics as pandas DataFrame
metrics_df = TrainingJobAnalytics(training_job_name).dataframe()
print(metrics_df)

# TODO: Plot metrics over time
# - Training vs validation metrics
# - Check for overfitting
# - Identify optimal stopping point
```

## 🔍 Debugging Training Jobs

**Training fails with data errors:**
- Check that your data format matches what the algorithm expects
- Verify channel names match between `.fit()` and your script
- Ensure CSV files have no headers if using built-in algorithm

**"ImportError: No module named X":**
- Add a `requirements.txt` in your source directory
- Make sure `source_dir` parameter includes it
- For built-in algorithms, all libraries must be pre-installed

**Model not saved correctly:**
- Verify you're saving to `SM_MODEL_DIR` environment variable
- Check that the model file has the expected name
- Look at CloudWatch logs for save operation

**Training is too slow:**
- Check if you're using appropriate instance type
- Consider distributed training for large datasets
- Look at I/O bottlenecks (data loading from S3)

## ✅ Success Criteria

**Technical Success:**
- [ ] Training job completes successfully
- [ ] Model artifacts saved to S3
- [ ] Training and validation metrics logged
- [ ] Can retrieve and load the trained model
- [ ] Validation metrics show the model is learning

**Understanding Success:**
- [ ] Can explain the difference between Estimator types
- [ ] Understand SageMaker environment variables and paths
- [ ] Know how to pass hyperparameters to training scripts
- [ ] Can access and analyze training metrics

## 💡 Advanced Exploration

**Distributed Training:**
- Research how to use multiple instances for training
- When is distributed training worth the added complexity?
- How does XGBoost handle distributed training?

**Spot Instances:**
- How much can you save using spot instances?
- What's the trade-off between cost and reliability?
- How do checkpoints enable spot training?

**Custom Metrics:**
- How do you define custom metrics in your training script?
- How do they appear in CloudWatch?
- Can you use custom metrics for early stopping?

## 📚 Research Resources
- [SageMaker Estimator Documentation](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html)
- [XGBoost Estimator Guide](https://sagemaker.readthedocs.io/en/stable/frameworks/xgboost/using_xgboost.html)
- [Training Script Requirements](https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-training-container.html)

---

Ready to optimize? Continue to [Assignment 3: Hyperparameter Tuning](Assignment_3_Tuning.md) →