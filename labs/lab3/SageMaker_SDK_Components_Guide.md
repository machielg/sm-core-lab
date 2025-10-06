# 🔧 SageMaker SDK Components - Learning Guide

## 📖 Introduction

This lab bridges the gap between individual SageMaker operations and full pipeline orchestration. You'll learn to use the **SageMaker Python SDK** - a higher-level abstraction than sagemaker-core - to work with the key components that make up ML workflows: processing, training, tuning, and model management.

**Learning Objectives:**
- Master the SageMaker SDK's high-level abstractions (Processors, Estimators, Tuners)
- Build production-ready preprocessing workflows
- Use Framework Estimators for flexible model training
- Implement automated hyperparameter optimization at scale
- Manage model lifecycle with Model Registry

**Prerequisites:**
- Complete Lab 1 (SageMaker Essentials with sagemaker-core)
- Complete Lab 2 (Bring Your Own Library)
- Understanding of the customer churn dataset
- Familiarity with scikit-learn and XGBoost

**Why This Lab Matters:**
- **Lab 1**: You learned the *low-level building blocks* (sagemaker-core)
- **This Lab**: You'll master *high-level SDK components* for production ML
- **Lab 4**: You'll *orchestrate these components* into automated pipelines

---

## 🏗️ Assignment 1: Data Processing with SageMaker Processors

### 🎯 Your Mission
Build a reusable data preprocessing workflow using the SageMaker SDK's Processor classes. Transform raw churn data into train/validation/test sets that are ready for model training.

### 🤔 Understanding Processors vs. Manual Preprocessing

#### Key Concept: Why Use SageMaker Processors?
**Research Questions:**
1. What's the difference between preprocessing locally vs. using SageMaker Processing Jobs?
2. How do Processors differ from the raw Processing Job shapes you used in Lab 1?
3. When would you choose `SKLearnProcessor` vs. `ScriptProcessor` vs. `FrameworkProcessor`?

**Real-World Scenario:** Your data is 50GB and growing. Local preprocessing crashes your laptop. How does SageMaker Processing solve this?

### 🛠️ What You Need to Build

#### 1. Choose Your Processor Type
**Your Task:** Research and select the right processor for your use case.

**Available Processor Types:**
```python
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ScriptProcessor, FrameworkProcessor
from sagemaker.xgboost.processing import XGBoostProcessor
```

**Decision Matrix:**
- **PyTorchProcessor**: ✅ Recommended - Modern framework processor, automatic requirements.txt support
- **XGBoostProcessor**: ✅ Good alternative - Automatic requirements.txt support, XGBoost-optimized
- **SKLearnProcessor**: ⚠️ Scikit-learn optimized, but NO automatic requirements.txt support
- **ScriptProcessor**: ⚠️ Legacy approach, manual dependency management required

**Critical Feature: requirements.txt Support**
- ✅ **PyTorchProcessor**, **XGBoostProcessor** and **FrameworkProcessor**: Automatically install packages from `requirements.txt` when using `source_dir`
- ❌ **SKLearnProcessor**: Does NOT support automatic requirements.txt installation (known limitation)
- ❌ **ScriptProcessor**: Generic processor, manual dependency management required

**Recommendation:** Use **PyTorchProcessor** for this assignment - it's based on the modern framework processor architecture, supports automatic dependency installation via `source_dir`, and provides excellent flexibility for preprocessing tasks.

**Think About:**
- Which Python libraries does your preprocessing code need?
- Do you need automatic dependency installation (requirements.txt)?
- What's the trade-off between ease of use and flexibility?

#### 2. Create Your Preprocessing Script
**Your Challenge:** Write `preprocessing.py` that works in the SageMaker Processing container environment.

**Critical Understanding - SageMaker Processing Paths:**
```python
# Input data location (SageMaker mounts S3 data here)
INPUT_PATH = "/opt/ml/processing/input"

# Output locations (SageMaker uploads these to S3)
TRAIN_OUTPUT_PATH = "/opt/ml/processing/train"
VALIDATION_OUTPUT_PATH = "/opt/ml/processing/validation"
TEST_OUTPUT_PATH = "/opt/ml/processing/test"
```

**Your Script Must:**
1. Read raw data from the input path
2. Perform feature engineering and data cleaning
3. Handle missing values and categorical encoding
4. Split data into train (70%), validation (20%), test (10%)
5. Save each dataset to the appropriate output path
6. **Critical**: Save in the format XGBoost expects (target column first, no headers)

**Example Skeleton:**
```python
# preprocessing.py
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def preprocess_data(input_path, train_path, val_path, test_path, train_split=0.7, val_split=0.2):
    """
    Your Task: Implement this function

    Research:
    - How do you handle categorical variables for XGBoost?
    - Should you normalize/scale features? Why or why not?
    - How do you ensure reproducible splits?
    """
    # Read raw data
    df = pd.read_csv(os.path.join(input_path, 'churn.csv'))

    # TODO: Your preprocessing logic here
    # - Drop unnecessary columns
    # - Handle missing values
    # - Encode categorical variables
    # - Create features

    # TODO: Split data
    # Hint: Use train_test_split twice for three-way split

    # TODO: Save data (target column first, no headers!)
    # train_df.to_csv(os.path.join(train_path, 'train.csv'), header=False, index=False)

    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-split', type=float, default=0.7)
    parser.add_argument('--validation-split', type=float, default=0.2)
    args = parser.parse_args()

    preprocess_data(
        "/opt/ml/processing/input",
        "/opt/ml/processing/train",
        "/opt/ml/processing/validation",
        "/opt/ml/processing/test",
        args.train_split,
        args.validation_split
    )
```

#### 3. Configure and Run the Processor
**Your Task:** Set up the PyTorchProcessor and execute the processing job.

**Getting Started Pattern:**
```python
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

# Create the processor (using PyTorchProcessor for modern framework support)
pytorch_processor = PyTorchProcessor(
    framework_version='2.0',
    role=role,  # Your execution role
    instance_type='ml.m5.xlarge',  # Research: How to choose instance size?
    instance_count=1,
    base_job_name='your-name-preprocessing',
    py_version='py310'  # Python 3.10
)
```

**Research Questions:**
1. How does `framework_version` relate to PyTorch versions?
2. When would you use `instance_count > 1`? (Hint: Look into distributed processing)
3. What's the difference between `base_job_name` and the actual job name?
4. Why specify `py_version`? What Python versions are available?

**Running the Processing Job:**
```python
# Your Task: Figure out how to configure ProcessingInput and ProcessingOutput
pytorch_processor.run(
    code='preprocessing.py',           # Your script
    source_dir='src/',                  # Directory with code AND requirements.txt
    inputs=[
        ProcessingInput(
            source=s3_input_data,       # Your raw data S3 URI
            destination='/opt/ml/processing/input'
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name='train',
            source='/opt/ml/processing/train',
            destination=s3_output_train_data
        ),
        # TODO: Add outputs for validation and test
    ],
    arguments=['--train-split', '0.7', '--validation-split', '0.2']
)
```

**Key Feature: Automatic Dependency Installation**
When you include a `requirements.txt` file in your `source_dir`, PyTorchProcessor automatically:
1. Uploads all files from `source_dir` to the container
2. Finds `requirements.txt` in that directory
3. Runs `pip install -r requirements.txt` before executing your code

**Advanced Challenge:**
- Add a `requirements.txt` in your `source_dir` to install additional libraries (e.g., `pandas`, `scikit-learn`)
- Pass the train/validation split ratios as configurable arguments
- Include preprocessing parameters (e.g., encoding strategy) as arguments

#### 4. Verify Your Processing Results
**Your Challenge:** Validate that preprocessing worked correctly.

**Verification Checklist:**
- [ ] Do output files exist in S3 at the specified locations?
- [ ] Is the target column in the first position?
- [ ] Are there no headers in the CSV files?
- [ ] Do the split ratios match what you specified?
- [ ] Are all values numeric (no strings or NaN)?
- [ ] Can you load and inspect the processed data?

**Debugging Pattern:**
```python
# Download and inspect your processed data
import pandas as pd

# Download one of the output files
!aws s3 cp {s3_output_train_data}/train.csv ./train_processed.csv

# Verify format
df = pd.read_csv('train_processed.csv', header=None)
print(f"Shape: {df.shape}")
print(f"First column (target): {df.iloc[:, 0].unique()}")  # Should be 0/1
print(f"Any NaNs: {df.isna().sum().sum()}")                # Should be 0
print(df.head())
```

### 🔍 Debugging Processing Jobs

**Job fails immediately:**
- Check CloudWatch logs for Python errors
- Verify your script runs locally first
- Ensure all imports are available in the container

**"No module named X" errors:**
- Add a `requirements.txt` in your `source_dir` with all dependencies
- PyTorchProcessor will automatically install them before running your code
- For SKLearnProcessor (doesn't support requirements.txt): manually install in script or use ProcessingInput workaround

**Output files not created:**
- Check that your script actually writes to output paths
- Verify the paths match between script and ProcessingOutput
- Look for exceptions in CloudWatch logs

**Data format issues:**
- Verify target column is first
- Ensure no headers in output CSVs
- Check data types are all numeric

### ✅ Success Criteria

**Technical Success:**
- [ ] Processing job completes successfully
- [ ] Three datasets created (train, validation, test) in correct S3 locations
- [ ] Data is in XGBoost-compatible format
- [ ] Split ratios are correct
- [ ] Can successfully use outputs for training

**Understanding Success:**
- [ ] Can explain why SageMaker Processing is better than local preprocessing
- [ ] Understand when to use each Processor type
- [ ] Know how to debug processing job failures
- [ ] Can add custom dependencies to processing jobs

### 💡 Advanced Topics to Explore

**Distributed Processing:**
- Research how to split processing across multiple instances
- When is distributed processing worth the complexity?

**Processing Job Monitoring:**
- How do you set up CloudWatch alarms for processing failures?
- What metrics should you monitor?

**Cost Optimization:**
- How do you choose the right instance type for your data size?
- Spot instances for processing - when are they appropriate?

### 📚 Research Resources
- [SageMaker Processing Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html)
- [SKLearnProcessor API Reference](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html#scikit-learn-processor)
- [Processing Container Environment](https://docs.aws.amazon.com/sagemaker/latest/dg/build-your-own-processing-container.html)

---

## 🎓 Assignment 2: Training with Framework Estimators

### 🎯 Your Mission
Use the SageMaker SDK's Estimator classes to train models with a simpler, more Pythonic API than the raw TrainingJob approach from Lab 1.

### 🤔 Understanding Estimators

#### What's Different from Lab 1?
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

### 🛠️ What You Need to Build

#### 1. Choose Your Estimator Type

**Available Options:**
```python
from sagemaker.estimator import Estimator
from sagemaker.xgboost import XGBoost
from sagemaker.sklearn import SKLearn
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
    framework_version='1.7-1',
    hyperparameters={
        'max_depth': 5,
        'eta': 0.2,
        'objective': 'binary:logistic',
        'num_round': 100
    }
)
```

**Your Task:** Find out why Pytorch is a subclass of Framework

**Think About:**
- How would you add custom evaluation metrics?

#### 2. Create a Custom Training Script

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

#### 3. Train Your Model with the Estimator

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

#### 4. Retrieve and Analyze Training Results

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

### 🔍 Debugging Training Jobs

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

### ✅ Success Criteria

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

### 💡 Advanced Exploration

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

### 📚 Research Resources
- [SageMaker Estimator Documentation](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html)
- [XGBoost Estimator Guide](https://sagemaker.readthedocs.io/en/stable/frameworks/xgboost/using_xgboost.html)
- [Training Script Requirements](https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-training-container.html)

---

## ⚡ Assignment 3: Hyperparameter Tuning with the SDK

### 🎯 Your Mission
Use the SageMaker SDK's `HyperparameterTuner` to automatically find optimal hyperparameters for your model - with a much simpler API than the raw tuning jobs from Lab 1.

### 🤔 Understanding SDK Tuning vs. Lab 1

#### What's Better with the SDK?
**In Lab 1 with sagemaker-core:**
- Manual configuration of `HyperParameterTuningJobConfig`
- Complex `ParameterRanges` shape definitions
- Verbose `HyperParameterTuningJobObjective` setup

**With the SDK:**
- Simple `HyperparameterTuner` class
- Clean parameter range definitions
- Automatic best model selection

**Research Questions:**
1. How does `HyperparameterTuner` simplify the tuning workflow?
2. What happens to your Estimator when you wrap it in a tuner?
3. How does the SDK help you find and deploy the best model?

### 🛠️ What You Need to Build

#### 1. Define Your Search Space

**Your Task:** Choose hyperparameters to tune and their ranges.

**For XGBoost, Consider Tuning:**
```python
from sagemaker.tuner import (
    IntegerParameter,
    ContinuousParameter,
    CategoricalParameter
)

hyperparameter_ranges = {
    'max_depth': IntegerParameter(3, 10),
    'eta': ContinuousParameter(0.01, 0.3),
    'min_child_weight': IntegerParameter(1, 10),
    'subsample': ContinuousParameter(0.5, 1.0),
    'gamma': ContinuousParameter(0, 5),
    'alpha': ContinuousParameter(0, 2)
}
```

**Research Questions:**
1. What does each hyperparameter control in XGBoost?
2. How did you determine the min/max values for each range?
3. Why not tune all possible hyperparameters?

**Strategic Thinking:**
- Start with 3-4 most impactful parameters
- Use domain knowledge to set reasonable ranges
- Consider computational budget (# of training jobs)

**Your Challenge:** Justify your hyperparameter choices.
- Why these parameters and not others?
- How did you determine the ranges?
- What's the expected impact on model performance?

#### 2. Configure the Tuner

**Your Task:** Set up a `HyperparameterTuner` with your estimator.

**Basic Tuner Setup:**
```python
from sagemaker.tuner import HyperparameterTuner

# Reuse your estimator from Assignment 2
# Remove the hyperparameters you want to tune from the estimator
# The tuner will inject them automatically

tuner = HyperparameterTuner(
    estimator=xgb_estimator,
    objective_metric_name='validation:auc',    # What to optimize
    objective_type='Maximize',                  # Maximize or Minimize
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=[                        # How to parse metrics from logs
        {'Name': 'validation:auc', 'Regex': 'validation-auc:([0-9\\.]+)'}
    ],
    max_jobs=20,                               # Total jobs to run
    max_parallel_jobs=3,                       # How many to run simultaneously
    strategy='Bayesian',                       # 'Bayesian', 'Random', or 'Grid'
    early_stopping_type='Auto'                 # Stop poor performers early
)
```

**Research Deep Dive:**

**1. Objective Metric:**
- How do you know what metrics your training emits?
- What's the regex pattern for extracting metrics from logs?
- Why optimize AUC vs. accuracy for imbalanced datasets?

**2. Tuning Strategy:**
- **Bayesian**: Smart exploration based on previous results (default)
- **Random**: Random sampling (good for baseline)
- **Grid**: Exhaustive search (expensive but thorough)

**Your Task:** Research and choose the best strategy for your use case.

**3. Resource Management:**
- `max_jobs`: Total training jobs (cost consideration)
- `max_parallel_jobs`: Concurrent jobs (time vs. cost trade-off)
- Early stopping: Automatically stop underperforming jobs

**Calculate Your Budget:**
```python
# Example calculation
instance_cost_per_hour = 0.269  # ml.m5.xlarge
estimated_job_duration_hours = 0.1  # 6 minutes
max_jobs = 20
max_parallel_jobs = 3

estimated_cost = max_jobs * estimated_job_duration_hours * instance_cost_per_hour
estimated_time_hours = (max_jobs / max_parallel_jobs) * estimated_job_duration_hours

print(f"Estimated cost: ${estimated_cost:.2f}")
print(f"Estimated time: {estimated_time_hours:.1f} hours")
```

#### 3. Run the Tuning Job

**Your Task:** Start tuning and monitor progress.

**Starting Tuning:**
```python
# Use the same data inputs as training
tuner.fit({
    'train': train_input,
    'validation': validation_input
})

# Wait for completion (optional - can run async)
tuner.wait()
```

**Monitoring Progress:**
```python
# Get tuning job status
tuning_job_name = tuner.latest_tuning_job.name
print(f"Tuning job: {tuning_job_name}")

# In SageMaker Studio or Console:
# - Watch the hyperparameter tuning job progress
# - See which hyperparameter combinations are being tried
# - Monitor the objective metric improvement over time
```

**Research Questions:**
1. How can you monitor tuning progress programmatically?
2. What information is available while tuning is running?
3. How do you know if tuning is making progress or stuck?

#### 4. Analyze Tuning Results

**Your Challenge:** Find the best model and understand why it's best.

**Getting the Best Model:**
```python
# Get best training job information
best_job = tuner.best_training_job()
print(f"Best training job: {best_job}")

# Get best hyperparameters
best_hyperparameters = tuner.best_estimator().hyperparameters()
print(f"Best hyperparameters: {best_hyperparameters}")

# Get best metric value
tuning_analytics = tuner.analytics()
best_metric = tuning_analytics.dataframe().sort_values('FinalObjectiveValue', ascending=False).iloc[0]
print(f"Best {tuner.objective_metric_name}: {best_metric['FinalObjectiveValue']}")
```

**Deep Analysis:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Get all tuning job results
tuning_df = tuner.analytics().dataframe()

# Analyze hyperparameter impact
print(tuning_df.head(10))  # Top 10 models

# Your Task: Visualize the results
# 1. Plot objective metric vs. each hyperparameter
# 2. Identify which hyperparameters had the most impact
# 3. Check for convergence (is best model at boundary of ranges?)

# Example: Impact of max_depth
plt.scatter(tuning_df['max_depth'], tuning_df['FinalObjectiveValue'])
plt.xlabel('max_depth')
plt.ylabel('Validation AUC')
plt.title('Impact of max_depth on Model Performance')
plt.show()

# TODO: Repeat for other hyperparameters
```

**Critical Thinking Questions:**
1. Which hyperparameter had the biggest impact on performance?
2. Did the tuning converge, or do you need wider ranges?
3. How much did the best model improve over your baseline?
4. Was the computational cost worth the improvement?

#### 5. Deploy the Best Model (Optional Preview)

**Your Task:** Deploy your best model to an endpoint.

**Deployment Pattern:**
```python
# The tuner's best model can be deployed directly
predictor = tuner.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)

# Test prediction
test_data = [...]  # Your test sample
prediction = predictor.predict(test_data)
print(f"Prediction: {prediction}")

# Don't forget to clean up!
predictor.delete_endpoint()
```

**Research:** How does this compare to manual deployment?

### 🔍 Debugging Hyperparameter Tuning

**All jobs fail with same error:**
- Check that your estimator works with a fixed set of hyperparameters first
- Verify metric regex matches your log output
- Ensure objective metric is actually emitted by training

**Tuning not improving:**
- Check if hyperparameter ranges are too narrow
- Verify you're tuning impactful parameters
- Consider if more jobs are needed
- Look for bugs in training script

**Jobs timing out:**
- Reduce dataset size for tuning experiments
- Decrease `num_round` or equivalent training parameter
- Increase `max_runtime_in_seconds`

**Costs running too high:**
- Reduce `max_jobs` or `max_parallel_jobs`
- Use smaller instance types for tuning
- Enable early stopping
- Consider spot instances for cost savings

### ✅ Success Criteria

**Technical Success:**
- [ ] Tuning job completes successfully
- [ ] Best model identified and metrics recorded
- [ ] Improvement over baseline model
- [ ] Can access and analyze tuning results
- [ ] Understand which hyperparameters mattered most

**Understanding Success:**
- [ ] Can explain why Bayesian optimization is effective
- [ ] Understand the cost/time trade-offs in tuning
- [ ] Know how to set appropriate hyperparameter ranges
- [ ] Can interpret tuning analytics and make data-driven decisions

### 💡 Advanced Challenges

**Warm Start Tuning:**
- Research how to continue tuning from a previous job
- When is this useful?
- How do you configure it?

**Multi-Objective Optimization:**
- Can you optimize for multiple metrics (e.g., accuracy AND inference latency)?
- How would you balance competing objectives?

**Transfer Learning with Tuning:**
- How might you use tuning results from one dataset to inform another?
- What assumptions would you need to validate?

### 📚 Research Resources
- [Hyperparameter Tuning Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html)
- [HyperparameterTuner API](https://sagemaker.readthedocs.io/en/stable/api/training/tuner.html)
- [Bayesian Optimization Explained](https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-automatic-model-tuning-becomes-more-efficient-with-warm-start-of-hyperparameter-tuning-jobs/)

---

## 📦 Assignment 4: Model Registry and Lifecycle Management

### 🎯 Your Mission
Learn to manage your ML models professionally using SageMaker Model Registry - tracking versions, approvals, lineage, and deployment status.

### 🤔 Understanding Model Registry

#### Why Do You Need Model Registry?
**Real-World Problem:**
- You've trained dozens of models over time
- Some are in production, some are experiments
- You need to track which model version is deployed where
- You need approval workflows before production deployment
- You need to know which data/code produced which model

**Model Registry Solves:**
- **Versioning**: Track all model versions with metadata
- **Lineage**: Connect models to training jobs, data, and code
- **Approval**: Formal approval workflow for production deployment
- **Deployment Tracking**: Know which models are deployed where
- **Model Comparison**: Compare metrics across versions

**Research Questions:**
1. How does Model Registry differ from just saving models to S3?
2. What's the relationship between Model Package Group, Model Package, and Model Version?
3. How does approval status affect model deployment?

### 🛠️ What You Need to Build

#### 1. Create a Model Package Group

**Your Task:** Set up a model package group to hold all versions of your model.

**Understanding the Hierarchy:**
```
Model Package Group (e.g., "customer-churn-models")
  └── Model Package Version 1 (first training run)
  └── Model Package Version 2 (after hyperparameter tuning)
  └── Model Package Version 3 (retrained on new data)
```

**Creating the Group:**
```python
import sagemaker
from sagemaker.model import ModelPackage

sm_client = boto3.client('sagemaker')

# Create a model package group
model_package_group_name = "your-name-customer-churn-models"

model_package_group_response = sm_client.create_model_package_group(
    ModelPackageGroupName=model_package_group_name,
    ModelPackageGroupDescription="Customer churn prediction models - all versions",
    Tags=[
        {'Key': 'Project', 'Value': 'CustomerChurn'},
        {'Key': 'Team', 'Value': 'YourName'}
    ]
)

print(f"Model Package Group ARN: {model_package_group_response['ModelPackageGroupArn']}")
```

**Research:**
- How do you list existing model package groups?
- Can you update the description or tags later?
- How do you delete a model package group?

#### 2. Register Your Model

**Your Task:** Register a trained model in the Model Registry with metrics and metadata.

**Option A: Register from Estimator (Recommended)**
```python
# From your training estimator (Assignment 2)
model_metrics = {
    "classification_metrics": {
        "validation:auc": {
            "value": 0.92,
            "standard_deviation": 0.01
        },
        "validation:accuracy": {
            "value": 0.89
        }
    }
}

# Register the model
model_package = xgb_estimator.register(
    model_package_group_name=model_package_group_name,
    approval_status="PendingManualApproval",  # Initial status
    inference_instances=["ml.t2.medium", "ml.m5.large"],
    transform_instances=["ml.m5.large"],
    content_types=["text/csv"],
    response_types=["text/csv"],
    model_metrics=model_metrics,
    description="XGBoost model for customer churn prediction - baseline",
)

print(f"Model Package ARN: {model_package.model_package_arn}")
```

**Option B: Register from Tuner Best Model**
```python
# From your hyperparameter tuning job (Assignment 3)
best_estimator = tuner.best_estimator()

# Get best metrics from tuning
best_training_job = tuner.best_training_job()
# TODO: Extract actual metrics from the training job

model_package = best_estimator.register(
    model_package_group_name=model_package_group_name,
    approval_status="PendingManualApproval",
    inference_instances=["ml.t2.medium", "ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_metrics={...},  # Add actual metrics here
    description="XGBoost model - best from hyperparameter tuning"
)
```

**Research Deep Dive:**

**1. Approval Status:**
- `PendingManualApproval`: Awaiting human review
- `Approved`: Ready for production deployment
- `Rejected`: Not suitable for deployment

**Your Task:** Design an approval workflow:
- Who should approve models?
- What criteria should they use?
- How do you automate approval for non-production environments?

**2. Model Metrics:**
- What metrics should you track?
- How do you structure metrics for the registry?
- Can you add custom metrics?

**3. Inference Specifications:**
- `inference_instances`: Instance types for real-time endpoints
- `transform_instances`: Instance types for batch transform
- `content_types`: Supported input formats
- `response_types`: Supported output formats

#### 3. Manage Model Versions

**Your Task:** Work with multiple model versions and update approval status.

**Listing Model Versions:**
```python
# List all versions in the group
model_versions = sm_client.list_model_packages(
    ModelPackageGroupName=model_package_group_name,
    SortBy='CreationTime',
    SortOrder='Descending'
)

print(f"Total versions: {len(model_versions['ModelPackageSummaryList'])}")
for version in model_versions['ModelPackageSummaryList']:
    print(f"Version {version['ModelPackageVersion']}: "
          f"Status={version['ModelApprovalStatus']}, "
          f"Created={version['CreationTime']}")
```

**Updating Approval Status:**
```python
# Approve a model for production
model_package_arn = model_package.model_package_arn

sm_client.update_model_package(
    ModelPackageArn=model_package_arn,
    ModelApprovalStatus='Approved',
    ApprovalDescription='Model approved after validation - AUC > 0.90 threshold met'
)

print(f"Model {model_package_arn} approved for production deployment")
```

**Your Challenge:** Implement a model promotion workflow:
1. Train and register model with `PendingManualApproval`
2. Run validation tests
3. If tests pass, approve model
4. If tests fail, reject model with explanation

**Example Validation Function:**
```python
def validate_and_approve_model(model_package_arn, validation_data, threshold_auc=0.85):
    """
    Your Task: Implement model validation logic

    Steps:
    1. Deploy model to temporary endpoint
    2. Run predictions on validation data
    3. Calculate metrics
    4. Compare to thresholds
    5. Approve or reject based on results
    6. Clean up temporary endpoint
    """
    # TODO: Implement validation logic
    pass
```

#### 4. Query Model Lineage

**Your Task:** Understand the full lineage of your model.

**Exploring Model Lineage:**
```python
# Get detailed model package information
model_details = sm_client.describe_model_package(
    ModelPackageName=model_package_arn
)

# Extract lineage information
training_job_arn = model_details.get('SourceAlgorithmSpecification', {}).get('SourceAlgorithms', [{}])[0].get('ModelDataUrl')

print(f"Model artifacts: {model_details['InferenceSpecification']['Containers'][0]['ModelDataUrl']}")
print(f"Created from training job: {training_job_arn}")

# Your Task: Trace back to:
# - Which training job created this model?
# - What hyperparameters were used?
# - What data was used for training?
# - What code version was used?
```

**Advanced Lineage Tracking:**
```python
# Use SageMaker Lineage Tracking (optional advanced topic)
from sagemaker.lineage.visualizer import LineageTableVisualizer

visualizer = LineageTableVisualizer(sagemaker.session.Session())

# Visualize lineage for the model
lineage_table = visualizer.show(model_package_arn)
print(lineage_table)
```

**Research:**
- What information is automatically captured in lineage?
- How can you add custom lineage information?
- How does this help with model governance and compliance?

#### 5. Deploy from Model Registry

**Your Task:** Deploy an approved model from the registry.

**Deployment Pattern:**
```python
from sagemaker import ModelPackage

# Get the latest approved model
approved_models = sm_client.list_model_packages(
    ModelPackageGroupName=model_package_group_name,
    ModelApprovalStatus='Approved',
    SortBy='CreationTime',
    SortOrder='Descending',
    MaxResults=1
)

if approved_models['ModelPackageSummaryList']:
    latest_approved_arn = approved_models['ModelPackageSummaryList'][0]['ModelPackageArn']

    # Create a model from the package
    model = ModelPackage(
        role=role,
        model_package_arn=latest_approved_arn,
        sagemaker_session=sagemaker.Session()
    )

    # Deploy to endpoint
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.t2.medium',
        endpoint_name='your-name-churn-prod'
    )

    print(f"Model deployed to endpoint: {predictor.endpoint_name}")
else:
    print("No approved models available for deployment")
```

**Research Questions:**
1. How does deploying from Model Registry differ from deploying from Estimator?
2. What guarantees does the approval workflow provide?
3. How do you handle model rollback if a deployed model performs poorly?

### 🔍 Real-World Model Registry Workflows

**Scenario 1: Model Retraining Pipeline**
```python
# Weekly retraining workflow
# 1. Train new model on latest data
# 2. Register in Model Registry (PendingManualApproval)
# 3. Run automated validation
# 4. If validation passes, approve
# 5. Deploy approved model to staging
# 6. After business validation, promote to production
```

**Scenario 2: A/B Testing with Model Registry**
```python
# Deploy two model versions for comparison
# 1. Register both models
# 2. Approve both for testing
# 3. Deploy to multi-variant endpoint
# 4. Monitor performance
# 5. Approve winning model for production
# 6. Reject losing model
```

**Scenario 3: Model Audit Trail**
```python
# Compliance requirement: Full audit trail for all models
# 1. Every model must be registered (not just saved to S3)
# 2. Approval must be documented with justification
# 3. Lineage must link model to training data and code
# 4. Deployment history must be tracked
```

**Your Challenge:** Design a complete model lifecycle workflow for your organization.

### ✅ Success Criteria

**Technical Success:**
- [ ] Model package group created
- [ ] At least one model registered with metrics
- [ ] Can update approval status
- [ ] Can list and compare model versions
- [ ] Can deploy from Model Registry

**Understanding Success:**
- [ ] Understand the model registry hierarchy
- [ ] Know when to use approval statuses
- [ ] Can explain model lineage and governance benefits
- [ ] Understand how registry fits into MLOps workflows

### 💡 Advanced Topics

**CI/CD Integration:**
- How do you integrate Model Registry with CI/CD pipelines?
- What automated tests should run before approval?
- How do you handle rollback in production?

**Cross-Account Model Sharing:**
- Research how to share models across AWS accounts
- What are the security considerations?
- How do you manage permissions?

**Model Monitoring Integration:**
- How does Model Registry integrate with Model Monitor?
- How do you track model performance drift?
- When should you automatically reject models?

### 📚 Research Resources
- [Model Registry Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
- [MLOps with Model Registry](https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-model-registry/)
- [Model Governance Best Practices](https://docs.aws.amazon.com/sagemaker/latest/dg/model-governance.html)

---

## 🎓 What You've Accomplished

### Technical Skills Mastered

**Processing:**
- Built reusable data processing workflows with SageMaker Processors
- Configured distributed processing jobs for large-scale data
- Managed processing job inputs, outputs, and dependencies

**Training:**
- Used Framework Estimators for flexible, high-level training
- Created custom training scripts following SageMaker conventions
- Configured and monitored training jobs with the SDK

**Hyperparameter Tuning:**
- Automated hyperparameter optimization with intelligent search strategies
- Analyzed tuning results to understand parameter importance
- Balanced computational budget with model improvement goals

**Model Management:**
- Implemented professional model versioning and lifecycle management
- Created approval workflows for production deployments
- Tracked model lineage for governance and compliance

### Conceptual Understanding

**Abstraction Layers:**
- **Lab 1 (sagemaker-core)**: Low-level building blocks, full control
- **This Lab (SDK)**: High-level components, productivity focused
- **Lab 4 (Pipelines)**: Orchestration layer, automation focused

**MLOps Maturity:**
You've progressed from:
- Running experiments → Building reusable components
- Manual tracking → Automated versioning and lineage
- Ad-hoc deployments → Formal approval workflows
- Individual jobs → Connected ML workflows

### 🤔 Reflection Questions

**Technical Decisions:**
1. When would you use sagemaker-core vs. SDK vs. Pipelines?
2. How do you choose between built-in algorithms and custom training scripts?
3. What factors determine your hyperparameter tuning strategy?

**Business Impact:**
1. How do you justify the infrastructure costs of automated tuning and processing?
2. What ROI do model registry and approval workflows provide?
3. How do you communicate model improvement to non-technical stakeholders?

**Production Readiness:**
1. What additional components are needed for production ML?
2. How do you ensure model quality before production deployment?
3. What monitoring and alerting should be in place?

---

## 🚀 Next Steps: Lab 4 - SageMaker Pipelines

Now that you've mastered individual components, you're ready to orchestrate them into automated ML pipelines!

**In Lab 4, You'll Learn:**
- Connect processing, training, evaluation, and registration into a single pipeline
- Use pipeline parameters for flexible, reusable workflows
- Implement conditional logic (e.g., only deploy if accuracy > threshold)
- Version and schedule pipeline executions
- Build end-to-end automated ML workflows

**The Bridge You've Built:**
- ✅ You know how each component works individually
- ✅ You understand the SDK abstractions
- ✅ You're familiar with model management
- ✅ You're ready to orchestrate it all!

---

## 📚 Continuous Learning Resources

### Official Documentation
- [SageMaker Python SDK Documentation](https://sagemaker.readthedocs.io/)
- [SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/)
- [SageMaker Examples Repository](https://github.com/aws/amazon-sagemaker-examples)

### Deep Dives
- [Distributed Training on SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training.html)
- [SageMaker Debugger for Training Insights](https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html)
- [Cost Optimization for ML Workloads](https://aws.amazon.com/blogs/machine-learning/ensuring-machine-learning-model-quality-at-scale-with-amazon-sagemaker/)

### MLOps Best Practices
- [ML Governance with Model Registry](https://aws.amazon.com/blogs/machine-learning/govern-your-machine-learning-models-with-amazon-sagemaker-model-registry/)
- [Building MLOps Pipelines](https://aws.amazon.com/blogs/machine-learning/building-automating-managing-and-scaling-ml-workflows-using-amazon-sagemaker-pipelines/)

**Remember:** The goal isn't just to complete assignments - it's to build the judgment to design production ML systems that are maintainable, scalable, and aligned with business objectives.

---

## 🎉 Congratulations!

You've completed the bridge between low-level SageMaker operations and high-level pipeline orchestration. You now have the skills to build production-ready ML components using the SageMaker SDK. Ready for Lab 4? Let's build some pipelines! 🚀
