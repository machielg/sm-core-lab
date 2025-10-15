# 🏗️ Assignment 1: Data Processing with SageMaker Processors

## 🎯 Your Mission
Build a reusable data preprocessing workflow using the SageMaker SDK's Processor classes. Transform raw churn data into train/validation/test sets that are ready for model training.

## 🤔 Understanding Processors vs. Manual Preprocessing

### Key Concept: Why Use SageMaker Processors?
**Research Questions:**
1. What's the difference between preprocessing locally vs. using SageMaker Processing Jobs?
2. How do Processors differ from the raw Processing Job shapes you used in Lab 1?
3. When would you choose `SKLearnProcessor` vs. `ScriptProcessor` vs `PyTorchProcessor`?

**Real-World Scenario:** Your data is 50GB and growing. Local preprocessing crashes your laptop. How does SageMaker Processing solve this?

## 🛠️ What You Need to Build

### 1. Choose Your Processor Type
**Your Task:** Research and select the right processor for your use case.

**Available Processor Types:**
```python
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.xgboost.processing import XGBoostProcessor
```

**Decision Matrix:**
- **PyTorchProcessor**: ✅ Recommended - Modern framework processor, automatic requirements.txt support
- **XGBoostProcessor**: ⚠️ Potential alternative - Automatic requirements.txt support, XGBoost-optimized but version is old
- **SKLearnProcessor**: ⚠️ Scikit-learn optimized, but NO automatic requirements.txt support
- **ScriptProcessor**: ⚠️ Bare bones, manual dependency management required

**Critical Feature: requirements.txt Support**
- ✅ **PyTorchProcessor**, **XGBoostProcessor**: Automatically install packages from `requirements.txt` when using `source_dir`
- ❌ **SKLearnProcessor**: Does NOT support automatic requirements.txt installation (known limitation)
- ❌ **ScriptProcessor**: Generic processor, manual dependency management required

**Recommendation:** Use **PyTorchProcessor** for this assignment - it's based on the modern framework processor architecture, supports automatic dependency installation via `source_dir`, and provides excellent flexibility for preprocessing tasks.

**Think About:**
- Which Python libraries does your preprocessing code need?
- Do you need automatic dependency installation (requirements.txt)?
- What's the trade-off between ease of use and flexibility?

### 2. Create Your Preprocessing Script
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

### 3. Configure and Run the Processor
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

If you want to use the AWS churn dataset directly you, the url is

`f"s3://sagemaker-example-files-prod-{lab_session.region}/datasets/tabular/synthetic/churn.txt"`

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

### 4. Verify Your Processing Results
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

## 🔍 Debugging Processing Jobs

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

## ✅ Success Criteria

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

## 💡 Advanced Topics to Explore

**Distributed Processing:**
- Research how to split processing across multiple instances
- When is distributed processing worth the complexity?

**Processing Job Monitoring:**
- How do you set up CloudWatch alarms for processing failures?
- What metrics should you monitor?

**Cost Optimization:**
- How do you choose the right instance type for your data size?
- Spot instances for processing - when are they appropriate?

## 📚 Research Resources
- [SageMaker Processing Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html)
- [SKLearnProcessor API Reference](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html#scikit-learn-processor)
- [Processing Container Environment](https://docs.aws.amazon.com/sagemaker/latest/dg/build-your-own-processing-container.html)

---

Ready to move on? Continue to [Assignment 2: Training with Framework Estimators](Assignment_2_Training.md) →