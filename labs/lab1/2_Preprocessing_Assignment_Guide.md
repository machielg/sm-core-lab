# 🔄 Data Preprocessing with SageMaker Core - Assignment Guide

## 📖 Introduction

This assignment teaches you how to run data preprocessing jobs in SageMaker using the **sagemaker-core** library. You'll transform raw customer churn data into ML-ready train/validation/test datasets using SageMaker Processing Jobs.

**Learning Objectives:**
- Understand SageMaker Processing Jobs architecture
- Use sagemaker-core shapes for explicit job configuration
- Create preprocessing scripts for containerized environments
- Manage input/output data channels
- Debug processing job failures

**Prerequisites:**
- Complete Lab 1 (SageMaker Essentials)
- Understanding of the customer churn dataset
- Basic knowledge of data preprocessing concepts

---

## 🎯 Assignment: Build a Data Preprocessing Pipeline

### Your Mission
Create a SageMaker Processing Job using sagemaker-core that takes raw churn data and outputs train/validation/test datasets ready for model training.

---

## 🤔 Understanding Processing Jobs

### What is a Processing Job?
**Research Questions:**
1. What's the difference between preprocessing locally vs. using SageMaker Processing?
2. When would you use Processing Jobs instead of local preprocessing?
3. How do Processing Jobs scale for large datasets?

**Real-World Scenario:**
Your customer churn dataset is 500GB and growing. Your laptop crashes when trying to load it. How can SageMaker Processing help?

### Processing Job Architecture

**Key Concept:** SageMaker Processing runs your code in a container with:
- **Input channels**: S3 data mounted to container paths
- **Processing script**: Your Python code running in the container
- **Output channels**: Results uploaded back to S3

```
S3 Input Data → Container (/opt/ml/processing/input) → Your Script → Container (/opt/ml/processing/output) → S3 Output
```

**Research:** What are the standard SageMaker Processing paths?

---

## 🏗️ Part 1: Understanding sagemaker-core Shapes

### Your Task: Understand the Building Blocks

Before building the processing job, research these sagemaker-core shapes:

#### 1. `AppSpecification`
```python
from sagemaker_core.main.shapes import AppSpecification
```

**Research Questions:**
- What does `image_uri` specify?
- What's the difference between `container_entrypoint` and `container_arguments`?
- How do you pass your preprocessing script to the container?

#### 2. `ProcessingResources`
```python
from sagemaker_core.main.shapes import ProcessingResources, ProcessingClusterConfig
```

**Research Questions:**
- How do you specify instance type and count?
- What's the minimum `volume_size_in_gb` needed?
- When would you use multiple instances?

#### 3. `ProcessingInput` and `ProcessingS3Input`
```python
from sagemaker_core.main.shapes import ProcessingInput, ProcessingS3Input
```

**Research Questions:**
- What's the difference between `s3_data_type="S3Prefix"` vs `"ManifestFile"`?
- What does `s3_input_mode="File"` vs `"Pipe"` mean?
- What is `local_path` used for?

#### 4. `ProcessingOutput` and `ProcessingOutputConfig`
```python
from sagemaker_core.main.shapes import (
    ProcessingOutput,
    ProcessingOutputConfig,
    ProcessingS3Output
)
```

**Research Questions:**
- What does `s3_upload_mode="EndOfJob"` mean?
- How do you create multiple output channels (train/validation/test)?
- What happens if your script doesn't write to the output path?

---

## 🛠️ Part 2: Create Your Preprocessing Script

### Your Challenge: Write `preprocessing.py`

Your script must work inside the SageMaker Processing container with these requirements:

#### Standard SageMaker Processing Paths
```python
# These paths are mounted by SageMaker automatically
INPUT_DATA_PATH = "/opt/ml/processing/input/data"
OUTPUT_TRAIN_PATH = "/opt/ml/processing/output/train"
OUTPUT_VALIDATION_PATH = "/opt/ml/processing/output/validation"
OUTPUT_TEST_PATH = "/opt/ml/processing/output/test"
```

#### Script Requirements
Your `preprocessing.py` must:
1. **Read** raw churn data from the input path
2. **Clean** data (handle missing values, remove duplicates)
3. **Transform** categorical variables (encode for XGBoost)
4. **Split** into train (70%), validation (20%), test (10%)
5. **Save** each dataset to appropriate output paths
6. **Format** for XGBoost: target column first, no headers, CSV

#### Getting Started Pattern
```python
# preprocessing.py
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def preprocess_data(input_path, train_output, val_output, test_output, train_split=0.7):
    """
    Your Task: Implement preprocessing logic

    Steps:
    1. Read data from input_path
    2. Clean and transform data
    3. Split into train/val/test
    4. Save to output paths (target first, no headers!)
    """
    # TODO: Load data
    # files = [os.path.join(input_path, f) for f in os.listdir(input_path)]
    # df = pd.concat([pd.read_csv(f) for f in files])

    # TODO: Data cleaning
    # - Handle missing values
    # - Drop unnecessary columns
    # - Encode categorical variables

    # TODO: Split data
    # train_df, temp_df = train_test_split(df, train_size=0.7, random_state=42)
    # val_df, test_df = train_test_split(temp_df, train_size=0.67, random_state=42)

    # TODO: Save data (XGBoost format: target first, no headers)
    # train_df.to_csv(os.path.join(train_output, 'train.csv'), header=False, index=False)

    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split', type=float, default=0.7)
    args, _ = parser.parse_known_args()

    preprocess_data(
        "/opt/ml/processing/input/data",
        "/opt/ml/processing/output/train",
        "/opt/ml/processing/output/validation",
        "/opt/ml/processing/output/test",
        args.train_test_split
    )
```

**Research Questions:**
1. How do you handle categorical columns for XGBoost?
2. Should you normalize numeric features? Why or why not for XGBoost?
3. How do you ensure reproducible data splits?

---

## 📦 Part 3: Build the Processing Job

### Your Task: Use sagemaker-core to Create the Job

#### Step 1: Import Required Modules
```python
from sagemaker_core.main.shapes import (
    AppSpecification,
    ProcessingResources,
    ProcessingInput,
    ProcessingClusterConfig,
    ProcessingOutput,
    ProcessingOutputConfig,
    ProcessingStoppingCondition,
    ProcessingS3Input,
    ProcessingS3Output
)
from sagemaker_core.main.resources import ProcessingJob
```

#### Step 2: Upload Your Code
```python
# Your Task: Upload preprocessing.py to S3
# Hint: Use your lab session's upload methods

code_s3_uri = # TODO: Upload your script
print(f"Code uploaded to: {code_s3_uri}")
```

#### Step 3: Define Data Locations
```python
# Input data (raw churn data)
data_s3_uri = f"s3://sagemaker-example-files-prod-{region}/datasets/tabular/synthetic/churn.txt"

# Output location for processed data
output_s3_uri = # TODO: Define your S3 output bucket/prefix
```

#### Step 4: Create the Processing Job
```python
job = ProcessingJob.create(
    processing_job_name=# TODO: Create unique job name,
    role_arn=# TODO: Your execution role,
    session=# TODO: Your boto session,
    region=# TODO: Your AWS region,

    # Application specification
    app_specification=AppSpecification(
        image_uri=# TODO: Get XGBoost container image,
        container_entrypoint=[
            "python3",
            "/opt/ml/processing/input/code/preprocessing.py"
        ],
        container_arguments=["--train-test-split", "0.7"]
    ),

    # Compute resources
    processing_resources=ProcessingResources(
        cluster_config=ProcessingClusterConfig(
            instance_type=# TODO: Choose instance type,
            instance_count=1,
            volume_size_in_gb=30
        )
    ),

    # Input channels
    processing_inputs=[
        # TODO: Add input for code
        ProcessingInput(
            input_name="code",
            app_managed=False,
            s3_input=ProcessingS3Input(
                s3_uri=code_s3_uri,
                local_path="/opt/ml/processing/input/code",
                s3_data_type="S3Prefix",
                s3_input_mode="File"
            )
        ),
        # TODO: Add input for data
        ProcessingInput(
            input_name="data",
            app_managed=False,
            s3_input=ProcessingS3Input(
                s3_uri=data_s3_uri,
                local_path="/opt/ml/processing/input/data",
                s3_data_type="S3Prefix",
                s3_input_mode="File"
            )
        )
    ],

    # Output configuration
    processing_output_config=ProcessingOutputConfig(
        outputs=[
            # TODO: Add outputs for train/validation/test
            ProcessingOutput(
                output_name="train",
                app_managed=False,
                s3_output=ProcessingS3Output(
                    s3_uri=f"{output_s3_uri}/train",
                    local_path="/opt/ml/processing/output/train",
                    s3_upload_mode="EndOfJob"
                )
            ),
            # TODO: Add validation and test outputs
        ]
    ),

    # Environment and stopping condition
    environment={"PYTHONUNBUFFERED": "1"},
    stopping_condition=ProcessingStoppingCondition(
        max_runtime_in_seconds=3600
    )
)

print(f"Processing Job created: {job.processing_job_name}")
print(f"Status: {job.processing_job_status}")
```

#### Step 5: Wait for Completion
```python
# Your Task: Wait for the job to complete
job.wait()

# Check final status
job.refresh()
print(f"Final status: {job.processing_job_status}")
```

---

## 🔍 Part 4: Debugging and Verification

### Common Issues and Solutions

#### Issue: Job fails immediately
**Debugging Steps:**
1. Check CloudWatch logs for errors
2. Verify script paths in `container_entrypoint`
3. Ensure input S3 URIs are accessible
4. Check IAM role permissions

**Research:** How do you access CloudWatch logs for a processing job?

#### Issue: "No such file or directory" errors
**Solutions:**
- Verify `local_path` in ProcessingInput matches script expectations
- Check that input data exists at S3 URI
- Ensure output directories are created by script

#### Issue: No output files created
**Solutions:**
- Add print statements in your script for debugging
- Check that script writes to correct output paths
- Verify ProcessingOutput paths match script output locations
- Look for exceptions in CloudWatch logs

#### Issue: Output format is wrong
**Solutions:**
- Verify target column is first in output CSV
- Ensure no headers in output files
- Check all values are numeric (no NaN or strings)
- Validate with `pd.read_csv(output_file, header=None)`

### Verification Checklist
```python
# Download and verify one of your outputs
import boto3
import pandas as pd

s3 = boto3.client('s3')
s3.download_file(bucket, f'{prefix}/train/train.csv', 'local_train.csv')

# Verify format
df = pd.read_csv('local_train.csv', header=None)
print(f"Shape: {df.shape}")
print(f"First 5 rows:\n{df.head()}")
print(f"Target column values: {df.iloc[:, 0].unique()}")  # Should be 0/1
print(f"Any NaN values: {df.isna().sum().sum()}")  # Should be 0
```

**Success Criteria:**
- [ ] Target column (0/1) is in position 0
- [ ] No headers in CSV
- [ ] All values are numeric
- [ ] No NaN or null values
- [ ] Train/validation/test split ratios are correct

---

## 📊 Part 5: Understanding the Results

### Your Analysis Tasks

1. **Data Quality Check:**
   - How many rows in each split (train/val/test)?
   - Are the split ratios what you specified?
   - Is the target distribution balanced in each split?

2. **Performance Analysis:**
   - How long did the processing job take?
   - How does this compare to local preprocessing?
   - What was the cost of the processing job?

3. **Scalability Questions:**
   - How would you handle 10x more data?
   - When would you use multiple instances?
   - What instance type would you choose for 100GB data?

---

## ✅ Success Criteria

### Technical Success
- [ ] Processing job completes with status "Completed"
- [ ] Three output datasets created (train, validation, test)
- [ ] Data is in XGBoost-compatible format (target first, no headers, all numeric)
- [ ] Split ratios match specifications
- [ ] No data leakage between splits

### Understanding Success
- [ ] Can explain each sagemaker-core shape's purpose
- [ ] Understand SageMaker Processing container paths
- [ ] Know how to debug processing job failures
- [ ] Can calculate processing job costs

### Code Quality
- [ ] Preprocessing script handles edge cases
- [ ] Code includes error handling
- [ ] Debugging print statements included
- [ ] Comments explain key decisions

---

## 💡 Advanced Challenges (Optional)

### Challenge 1: Parameterize Everything
Make your preprocessing fully configurable:
- Train/validation/test split ratios as arguments
- Feature selection strategy as argument
- Encoding method as argument

### Challenge 2: Add Data Quality Checks
Implement validation in your preprocessing script:
- Check for data drift
- Validate expected columns exist
- Ensure no data leakage
- Log data statistics

### Challenge 3: Multi-Instance Processing
Research and implement distributed processing:
- How to split data across instances?
- How to combine outputs?
- When is it worth the complexity?

---

## 🔗 Research Resources

### Essential Reading
- [SageMaker Processing Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html)
- [sagemaker-core ProcessingJob Reference](https://sagemaker-core.readthedocs.io/)
- [SageMaker Processing Container Paths](https://docs.aws.amazon.com/sagemaker/latest/dg/build-your-own-processing-container.html)

### Deep Dives
- [Data Preprocessing Best Practices](https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/sagemaker/amazon-sagemaker-processing/)
- [XGBoost Data Format Requirements](https://xgboost.readthedocs.io/en/stable/tutorials/input_format.html)

---

## 🎓 What You've Learned

After completing this assignment, you should understand:

**Technical Skills:**
- Creating Processing Jobs with sagemaker-core explicit shapes
- Configuring container specifications and resource allocation
- Managing input/output channels for data processing
- Debugging processing failures with CloudWatch logs

**Conceptual Understanding:**
- When to use SageMaker Processing vs. local preprocessing
- How containers and data mounting work in SageMaker
- The importance of reproducible data preprocessing
- Cost and performance trade-offs in data processing

**MLOps Foundation:**
- Building reusable preprocessing pipelines
- Preparing data for distributed training
- Ensuring data quality and format compliance
- Scaling preprocessing for production workloads

---

## 🚀 Next Steps

Now that you can preprocess data with SageMaker Processing:
1. Use your processed datasets in Lab 1 training jobs
2. Build the BYOL assignment with custom preprocessing
3. Learn the higher-level SDK abstractions in Lab 3
4. Orchestrate preprocessing in pipelines in Lab 4

**Remember:** The sagemaker-core approach gives you full control and understanding of how SageMaker works. This foundation will help you appreciate the SDK abstractions you'll learn next!
