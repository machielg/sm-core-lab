# 🚀 SageMaker Pipelines - Orchestrated ML Workflows Learning Guide

## 📖 Introduction

Welcome to the SageMaker Pipelines learning guide! This assignment will teach you how to build production-ready, automated ML workflows that handle data processing, model training, evaluation, and deployment in an orchestrated pipeline.

**Learning Objectives:**
- Understand SageMaker Pipeline architecture and components
- Build multi-step ML workflows with automatic orchestration
- Implement data processing, training, evaluation, and model registration steps
- Master pipeline parameters for flexible, reusable workflows
- Learn pipeline versioning and execution monitoring

**Prerequisites:**
- Complete (most) assignments in **lab3**
- Understanding of SageMaker Training Jobs and Processing Jobs
- Access to SageMaker Studio or notebook environment
- Familiarity with the customer churn dataset

---

## 🏗️ Assignment Overview: Build a Complete ML Pipeline

### 🎯 Your Mission
Transform your standalone training jobs into a production-ready ML pipeline that automatically processes data, trains models, evaluates performance, and registers successful models.

### Pipeline Architecture
Your pipeline will consist of 4-5 connected steps:
```
Data Processing → Model Training → Model Evaluation → Model Creation → [Optional] Model Registration
```

---

## 📚 Part 1: Understanding Pipeline Components

### 🤔 Key Concepts to Research

#### Pipeline Session vs Regular Session
**Research Questions:**
- What's the difference between `PipelineSession` and regular `Session`?
- Why do pipeline steps not execute immediately when defined?
- How does `PipelineSession` handle S3 paths differently?

**Hint:** Pipeline sessions delay execution until `pipeline.start()` is called.

#### Pipeline Parameters
**Your Task:** Understand how to make pipelines reusable with parameters.

**Research These Parameter Types:**
```python
from sagemaker.workflow.parameters import (
    ParameterFloat,    # For numeric values like train/test split
    ParameterString,   # For instance types, S3 paths
    ParameterInteger   # For counts, epochs
)
```

**Key Questions:**
- How do you reference parameters in step configurations?
- Why use `parameter.to_string()` in some contexts?
- How do you override default values at execution time?

---

## 📊 Part 2: Data Processing Step

### 🎯 Challenge: Build a Flexible Processing Step

#### Your Tasks:
1. **Choose a Processor**: Research and select between:
   - `XGBoostProcessor` - Framework-aware, handles dependencies better
   - `ScriptProcessor` - More flexible, any container image
   - `SKLearnProcessor` - Good for scikit-learn preprocessing

2. **Create Processing Script** (`preprocessing.py`):
   ```python
   # Your script should:
   # 1. Read raw data from /opt/ml/processing/input
   # 2. Perform feature engineering
   # 3. Split into train/validation/test sets
   # 4. Save to /opt/ml/processing/output/[train|validation|test]
   ```

3. **Configure Processing I/O**:
   **Research:** `ProcessingInput` and `ProcessingOutput` classes

   **Key Decisions:**
   - Where does input data come from? (S3 URI)
   - How many output channels do you need?
   - How do you reference outputs in later steps?

#### Implementation Pattern:
```python
# Create processor
processor = XGBoostProcessor(...)

# Define step arguments (not executed immediately!)
step_args = processor.run(
    code="preprocessing.py",
    inputs=[ProcessingInput(...)],
    outputs=[ProcessingOutput(...), ...],
    arguments=["--train-test-split", parameter.to_string()]
)

# Create the step
step_process = ProcessingStep(
    name="ProcessData",
    step_args=step_args
)
```

**Common Pitfalls:**
- Forgetting to use `step_args` pattern
- Hard-coding S3 paths instead of using step properties
- Missing source_dir for requirements.txt

---

## 🤖 Part 3: Training Step

### 🎯 Challenge: Dynamic Training with Pipeline Properties

#### Your Tasks:

1. **Create an Estimator**:
   - Use generic `Estimator` class with XGBoost image URI
   - Configure hyperparameters (some static, some from parameters)
   - Set output path for model artifacts

2. **Connect to Processing Outputs**:
   **Key Concept:** Use step properties to reference previous outputs
   ```python
   TrainingInput(
       s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
   )
   ```

3. **Research Questions:**
   - How do you reference outputs from previous steps?
   - What's the difference between `TrainingInput` and regular S3 paths?
   - Why use `content_type="text/csv"`?

#### Advanced Challenge:
Make hyperparameters configurable via pipeline parameters:
- max_depth
- num_round (number of boosting rounds)
- learning rate (eta)

---

## 📈 Part 4: Model Evaluation Step

### 🎯 Challenge: Create Custom Evaluation Metrics

#### Your Tasks:

1. **Write Evaluation Script** (`evaluate.py`):
   Your script must:
   - Extract and load the trained model from tar.gz
   - Load validation data from processing step
   - Calculate metrics (accuracy, precision, recall, F1, ROC-AUC)
   - Save results as JSON for property files

2. **Use PropertyFile for Metrics**:
   ```python
   from sagemaker.workflow.properties import PropertyFile

   evaluation_report = PropertyFile(
       name="EvaluationReport",
       output_name="evaluation",
       path="evaluation.json"
   )
   ```

3. **Research Questions:**
   - How do you extract a model from `model.tar.gz`?
   - What format should the evaluation JSON follow?
   - How can you use PropertyFile values in conditions?

#### Evaluation Output Format:
```json
{
  "binary_classification_metrics": {
    "accuracy": {"value": 0.95, "standard_deviation": "NaN"},
    "precision": {"value": 0.93, "standard_deviation": "NaN"},
    "recall": {"value": 0.91, "standard_deviation": "NaN"},
    "f1": {"value": 0.92, "standard_deviation": "NaN"},
    "roc_auc": {"value": 0.97, "standard_deviation": "NaN"}
  }
}
```

---

## 🚀 Part 5: Model Creation and Registration

### 🎯 Challenge: Create Deployable Models

#### Model Creation Step:
```python
from sagemaker.model import Model
from sagemaker.workflow.model_step import ModelStep

model = Model(
    image_uri=xgboost_image_uri,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    sagemaker_session=pipeline_session
)

step_create_model = ModelStep(
    name="CreateModel",
    step_args=model.create(instance_type="ml.m5.large")
)
```

#### Optional: Model Registry
**Research:** How to register models with metrics
- Model package groups
- Approval status (Approved, PendingManualApproval, Rejected)
- Model metrics attachment
- Version management

---

## 🔧 Part 6: Pipeline Assembly and Execution

### 🎯 Challenge: Build and Run Your Pipeline

#### Assembly Tasks:

1. **Create the Pipeline**:
   ```python
   pipeline = Pipeline(
       name="your-name-customer-churn-pipeline",
       parameters=[...],
       steps=[...],
       sagemaker_session=pipeline_session
   )
   ```

2. **Validate Before Execution**:
   ```python
   # Parse and validate pipeline definition
   pipeline_definition = json.loads(pipeline.definition())
   ```

3. **Execute with Parameters**:
   ```python
   execution = pipeline.start(
       parameters={
           "ProcessingInstanceType": "ml.m5.large",
           "TrainingInstanceType": "ml.m5.xlarge",
           # Override other parameters...
       }
   )
   ```

#### Monitoring and Debugging:
- Use `execution.wait()` or poll status
- Check individual step statuses
- View in SageMaker Studio Pipelines UI
- Access CloudWatch logs for debugging

---

## 🎯 Bonus Challenges

### Advanced Pipeline Features

1. **Conditional Steps**:
   Research `ConditionStep` to add conditional logic:
   - Only register model if accuracy > threshold
   - Choose different instance types based on data size

2. **Parallel Steps**:
   Run multiple training jobs in parallel with different algorithms

3. **Cache Steps**:
   Research step caching to avoid re-running unchanged steps

4. **Pipeline Versioning**:
   - How do versions get created?
   - How to execute specific versions?
   - Best practices for version management

5. **Custom Containers**:
   Replace XGBoostProcessor with your own Docker container

---

## 📝 Deliverables

### Your Complete Pipeline Should:
1. ✅ Process raw churn data into train/validation/test sets
2. ✅ Train an XGBoost model with configurable hyperparameters
3. ✅ Evaluate the model and generate metrics report
4. ✅ Create a deployable SageMaker Model
5. ✅ (Optional) Register model in Model Registry with metrics

### Success Criteria:
- Pipeline executes successfully end-to-end
- All parameters are configurable
- Steps properly reference outputs from previous steps
- Evaluation metrics are captured and reportable
- Pipeline can be re-run with different parameters

---

## 🐛 Common Issues and Solutions

### Issue: "No module named 'sagemaker_training'"
**Solution:** Your processing/training container might not have the right dependencies. Use framework processors or add to requirements.txt.

### Issue: Step outputs not found
**Solution:** Check you're using `.properties` to reference previous step outputs, not hard-coded S3 paths.

### Issue: Pipeline validation fails
**Solution:** Ensure all parameters have default values and check for circular dependencies.

### Issue: "PipelineSession" warnings about no logs
**Solution:** This is normal - pipeline steps don't execute until `pipeline.start()`.

### Issue: Model extraction fails in evaluation
**Solution:** Check your tar.gz extraction logic and model file naming conventions.

---

## 📚 Additional Resources

### Key Documentation:
- [SageMaker Pipelines Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
- [Pipeline Parameters](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-parameters.html)
- [PropertyFiles for Metrics](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-step-properties.html)

### Debugging Tips:
1. Start with a simple 2-step pipeline (process → train)
2. Add evaluation and model steps incrementally
3. Test each script locally first if possible
4. Use SageMaker Studio for visual debugging
5. Check CloudWatch logs for detailed error messages

---

## 🎉 Congratulations!

Once you complete this assignment, you'll have mastered:
- Building production ML pipelines
- Orchestrating complex ML workflows
- Creating reusable, parameterized pipelines
- Implementing proper ML evaluation and model management

Next steps: Explore SageMaker Model Monitor for continuous model quality monitoring in production!