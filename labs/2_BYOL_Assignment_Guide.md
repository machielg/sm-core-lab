# 📚 Bring Your Own Library (BYOL) - SageMaker Training Assignment

## Overview
Learn how to bring any machine learning library to SageMaker by creating a custom training job with your own dependencies and inference code.

---

## 🎯 Assignment 1: Custom Model Training & Deployment

### Objective
Deploy a custom ML model to SageMaker using your preferred library (e.g., RandomForest, XGBoost, LightGBM, CatBoost).

### Part 1: Prepare Your Training Environment

#### 1.1 Select Your Components
- **Base Image**: Use the sklearn SageMaker container image
  ```python
  image = image_uris.retrieve('sklearn', region, version='1.2-1')
  ```
  
- **Training Script**: Create a SageMaker-compatible `train.py`
  - Must use `sagemaker_training.environment` to access paths
  - Should save model to `model_dir`
  - Can use any ML library (RandomForest, XGBoost, etc.)

- **Dependencies**: Create `requirements.txt` for additional libraries
  ```txt
  xgboost==1.7.0
  lightgbm==3.3.0
  # Add any other libraries you need
  ```

### Part 2: Create and Run Training Job

#### 2.1 Configure the Training Job
Using `sagemaker-core`, create a TrainingJob that:
- References your custom `train.py` script
- Includes your `requirements.txt` for dependency installation
- Specifies hyperparameters if needed

**Key Steps:**
```python
# 1. Upload your code to S3
# 2. Configure AlgorithmSpecification with your image
# 3. Set up input/output data channels
# 4. Launch the training job
```

### Part 3: Create Model with Custom Inference

#### 3.1 Write Your Inference Handler (`inference.py`)

Your inference script must implement four key functions:

##### 📥 **Load the Model**
```python
def model_fn(model_dir):
    """
    Load your trained model from disk.
    
    Args:
        model_dir: Path where model artifacts are stored
    
    Returns:
        Loaded model object
    """
    # Example: return joblib.load(os.path.join(model_dir, 'model.joblib'))
```

##### 🔄 **Process Input Data**
```python
def input_fn(request_body, content_type):
    """
    Deserialize and prepare the prediction input.
    
    Args:
        request_body: Raw request data
        content_type: MIME type (e.g., 'text/csv', 'application/json')
    
    Returns:
        Processed input ready for prediction
    """
    # Example: Parse CSV or JSON into format your model expects
```

##### 🎯 **Make Predictions**
```python
def predict_fn(input_data, model):
    """
    Run prediction using your model.
    
    Args:
        input_data: Processed input from input_fn
        model: Model loaded by model_fn
    
    Returns:
        Model predictions
    """
    # Example: return model.predict(input_data)
```

##### 📤 **Format Output**
```python
def output_fn(prediction, accept):
    """
    Serialize predictions for the response.
    
    Args:
        prediction: Output from predict_fn
        accept: Requested MIME type for response
    
    Returns:
        Serialized predictions
    """
    # Example using sagemaker-inference toolkit:
    from sagemaker_inference import encoder
    return encoder.encode(prediction, accept)
```

#### 3.2 Package and Deploy Your Model
1. Include `inference.py` in your model package
2. Create a SageMaker Model resource
3. Deploy to an endpoint (real-time or serverless)

---

## 🌟 Bonus Assignment: Add Training Metrics

### Objective
Make your training job report metrics that appear in:
- SageMaker Training Job console
- CloudWatch Metrics

### Implementation Steps

#### Step 1: Define Metrics in Training Job
```python
from sagemaker_core.shapes import MetricDefinition

metric_definitions = [
    MetricDefinition(
        name="validation:accuracy",
        regex="validation:accuracy=([0-9\\.]+)"
    ),
    MetricDefinition(
        name="validation:auc",
        regex="validation:auc=([0-9\\.]+)"
    )
]
```

#### Step 2: Emit Metrics in train.py
```python
# In your train.py, print metrics in the expected format
print(f"validation:accuracy={accuracy:.4f}")
print(f"validation:auc={auc_score:.4f}")
```

#### Step 3: View Metrics
- Check SageMaker console → Training jobs → Metrics tab
- Query CloudWatch Metrics for detailed analysis

---

## 📋 Success Criteria

### Assignment 1 Checklist
- [ ] Training job completes successfully
- [ ] Model artifacts saved to S3
- [ ] Custom dependencies installed correctly
- [ ] Model endpoint deployed and responding
- [ ] Inference handles CSV/JSON inputs
- [ ] Predictions return in expected format

### Bonus Assignment Checklist
- [ ] Metrics visible in Training Job console
- [ ] Metrics queryable in CloudWatch
- [ ] At least 2 different metrics tracked
- [ ] Metrics update during training (not just final)

---

## 💡 Tips & Best Practices

1. **Test Locally First**: Run your `train.py` locally before SageMaker
2. **Use Logging**: Add print statements for debugging
3. **Handle Errors Gracefully**: Add try-catch blocks in inference.py
4. **Version Your Models**: Include version info in model metadata
5. **Document Dependencies**: Specify exact versions in requirements.txt

---

## 🔗 Resources

- [SageMaker Training Toolkit Docs](https://github.com/aws/sagemaker-training-toolkit)
- [SageMaker Inference Toolkit Docs](https://github.com/aws/sagemaker-inference-toolkit)
- [Custom Container Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms.html)