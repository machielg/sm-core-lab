# 🚀 Assignment 4: Model Deployment and Inference

## 🎯 Your Mission
Deploy your trained models for both batch and real-time inference using the SageMaker SDK's simplified deployment abstractions - much cleaner than the raw API shapes from Lab 1!

## 🤔 Understanding SDK Deployment vs. Lab 1

### What's Different from Lab 1?
**In Lab 1 with sagemaker-core:**
- Create separate `Model` resource with complex shape
- Configure `EndpointConfig` with instance specifications
- Deploy `Endpoint` referencing the config
- Manage batch transform job configurations manually

**With the SDK:**
- Deploy directly from estimator: `estimator.deploy()`
- Create models from artifacts: `Model()` class
- Simple batch transform: `transformer.transform()`
- Automatic resource management and cleanup

**Research Questions:**
1. How does the SDK abstract Model, EndpointConfig, and Endpoint creation?
2. What's the difference between deploying from an Estimator vs. creating a Model?
3. When should you use batch transform vs. real-time endpoints?

## 🛠️ What You Need to Build

### Part 1: Create a Model from Training Artifacts

**Your Task:** Create a deployable model from your training job artifacts.

**Option A: From Estimator (Simplest)**
```python
# If you still have your estimator from Assignment 2
# The model is automatically created from the training job
model = xgb_estimator.create_model()

# Or from a tuning job (Assignment 3)
best_model = tuner.create_model()
```

**Option B: From S3 Artifacts (More Control)**
```python
from sagemaker.model import Model
from sagemaker import get_execution_role

# Create a model from S3 artifacts
xgb_model = Model(
    model_data='s3://your-bucket/path/to/model.tar.gz',  # From training job
    image_uri=sagemaker.image_uris.retrieve('xgboost', region, version='1.5-1'),
    role=get_execution_role(),
    predictor_cls=sagemaker.predictor.Predictor  # Optional: custom predictor
)
```

**Option C: Using PyTorchModel for XGBoost (Container Strategy)**
```python
from sagemaker.pytorch import PyTorchModel

# Use PyTorch container for Python 3.12+ support (even for XGBoost!)
model = PyTorchModel(
    model_data='s3://your-bucket/path/to/model.tar.gz',
    role=role,
    framework_version='2.0',
    py_version='py310',
    entry_point='inference.py',  # Your inference script
    source_dir='src/'  # Directory with inference code
)
```

**Research Deep Dive:**
1. Why would you use a PyTorchModel for XGBoost? (Hint: Python version support)
2. What goes into the model.tar.gz file?
3. How does SageMaker know which inference code to run?

### Part 2: Batch Transform for Large-Scale Inference

**Your Challenge:** Process a large dataset efficiently using batch transform.

**Basic Batch Transform:**
```python
from sagemaker.transformer import Transformer

# Create transformer from your model
transformer = xgb_model.transformer(
    instance_count=1,
    instance_type='ml.m5.xlarge',
    strategy='MultiRecord',  # Process multiple records per request
    max_payload=6,  # MB
    max_concurrent_transforms=10,
    output_path='s3://your-bucket/batch-output/',
    accept='text/csv'  # Output format
)

# Run the batch transform job
transformer.transform(
    data='s3://your-bucket/test-data/',
    content_type='text/csv',
    split_type='Line',
    join_source='Input'  # Include input data in output
)

# Wait for completion
transformer.wait()
```

**Advanced Batch Configuration:**
```python
# Configure for optimal performance
transformer = xgb_model.transformer(
    instance_count=2,  # Use multiple instances for parallelism
    instance_type='ml.m5.4xlarge',
    strategy='MultiRecord',
    max_payload=100,  # Larger batches for efficiency
    max_concurrent_transforms=20,
    env={
        'MODEL_CACHE_ROOT': '/opt/ml/model',
        'SAGEMAKER_MODEL_SERVER_TIMEOUT': '3600'
    },
    output_path='s3://your-bucket/batch-predictions/'
)

# Transform with data filters
transformer.transform(
    data='s3://your-bucket/large-dataset/',
    data_type='S3Prefix',  # Process all files with prefix
    content_type='text/csv',
    split_type='Line',
    input_filter='$[1:]',  # Skip header
    output_filter='$[0,-1]',  # First and last columns
    join_source='None'  # Don't include input in output
)
```

**Research Questions:**
1. What's the difference between `SingleRecord` and `MultiRecord` strategies?
2. How do `input_filter` and `output_filter` work with JSONPath?
3. When would you use multiple instances for batch transform?

**Cost Optimization:**
```python
# Calculate batch transform costs
instance_cost_per_hour = 0.269  # ml.m5.xlarge
data_size_gb = 10
throughput_mb_per_sec = 5  # Estimated
processing_time_hours = (data_size_gb * 1024) / (throughput_mb_per_sec * 3600)
total_cost = processing_time_hours * instance_cost_per_hour * instance_count

print(f"Estimated time: {processing_time_hours:.2f} hours")
print(f"Estimated cost: ${total_cost:.2f}")
```

### Part 3: Real-Time Endpoints for Online Inference

**Your Task:** Deploy a real-time endpoint for low-latency predictions.

**Simple Deployment from Estimator:**
```python
# Deploy directly from estimator (easiest!)
predictor = xgb_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    endpoint_name='your-name-xgboost-endpoint'  # Optional: custom name
)

# Test the endpoint
import numpy as np
test_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Your feature vector
prediction = predictor.predict(test_data)
print(f"Prediction: {prediction}")
```

**Deployment from Model Object:**
```python
# Deploy from model object (more control)
predictor = xgb_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='your-custom-endpoint',
    wait=True  # Wait for endpoint to be ready
)

# Configure serialization/deserialization
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

predictor.serializer = CSVSerializer()
predictor.deserializer = JSONDeserializer()

# Make predictions
result = predictor.predict("5.1,3.5,1.4,0.2")
print(f"Result: {result}")
```


**Research Questions:**
1. How does autoscaling work for SageMaker endpoints?
2. How do you update a deployed model without downtime?

### Part 4: Custom Inference Scripts

**Your Challenge:** Write custom inference logic for preprocessing and postprocessing.

**Custom Inference Script (inference.py):**
```python
import os
import json
import pickle
import numpy as np
import xgboost as xgb

def model_fn(model_dir):
    """
    Load the model for inference

    Args:
        model_dir: Directory where model is saved

    Returns:
        Loaded model object
    """
    model_file = os.path.join(model_dir, 'xgboost-model')
    model = xgb.Booster()
    model.load_model(model_file)
    return model

def input_fn(request_body, content_type='text/csv'):
    """
    Parse input data

    Args:
        request_body: Raw input data
        content_type: Content type of input

    Returns:
        Processed input ready for prediction
    """
    if content_type == 'text/csv':
        # Parse CSV input
        lines = request_body.strip().split('\n')
        return np.array([line.split(',') for line in lines], dtype=np.float32)
    elif content_type == 'application/json':
        # Parse JSON input
        return np.array(json.loads(request_body), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """
    Make predictions

    Args:
        input_data: Processed input data
        model: Loaded model

    Returns:
        Model predictions
    """
    dmatrix = xgb.DMatrix(input_data)
    predictions = model.predict(dmatrix)

    # Add custom business logic
    # Example: Convert probabilities to classes
    return (predictions > 0.5).astype(int)

def output_fn(prediction, accept='text/csv'):
    """
    Format output

    Args:
        prediction: Model predictions
        accept: Desired output format

    Returns:
        Formatted output
    """
    if accept == 'text/csv':
        return '\n'.join([str(pred) for pred in prediction])
    elif accept == 'application/json':
        return json.dumps(prediction.tolist())
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
```

**Deploy with Custom Inference:**
```python
from sagemaker.pytorch import PyTorchModel

# Deploy with custom inference script
model_with_custom = PyTorchModel(
    model_data=model_artifacts_uri,
    role=role,
    framework_version='2.0',
    py_version='py310',
    entry_point='inference.py',
    source_dir='src/',
    predictor_cls=sagemaker.predictor.Predictor
)

predictor = model_with_custom.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)
```

**Research Questions:**
1. What are the four handler functions and when is each called?
2. How do you add preprocessing logic that wasn't in training?
3. How do you handle different input/output formats?

### Part 5: Endpoint Management

**Your Task:** Monitor, update, and clean up endpoints.

**Monitoring and Metrics:**
```python
# Get endpoint description
endpoint_desc = predictor.describe()
print(f"Endpoint status: {endpoint_desc['EndpointStatus']}")
print(f"Created: {endpoint_desc['CreationTime']}")

# Monitor with CloudWatch (via boto3)
import boto3
cloudwatch = boto3.client('cloudwatch', region_name=region)

# Get invocation metrics
response = cloudwatch.get_metric_statistics(
    Namespace='AWS/SageMaker',
    MetricName='Invocations',
    Dimensions=[
        {'Name': 'EndpointName', 'Value': predictor.endpoint_name},
        {'Name': 'VariantName', 'Value': 'AllTraffic'}
    ],
    StartTime=datetime.now() - timedelta(hours=1),
    EndTime=datetime.now(),
    Period=300,
    Statistics=['Sum']
)
```

**Update Endpoint (Blue/Green Deployment):**
```python
# Create new model version
new_model = xgb_estimator_v2.create_model()

# Update endpoint with new model
predictor.update_endpoint(
    model_name=new_model.name,
    initial_instance_count=1,
    instance_type='ml.m5.large',
    wait=True
)

print("Endpoint updated with new model version!")
```

**Autoscaling Configuration:**
```python
# Configure autoscaling
client = boto3.client('application-autoscaling', region_name=region)

# Register scalable target
client.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{predictor.endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=4
)

# Create scaling policy
client.put_scaling_policy(
    PolicyName='target-tracking-policy',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{predictor.endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        }
    }
)
```

**Clean Up Resources:**
```python
# IMPORTANT: Always clean up to avoid charges!

# Delete endpoint
predictor.delete_endpoint(delete_endpoint_config=True)

# Delete model
xgb_model.delete()

# Verify deletion
try:
    predictor.describe()
except Exception as e:
    print("Endpoint successfully deleted!")
```

## 🔍 Debugging Deployment Issues

**Endpoint fails to start:**
- Check CloudWatch logs for the endpoint
- Verify model artifacts exist in S3
- Ensure IAM role has necessary permissions
- Check if instance type supports your model size

**Predictions fail:**
- Verify input format matches training data
- Check serializer/deserializer configuration
- Look at endpoint logs for stack traces
- Test inference script locally first

**Batch transform issues:**
- Ensure input data format is correct
- Check if output path has write permissions
- Verify instance type has enough memory
- Monitor CloudWatch for transform job logs

**Performance problems:**
- Use CloudWatch metrics to identify bottlenecks
- Consider larger instance types
- Enable autoscaling for variable loads
- Optimize inference script for speed

## ✅ Success Criteria

**Technical Success:**
- [ ] Model created from training artifacts
- [ ] Batch transform job completes successfully
- [ ] Real-time endpoint deployed and responding
- [ ] Custom inference script working if used
- [ ] Endpoints properly cleaned up after testing

**Understanding Success:**
- [ ] Can explain SDK deployment abstractions
- [ ] Understand batch vs. real-time trade-offs
- [ ] Know how to monitor and update endpoints
- [ ] Can implement custom inference logic

## 💡 Advanced Challenges

**A/B Testing with Production Variants:**
- Research how to split traffic between model versions
- Implement canary deployments for safe rollouts
- Monitor variant-specific metrics

**Cost Optimization:**
- Compare costs: batch transform vs. endpoints
- Implement endpoint scheduling for non-24/7 usage
- Use Savings Plans for predictable workloads

**Model Serving Patterns:**
- Implement feature stores for real-time features
- Add caching layers for frequently requested predictions
- Create ensemble endpoints with multiple models

## 📚 Research Resources
- [SageMaker Model Deployment](https://sagemaker.readthedocs.io/en/stable/api/inference/model.html)
- [Batch Transform Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)
- [Real-time Inference Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html)
- [Custom Inference Scripts](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#deploy-pytorch-models)

---

Ready for model management? Continue to [Assignment 5: Model Registry](Assignment_5_Registry.md) →