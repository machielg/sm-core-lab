# 📦 Assignment 4: Model Registry and Lifecycle Management

## 🎯 Your Mission
Learn to manage your ML models professionally using SageMaker Model Registry - tracking versions, approvals, lineage, and deployment status.

## 🤔 Understanding Model Registry

### Why Do You Need Model Registry?
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

## 🛠️ What You Need to Build

### 1. Create a Model Package Group

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

### 2. Register Your Model

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

### 3. Manage Model Versions

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

### 4. Query Model Lineage

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

### 5. Deploy from Model Registry

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

## 🔍 Real-World Model Registry Workflows

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

## ✅ Success Criteria

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

## 💡 Advanced Topics

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

## 📚 Research Resources
- [Model Registry Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
- [MLOps with Model Registry](https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-model-registry/)
- [Model Governance Best Practices](https://docs.aws.amazon.com/sagemaker/latest/dg/model-governance.html)

---

Congratulations on completing all assignments! Continue to [Conclusion & Next Steps](Conclusion.md) →