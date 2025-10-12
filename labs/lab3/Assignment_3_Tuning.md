# ⚡ Assignment 3: Hyperparameter Tuning with the SDK

## 🎯 Your Mission
Use the SageMaker SDK's `HyperparameterTuner` to automatically find optimal hyperparameters for your model - with a much simpler API than the raw tuning jobs from Lab 1.

## 🤔 Understanding SDK Tuning vs. Lab 1

### What's Better with the SDK?
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

## 🛠️ What You Need to Build

### 1. Define Your Search Space

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

### 2. Configure the Tuner

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

### 3. Run the Tuning Job

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

### 4. Analyze Tuning Results

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

## 🔍 Debugging Hyperparameter Tuning

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

## ✅ Success Criteria

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

## 💡 Advanced Challenges

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

## 📚 Research Resources
- [Hyperparameter Tuning Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html)
- [HyperparameterTuner API](https://sagemaker.readthedocs.io/en/stable/api/training/tuner.html)
- [Bayesian Optimization Explained](https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-automatic-model-tuning-becomes-more-efficient-with-warm-start-of-hyperparameter-tuning-jobs/)

---

Ready to deploy your model? Continue to [Assignment 4: Deployment and Inference](Assignment_4_Deployment.md) →