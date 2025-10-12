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

## 📚 Lab Assignments

### [Assignment 1: Data Processing with SageMaker Processors](Assignment_1_Processing.md)
Build reusable data preprocessing workflows using the SageMaker SDK's Processor classes.

### [Assignment 2: Training with Framework Estimators](Assignment_2_Training.md)
Use the SageMaker SDK's Estimator classes to train models with a simpler, more Pythonic API.

### [Assignment 3: Hyperparameter Tuning with the SDK](Assignment_3_Tuning.md)
Automate hyperparameter optimization with HyperparameterTuner.

### [Assignment 4: Model Deployment and Inference](Assignment_4_Deployment.md)
Deploy models for batch and real-time inference with simplified SDK abstractions.

### [Assignment 5: Model Registry and Lifecycle Management](Assignment_5_Registry.md)
Manage ML models professionally using SageMaker Model Registry.

### [Conclusion & Next Steps](Conclusion.md)
What you've accomplished and where to go next.

---

Ready to begin? Start with [Assignment 1: Data Processing](Assignment_1_Processing.md)! 🚀