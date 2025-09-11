# Introduction

These assignements will make you familair with the basic building blocks of SageMaker as a training and inference platform.

You will create a trained model using a Training job and create an HTTP Endpoints which will serve the  

## Essentials 1
1. Read the xgboost_local.ipynb notebook
2. Create a new notebook  
3. Create a training job using SageMaker Core to *train* the xgboost model using a SageMaker Training job.
   1. Use the SageMaker XGBoost image version 1.7-1
   2. Upload the csv to S3
   3. Use your name in the training job name so you can find it easily
4. After training job run inspect the training job in [SagemMaker training jobs](https://eu-central-1.console.aws.amazon.com/sagemaker/home?region=eu-central-1#/jobs)

## Essentials 2
1. Using SageMaker core create a second job of type 'hyperparameter training job' which explorers a parameters space to find the best model parameters
2. Get the `best_training_job` from the tuning run and print the hyperparams found

## Essentials 3
1. Create a batch transform job based on either the training job (or hyperparameter `best_training_job`)
2. Inspect the output on S3

_Hint_: transform jobs need a Model  

## Essentials 4
1. Create a serverless endpoint for the trained model (either regular or hyper tuned)
2. Invoke the enpoint using the created "serverless_endpoint" object

_Hint_: For no-boilerplate invokation use `sagemaker.predictor.Predictor`

## Outtakes
Training, Tuning and Batch prediction jobs can be used in SageMaker Studio _and_ from local notebooks. 
It offloads compute, possibly with more (GPU) resources than your local laptop or notebook instance, with some minimal overhead but with added traceability and clarity.
Metrics are visible and can be queried across runs. This gives you a basic level of MLOps maturity.


### Tips
- Use versioned folders (timestamp) to store both data and model artefacts _per run_ so results can be reproduced at a later stage
- Add tags to the TrainingJob to record why ran it, what was the reason/changes going into the training run
- Put code into functions from an early stage, future you/team will thank you