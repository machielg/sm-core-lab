
## Assignment 1
1. Read the xgboost.ipynb notebook
1. Rewrite/ammend the code to use SageMaker Core to *train* the xgboost model using a SageMaker Training job.

## Assignment 2
1. Add a second job of type 'hyperparameter training job' which explorers a parameters space to find the best model parameters

## Assignment 3
1. Add a batch transform job based on either the training job or hyper parameter job run 'best_training_job'

## Assignment 4
1. Create a serverless endpoint for the trained model (either regular or hyper tuned)
2. Invoke the enpoint using the created "serverless_endpoint" object

## Outtakes
Training, Tuning and Batch prediction jobs can be used in SageMaker Studio _and_ from local notebooks. 
It offloads compute, possibly with more (GPU) resources than your local laptop or notebook instance, with some minimal overhead but with added traceability and clarity.
Metrics are visible and can be queried across runs. This gives you a basic level of MLOps maturity.


### Tips
- Use versioned folders (timestamp) to store both data and model artefacts _per run_ so results can be reproduced at a later stage
- Add tags to the TrainingJob to record why ran it, what was the reason/changes going into the training run
- Put code into functions from an early stage, future you/team will thank you