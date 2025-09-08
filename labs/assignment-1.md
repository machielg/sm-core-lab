
## Assignment
1. Read the xgboost.ipynb notebook
1. Rewrite the code to use SageMaker Core to train the xgboost model using a SageMaker Training job.



## Outtakes
Training jobs can be used in SageMaker Studio and from local notebooks. 
It offloads compute, possibly with more (GPU) resources than your local laptop or notebook instance, with some minimal overhead but with added traceability and clarity.
Metrics are visible and can be queried across runs. This gives you a basic level of MLOps maturity.


### Tips
- Use versioned folders (timestamp) to store both data and model artefacts _per run_ so results can be reproduced at a later stage
- Add tags to the TrainingJob to record why ran it, what was the reason/changes going into the training run
- Put code into functions from an early stage, future you/team will thank you