# CvM_Fairness_Regularizers

This project contains the experiments for the project [...] where a CvM-based regularizer is used to mitigate unfairness in deep learning models.

Following is a description of the main files in the project:

- main.py: contains the function to train models with a given mitigation method and several multipliers which determine the strength of regularization. Logging of experiments available using Wandb.

- tune.py: [not fully implemented] contains the main function to perform hyperparameter optimization for a given model (the spaces to search in must be specified using a .yaml file). Hyperparameter optimization done using Optuna.
  
- train.py: file containing the training functions compatible with different regularization methods.

- regularizers: folder containing the novel CvM regularizer and other baselines:
  
  - CvM: Our method
    
  - DC: from https://arxiv.org/abs/2412.00720
    
  - CDC: from https://arxiv.org/abs/2412.00720
    
  - DISCO: from https://arxiv.org/html/2506.11653v1
    
  - finetuning: code enabling to use any of the previous methods for finetuning
    
- datasets.py: downloading and prepocessing of datasets + building of dataloaders

## How to use:
After downloading the necessary libraries, the different methods can be tried by running in the terminal the following line:

python main.py --method=DesiredMethodName --dataset=adult --seed=0 --offline

Some relevant flags that can be added to control the training and application of the method:

--finetuning (when added indicates to use the chosen method for finetuning)

--batchsize=desired_batch_size (default 1024)

--epochs=desired_number_of_epochs (default 40)

--fairness_metric_goal="DP" or "EO" (which one to optimize for, default "DP")

--pareto (when added indicates that the optuna optimization should be done using a pareto front type sampler)

--wandb_logging  (when added indicates that the results should be logged on wandb)

--wandb_project=name_of_the_wandb_project 

--lam=desired_lambda (if a value is given, only a model with that value of lambda as multiplier will be computed, otherwise multiple models will be computed for several predefined values of the multiplier)
