import torch
import wandb
import optuna
import argparse

from .config import TrainConfig, WandbConfig, OptunaConfig
from .optuna_utils import create_study, suggest_from_yaml, log_study_config_once, load_search_space
from .wandb_utils import init_wandb_run
from .train import train_model, set_seed
from pathlib import Path
import utils
import datasets
from pathlib import Path
import optuna
from .wandb_utils import init_wandb_run, log_yaml_artifact
from .train import train_model, set_seed

SEARCH_SPACE_FILE = Path("search_space.yaml")


def run_optuna(args, train_dataset, val_dataset):

    optuna_cfg = OptunaConfig()
    wandb_cfg = WandbConfig()

    # Log search space once
    log_study_config_once(
        project=wandb_cfg.project,
        study_name=optuna_cfg.study_name,
        yaml_path=SEARCH_SPACE_FILE,
    )

    study = create_study(optuna_cfg)

    objective_fn = lambda trial: objective(trial, args, train_dataset, val_dataset)

    if optuna_cfg.timeout is not None:
        study.optimize(objective_fn, timeout=optuna_cfg.timeout)
    else:
        study.optimize(objective_fn, n_trials=optuna_cfg.n_trials)

    print("Best params:", study.best_params)
    print("Best value:", study.best_value)


def objective(trial, args, train_dataset, val_dataset):


    base_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        hidden_dim=args.MLP_hidden_layers[0],
        num_layers=len(args.MLP_hidden_layers),
        seed=args.seed,
        device=args.device,
    )
    
    search_space = load_search_space(SEARCH_SPACE_FILE)

    suggested = suggest_from_yaml(trial, search_space)

    train_cfg = vars(base_cfg) | suggested

    wandb_cfg = WandbConfig(
        group=f"optuna-trial-{trial.number}"
    )

    run = init_wandb_run(
        args,
        train_cfg,
        wandb_cfg,
        run_name=f"trial-{trial.number}"
    )

    metric = train_model(
        args=train_cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lam = args.lam,
        trial=trial,
    )

    run.finish()

    return metric

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="adult", choices=["adult","acs"],
                        help="Choose a tabular dataset.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="gpu", choices=["cuda","cpu"],
                        help="Choose a device.")
    parser.add_argument('--method', type=str, default='CvM', choices=['ERM', 'CvM', 'DC', 'CDC', 'DISCO', None], help="Choose an unfairness mitigation method.")
    parser.add_argument('--beta', type=float, default=0, 
                        help="balanced hyperparameter: ascent rate of lambda")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="fairness_regularizers", 
                        help="Weights & Biases project name")
    parser.add_argument("--goal", type=str, default="DP", choices=["DP", "EO"], help="Fairness goal")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for MLP model")
    parser.add_argument("--MLP_hidden_layers", type=list, default=[200, 200, 20], help="Hidden layer sizes for MLP model")
    parser.add_argument("--split_list", type=list, default=[0.7, 0.5], help="Train/Val/Test split ratios in the form [train_size, val_size] where val_size is the proportion of the testvalid set to be used as validation set. Default is [0.7, 0.5], which means 70% train, 15% val, 15% test.")
    parser.add_argument("--fairness_metric_goal", type=str, default="DP", choices=["DP", "EO"], help="Fairness metric to optimize for when using a regularization method. Default is 'DP' (Demographic Parity). If the goal is 'EO' (Equalized Odds), the regularizer will optimize for equalized odds instead.")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Small constant for numerical stability in regularization terms (if necessary).")
    parser.add_argument("--finetuning", type=bool, default=False, help="whether the mitigation method should be used fro finetuning")
    parser.add_argument("--finetuning_epochs", type=int, default=20, help="number of finetuning epochs for finetuning")

    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu"
    print(f"Using device: {args.device}")

    utils.seed_everything(args.seed)
    train_dataset, val_dataset, test_dataset, dataset_info = datasets.datasetPreprocessing("../dataset/", dataset_name=args.dataset, split_list=args.split_list, seed=args.seed, sensitive_attribute="sex", to_tensors=True, verbose=False)
    args.dataset_info = dataset_info
    run_optuna(args, train_dataset, val_dataset)
