import optuna
import yaml
import wandb
from pathlib import Path
from .config import OptunaConfig


def create_study(cfg: OptunaConfig):
    """
    Creates or loads an Optuna study.
    """

    if cfg.pareto:

        sampler = optuna.samplers.NSGAIISampler(
            seed=42,
        )

        study = optuna.create_study(
            study_name=cfg.study_name,
            storage=cfg.storage,
            directions=cfg.direction,
            sampler=sampler,
            load_if_exists=True
        )
    else: 

        sampler = optuna.samplers.TPESampler(seed=42)

        study = optuna.create_study(
            study_name=cfg.study_name,
            storage=cfg.storage,
            direction=cfg.direction,
            load_if_exists=True,
        )

    return study

def load_search_space(path: str):
    """Load YAML hyperparameter search space."""

    with open(path, "r") as f:
        return yaml.safe_load(f)


def suggest_from_yaml(trial, search_space: dict):

    """
    Convert YAML spec → Optuna suggestions.
    """

    params = {}

    for name, spec in search_space.items():

        ptype = spec["type"]

        if ptype == "float":

            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
                step=spec.get("step", None),
            )

        elif ptype == "int":

            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
            )

        elif ptype == "categorical":

            params[name] = trial.suggest_categorical(
                name,
                spec["choices"],
            )

        else:
            raise ValueError(f"Unsupported parameter type: {ptype}")

    return params


def log_study_config_once(project, study_name, yaml_path):
    """
    Logs the search space YAML once for a study.

    Uses a dedicated W&B run whose only job is to store
    the search configuration.
    """

    run = wandb.init(
        project=project,
        name=f"{study_name}-search-space",
        job_type="study-config",
        group=study_name,
        reinit=True,
    )

    artifact = wandb.Artifact(
        name=f"{study_name}-search-space",
        type="config",
    )

    artifact.add_file(str(yaml_path))

    run.log_artifact(artifact)

    run.finish()