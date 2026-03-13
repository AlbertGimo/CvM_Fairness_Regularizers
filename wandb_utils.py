import wandb
from typing import Any, Dict
from config import WandbConfig
from typing import Optional


def init_wandb_run(
    config: Dict[str, Any],
    wandb_cfg: WandbConfig,
    run_name: Optional[str] = None,
):
    """
    Initializes a W&B run safely.

    - Does NOT overwrite projects
    - Works both inside and outside Optuna
    """

    run = wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        group=wandb_cfg.group,
        tags=wandb_cfg.tags,
        config=config,
        name=run_name,
        mode=wandb_cfg.mode,
        reinit=True,
    )

    return run


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """Safe metric logging."""
    wandb.log(metrics, step=step)


def log_model_artifact(model_path: str, name: str = "model"):
    """Logs a trained model as a W&B artifact."""

    artifact = wandb.Artifact(name=name, type="model")
    artifact.add_file(model_path)

    wandb.log_artifact(artifact)