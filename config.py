from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainConfig:
    # training
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3

    # model
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1

    # misc
    seed: int = 42
    device: str = "cuda"


# apparently this doesn't work in python <= 3.9 because of the mutable default argument, so we need to use Optional and field(default_factory=list) instead
# @dataclass
# class WandbConfig:
#     project: str = "ml-project"
#     entity: str | None = None
#     group: str | None = None
#     tags: list[str] | None = None
#     mode: str = "online"  # "offline" if needed

@dataclass
class WandbConfig:
    project: str = "ml-project"
    entity: Optional[str] = None
    group: Optional[str] = None
    tags: Optional[List[str]] = field(default_factory=list)
    mode: str = "online"


@dataclass
class OptunaConfig:
    study_name: str = "ml-study"
    storage: str = "sqlite:///optuna_study.db"
    direction: str = "minimize"
    timeout: Optional[int] = None
    pareto: bool = True
    n_trials: Optional[int] = None