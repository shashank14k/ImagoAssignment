from dataclasses import dataclass
from typing import List, Optional, Dict
import yaml


@dataclass
class PreprocessParams:
    fill_na: str
    null_thresh: float
    iqr_factor: float
    iqr_limits: List[float]
    fill_outlier: str
    n_components: float
    target_outlier: bool = False


@dataclass
class ModelParams:
    name: str
    params: Optional[Dict] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        else:
            assert isinstance(self.params, dict)


@dataclass
class TrainParams:
    lr: float = 1e-4
    epochs: int = 2
    n_splits: int = 5
    batch_size: int = 64

    def __post_init__(self):
        self.lr = float(self.lr)
        self.epochs = int(self.epochs)
        self.n_splits = int(self.n_splits)
        self.batch_size = int(self.batch_size)


@dataclass
class Config:
    log_dir: str
    ignore_col_names: List[str]
    target_col_name: str
    preprocess_params: PreprocessParams
    model_params: ModelParams
    train_params: TrainParams

    @staticmethod
    def from_file(config_path: str) -> "Config":
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return Config(
            log_dir=config_dict["log_dir"],
            ignore_col_names=config_dict["ignore_col_names"],
            target_col_name=config_dict["target_col_name"],
            preprocess_params=PreprocessParams(**config_dict["preprocess_params"]),
            model_params=ModelParams(**config_dict["model_params"]),
            train_params=TrainParams(**config_dict["train_params"]),
        )
