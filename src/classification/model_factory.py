from omegaconf import DictConfig

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def load_model(cfg: DictConfig) -> ClassifierMixin:
    """Load model according to the config dict"""

    if cfg.model_name == "random_forest":
        model = RandomForestClassifier(**cfg.model_kwargs, random_state=cfg.seed)
    elif cfg.model_name == "gradient_boosting":
        model = GradientBoostingClassifier(**cfg.model_kwargs, random_state=cfg.seed)
    else:
        raise ValueError(f"Invalid model name {cfg.model_name}")
    return model
