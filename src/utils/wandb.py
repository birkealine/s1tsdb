from omegaconf import OmegaConf
import os
import wandb

from src.constants import WANDB_PROJECT, WANDB_USERNAME

os.environ["WANDB__SERVICE_WAIT"] = "300"


def init_wandb(run_name, folder, cfg=None):
    # WandB initialization
    if cfg:
        # Save config
        wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    wandb.init(
        entity=WANDB_USERNAME,
        project=WANDB_PROJECT,
        name=run_name,
        dir=folder,
    )
