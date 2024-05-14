"""Full run (5-fold cross-validation + test set) + dense inference"""

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import time
import wandb

from src.constants import HYDRA_CONFIG_NAME, HYDRA_CONFIG_PATH, LOGS_PATH
from src.data.old_datasets import load_datasets
from src.dense_inference import dense_inference
from src.classification.model_factory import load_model
from src.old_trainer import CVTrainer
from src.utils.time import print_sec
from src.utils.wandb import init_wandb
from src.visualization.dense_predictions import plot_dense_predictions


def full_run(cfg: DictConfig, logging_folder: Path = None):
    """5-fold cross-validation, compute metrics on test set and perform dense inference on selected AOIs"""

    # Make sure wandb is initialized, otherwise set to False
    if cfg.use_wandb and not wandb.run:
        print("WandB is not initialized, setting use_wandb to False.")
        cfg.use_wandb = False

    # Default logging folder for quick runs
    if logging_folder is None:
        logging_folder = Path("./tmp_logs")
        logging_folder.mkdir(exist_ok=True)

    # Load data
    df, df_test = load_datasets(cfg)

    # Load model
    model = load_model(cfg)

    # Get features extractor (for now function below)
    extract_features = get_features_extractor(cfg)

    # Load trainer
    trainer = CVTrainer(
        df,
        df_test,
        model,
        features_extractor=extract_features,
        aggregation=cfg.aggregation_method,
        logging_folder=logging_folder,
        use_wandb=cfg.use_wandb,
    )

    # # 5-fold cross-validation
    print("Starting 5-fold cross-validation...")
    _ = trainer.train_cv()

    # Test set
    print("Starting testing on test set...")
    _ = trainer.train_and_test()

    # Save model trained on full training set
    model_fp = logging_folder / "model" / f"{cfg.run_name}.joblib"
    model_fp.parent.mkdir(exist_ok=True)
    trainer.save_model(model_fp)

    # Dense inference
    preds_logging_folder = logging_folder / "predictions"
    preds_logging_folder.mkdir(exist_ok=True)
    start_dates = pd.date_range("2020-06-01", "2022-06-01", freq="MS").strftime("%Y-%m-%d").tolist()
    for aoi in cfg.aois_dense_preds:
        preds_aoi_folder = preds_logging_folder / aoi
        print(f"Starting dense inference for {aoi}...")
        dense_inference(
            aoi,
            trainer.model,
            folder=preds_aoi_folder,
            features_extractor=extract_features,
            start_dates=start_dates,
            n_tiles=cfg.n_tiles,
            extraction_strategy=cfg.extraction_strategy,
        )

        # Plot dense predictions and save figs
        for start_date in start_dates:
            fig = plot_dense_predictions(aoi, preds_aoi_folder, start_date, show=False)
            fig.savefig(preds_aoi_folder / f"{start_date}.png")
            plt.close(fig)

    # Log to each wandb (one step per start_date)
    for start_date in start_dates:
        d_wandb = {}
        for aoi in cfg.aois_dense_preds:
            # read fig from file
            img_path = str(preds_logging_folder / aoi / f"{start_date}.png")
            fig = wandb.Image(img_path, caption=f"{aoi} - {start_date}")
            d_wandb[f"preds_{aoi}"] = fig
        wandb.log(d_wandb)


def get_features_extractor(cfg: DictConfig):
    if cfg.features == "all":
        return extract_features
    elif cfg.features is None:
        print("here")
        return signal_as_features


def signal_as_features(df_ts):
    """No manual features, just raw signal."""

    time_columns = [c for c in df_ts.columns if c.startswith("T")]
    df = df_ts.copy()
    for i in range(32):
        df[f"{i}"] = df_ts[time_columns].iloc[:, i]

    df = df.drop(time_columns, axis=1)  # Drop columns with raw signal

    # Split into VV and VH and combine
    df_vv = df.xs("VV", level="band")
    df_vh = df.xs("VH", level="band")
    df_vv.columns = [c if c in df_ts.columns else f"VV_{c}" for c in df_vv.columns]
    df_vh.columns = [c if c in df_ts.columns else f"VH_{c}" for c in df_vh.columns]
    df_vh = df_vh.drop([c for c in df_vh.columns if c in df_vv.columns], axis=1)
    df_stats = pd.concat([df_vv, df_vh], axis=1)
    return df_stats.reset_index()


# TODO: Refactor and make it modular
def extract_features(df_ts):
    """The dataframe must have a multiindex with at least bands (VV and VH)"""

    time_columns = [c for c in df_ts.columns if c.startswith("T")]

    df = df_ts.copy()

    # Global features
    df["mean"] = df_ts[time_columns].mean(axis=1)
    df["std"] = df_ts[time_columns].std(axis=1)
    df["median"] = df_ts[time_columns].median(axis=1)
    df["max"] = df_ts[time_columns].max(axis=1)
    df["min"] = df_ts[time_columns].min(axis=1)
    df["ptp"] = df["max"] - df["min"]
    df["skew"] = df_ts[time_columns].skew(axis=1)
    df["kurtosis"] = df_ts[time_columns].kurtosis(axis=1)
    df["var"] = df_ts[time_columns].var(axis=1)

    # Time-dependant features
    # df['mean_start'] = df_ts[time_columns].iloc[:,:8].mean(axis=1)
    # df['std_start'] = df_ts[time_columns].iloc[:,-8:].mean(axis=1)
    # df['mean_end'] = df_ts[time_columns].iloc[:,:8].mean(axis=1)
    # df['std_end'] = df_ts[time_columns].iloc[:,-8:].mean(axis=1)
    # Mean and std for each slice of 4 dates
    for i in range(int(np.ceil(len(time_columns) / 4))):  # last slice might be smaller if not multiple of 4
        df[f"mean_slice_{i}"] = df_ts[time_columns].iloc[:, i * 4 : (i + 1) * 4].mean(axis=1)  # noqa E203
        df[f"std_slice_{i}"] = df_ts[time_columns].iloc[:, i * 4 : (i + 1) * 4].std(axis=1)  # noqa E203

    # Postprcessing
    df = df.drop(time_columns, axis=1)  # Drop columns with raw signal

    # Split into VV and VH and combine
    df_vv = df.xs("VV", level="band")
    df_vh = df.xs("VH", level="band")
    df_vv.columns = [c if c in df_ts.columns else f"VV_{c}" for c in df_vv.columns]
    df_vh.columns = [c if c in df_ts.columns else f"VH_{c}" for c in df_vh.columns]
    df_vh = df_vh.drop([c for c in df_vh.columns if c in df_vv.columns], axis=1)
    df_stats = pd.concat([df_vv, df_vh], axis=1)
    return df_stats.reset_index()


@hydra.main(version_base=None, config_path=str(HYDRA_CONFIG_PATH), config_name=HYDRA_CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    print(f" ====== Run: {cfg.run_name} ======")
    print(OmegaConf.to_yaml(cfg))

    # Logging folder has run_name
    logging_folder = LOGS_PATH / cfg.run_name
    logging_folder.mkdir(exist_ok=True, parents=True)

    if cfg.use_wandb:
        init_wandb(cfg.run_name, logging_folder, cfg)

    # Run
    start_time = time.time()
    full_run(cfg, logging_folder)
    print(f"Full run done in {print_sec(time.time() - start_time)}.")

    if cfg.use_wandb:
        # Finish WandB run
        wandb.finish()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
