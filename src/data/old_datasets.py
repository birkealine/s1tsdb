"""Functions to load datasets ready for training/testing"""
from argparse import ArgumentParser
from omegaconf import DictConfig
import pandas as pd
import random
from typing import Tuple, Union

from src.constants import READY_PATH

INDEX_NAMES = ["aoi", "orbit", "unosat_id", "start_month", "label", "band"]


def load_datasets(cfg: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and test datasets"""

    # Get name of file to load and read
    fp = get_dataset_ready_name(cfg)
    assert fp.exists(), f"File {fp} does not exist"
    df_full = pd.read_csv(fp)

    # Keep only some type of damage
    labels_to_keep = [1, 2, 3, 4, 5, 7] if cfg.labels_to_keep in ["all", None] else cfg.labels_to_keep
    df_full = df_full[df_full.damage.isin(labels_to_keep)]

    # Split into training and test
    df = df_full[~df_full.aoi.isin(cfg.aois_test)]
    df_test = df_full[df_full.aoi.isin(cfg.aois_test)]

    # Add random negative samples in training set if needed
    if cfg.add_random_neg_labels > 0:
        print(f"Adding {cfg.add_random_neg_labels*100}% random negative samples")
        df = add_random_neg_samples(df, cfg)

    # Post-processing according to config
    df = postprocess_df(df, cfg.train_cfg)
    df_test = postprocess_df(df_test, cfg.test_cfg)

    print(f"Training dataset: {len(df)//2} time-series")  # div. by 2 because VV and VH
    print(f"Test dataset: {len(df_test)//2} time-series")  # idem
    return df, df_test


def add_random_neg_samples(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Add a certain percentage of random negative samples"""

    # Read csv file
    fp = get_dataset_ready_name(cfg)
    fp_random = fp.parent / f"{fp.stem}_random_locations{fp.suffix}"
    assert fp_random.exists(), f"File {fp_random} does not exist"
    df_random = pd.read_csv(fp_random)

    # Keep same AOIs
    df_random = df_random[df_random.aoi.isin(df.aoi.unique())]

    # Select certain percentage of random samples to add (vs number of UNOSAT labels)
    n_original_labels = len(df.groupby(["aoi", "unosat_id"]))
    n_to_add = int(n_original_labels * cfg.add_random_neg_labels)
    groups = list(df_random.groupby(["aoi", "unosat_id"]).groups.keys())
    selected_groups = sorted(random.sample(groups, n_to_add))
    df_random = df_random.set_index(["aoi", "unosat_id"]).loc[selected_groups].reset_index()

    # concat with original df
    df = pd.concat([df, df_random], ignore_index=True)
    df.fillna(0, inplace=True)  # fill NaNs with 0 for date_of_analysis and flag random locations
    return df


def postprocess_df(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Post-process dataframe according to config"""

    # remove time-series for which we don't know for sure the label
    if cfg.remove_unknown_labels:
        df = df[df.label.isin([0, 1])]

    # if we want only starting month per label (instead of all months)
    if not cfg.sliding_window:
        df = df[df.start_month.isin([cfg.start_month_pre, cfg.start_month_post])]

    # Set multi-index
    return df.set_index(INDEX_NAMES)


def get_dataset_ready_name(cfg: Union[DictConfig, ArgumentParser]):
    """Folder to store individual AOI files (same name as the main file)"""
    base_folder = READY_PATH / "time_series_datasets"
    filename = f"ts_{cfg.n_tiles}d_{cfg.extraction_strategy.replace('-','_')}.csv"
    filepath = base_folder / filename
    return filepath


def get_folder_ready_name(cfg: Union[DictConfig, ArgumentParser]):
    """Folder to store individual AOI files (same name as the main file)"""
    base_folder = READY_PATH / "time_series_datasets"
    folder_name = f"ts_{cfg.n_tiles}d_{cfg.extraction_strategy.replace('-','_')}"
    folder = base_folder / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder
