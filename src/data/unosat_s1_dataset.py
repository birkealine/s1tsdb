"""Create datasets of features and labels"""

import pandas as pd
import time
from typing import Mapping

from src.classification.features import default_extract_features, get_df_features
from src.classification.utils import assign_labels
from src.data import load_unosat_labels
from src.data.utils import aoi_orbit_iterator
from src.utils.time import print_sec


class UNOSAT_S1TS_Dataset:
    """
    DataModule to train and test on UNOSAT labels.

    Creates datasets of features and labels from UNOSAT labels and Sentinel-1 time-series.
    """

    def __init__(self, cfg, extract_features=default_extract_features):

        time_start = time.time()

        # Configuration
        self.aois_test = cfg.aois_test
        self.damages_to_keep = cfg.damages_to_keep
        self.time_periods = cfg.time_periods  # the time periods
        self.extract_winds = cfg.extract_winds
        self.random_neg_labels = cfg.random_neg_labels

        # Load feature extractor method
        self.extract_features = extract_features

        # Load unosat labels
        self.unosat_labels = load_unosat_labels(labels_to_keep=self.damages_to_keep)
        self.unosat_labels.index.name = "unosat_id"

        # Create datasets
        self.df, self.df_test = self.create_datasets()
        print(f"Dataset created in {print_sec(time.time()-time_start)}.")
        print(f"Training set: {self.df.shape}, Test set: {self.df_test.shape}")
        print(f"Training set labels: {self.df['label'].value_counts().to_dict()}")
        print(f"Test set labels: {self.df_test['label'].value_counts().to_dict()}")

    def get_datasets(self, split="test", fold=None, remove_unknown_labels=True):

        # Remove unknown labels if specified
        df = self.df[self.df["label"] != -1].copy() if remove_unknown_labels else self.df
        df_test = self.df_test[self.df_test["label"] != -1].copy() if remove_unknown_labels else self.df_test

        if split == "test":
            return df, df_test
        elif split == "valid":
            assert fold is not None, "Fold must be specified for validation set"
            return df[df["bin"] != fold], df[df["bin"] == fold]
        else:
            raise ValueError(f"Invalid split: {split}")

    def create_datasets(self):
        """
        Create train and test datasets.

        Each dataset has X columns of features (all starting with "VV" or "VH"), and some metadata
        columns such as aoi, orbit, label, etc."
        """

        # Extract features for all time periods
        # time periods can be a dict of list (typically {'pre': [pre1, pre2, ...], 'post':  [post1, post2, ..]},
        # then each combination of pre/post will be a dataset, or a list of dict, then each dict
        # (typically {'pre: pre1, 'post': post1}) will be a dataset
        # pre1 or post1 are tuples with start and end dates (eg ('2021-01-01', '2021-02-01'))

        dfs = []
        if isinstance(self.time_periods, Mapping):  # dict or omegaconf.DictConfig
            for pre in self.time_periods["pre"]:
                for post in self.time_periods["post"]:
                    d_periods = {"pre": pre, "post": post}
                    dfs.append(self.extract_all_features(d_periods, self.extract_winds))
        else:
            for d_periods in self.time_periods:
                dfs.append(self.extract_all_features(d_periods, self.extract_winds))
        df = pd.concat(dfs, axis=0)

        # Add metadata (damage type, date of analysis, etc.) from the original dataset
        df = df.loc[self.unosat_labels.index]
        df = df.merge(
            self.unosat_labels[["damage", "date", "geometry", "bin"]], left_index=True, right_index=True, how="left"
        )
        df.reset_index(inplace=True)

        # filter out damages
        df = df[df["damage"].isin(self.damages_to_keep)]

        # Add label based on date and pre/post time periods
        df = assign_labels(df)

        # Set damage to 0 for negative labels and to -1 for unknown labels
        df.loc[df["label"] == 0, "damage"] = 0
        df.loc[df["label"] == -1, "damage"] = -1

        # columns in nicer order (meta and then features)
        feat_col = [c for c in df.columns if c.startswith(("VV", "VH"))]
        df = df[[c for c in df.columns if c not in feat_col] + feat_col]

        # remove rows with NaNs
        if df.isna().any().any():
            print("Warning: NaNs in features")
            df = df[~df.isna().any(axis=1)]

        # split into train and test
        df_train = df[~df["aoi"].isin(self.aois_test)]
        df_test = df[df["aoi"].isin(self.aois_test)]

        if self.random_neg_labels > 0:
            df_train = self.add_random_neg_samples(df_train)
        return df_train, df_test

    def extract_all_features(self, time_periods, extract_winds):
        """Iterate over AOIs and orbits (cf get_df_features for docs)"""
        dfs = []
        for aoi, orbit in aoi_orbit_iterator():

            df = get_df_features(
                aoi, orbit, time_periods, extract_winds, dense_inference=False, extract_features=self.extract_features
            )
            dfs.append(df)

        df = pd.concat(dfs, axis=0)
        return df

    def add_random_neg_samples(self, df):
        """Augment the training set with random locations, to make it more diverse"""
        print("Random locations not implemented yet")
        return df
