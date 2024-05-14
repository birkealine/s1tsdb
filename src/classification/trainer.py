"""Trainer to perform 5-fold cross-validation and inference on test set"""

from collections import defaultdict
import numpy as np
import time
from tqdm import tqdm
from typing import List
from sklearn.base import ClassifierMixin
import wandb

from src.data.unosat_s1_dataset import UNOSAT_S1TS_Dataset
from src.utils.time import print_sec
from src.classification.utils import aggregate_predictions, compute_metrics

VALID_METRICS = ["accuracy", "precision", "recall", "f1", "f05", "auc"]


class S1TSDD_Trainer:
    def __init__(
        self,
        ds: UNOSAT_S1TS_Dataset,
        model: ClassifierMixin,
        metrics: List[str] = VALID_METRICS,
        aggregation: str = "mean",
        seed: int = 123,
        verbose: int = 0,
        use_wandb: bool = False,
    ):
        """
        Trainer to perform 5-fold cross-validation and inference on test set

        Args:
            ds (UNOSAT_S1TS_Dataset): The dataset
            model (ClassifierMixin): The model. eg RandomForestClassifier
            metrics (List[str]): The metrics. Defaults to VALID_METRICS.
            aggregation (str): The aggregation method. Defaults to "mean".
            seed (int): The random seed. Defaults to 123.
            verbose (int): The verbosity level. Defaults to 0.
            use_wandb (bool): Whether to use wandb. Defaults to False.
        """

        self.ds = ds

        # check columnsa re valid
        df, df_test = ds.get_datasets("test")

        COLUMNS_NAMES = ["unosat_id", "aoi", "label"]
        assert all([c in df.columns for c in COLUMNS_NAMES]), "Invalid columns in training set"
        assert all([c in df_test.columns for c in COLUMNS_NAMES]), "Invalid columns in test set"
        assert all([m in VALID_METRICS for m in metrics]), "Invalid metric"

        # Make sure wandb is initialized, otherwise set to False
        if use_wandb and not wandb.run:
            print("WandB is not initialized, setting use_wandb to False.")
            use_wandb = False

        self.model = model
        self.metrics = metrics
        self.aggregation = aggregation
        self.seed = seed
        self.verbose = verbose
        self.use_wandb = use_wandb

    def print_summary_metrics(self, d_metrics, aggregated=False, threshold=0.5):
        """For cross-validation, need to take mean and std of metrics over the 5 folds."""
        suffix = "_agg" if aggregated else ""
        agg = "aggregated" if aggregated else ""
        print(f"Average metrics {agg} with threshold={threshold} (N = {np.mean(d_metrics['N_support' + suffix]):.0f}):")
        for metric in self.metrics:
            metric_mean = np.mean(d_metrics[metric + suffix])
            metric_std = np.std(d_metrics[metric + suffix])
            print(f"{metric} {agg}: {metric_mean:.2f} +/- {metric_std:.2f}")

    def get_metrics(self, df_preds, d_metrics, aggregated=False, threshold=0.5):
        suffix = "_agg" if aggregated else ""

        # Get true label and predictions
        y_true = df_preds.label.values
        y_preds_proba = df_preds.preds_proba.values

        # Compute metrics
        d_metrics = compute_metrics(
            y_true, y_preds_proba, self.metrics, threshold, suffix=suffix, d_metrics=d_metrics, verbose=self.verbose
        )
        return d_metrics

    def get_X_y_from_df(self, df):
        """Get X (the features starting with VV and VH) and y (the label) from a dataframe"""
        X = df[[c for c in df.columns if c.startswith(("VV", "VH"))]].values
        y = df["label"].values
        return X, y

    def fit(self, df):
        X, y = self.get_X_y_from_df(df)
        if self.verbose:
            print(f"Fitting model with: {X.shape}, y.shape: {y.shape}")
        self.model.fit(X, y)

    def inference(self, df):
        """Inference. Returns a dataframe with columns preds and preds_proba"""

        # Get X and y from features
        X, _ = self.get_X_y_from_df(df)

        # Predict
        preds_proba = self.model.predict_proba(X)[:, 1]  # Only keep label 'destroyed'

        # Store in dataframe
        col_to_keep = ["aoi", "unosat_id", "orbit", "label"]
        df_preds = df[col_to_keep].copy()
        df_preds["preds_proba"] = preds_proba
        return df_preds

    def train_cv(self, threshold_for_metrics: float = 0.5):
        """5-fold cross-validation"""
        start_time = time.time()
        d_metrics = defaultdict(list)

        self.df_preds_cv = []
        self.df_preds_agg_cv = []

        for fold in tqdm(range(1, 6)):
            if self.verbose:
                print(f"Fold {fold}")

            # Split data into training and validation
            df_train, df_val = self.ds.get_datasets("valid", fold=fold, remove_unknown_labels=True)

            # Fit model
            self.fit(df_train)

            # Perform inference on validation set
            df_preds = self.inference(df_val)
            self.df_preds_cv.append(df_preds)  # store here for further analysis

            # Aggregate predictions
            df_preds_agg = aggregate_predictions(df_preds, aggregation=self.aggregation)
            self.df_preds_agg_cv.append(df_preds_agg)

            # Compute (and print) metrics
            d_metrics = self.get_metrics(df_preds, d_metrics, aggregated=False, threshold=threshold_for_metrics)
            d_metrics = self.get_metrics(df_preds_agg, d_metrics, aggregated=True, threshold=threshold_for_metrics)

            if self.verbose:
                print(f"Time elapsed: {print_sec(time.time() - start_time)}")

        # Print summary results
        self.print_summary_metrics(d_metrics, aggregated=False, threshold=threshold_for_metrics)
        self.print_summary_metrics(d_metrics, aggregated=True, threshold=threshold_for_metrics)

        if self.use_wandb:
            # Log to wandb as summary metrics
            for metric in self.metrics:
                wandb.run.summary[f"valid_{metric}"] = np.mean(d_metrics[metric])
                wandb.run.summary[f"valid_{metric}_agg"] = np.mean(d_metrics[metric + "_agg"])

        print(f"Total time: {print_sec(time.time() - start_time)}")
        return d_metrics

    def train_and_test(self, threshold_for_metrics=0.5):
        """Train on full training set and test on test set"""
        start_time = time.time()
        d_metrics = defaultdict(list)

        df, df_test = self.ds.get_datasets("test", remove_unknown_labels=True)

        # Fit model
        self.fit(df)

        # Inference on test set
        df_preds = self.inference(df_test)
        self.df_preds = df_preds  # store here for further analysis

        # Aggregate predictions
        print(f"Aggregating predictions with method: {self.aggregation}")
        df_preds_agg = aggregate_predictions(df_preds, aggregation=self.aggregation)
        self.df_preds_agg = df_preds_agg

        # Compute (and print) metrics
        d_metrics = self.get_metrics(df_preds, d_metrics, aggregated=False, threshold=threshold_for_metrics)
        d_metrics = self.get_metrics(df_preds_agg, d_metrics, aggregated=True, threshold=threshold_for_metrics)
        print(f"Total time (training + testing): {print_sec(time.time() - start_time)}")

        if self.use_wandb:
            # Log to wandb as summary metrics
            for metric in self.metrics:
                wandb.run.summary[f"test_{metric}"] = d_metrics[metric][0]
                wandb.run.summary[f"test_{metric}_agg"] = d_metrics[metric + "_agg"][0]
        return d_metrics
