"""Trainer to perform 5-fold cross-validation and inference on test set"""

from collections import defaultdict
import joblib
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from typing import List, Tuple
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
)
from sklearn.base import ClassifierMixin
import wandb

from src.utils.time import print_sec


class CVTrainer:
    def __init__(
        self,
        df: pd.DataFrame,
        df_test: pd.DataFrame,
        model: ClassifierMixin,
        features_extractor: callable,
        metrics: List[str] = ["accuracy", "precision", "recall", "f1"],
        aggregation: str = "mean",
        seed: int = 123,
        logging_folder: str = None,
        use_wandb: bool = False,
    ):
        """
        Trainer to perform 5-fold cross-validation and inference on test set

        Args:
            df (pd.DataFrame): The dataframe with the training data. It must have a multi-index with the following
                levels: aoi, orbit, unosat_id, label, band. And the data must be given in columns T0, T1, ..., Tn.
            df_test (pd.DataFrame): Same as df but for the test set.
            model (ClassifierMixin): The model. eg RandomForestClassifier
            features_extractor (callable): The function to extract features from the time series.
            metrics (List[str]): The metrics. Defaults to ["accuracy", "precision", "recall", "f1"].
            aggregation (str): The aggregation method. Defaults to "mean".
            seed (int): The random seed. Defaults to 123.
            logging_folder (str): The name of the log folder where to save the model. The model will have same name
                as folder or just 'model' if logging_folder is None. Defaults to None.
            use_wandb (bool): Whether to use wandb. Defaults to False.
        """

        INDEX_NAMES = ["aoi", "orbit", "unosat_id", "start_month", "label", "band"]
        VALID_METRICS = ["accuracy", "precision", "recall", "f1"]
        assert df.index.names == INDEX_NAMES, "Invalid index"
        assert df_test.index.names == INDEX_NAMES, "Invalid index"
        assert all([m in VALID_METRICS for m in metrics]), "Invalid metric"

        self.df = df
        self.df_test = df_test
        self.model = model
        self.features_extractor = features_extractor
        self.metrics = metrics
        self.aggregation = aggregation
        self.seed = seed
        self.logging_folder = logging_folder
        self.run_name = logging_folder.name if logging_folder else "model"
        self.use_wandb = use_wandb

    def save_model(self, fp):
        """Save model"""
        joblib.dump(self.model, fp)
        print(f"Model saved at {fp}")

    def load_model(self, fp):
        """Load model"""
        self.model = joblib.load(fp)
        print(f"Model loaded from {fp}")

    def split_train_valid(self, df: pd.DataFrame, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and validation"""

        df_train = df[df.bin != fold].copy()
        df_val = df[df.bin == fold].copy()
        return df_train, df_val

    def compute_metrics(self, y_true, y_preds, d_metrics, suffix=""):
        """Compute metrics (suffix is _agg or nothing)"""
        for metric in self.metrics:
            if metric == "accuracy":
                d_metrics[metric + suffix].append(accuracy_score(y_true, y_preds))
            elif metric == "precision":
                d_metrics[metric + suffix].append(precision_score(y_true, y_preds))
            elif metric == "recall":
                d_metrics[metric + suffix].append(recall_score(y_true, y_preds))
            elif metric == "f1":
                d_metrics[metric + suffix].append(f1_score(y_true, y_preds))
            else:
                raise ValueError(f"Invalid metric: {metric}")

        d_metrics["N_support" + suffix].append(len(y_true))  # add number of samples
        return d_metrics

    def print_summary_metrics(self, d_metrics, aggregated=False):
        suffix = "_agg" if aggregated else ""
        agg = "aggregated" if aggregated else ""
        print(f"Average metrics {agg} (N = {np.mean(d_metrics['N_support' + suffix]):.0f}):")
        for metric in self.metrics:
            metric_mean = np.mean(d_metrics[metric + suffix])
            metric_std = np.std(d_metrics[metric + suffix])
            print(f"{metric} {agg}: {metric_mean:.2f} +/- {metric_std:.2f}")

    def get_metrics(self, df_preds, d_metrics, aggregated=False, verbose=0):
        suffix = "_agg" if aggregated else ""
        agg = "Aggregated" if aggregated else "Non-aggregated"

        # Get true label and predictions
        y_true = df_preds.label.values
        y_preds = df_preds.preds.values

        # Compute metrics
        d_metrics = self.compute_metrics(y_true, y_preds, d_metrics, suffix=suffix)
        if verbose:
            print(f"{agg} metrics (N={len(df_preds)})")
            print(classification_report(y_true, y_preds, target_names=["intact", "destroyed"]))
        return d_metrics

    def aggregate_predictions(self, df):
        """Aggregate predictions (must be a column 'preds_proba') for the same satellite pass"""
        df_agg = df.groupby(["aoi", "unosat_id", "start_month", "label"]).agg({"preds_proba": self.aggregation})
        df_agg["preds"] = (df_agg["preds_proba"] > 0.5).astype(int)
        df_agg = df_agg.reset_index()
        return df_agg

    def get_X_y_from_features(self, features):
        X = features[[c for c in features.columns if c.startswith(("VV", "VH"))]].values
        y = features["label"].values
        return X, y

    def get_X_y_from_df(self, df):
        features = self.features_extractor(df)
        X, y = self.get_X_y_from_features(features)
        return X, y

    def fit(self, df, verbose=0):
        X, y = self.get_X_y_from_df(df)
        if verbose:
            print(f"Fitting model with: {X.shape}, y.shape: {y.shape}")
        self.model.fit(X, y)

    def inference(self, df):
        """Inference. Returns a dataframe with columns preds and preds_proba"""

        # Get X and y from time series
        features = self.features_extractor(df)
        X, _ = self.get_X_y_from_features(features)

        # Predict
        preds_proba = self.model.predict_proba(X)[:, 1]  # Only keep label 'destroyed'
        preds = (preds_proba > 0.5).astype(int)  # binary classification

        # Store in dataframe
        col_to_keep = ["aoi", "unosat_id", "orbit", "start_month", "label"]
        df_preds = features[col_to_keep].copy()
        df_preds["preds_proba"] = preds_proba
        df_preds["preds"] = preds
        return df_preds

    def train_cv(self, verbose=0):
        """5-fold cross-validation"""
        start_time = time.time()
        d_metrics = defaultdict(list)

        for fold in tqdm(range(1, 6)):
            if verbose:
                print(f"Fold {fold}")

            # Split data into training and validation
            df_train, df_val = self.split_train_valid(self.df, fold)

            # Fit model
            self.fit(df_train, verbose=verbose)

            # Perform inference on validation set
            df_preds = self.inference(df_val)

            # Aggregate predictions
            df_preds_agg = self.aggregate_predictions(df_preds)

            # Compute (and print) metrics
            d_metrics = self.get_metrics(df_preds, d_metrics, aggregated=False, verbose=verbose)
            d_metrics = self.get_metrics(df_preds_agg, d_metrics, aggregated=True, verbose=verbose)

            if verbose:
                print(f"Time elapsed: {print_sec(time.time() - start_time)}")

        # Print summary results
        self.print_summary_metrics(d_metrics, aggregated=False)
        self.print_summary_metrics(d_metrics, aggregated=True)

        if self.use_wandb:
            # Log to wandb as summary metrics
            for metric in self.metrics:
                wandb.run.summary[f"valid_{metric}"] = np.mean(d_metrics[metric])
                wandb.run.summary[f"valid_{metric}_agg"] = np.mean(d_metrics[metric + "_agg"])

        print(f"Total time: {print_sec(time.time() - start_time)}")
        return d_metrics

    def train_and_test(self):
        """Train on full training set and test on test set"""
        start_time = time.time()
        d_metrics = defaultdict(list)

        # Fit model
        self.fit(self.df)

        # Inference on test set
        df_preds = self.inference(self.df_test)
        self.df_preds = df_preds  # save for further analysis

        # Aggregate predictions
        df_preds_agg = self.aggregate_predictions(df_preds)
        self.df_preds_agg = df_preds_agg

        # Compute (and print) metrics
        d_metrics = self.get_metrics(df_preds, d_metrics, aggregated=False, verbose=1)
        d_metrics = self.get_metrics(df_preds_agg, d_metrics, aggregated=True, verbose=1)
        print(f"Total time: {print_sec(time.time() - start_time)}")

        if self.use_wandb:
            # Log to wandb as summary metrics
            for metric in self.metrics:
                wandb.run.summary[f"test_{metric}"] = d_metrics[metric][0]
                wandb.run.summary[f"test_{metric}_agg"] = d_metrics[metric + "_agg"][0]
        return d_metrics
