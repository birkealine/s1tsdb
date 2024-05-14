from collections import defaultdict
import joblib
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    fbeta_score,
    roc_auc_score,
    classification_report,
)
from typing import Tuple

from src.constants import UKRAINE_WAR_START


def save_model(model, fp):
    """Save model"""
    joblib.dump(model, fp)
    print(f"Model saved at {fp}")


def load_model(model, fp):
    """Load model"""
    model = joblib.load(fp)
    print(f"Model loaded from {fp}")
    return model


def compute_metrics(
    y_true,
    y_preds_proba,
    metrics=["accuracy", "precision", "recall", "f1", "f05", "auc"],
    threshold=0.5,
    suffix="",
    d_metrics=None,
    verbose=1,
):
    """Compute metrics (suffix is _agg or nothing)"""

    y_preds = (y_preds_proba >= threshold).astype(int)

    d_metrics = d_metrics if d_metrics else defaultdict(list)
    for metric in metrics:
        if metric == "accuracy":
            d_metrics[metric + suffix] = accuracy_score(y_true, y_preds)
        elif metric == "precision":
            d_metrics[metric + suffix] = precision_score(y_true, y_preds)
        elif metric == "recall":
            d_metrics[metric + suffix] = recall_score(y_true, y_preds)
        elif metric == "f1":
            d_metrics[metric + suffix] = f1_score(y_true, y_preds)
        elif metric == "f05":
            d_metrics[metric + suffix] = fbeta_score(y_true, y_preds, beta=0.5)
        elif metric == "auc":
            d_metrics[metric + suffix] = roc_auc_score(y_true, y_preds_proba)
        else:
            raise NotImplementedError(f"Metric: {metric} not implemented")

    d_metrics["N_support" + suffix] = len(y_true)  # add number of samples

    if verbose:
        agg = "Aggregated" if "agg" in suffix else "Non-aggregated"
        print(f"{agg} metrics with threshold={threshold} (N = {len(y_true)}):")
        print(classification_report(y_true, y_preds, target_names=["intact", "destroyed"], digits=3))
    return d_metrics


def aggregate_predictions(
    df: pd.DataFrame,
    aggregation: str = "mean",
    columns=["aoi", "unosat_id", "label"],
) -> pd.DataFrame:
    """Aggregate predictions on columns"""
    df_agg = df.groupby(columns).agg({"preds_proba": aggregation})
    df_agg = df_agg.reset_index()
    return df_agg


def split_train_valid(df: pd.DataFrame, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into training and validation"""
    assert "bin" in df.columns, "Column 'bin' not found in dataframe."
    df_train = df[df.bin != fold].copy()
    df_val = df[df.bin == fold].copy()
    return df_train, df_val


def assign_labels(df):
    """Define labels based on dates."""

    def _assign_label(row):

        assert (
            row.pre_start < row.pre_end and row.post_start < row.post_end and row.pre_end <= row.post_start
        ), f"Invalid dates: {row.pre_start} < {row.pre_end} < {row.post_start} < {row.post_end}"
        assert (
            row.pre_end < UKRAINE_WAR_START
        ), f"pre period should be before war start: {row.pre_end} < {UKRAINE_WAR_START}"

        if row.post_end <= UKRAINE_WAR_START:
            return 0  # negative labels
        if row.post_end < row.date.strftime("%Y-%m-%d"):
            return -1  # unknown
        else:
            return 1

    df["label"] = df.apply(_assign_label, axis=1)
    return df
