from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, fbeta_score
from tqdm import tqdm

from src.data import UNOSAT_S1TS_Dataset
from src.utils.time import timeit


@timeit
def compute_grid_search_cv(df, param_grid):
    """Compute grid search with cross validation for a given grid of hyperparameters."""

    assert "bin" in df.columns, "bin column must be present in dataframe for cross validation."

    d_results = []
    print(f"Starting CV grid search on {len(param_grid)} combinations of hyperparameters.")
    for param in tqdm(param_grid):

        if "bootstrap" in param and not param["bootstrap"]:
            param["max_samples"] = None

        d_metrics = defaultdict(list)
        for fold in range(5):

            # Get datasets for fold
            df_train, df_valid = df[df.bin != fold + 1], df[df.bin == fold + 1]
            X_train = df_train[[c for c in df_train.columns if c.startswith(("VV", "VH"))]].values
            y_train = df_train["label"].values
            X_valid = df_valid[[c for c in df_valid.columns if c.startswith(("VV", "VH"))]].values

            # fit and inference
            clf = RandomForestClassifier(**param, n_jobs=16, random_state=123)
            clf.fit(X_train, y_train)
            y_preds = clf.predict_proba(X_valid)[:, 1]

            # store in dataframe
            df_preds = df_valid[["aoi", "unosat_id", "orbit", "label"]].copy()
            df_preds["preds_proba"] = y_preds

            # aggregate predictions
            df_agg = df_preds.groupby(["aoi", "unosat_id", "label"]).agg({"preds_proba": "mean"}).reset_index()
            y_true = df_agg.reset_index()["label"]

            # compute metrics for different thresholds
            for thresh in [0.4, 0.5, 0.6, 0.75]:
                y_pred_agg_thresh = (df_agg["preds_proba"] > thresh).astype(int)
                d_metrics[f"acc_{thresh:.2f}"].append(accuracy_score(y_true, y_pred_agg_thresh))
                d_metrics[f"precision_{thresh:.2f}"].append(precision_score(y_true, y_pred_agg_thresh))
                d_metrics[f"recall_{thresh:.2f}"].append(recall_score(y_true, y_pred_agg_thresh))
                d_metrics[f"f1_{thresh:.2f}"].append(f1_score(y_true, y_pred_agg_thresh))
                d_metrics[f"f0.5_{thresh:.2f}"].append(fbeta_score(y_true, y_pred_agg_thresh, beta=0.5))

        # store metrics
        for k, v in d_metrics.items():
            param[k] = np.mean(v)
            param[f"{k}_std"] = np.std(v)

        d_results.append(param)

    return pd.DataFrame(d_results)


@timeit
def compute_grid_search(df, df_test, param_grid):
    """Compute grid search between training and test set for a given grid of hyperparameters."""

    # Get datasets
    X = df[[c for c in df.columns if c.startswith(("VV", "VH"))]].values
    y = df["label"].values
    X_test = df_test[[c for c in df_test.columns if c.startswith(("VV", "VH"))]].values

    # Grid search
    d_results = []
    print(f"Starting grid search on {len(param_grid)} combinations of hyperparameters.")
    for i, param in tqdm(enumerate(param_grid)):
        if i % 10 == 0:
            print(f"Grid search: {i}/{len(param_grid)}: {param}")

        if "bootstrap" in param and not param["bootstrap"]:
            param["max_samples"] = None

        # fit and inference
        clf = RandomForestClassifier(**param, n_jobs=12, random_state=123)
        clf.fit(X, y)
        y_preds = clf.predict_proba(X_test)[:, 1]

        # store in dataframe
        df_preds = df_test[["aoi", "unosat_id", "orbit", "label"]].copy()
        df_preds["preds_proba"] = y_preds

        # aggregate predictions
        df_agg = df_preds.groupby(["aoi", "unosat_id", "label"]).agg({"preds_proba": "mean"}).reset_index()
        y_true = df_agg.reset_index()["label"]

        # compute metrics for different thresholds
        for thresh in [0.5, 0.6, 0.75]:
            y_pred_agg_thresh = (df_agg["preds_proba"] > thresh).astype(int)
            param[f"acc_{thresh:.2f}"] = accuracy_score(y_true, y_pred_agg_thresh)
            param[f"precision_{thresh:.2f}"] = precision_score(y_true, y_pred_agg_thresh)
            param[f"recall_{thresh:.2f}"] = recall_score(y_true, y_pred_agg_thresh)
            param[f"f1_{thresh:.2f}"] = f1_score(y_true, y_pred_agg_thresh)
        d_results.append(param)

    return pd.DataFrame(d_results)


if __name__ == "__main__":

    import argparse
    from omegaconf import OmegaConf
    from sklearn.model_selection import ParameterGrid

    from src.classification.features import default_extract_features

    parser = argparse.ArgumentParser(description="Compute grid search for random forest hyperparameters.")
    parser.add_argument("-cv", "--cross_validation", action="store_true", help="Use cross validation.")
    parser.add_argument("-a", "--all_data", action="store_true", help="Use all data for cross validation.")
    args = parser.parse_args()

    if args.all_data and not args.cross_validation:
        raise ValueError("Use --all_data only for cross validation.")

    cfg = OmegaConf.create(
        dict(
            aggregation_method="mean",
            data=dict(
                aois_test=[f"UKR{i}" for i in range(1, 19) if i not in [1, 2, 3, 4]],
                damages_to_keep=[1, 2, 3],
                extract_winds=["3x3"],  # ['1x1', '3x3', '5x5']
                random_neg_labels=0.0,  # percentage of negative labels to add in training set (eg 0.1 for 10%)
                time_periods=[
                    dict(pre=("2021-02-24", "2022-02-23"), post=("2022-02-24", "2023-02-23")),
                    dict(pre=("2020-02-24", "2021-02-23"), post=("2021-02-24", "2022-02-23")),
                ],
            ),
            seed=123,
            run_name=None,
        )
    )

    ds = UNOSAT_S1TS_Dataset(cfg.data, extract_features=default_extract_features)
    df, df_test = ds.get_datasets("test")

    param_grid = ParameterGrid(
        {
            "n_estimators": [10, 20, 50, 100, 200],
            "max_depth": [20, 40, None],
            # "min_samples_split": [2, 4, 8, 16],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_samples": [0.5, 1.0],
        }
    )

    if args.cross_validation:
        if args.all_data:
            df = pd.concat([df, df_test])
            print("Cross-validating on full dataset (training + test)")
            fp = "results_rf_cv_all_data.csv"
        else:
            print("Cross-validating on training set")
            fp = "results_rf_cv.csv"
        df_results = compute_grid_search_cv(df, param_grid)
    else:
        print("Training on full training set, testing on test set")
        df_results = compute_grid_search(df, df_test, param_grid)
        fp = "results_rf.csv"

    df_results.to_csv(fp, index=False)
    print(f"Results saved at {fp}")
