"""Perform dense inference using a trained model."""

import numpy as np
import pandas as pd
from pathlib import Path
from rasterio.enums import Resampling
from scipy.signal import convolve
from sklearn.base import ClassifierMixin
from tqdm import tqdm
import time
from typing import List, Union
import xarray as xr

from src.data.sentinel1.orbits import get_valid_orbits
from src.data.sentinel1.time_series import get_s1_ts
from src.data.sentinel1.utils import get_target_s1
from src.utils.time import print_sec


def dense_inference(
    aoi: str,
    model: ClassifierMixin,
    folder: Path,
    features_extractor: callable,
    start_dates: Union[List[str], str] = "2021-10-01",
    n_tiles=32,
    extraction_strategy: str = "3x3",
) -> None:
    """
    Compute full 2D inference for a given AOI and a given model.

    Aggregates the predictions of each orbit and each start date.

    Args:
        aoi (str): The Area of Interest
        model (ClassifierMixin): The model already trained
        folder (Path): The folder where to save the predictions
        features_extractor (callable): The function to extract features from the time series.
        start_dates (Union[List[str], str], optional): Date(s) to start. Defaults to "2021-10-01".
        n_tiles (int, optional): Number of tiles in the time series to predict. Defaults to 32.
        extraction_strategy (str, optional): Either pixel-wise or 3x3. Defaults to "3x3".
    """

    print(f"Full inference for {aoi}")
    start_time = time.time()

    # Get orbits
    orbits = get_valid_orbits(aoi)
    print(f"Orbits: {orbits}")

    # Start dates
    start_dates = [start_dates] if isinstance(start_dates, str) else start_dates

    # Get target georeferenced tile
    global_target_img = get_target_s1(aoi, chunks=None).sel(band="VV")
    global_target_img["band"] = "preds_proba"

    # Predict for each orbit and each start date. The dict is made of start_date: preds, where
    # preds is a 3D array of shape (n_orbits, W, H) (so that aggregating over orbits is easy)
    d_preds = {d: np.zeros([len(orbits), *global_target_img.shape[-2:]]) for d in start_dates}

    for i, orbit in enumerate(orbits):
        print(f"Predicting for orbit {orbit}")

        # Only read once the entire time series, and slice it later
        s1_ts = get_s1_ts(aoi, orbit, chunks=None)

        for start_date in tqdm(start_dates):
            # time series to features
            df_stats = s1_ts_to_stats(
                s1_ts,
                features_extractor,
                start_date=start_date,
                n_tiles=n_tiles,
                extraction_strategy=extraction_strategy,
            )
            # Add metadata
            df_stats["aoi"] = aoi
            df_stats["orbit"] = orbit
            df_stats["has_nan"] = df_stats.isnull().any(axis=1)  # keep track of NaNs
            df_stats.fillna(0, inplace=True)

            # Inference
            X = df_stats[[c for c in df_stats.columns if c.startswith(("VV", "VH"))]].values
            preds_proba = model.predict_proba(X)[:, 1]

            # Fill with 0 where there were NaNs
            preds_proba[df_stats["has_nan"]] = 0

            # Reshape to original 2D shape
            preds_proba = preds_proba.reshape(*s1_ts.shape[-2:])

            # Reproject to match general target if needed
            if global_target_img.shape != preds_proba.shape:
                orbit_target_img = get_target_s1(aoi, orbit, chunks=None).sel(band="VV")
                orbit_target_img.values = preds_proba
                preds_proba = orbit_target_img.rio.reproject_match(
                    global_target_img, resampling=Resampling.bilinear
                ).values

            # Store in dict
            d_preds[start_date][i] = preds_proba

    # Combine all predictions
    print("Combining predictions")
    d_preds_agg = {}
    for start_date, preds in d_preds.items():
        preds_agg = global_target_img.copy()
        preds_agg.values = preds.mean(axis=0)
        d_preds_agg[start_date] = preds_agg

    # Save
    print(f"Saving predictions in {folder}")
    folder.mkdir(exist_ok=True, parents=True)
    for start_date, preds in d_preds_agg.items():
        preds.rio.to_raster(folder / f"{start_date}.tif")

    print(f"Done in {print_sec(time.time() - start_time)}.")


def s1_ts_to_stats(
    s1_ts: xr.DataArray,
    features_extractor: callable,
    start_date: str = "2021-10-01",
    n_tiles: int = 32,
    extraction_strategy: str = "pixel-wise",
) -> pd.DataFrame:
    """
    Transform the Sentinel-1 time series into a dataframe of features.

    Works by stacking each pixel individually and extracting features from the resulting dataframe.

    Args:
        s1_ts (xr.DataArray): The full Sentinel-1 time-series
        features_extractor (callable): The function to extract features from the dataframe of stacked pixels.
        start_date (str, optional): The date to start. Defaults to "2021-10-01".
        n_tiles (int, optional): Number of tiles in the time series. Defaults to 32.
        extraction_strategy (str, optional): Either pixel-wise or 3x3. Defaults to "pixel-wise".

    Returns:
        pd.DataFrame: The (big) dataframe of features.
    """
    # Only keep dates of interest
    s1_ts = s1_ts.sel(date=slice(start_date, None))[:n_tiles]

    # if extraction strategy not pixel-wise, need to convolve
    if extraction_strategy == "3x3":
        kernel = np.ones((3, 3)) / 9
        method = "direct" if s1_ts.isnull().sum() else "auto"  # can't use fft if there are NaNs
        s1_ts.values = convolve(s1_ts, kernel[None, None, :, :], mode="same", method=method)

    # Flatten along the time dimension
    T, B, W, H = s1_ts.shape
    s1_ts_reshaped = s1_ts.values.transpose(1, 2, 3, 0).reshape(B * W * H, T)

    # Store in dataframe
    df = pd.DataFrame(s1_ts_reshaped, columns=[f"T{i}" for i in range(T)])
    band_pixel_index = pd.MultiIndex.from_product([["VV", "VH"], range(W * H)], names=["band", "pixel"])
    df.index = band_pixel_index
    df = df.swaplevel().sort_index()

    # Extract features
    df_stats = features_extractor(df)

    return df_stats


# TODO: Refactor this function to be more modular
def extract_features(df_ts):
    """The dataframe must have a multiindex with at least bands (VV and VH)"""

    time_columns = [c for c in df_ts.columns if c.startswith("T")]

    df = df_ts.copy()

    # for i in range(32):
    #     df[f'{i}'] = df_ts[time_columns].iloc[:,i]
    df["mean"] = df_ts[time_columns].mean(axis=1)
    df["std"] = df_ts[time_columns].std(axis=1)
    df["median"] = df_ts[time_columns].median(axis=1)
    df["max"] = df_ts[time_columns].max(axis=1)
    df["min"] = df_ts[time_columns].min(axis=1)
    df["ptp"] = df["max"] - df["min"]
    df["skew"] = df_ts[time_columns].skew(axis=1)
    df["kurtosis"] = df_ts[time_columns].kurtosis(axis=1)
    df["var"] = df_ts[time_columns].var(axis=1)

    # df['mean_start'] = df_ts[time_columns].iloc[:,:8].mean(axis=1)
    # df['std_start'] = df_ts[time_columns].iloc[:,-8:].mean(axis=1)
    # df['mean_end'] = df_ts[time_columns].iloc[:,:8].mean(axis=1)
    # df['std_end'] = df_ts[time_columns].iloc[:,-8:].mean(axis=1)
    for i in range(int(np.ceil(len(time_columns) / 4))):  # last slice might be smaller if not multiple of 4
        df[f"mean_slice_{i}"] = df_ts[time_columns].iloc[:, i * 4 : (i + 1) * 4].mean(axis=1)  # noqa E203
        df[f"std_slice_{i}"] = df_ts[time_columns].iloc[:, i * 4 : (i + 1) * 4].std(axis=1)  # noqa E203

    # Drop time columns
    df = df.drop(time_columns, axis=1)

    # Split into VV and VH and combine
    df_vv = df.xs("VV", level="band")
    df_vh = df.xs("VH", level="band")
    df_vv.columns = [c if c in df_ts.columns else f"VV_{c}" for c in df_vv.columns]
    df_vh.columns = [c if c in df_ts.columns else f"VH_{c}" for c in df_vh.columns]
    df_vh = df_vh.drop([c for c in df_vh.columns if c in df_vv.columns], axis=1)
    df_stats = pd.concat([df_vv, df_vh], axis=1)
    return df_stats.reset_index()


# if __name__ == "__main__":
#     import joblib
#     from src.constants import AOIS_TEST, SRC_PATH

#     base_folder = SRC_PATH / "time_series_rf"

#     aoi = "UKR6"
#     model_name = "rf_sliding_window_3x3"
#     model = joblib.load(base_folder / f"models/{model_name}.joblib")

#     start_dates = pd.date_range("2020-06-01", "2022-06-01", freq="MS").strftime("%Y-%m-%d").tolist()
#     n_tiles = 32
#     extraction_strategy = "3x3"

#     for aoi in AOIS_TEST:
#         folder = Path(base_folder / "predictions" / f"{aoi}_{extraction_strategy}")

#         d_preds = dense_inference(
#             aoi,
#             model,
#             folder=folder,
#             features_extractor=extract_features,
#             start_dates=start_dates,
#             n_tiles=n_tiles,
#             extraction_strategy=extraction_strategy,
#         )
