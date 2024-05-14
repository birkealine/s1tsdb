"""Extract features from time-series"""

from collections.abc import Mapping
import numpy as np
import pandas as pd
from scipy.signal import convolve
from typing import List, Union

from src.data.sentinel1.time_series import get_s1_ts
from src.data.time_series.stacked_ts import load_stacked_ts


# The default feature extractor
def default_extract_features(df: pd.DataFrame, start: str, end: str, prefix: str = "") -> pd.DataFrame:
    """
    Transform a dataframe of time-series into a dataframe of features.

    Args:
        df (pd.DataFrame): The dataframe with time-series. Columns must be datetime.
        start (str): First date for slicing
        end (str): Last date for slicing
        prefix (str, optional): To be added in features names, eg pre, post, pre_3x3, ... Defaults to "".

    Returns:
        pd.DataFrame: The dataframes with the features.
    """

    # columns are datetime -> can slice directly between two dates
    df = df.loc[:, start:end]

    # features
    df_features = pd.DataFrame(index=df.index)
    df_features["mean"] = df.mean(axis=1)
    df_features["std"] = df.std(axis=1)
    df_features["median"] = df.median(axis=1)
    df_features["min"] = df.min(axis=1)
    df_features["max"] = df.max(axis=1)
    df_features["skew"] = df.skew(axis=1)
    df_features["kurt"] = df.kurt(axis=1)

    # rename columns using band, prefix (eg pre/post/pre_3x3, ...)
    df_vv = df_features.xs("VV", level="band")
    df_vh = df_features.xs("VH", level="band")
    df_vv.columns = [f"VV_{prefix}_{col}" for col in df_vv.columns]
    df_vh.columns = [f"VH_{prefix}_{col}" for col in df_vh.columns]
    return pd.concat([df_vv, df_vh], axis=1)


def get_df_features(
    aoi: str,
    orbit: str,
    time_periods: Union[Mapping, List],
    extract_winds: List,
    dense_inference: bool = False,
    extract_features: callable = default_extract_features,
    remove_nan: bool = True,
) -> pd.DataFrame:
    """
    Extract features from time-series for a given AOI, orbit, periods and windows.

    Each combination of period and window will be a prefix for the features. For instance, input can be:

    time_periods = {"pre": ("2019-01-01", "2019-01-31"), "post": ("2019-02-01", "2019-02-28")}
    extract_winds = ["3x3"]

    Args:
        aoi (str): The AOI.
        orbit (str): The orbit.
        time_periods (Union[Mapping, List]): The periods must all be like (start, end).
            If dict, will use the keys as prefix for the features. Otherwise it will be P1, P2, P3, ...
        extract_winds (List): The extracting windows. Eg ["1x1", "3x3"].
        dense_inference (bool, optional): Whether to use pre-extracted time-series (eg unosat points),
            or from stack of images ('dense' inference). Defaults to False.
        extract_features (callable, optional): The method to extract features from time-series.
            See defaults_extract_features docstring. Defaults to default_extract_features.
        remove_nan (bool, optional): Whether to remove NaNs from the dataframe. Defaults to True.

    Returns:
        pd.DataFrame: The dataframe with the features and columns 'aoi' and 'orbit'
    """

    # If list, gives arbitrary names to the periods (P1, P2, ...)
    time_periods = preprocess_time_periods(time_periods)

    dfs = []
    lengths = {}
    for window in extract_winds:
        for name_period, (start, end) in time_periods.items():

            if dense_inference:
                df = get_df_from_dense_ts(aoi, orbit, start, end, window)
            else:
                df = load_stacked_ts(aoi, orbit, window)

            # keep track of length once
            # if name_period not in lengths:
            #     lengths[name_period] = df.loc[:,start:end].shape[1]

            df_features = extract_features(df, start, end, prefix=f"{name_period}_{window}")

            df_features[f"{name_period}_start"] = start
            df_features[f"{name_period}_end"] = end

            dfs.append(df_features)
    df_final = pd.concat(dfs, axis=1)
    if remove_nan:
        df_final = df_final[~df_final.isna().any(axis=1)]

    for name_period, length in lengths.items():
        # For now we do not add the length of the period as features, as we must ensure that
        # the length is the same for positive and negative samples
        pass  # df_final[f'{name_period}_length'] = length

    df_final["aoi"] = aoi
    df_final["orbit"] = orbit

    return df_final


def preprocess_time_periods(time_periods):
    """Transform in a dict with name P1, P2, ... if not already a dict (or dict-equivalent)"""

    if isinstance(time_periods, Mapping):
        return time_periods
    else:
        return {f"P{i}": period for i, period in enumerate(time_periods, 1)}


def get_df_from_dense_ts(aoi: str, orbit: str, start: str, end: str, window: str) -> pd.DataFrame:
    """
    Flatten a dense time-series into a dataframe.

    Performs a convolution to mimic larger extraction windows.

    Args:
        aoi (str): The AOI
        orbit (str): The orbit
        start (str): The first date to consider
        end (str): The last date to consider
        window (str): The window (eg "1x1", "3x3")

    Returns:
        pd.DataFrame: The (very large) dataframe with the time-series
    """

    # Load time-series
    s1_ts = get_s1_ts(aoi, orbit, start_date=start, end_date=end, chunks=None)

    # Convolve if necessary to mimic larger extraction windows
    ws = int(window[0])
    if ws > 1:
        kernel = np.ones((ws, ws)) / ws**2
        method = "direct" if s1_ts.isnull().sum() else "auto"  # can't use fft if there are NaNs
        s1_ts.values = convolve(s1_ts, kernel[None, None, :, :], mode="same", method=method)

    # Flatten into a dataframe
    T, B, W, H = s1_ts.shape
    s1_ts_reshaped = s1_ts.values.transpose(1, 2, 3, 0).reshape(B * W * H, T)
    df = pd.DataFrame(s1_ts_reshaped, columns=s1_ts.date)
    band_pixel_index = pd.MultiIndex.from_product([["VV", "VH"], range(W * H)], names=["band", "pixel"])
    df.index = band_pixel_index
    df = df.swaplevel().sort_index()
    return df
