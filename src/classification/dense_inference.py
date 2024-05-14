from collections.abc import Mapping
import numpy as np
import pandas as pd
from rasterio.enums import Resampling
from sklearn.base import ClassifierMixin
import time
from typing import List, Union

from src.classification.features import get_df_features, default_extract_features
from src.data.sentinel1.orbits import get_valid_orbits
from src.data.sentinel1.utils import get_target_s1
from src.utils.time import print_sec


def perform_dense_inference(
    aoi: str,
    model: ClassifierMixin,
    time_periods: Union[Mapping, List],
    extract_winds: List,
    extract_features: callable = default_extract_features,
    verbose: int = 0,
) -> pd.DataFrame:
    """
    Perform dense inference for a given AOI and model.

    Args:
        aoi (str): The AOI
        model (ClassifierMixin): The model
        time_periods (Union[Mapping, List]): The time periods
        extract_winds (List): The extract windows
        extract_features (callable, optional): The feature extractor. Defaults to default_extract_features.
        verbose (int, optional): The verbosity. Defaults to 0.

    Returns:
        pd.DataFrame: The predictions
    """

    if verbose:
        start_time = time.time()
        print(f"Dense inference for {aoi} with time_periods={time_periods} and extract_winds={extract_winds}...")

    # Get all valid orbits
    orbits = get_valid_orbits(aoi)

    # Get target georeferenced tile
    global_target_img = get_target_s1(aoi, chunks=None).sel(band="VV")
    global_target_img["band"] = "preds_proba"

    # Predict for each orbit and store in a 3D array (then aggregating is easy)
    preds = np.zeros([len(orbits), *global_target_img.shape[-2:]])

    for i, orbit in enumerate(orbits):

        # EXtract features for each pixel
        df_features = get_df_features(
            aoi=aoi,
            orbit=orbit,
            time_periods=time_periods,
            extract_winds=extract_winds,
            dense_inference=True,
            extract_features=extract_features,
            remove_nan=False,
        )
        df_features["has_nan"] = df_features.isnull().any(axis=1)  # keep track of NaNs
        df_features.fillna(0, inplace=True)

        # Inference
        X = df_features[[c for c in df_features.columns if c.startswith(("VV", "VH"))]].values
        preds_proba = model.predict_proba(X)[:, 1]

        # Fill with 0 where there were NaNs
        preds_proba[df_features["has_nan"]] = 0

        # Reshape to original 2D shape
        orbit_target_img = get_target_s1(aoi, orbit, chunks=None).sel(band="VV")
        preds_proba = preds_proba.reshape(orbit_target_img.shape)

        # Reproject to match general target if needed
        if global_target_img.shape != preds_proba.shape:
            orbit_target_img.values = preds_proba
            preds_proba = orbit_target_img.rio.reproject_match(global_target_img, resampling=Resampling.bilinear).values

        preds[i] = preds_proba

    # Aggregate and store in georeferenced array
    preds_agg = global_target_img.copy()
    preds_agg.values = preds.mean(axis=0)

    if verbose:
        print(f"...done in {print_sec(time.time()-start_time)}")
    return preds_agg, preds
