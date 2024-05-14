"""Intersect predictions with Microsoft buildings footprints"""

import geopandas as gpd
import pandas as pd
import warnings
import xarray as xr

from src.data.buildings.microsoft_unosat import load_buildings_aoi
from src.data import get_unosat_geometry
from src.postprocessing.utils import vectorize_xarray, read_fp_within_geo
from src.constants import PREDS_PATH


def load_buildings_with_preds(aoi: str, run_name: str) -> gpd.GeoDataFrame:

    folder_preds = PREDS_PATH / run_name
    fp_global_preds = folder_preds / f"{run_name}_global_ukraine_preds.tif"

    fp_aoi_preds = folder_preds / "buildings_with_preds" / f"{aoi}_buildings_with_preds_{run_name}.geojson"
    fp_aoi_preds.parents[0].mkdir(exist_ok=True, parents=True)

    if fp_aoi_preds.exists():
        buildings_with_preds = gpd.read_file(fp_aoi_preds)
    else:
        # Vectorize the predictions with buildings footprints
        print(f"Computing predictions for {aoi}...")
        buildings_with_preds = compute_preds_buildings_aoi(fp_global_preds, aoi)
        buildings_with_preds.to_file(fp_aoi_preds, driver="GeoJSON")
        print(f"Buildings with predictions from {run_name} saved for {aoi}")
    return buildings_with_preds


def compute_preds_buildings_aoi(fp_preds, aoi):
    """Intersect predictions with Microsoft buildings footprints"""

    geo = get_unosat_geometry(aoi)

    preds = read_fp_within_geo(fp_preds, geo)

    buildings = load_buildings_aoi(aoi)
    preds_vectorized = vectorize_xarray_with_gdf(preds, buildings, name_id="building_id")
    buildings_with_preds = buildings.merge(preds_vectorized, on="building_id")

    return buildings_with_preds


def vectorize_xarray_with_gdf(xa: xr.DataArray, gdf: gpd.GeoDataFrame, name_id: str, verbose: int = 1) -> pd.DataFrame:

    # get the pixels geometry
    gdf_pixels = vectorize_xarray(xa)
    if verbose:
        print("Pixels vectorized.")

    # Intersect the pixels with the geodataframe
    overlap = gpd.overlay(gdf, gdf_pixels, how="intersection")
    if verbose:
        print("Pixels intersected with polygons.")

    # Compute the area for each intersection (i.e. the weights)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        overlap["polygon_area"] = overlap.area

    # Aggregate (weighted mean, max, etc.) the predictions for each polygon
    preds_agg = (
        overlap.assign(weighted_value=lambda df: df["value"] * df["polygon_area"])
        .groupby(name_id)
        .pipe(lambda grp: grp["weighted_value"].sum() / grp["polygon_area"].sum())
        .reset_index(name="weighted_mean")
        .set_index(name_id)
    )
    preds_agg["max"] = overlap.groupby(name_id)["value"].max()  # max
    if verbose:
        print("Prediction aggregated.")

    return preds_agg.reset_index()


if __name__ == "__main__":

    from src.utils.time import timeit

    @timeit
    def main():
        run_name = "240212"
        aoi = "UKR3"
        load_buildings_with_preds(aoi, run_name)

    main()
