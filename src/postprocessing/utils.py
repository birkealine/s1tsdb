import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
import rioxarray as rxr
import shapely
from shapely.geometry import Polygon
import warnings
import xarray as xr


def get_prediction_for_geo(fp_preds: Path, geo: Polygon, agg: str = "weighted_mean") -> float:

    # Read the predictions within the geometry and vectorize in pixels
    preds = read_fp_within_geo(fp_preds, geo)
    gdf_preds = vectorize_xarray(preds)

    if agg == "weighted_mean":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            gdf_preds["polygon_area"] = gdf_preds.intersection(geo).area
        return (gdf_preds["value"] * gdf_preds["polygon_area"]).sum() / gdf_preds["polygon_area"].sum()
    elif agg == "max":
        return gdf_preds["value"].max()


def vectorize_xarray(xa: xr.DataArray) -> pd.DataFrame:
    """Returns a dataframe with one geometry per pixel."""

    if len(xa.shape) != 2:
        xa = xa.squeeze(dim="band")
        assert len(xa.shape) == 2, "xarray should be 2D"

    # Construct dataframe with one geometry per pixel
    x, y, v = xa.x.values, xa.y.values, xa.values
    x, y = np.meshgrid(x, y)
    x, y, v = x.flatten(), y.flatten(), v.flatten()
    df = pd.DataFrame.from_dict({"x": x, "y": y, "v": v})
    gdf_pixels = gpd.GeoDataFrame(v, geometry=gpd.GeoSeries.from_xy(df.x, df.y), columns=["value"], crs=xa.rio.crs)
    gdf_pixels.index.name = "pixel_id"
    gdf_pixels.reset_index(inplace=True)

    # Buffer the pixels to get one polygon per pixel
    res = xa.rio.resolution()
    buffer = res[0] / 2  # half the pixel size, assuming square pixels
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        gdf_pixels["geometry"] = gdf_pixels.buffer(buffer, cap_style=3)
    return gdf_pixels


def vectorize_xarray_3d(xa: xr.DataArray, dates) -> pd.DataFrame:

    assert "date" in xa.dims, "xarray should have a 'date' dimension"
    if len(xa.shape) > 3:
        xa = xa.squeeze(dim="band")
        assert len(xa.shape) == 3, "xarray should be 3D"

    x, y = xa.x.values, xa.y.values
    x, y = np.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()

    vs = {d: xa.sel(date=d).values.flatten() for d in dates}
    gdf_pixels = gpd.GeoDataFrame(vs, geometry=gpd.GeoSeries.from_xy(x, y), columns=vs.keys(), crs=xa.rio.crs)
    gdf_pixels.index.name = "pixel_id"
    gdf_pixels.reset_index(inplace=True)

    # Buffer the pixels to get one polygon per pixel
    res = xa.rio.resolution()
    buffer = res[0] / 2  # half the pixel size, assuming square pixels
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        gdf_pixels["geometry"] = gdf_pixels.buffer(buffer, cap_style=3)
    return gdf_pixels


def read_fp_within_geo(fp: Path, geo: shapely.Geometry) -> xr.DataArray:
    """Read a raster file within a given geometry."""

    assert fp.exists(), f"File {fp} does not exist"

    with rasterio.open(fp) as src:

        wind = rasterio.windows.from_bounds(*geo.bounds, src.transform)
        # data = src.read(window=wind)
    xa = rxr.open_rasterio(fp).rio.isel_window(wind)
    return xa
