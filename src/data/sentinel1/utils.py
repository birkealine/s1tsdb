import datetime as dt
import math
import pandas as pd
from pathlib import Path
import re
import rioxarray as rxr
from typing import Dict, List
import xarray as xr

from src.constants import S1_PATH, NO_DATA_VALUE


def read_s1(
    fp: Path, bands: List[str] = ["VV", "VH"], chunks: Dict[str, int] = {"x": 64, "y": 64}, **kwargs
) -> xr.DataArray:
    """
    Read a Sentinel-1 tif file using rioxarray.

    If desired, read it by chunks and only returns some bands. Also make sure the
    nodata is correctly set.

    Args:
        fp (Path): Path to the tif file
        bands (List[str], optional): The bands to keep. Defaults to ["VV", "VH"].
        chunks (_type_, optional): Size of chunks to read.
            Defaults to {"x": 64, "y": 64}.

    Returns:
        xr.DataArray: The DataArray
    """

    # Add chunks parameter to the kwargs if given
    if chunks is not None:
        kwargs["chunks"] = chunks

    # Read file and write nodata value
    xar = rxr.open_rasterio(fp, **kwargs)
    xar.rio.write_nodata(NO_DATA_VALUE, inplace=True)
    xar = xar.fillna(xar.rio.nodata)
    # dataarray -> dataset -> dataarray (only way found to have the name of each band
    # correctly written)
    long_names = [xar.attrs["long_name"]] if isinstance(xar.attrs["long_name"], str) else xar.attrs["long_name"]
    xar = xar.to_dataset("band").rename({k: v for k, v in enumerate(long_names, start=1)})
    del xar.attrs["long_name"]
    if bands is not None:
        try:
            xar = xar[bands]
        except KeyError:
            # not all bands are present
            xar = xar[[b for b in bands if b in xar.variables.keys() if b in bands]]

    xar = xar.to_array("band")

    # modify chunk for band
    if chunks is not None:
        xar = xar.chunk({"band": 2})
    return xar


def save_s1(xar: xr.DataArray, fp: Path) -> None:
    """
    Save the Sentinel-1 DataArray

    Args:
        xar (xr.DataArray): The DataArray to save
        fp (Path): The path.
    """
    xar_ = xar.to_dataset("band")
    for band in xar_:
        xar_[band].rio.write_nodata(xar.rio.nodata, inplace=True)
    xar_.rio.to_raster(fp)


def get_target_s1(aoi: str, orbit=None, **kwargs) -> xr.DataArray:
    """Simply read the first S1 tile for the given AOI"""
    from src.data.sentinel1.orbits import get_best_orbit

    orbit = orbit or get_best_orbit(aoi)
    fps = sorted((S1_PATH / aoi / f"orbit_{orbit}").glob("*.tif"))
    return read_s1(fps[0], **kwargs)


def pad_multiple(xar: xr.DataArray, multiple: int = 64) -> xr.DataArray:
    """
    Pad the DataArray to become an exact multiple of the value given.

    Padded pixels will have NO_DATA_VALUE value

    Args:
        xar (xr.DataArray): The DataArray of arbitrary shape
        multiple (int, optional): The value. Defaults to 64.

    Returns:
        xr.DataArray: The DataArray whose shape is an exact multiple of the value given.
    """

    # Get resolution and size of each dimension
    x_res, y_res = xar.rio.resolution()
    x_size, y_size = xar["x"].size, xar["y"].size

    # What to add in each dimension
    to_add_x = multiple - x_size % multiple if x_size % multiple else 0
    to_add_y = multiple - y_size % multiple if y_size % multiple else 0

    # New min and max after adding values (sorted if resolution is negative)
    xmin, xmax = sorted([xar.x[0] - math.ceil(to_add_x / 2) * x_res, xar.x[-1] + math.floor(to_add_x / 2) * x_res])
    ymin, ymax = sorted([xar.y[0] - math.ceil(to_add_y / 2) * y_res, xar.y[-1] + math.floor(to_add_y / 2) * y_res])

    # Pad
    return xar.rio.pad_box(xmin, ymin, xmax, ymax)


def crop_multiple(xar: xr.DataArray, multiple: int = 64):
    """
    Crop the DataArray to become an exact multiple of the value given.

    Tiles should have been downloaded with buffer, otherwise loss of information.

    Args:
        xar (xr.DataArray): The DataArray of arbitrary shape
        multiple (int, optional): The value. Defaults to 64.
    """
    _, H, W = xar.shape
    if H % multiple:
        h_offset = (H - H // multiple * multiple) // 2
        hmin, hmax = h_offset, -h_offset
    else:
        hmin, hmax = None, None

    if W % multiple:
        w_offset = (W - W // multiple * multiple) // 2
        wmin, wmax = w_offset, -w_offset
    else:
        wmin, wmax = None, None
    return xar[:, hmin:hmax, wmin:wmax]
