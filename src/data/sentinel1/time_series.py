""" Stack all tiles for same AOI (and orbit) and save into HDF5 file"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Union
import xarray as xr

from src.constants import S1_PATH, NO_DATA_VALUE
from src.data.sentinel1.utils import read_s1


def get_s1_ts(
    aoi: str,
    orbit: int = None,
    return_dates: bool = False,
    start_date: str = None,
    end_date: str = None,
    n_dates: int = None,
    base_folder: Union[str, Path] = S1_PATH,
    **kwargs,
) -> Union[xr.DataArray, Tuple[xr.DataArray, List[str]]]:
    """
    Read all Sentinel-1 tiles corresponding to AOI, and stack along date dim.

    Output is T x C x H x W (Time, Channels, Height, Weight). By default, read tiles in chunks.

    Args:
        aoi (str): The Area of Interest
        orbit (int, optional): The orbit. If None, read the one from the symlink. Defaults to None.
        return_dates (bool, optional): If true, return dates (str format "%Y-%m-%d")
            along with the DataDarray. Defaults to False.
        start_date (str, optional): If given, only read dates after this date. Defaults to None.
        end_date (str, optional): If given, only read dates before this date. Defaults to None.
        n_dates (int, optional): If given, only read the n_dates up to end_date or from start_date.
            Can't be used with both start_date and end_date. If Neither are given, read n_dates
            until most recent. Defaults to None.
        base_folder (Union[str, Path], optional): The folder where to read the tiles. Defaults to S1_PATH.

    Returns:
        Union[xr.DataArray, Tuple[xr.DataArray, List[str]]]: The time-series (and dates)
    """

    base_folder = Path(base_folder) if not isinstance(base_folder, Path) else base_folder

    if n_dates is not None:
        assert start_date is None or end_date is None, "Can't use n_dates with both start/end_date"

    orbit = orbit or "best"  # if not given, take the default one
    orbit = f"orbit_{orbit}" if isinstance(orbit, int) else orbit
    folder = base_folder / aoi / orbit or "best"
    assert folder.exists(), f"The folder {folder} does not exist..."

    # Get all tiles corresponding to AOI/orbit and fitler by dates
    start_date = start_date or "0000-00-00"
    end_date = end_date or "9999-99-99"

    fps = sorted(folder.glob("*.tif"))
    fps = [fp for fp in fps if start_date <= fp.stem <= end_date]
    dates_ = [fp.stem for fp in fps]

    if n_dates is not None:
        if start_date != "0000-00-00":
            dates_ = dates_[:n_dates]
            fps = fps[:n_dates]
        else:
            dates_ = dates_[-n_dates:]
            fps = fps[-n_dates:]

    # Concatenate along new dimension "date"
    dates = xr.Variable("date", pd.to_datetime(dates_))
    s1_ts = xr.concat([read_s1(fp, **kwargs) for fp in fps], dim=dates)

    # Interpolate NODATA values if any and no chunks
    if (s1_ts == NO_DATA_VALUE).any():
        if s1_ts.chunks is None:
            s1_ts = s1_ts.where(s1_ts != NO_DATA_VALUE, np.nan)
            s1_ts = s1_ts.interpolate_na(dim="date", method="nearest")
        else:
            print("WARNING: NODATA values but chunks -> no interpolation done !")

    # Add metadata
    s1_ts.attrs["aoi"] = aoi
    s1_ts.attrs["orbit"] = orbit

    if return_dates:
        return s1_ts, dates_
    else:
        return s1_ts
