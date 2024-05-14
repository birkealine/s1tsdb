"""Create a .nc file for each time-series from the local data"""

import geopandas as gpd
from shapely.geometry import Point, Polygon
from tqdm import tqdm
from typing import List
import xarray as xr

from src.data.utils import get_folder_ts, load_ps_mask
from src.data import load_unosat_labels, load_unosat_aois
from src.data.sentinel1.time_series import get_s1_ts


def preprocess_all_ts(
    aoi: str,
    orbit: int,
    overwrite: bool = False,
    extraction_strategy: str = "pixel-wise",
    labels_to_keep: List[int] = [1, 2],
):
    """
    Extract a time series for all UNOSAT labels in a given AOI and orbit.

    For different extraction strategy, please see extract_ts. Each time series is saved with its metadata as a netCDF
    file. The filename is its UNOSAT ID.

    Args:
        aoi (str): The AOI
        orbit (int): The orbit
        overwrite (bool, optional): Whether to overwrite if already exists. Defaults to False.
        extraction_strategy (str, optional): The extraction strategy. Defaults to "pixel-wise".
    """
    # Load labels and assign bins
    geo = load_unosat_aois().set_index("aoi").loc[aoi].geometry
    labels = load_unosat_labels(aoi, labels_to_keep=labels_to_keep)

    labels_with_bins = assign_bins(labels, geo, n_bins=5)

    # Define folder where to save the time series
    _folder = get_folder_ts(extraction_strategy)
    folder = _folder / aoi / f"orbit_{orbit}"
    folder.mkdir(parents=True, exist_ok=True)

    # Only keep files that do not exist
    ids = labels_with_bins.index.to_list()
    if not overwrite:
        ids = [id_ for id_ in ids if not (folder / f"{id_}.nc").exists()]
        if not ids:
            print(f"All time series already preprocessed for {aoi} orbit {orbit}")
            return
        if len(ids) < len(labels_with_bins):
            print(f"Only {len(ids)}/{len(labels_with_bins)} time series to preprocess for {aoi} orbit {orbit}")
        labels_with_bins = labels_with_bins.loc[ids]

    # Load stack of Sentinel-1 and reproject labels
    s1_ts = get_s1_ts(aoi, orbit, chunks=None)
    labels_with_bins.to_crs(s1_ts.rio.crs, inplace=True)

    # Load PS Mask
    # ps_mask = load_ps_mask(aoi, orbit)

    for id_, row in tqdm(labels_with_bins.iterrows()):
        try:
            # Extract time-series
            ts = extract_ts(row.geometry, s1_ts, extraction_strategy)

            # # Extract Persistent Scatterers value (for filtering later)
            # ps = extract_ps(row.geometry, ps_mask, extraction_strategy)
        except Exception as e:
            print(f"Error for {id_}: {e}")
            print("Ignoring it...")
            continue

        # Store useful metadata as attributes(retrieve from original labels DataFrame)
        ts.attrs = {
            "aoi": aoi,
            "orbit": orbit,
            "geometry": row.geometry.wkt,
            "crs": s1_ts.rio.crs.to_string(),
            "damage": row.damage,
            "bin": row.bin,
            "unosat_id": id_,
            # "ps": ps,
            "date_of_analysis": row.date.strftime("%Y-%m-%d"),
            "had_nans": 0,
            "extraction_strategy": extraction_strategy,
        }

        # Fill NaNs if any
        if ts.isnull().any():
            ts = ts.ffill(dim="date").bfill(dim="date")
            ts.attrs["had_nans"] = 1

        # Save as netCDF
        ts.to_netcdf(folder / f"{id_}.nc")

    print(f"All time series preprocessed for {aoi} orbit {orbit}")


def extract_ts(geometry: Point, s1_ts: xr.DataArray, extraction_strategy: str = "pixel-wise") -> xr.DataArray:
    """
    Extracts pixel-wise time series from s1_ts and saves it as a netCDF file.

    Extraction strategy can be:
        - "pixel-wise": the time series is extracted at the pixel corresponding to the geometry
        - "3x3": the time series is extracted as the mean of a 3x3 window centered on the geometry
        TODO:
        - "buildings_fp": the time series is extracted as the mean of the pixels corresponding to the buildings

    Args:
        geometry (Point): The geometry
        s1_ts (xr.DataArray): The stack of Sentinel-1 time-series
        extraction_strategy (str, optional): The extraction strategy. Defaults to "pixel-wise".

    Returns:
        xr.DataArray: The extracted time-series
    """

    if extraction_strategy == "pixel-wise":
        ts = s1_ts.sel(x=geometry.x, y=geometry.y, method="nearest")
    elif extraction_strategy in ["3x3", "5x5"]:
        size = int(extraction_strategy.split("x")[0])
        window_size = [size * r for r in s1_ts.rio.resolution()]  # 3x3 pixels -> 30m, 5x5 = 50m, ...
        half_ws = [r // 2 for r in window_size]
        ts = s1_ts.sel(
            x=slice(geometry.x - half_ws[0], geometry.x + half_ws[0]),
            y=slice(geometry.y - half_ws[1], geometry.y + half_ws[1]),
        ).mean(dim=["x", "y"])
    elif extraction_strategy == "pixel-wise-3x3":
        # Same as 3x3 but we do not aggregate (we keep the 3x3 pixels)
        window_size = [3 * r for r in s1_ts.rio.resolution()]  # 3x3 pixels -> 30m
        half_ws = [r // 2 for r in window_size]
        ts = s1_ts.sel(
            x=slice(geometry.x - half_ws[0], geometry.x + half_ws[0]),
            y=slice(geometry.y - half_ws[1], geometry.y + half_ws[1]),
        )
    elif extraction_strategy == "buildings_fp":
        raise NotImplementedError('extraction_strategy="buildings_fp" not implemented yet')
    else:
        raise ValueError(f"Unknown extraction strategy: {extraction_strategy}")

    return ts


def extract_ps(geometry: Point, ps_mask: xr.DataArray, extraction_strategy: str = "pixel-wise") -> xr.DataArray:
    """Same as extract_ts, but max aggregation instead of mean"""

    if extraction_strategy == "pixel-wise":
        ps = ps_mask.sel(x=geometry.x, y=geometry.y, method="nearest").item()
    elif "3x3" in extraction_strategy:
        # Keep only scalar value, even if pixel-wise-3x3
        window_size = [3 * r for r in ps_mask.rio.resolution()]  # 3x3 pixels -> 30m
        half_ws = [r // 2 for r in window_size]
        _ps = ps_mask.sel(
            x=slice(geometry.x - half_ws[0], geometry.x + half_ws[0]),
            y=slice(geometry.y - half_ws[1], geometry.y + half_ws[1]),
        ).max()  # max value -> PS if any pixel within the window is PS
        ps = _ps.item()
    return ps


def assign_bins(points_gdf: gpd.GeoDataFrame, polygon: Polygon, n_bins: int = 5) -> gpd.GeoDataFrame:
    """
    Assigns a bin to each point based on its longitude.

    The AOI is divided vertically into 5 bins of equal width. The bin number is assigned based on
    the bin in which the point is located.

    Args:
        points_gdf (gpd.GeoDataFrame): The GeoDataFrame containing the points
        polygon (Polygon): The polygon defining the AOI
        n_bins (int, optional): The number of bins. Defaults to 5.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with the bin column
    """
    min_lon, _, max_lon, _ = polygon.bounds

    delta = max_lon - min_lon
    bin_size = delta / n_bins

    # Function to assign a bin based on longitude
    def assign_bin(point):
        longitude = point.x
        bin_index = int((longitude - min_lon) // bin_size) + 1
        # Ensure the maximum value is n_bins
        bin_index = min(bin_index, n_bins)
        return bin_index

    points_gdf["bin"] = points_gdf.geometry.apply(assign_bin)
    return points_gdf


if __name__ == "__main__":
    import time
    from src.data.utils import aoi_orbit_iterator
    from src.utils.time import print_sec
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction_strategy", type=str, default="pixel-wise")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print(f'Extracting time-series with extraction strategy "{args.extraction_strategy}"')

    start_time = time.time()
    for aoi, orbit in aoi_orbit_iterator():
        if not aoi.startswith("UKR"):
            continue

        preprocess_all_ts(
            aoi, orbit, overwrite=args.overwrite, extraction_strategy=args.extraction_strategy, labels_to_keep=None
        )

    print(f"Preprocessing all time series took {print_sec(time.time() - start_time)}")
