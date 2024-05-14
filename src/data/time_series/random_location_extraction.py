"""Create a .nc file for each time-series from the random points"""

import geopandas as gpd
from tqdm import tqdm

from src.data.utils import get_folder_ts, load_ps_mask
from src.data import load_unosat_aois
from src.data.sentinel1.time_series import get_s1_ts
from src.data.time_series.unosat_extraction import extract_ts, extract_ps, assign_bins
from src.constants import PROCESSED_PATH


def preprocess_all_ts(
    aoi: str,
    orbit: int,
    overwrite: bool = False,
    extraction_strategy: str = "pixel-wise",
    min_distance: float = 100,
):
    """
    Extract a time series for all random points in a given AOI and orbit.

    For different extraction strategy, please see extract_ts. Each time series is saved with its metadata as a netCDF
    file. The filename is its ID in the corresponding geojson file.

    Args:
        aoi (str): The AOI
        orbit (int): The orbit
        overwrite (bool, optional): Whether to overwrite if already exists. Defaults to False.
        extraction_strategy (str, optional): The extraction strategy. Defaults to "pixel-wise".
        min_distance (float, optional): The minimum distance between the random location and the UNOSAT labels.
            Default to 100m.
    """
    # Load random points
    random_points_path = PROCESSED_PATH / "random_points" / f"{aoi}_{min_distance}m_far.geojson"
    assert random_points_path.exists(), f"File {random_points_path} does not exist."
    labels = gpd.read_file(random_points_path, driver="GeoJSON")

    # Assign bins
    geo = load_unosat_aois().set_index("aoi").loc[aoi].geometry
    labels_with_bins = assign_bins(labels, geo, n_bins=5)

    # Define folder where to save the time series
    _folder = get_folder_ts(extraction_strategy)
    folder = _folder / f"random_{aoi}" / f"orbit_{orbit}"
    folder.mkdir(parents=True, exist_ok=True)

    # Only keep files that do not exist
    ids = labels_with_bins.index.to_list()
    if not overwrite:
        ids = [id_ for id_ in ids if not (folder / f"{id_}.nc").exists()]
        if not ids:
            print(f"All time series already preprocessed for {aoi} orbit {orbit}")
            return

    # Load stack of Sentinel-1 and reproject labels
    s1_ts = get_s1_ts(aoi, orbit, chunks=None)
    labels_with_bins.to_crs(s1_ts.rio.crs, inplace=True)

    # Load PS Mask
    ps_mask = load_ps_mask(aoi, orbit)

    for id_, row in tqdm(labels_with_bins.iterrows()):
        # Extract time-series
        ts = extract_ts(row.geometry, s1_ts, extraction_strategy)

        # Extract Persistent Scatterers value (for filtering later)
        ps = extract_ps(row.geometry, ps_mask, extraction_strategy)

        # Store useful metadata as attributes(retrieve from original labels DataFrame)
        ts.attrs = {
            "aoi": aoi,
            "orbit": orbit,
            "geometry": row.geometry.wkt,
            "crs": s1_ts.rio.crs.to_string(),
            "damage": -1,  # Flag for random points
            "bin": row.bin,
            "unosat_id": 1e6 + id_,  # make sure it is different from UNOSAT IDs
            "ps": ps,
            "had_nans": 0,
            "extraction_strategy": extraction_strategy,
            "random_location": 1,
        }

        # Fill NaNs if any
        if ts.isnull().any():
            ts = ts.ffill(dim="date").bfill(dim="date")
            ts.attrs["had_nans"] = 1

        # Save as netCDF
        ts.to_netcdf(folder / f"{id_}.nc")

    print(f"All time series preprocessed for {aoi} orbit {orbit}")


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

        preprocess_all_ts(aoi, orbit, overwrite=args.overwrite, extraction_strategy=args.extraction_strategy)

    print(f"Preprocessing all time series took {print_sec(time.time() - start_time)}")
