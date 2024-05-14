import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr

from src.gee.constants import RAW_TS_GEE_PATH, TS_GEE_PATH
from src.data import load_unosat_labels, load_unosat_aois


def preprocess_all_ts(aoi, orbit, overwrite=False):
    geo = load_unosat_aois().set_index("aoi").loc[aoi].geometry
    labels = load_unosat_labels(aoi)
    labels_with_bins = assign_bins(labels, geo, n_bins=5)
    folder = TS_GEE_PATH / aoi / f"orbit_{orbit}"
    folder.mkdir(parents=True, exist_ok=True)

    def _save_csv_as_xarray(raw_fp):
        """Read CSV file, transforms it into an xarray and saves it as a netCDF file"""

        # Read CSV
        df = pd.read_csv(raw_fp)
        vv = df.VV.values
        vh = df.VH.values
        dates = df.date.values
        dates_dt = pd.to_datetime(dates)

        # Store useful metadata as attributes(retrieve from original labels DataFrame)
        id_ = int(raw_fp.stem)
        row = labels_with_bins.loc[id_]
        attrs = {
            "aoi": aoi,
            "orbit": orbit,
            "geometry": row.geometry.wkt,
            "damage": row.damage,
            "bin": row.bin,
            "unosat_id": id_,
            "date_of_analysis": row.date.strftime("%Y-%m-%d"),
            "had_nans": 0,
        }

        # Create DataArray
        ts = xr.DataArray(
            np.stack([vv, vh], axis=1),
            coords=dict(date=dates_dt, band=["VV", "VH"]),
            dims=["date", "band"],
            attrs=attrs,
        )

        # Fill NaNs if any
        if ts.isnull().any():
            ts = ts.ffill(dim="date").bfill(dim="date")
            ts.attrs["had_nans"] = 1

        # Save as netCDF
        ts.to_netcdf(folder / f"{id_}.nc")

    # Preprocess all time series
    raw_fps = sorted((RAW_TS_GEE_PATH / aoi / f"orbit_{orbit}").glob("*.csv"))

    # Only keep files that do not exist
    if not overwrite:
        folder
        raw_fps = [fp for fp in raw_fps if not (folder / f"{fp.stem}.nc").exists()]
        if not raw_fps:
            print(f"All time series already preprocessed for {aoi} orbit {orbit}")
            return

    for raw_fp in tqdm(raw_fps):
        _save_csv_as_xarray(raw_fp)
    print(f"All time series preprocessed for {aoi} orbit {orbit}")


def assign_bins(points_gdf, polygon, n_bins=5):
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

    start_time = time.time()
    for aoi, orbit in aoi_orbit_iterator():
        if not aoi.startswith("UKR"):
            continue

        preprocess_all_ts(aoi, orbit, overwrite=True)

    print(f"Preprocessing all time series took {print_sec(time.time() - start_time)}")
