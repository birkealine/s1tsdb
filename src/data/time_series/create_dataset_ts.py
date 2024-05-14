import datetime as dt
import geopandas as gpd
import multiprocessing as mp
import pandas as pd
from shapely import wkt
from typing import List, Union
import xarray as xr
from src.constants import CRS_GLOBAL, UKRAINE_WAR_START

from src.data.utils import get_folder_ts


def get_df_ts(full_ts, start_date, n_dates):
    """Store time-series in a dataframe with one column per date (+ other attributes)"""

    # Select dates
    ts = full_ts.sel(date=slice(start_date, None)).isel(date=slice(0, n_dates))

    # Get label
    first_date = ts.date.dt.strftime("%Y-%m-%d")[0].item()
    last_date = ts.date[-1].dt.strftime("%Y-%m-%d").item()
    if full_ts.attrs["damage"] == -1:
        # flag for random locations, label is always 0
        label = 0
    else:
        # find label according to dates spanned:
        # 0 (not destroyed) if before war
        # 1 (destroyed) if spans start of war and date of analysis
        # -1 (unknown) if ends between start of war and date of analysis
        label = (
            0
            if last_date < UKRAINE_WAR_START
            else 1 if last_date > ts.date_of_analysis and first_date < UKRAINE_WAR_START else -1
        )

    # Update metadata
    ts.attrs["label"] = label
    ts.attrs["first_date"] = first_date
    ts.attrs["last_date"] = last_date
    ts.attrs["start_month"] = dt.date.fromisoformat(start_date).strftime("%Y-%m")

    # Transform in dataframe with columns T0 to TN
    index = [f"T{i}" for i in range(ts.sizes["date"])]  # T0, T1, ... TN
    df = pd.DataFrame(ts, columns=["VV", "VH"], index=index).transpose()
    df.index.name = "band"
    df.reset_index(inplace=True)

    # Store all attrs as new column
    for k, v in ts.attrs.items():
        df[k] = v

    # put all columns T0, T1, ... at the end
    df = df[[c for c in df.columns if c not in index] + index]

    return df


def get_df_ts_from_fp(fp, start_date, n_dates):
    """Store time-series in a dataframe with one column per date (+ other attributes)"""

    # Read xarray
    ts = xr.open_dataarray(fp)
    return get_df_ts(ts, start_date, n_dates)


def get_df_ts_aoi_orbit(
    aoi,
    orbit,
    extraction_strategy="pixel-wise",
    start_date: Union[str, List[str]] = "2021-10-01",
    n_tiles=32,
):
    # Get all files
    folder = get_folder_ts(extraction_strategy)
    fps = sorted((folder / aoi / f"orbit_{orbit}").glob("*.nc"))

    # Create list of args for multiprocessing
    if isinstance(start_date, str):
        args_list = [(fp, start_date, n_tiles) for fp in fps]
    else:
        args_list = [(fp, date, n_tiles) for date in start_date for fp in fps]

    print(f"Reading all time-series for {aoi} orbit {orbit}... (N={len(args_list)}))")
    # Transform each file into a dataframe
    with mp.Pool(processes=mp.cpu_count()) as pool:
        dfs = pool.starmap(get_df_ts_from_fp, args_list)
    df = pd.concat(dfs)
    # return df_to_gdf(df)
    return df


def df_to_gdf(df, geo_col="geometry"):
    df[geo_col] = df[geo_col].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry=geo_col, crs=CRS_GLOBAL)


if __name__ == "__main__":
    import argparse
    import time

    from src.data.utils import get_all_aois
    from src.data.sentinel1.orbits import get_valid_orbits
    from src.utils.time import print_sec
    from src.data.old_datasets import get_folder_ready_name, get_dataset_ready_name

    parser = argparse.ArgumentParser()
    parser.add_argument("-ex", "--extraction_strategy", type=str, default="pixel-wise")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n_tiles", type=int, default=32)
    parser.add_argument("--first_start_date", type=str, default="2020-06-01")
    parser.add_argument("--last_start_date", type=str, default="2021-12-01")
    parser.add_argument("--random_locations", action="store_true")
    args = parser.parse_args()

    # Get folder where to store all ready files
    folder = get_folder_ready_name(args)
    if args.random_locations:
        folder = folder.parent / f"{folder.stem}_random_locations"

    # Get list of start dates, one per month
    start_dates = pd.date_range(args.first_start_date, args.last_start_date, freq="MS").strftime("%Y-%m-%d").tolist()
    print(f"Extracting time series with extraction strategy {args.extraction_strategy}...")
    print(f"With a monthly sliding window from {start_dates[0]} to {start_dates[-1]}")

    # Save one file per AOI first
    start_time = time.time()
    for aoi in get_all_aois():
        filepath = folder / f"{aoi}.csv"
        if filepath.exists() and not args.overwrite:
            print(f"File {filepath} already exists, skipping")
            continue

        list_dfs = []
        for orbit in get_valid_orbits(aoi):
            if args.random_locations:
                aoi = f"random_{aoi}"
            _df_ts = get_df_ts_aoi_orbit(
                aoi, orbit, extraction_strategy=args.extraction_strategy, start_date=start_dates, n_tiles=args.n_tiles
            )
            list_dfs.append(_df_ts)
        df_ts = pd.concat(list_dfs)
        df_ts.to_csv(filepath, index=False)

    # Merge all AOIs in one final file and save also each AOI in a separate file
    # (not very efficient but ok for now)
    print(f"Merging all AOIs in one file...")
    fp = get_dataset_ready_name(args)
    if fp.exists() and not args.overwrite:
        print(f"File {fp} already exists, skipping")
    else:
        list_df = []
        for aoi in get_all_aois():
            list_df.append(pd.read_csv(folder / f"{aoi}.csv"))
        df = pd.concat(list_df)
        df.to_csv(fp, index=False)

    print(f"Time-series stacked in {print_sec(time.time() - start_time)}")
