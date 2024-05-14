"""Stack time-series for a given aoi/orbit/extract"""

import multiprocessing as mp
import pandas as pd
import xarray as xr

from src.constants import TS_PATH, PROCESSED_PATH

TS_STACKED_PATH = PROCESSED_PATH / "stacked_ts"


def load_stacked_ts(aoi, orbit, extract_strategy):
    """Load stacked time series for a given aoi/orbit/extract"""

    fp = TS_STACKED_PATH / extract_strategy / f"{aoi}_orbit_{orbit}.csv"

    if not fp.exists():
        create_stacked_ts(aoi, orbit, extract_strategy)

    df = pd.read_csv(fp)
    df.set_index(["unosat_id", "band"], inplace=True)
    df.columns = pd.to_datetime(df.columns)
    return df


def create_stacked_ts(aoi, orbit, extract_strategy):
    """Create stacked time series for a given aoi/orbit/extract"""

    _extract_strategy = "pixel_wise" if extract_strategy == "1x1" else extract_strategy
    folder = TS_PATH / _extract_strategy / aoi / f"orbit_{orbit}"
    fps = sorted(folder.glob("*.nc"), key=lambda x: int(x.stem))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        dfs = pool.map(get_df_ts_from_fp, fps)
    df = pd.concat(dfs)

    folder = TS_STACKED_PATH / extract_strategy
    filename = f"{aoi}_orbit_{orbit}.csv"
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / filename)
    print(f"Stacked time series saved to {folder / filename}")
    return df


def get_df_ts_from_fp(fp):
    ts = xr.open_dataarray(fp)
    return get_df_from_single_ts(ts)


def get_df_from_single_ts(ts: xr.DataArray):
    id_ = ts.unosat_id
    bands = ts.band.values
    index = pd.MultiIndex.from_tuples([(id_, b) for b in bands], name=["unosat_id", "band"])
    return pd.DataFrame(ts.values.T, index=index, columns=pd.to_datetime(ts.date.values))


if __name__ == "__main__":

    import time
    from src.data.utils import aoi_orbit_iterator
    from src.utils.time import print_sec

    start_time = time.time()
    for extract_strategy in ["3x3"]:
        for aoi, orbit in aoi_orbit_iterator():
            create_stacked_ts(aoi, orbit, extract_strategy)
            print(f"Stacked time series for {aoi} orbit {orbit} created")
    print_sec(f"Stacked time series created in {print_sec(time.time() - start_time)}")
