import h5py
import numpy as np
import time
from tqdm import tqdm
import xarray as xr

from src.constants import AOIS_TEST, READY_PATH
from src.data.utils import aoi_orbit_iterator, get_folder_ts
from src.utils.time import print_sec


def create_h5_dataset(extraction_strategy: str = "3x3", force_recreate: bool = False):
    start_time = time.time()

    # All time-series
    fps = []
    for aoi, orbit in aoi_orbit_iterator():
        folder = get_folder_ts(extraction_strategy) / aoi / f"orbit_{orbit}"
        fps += list(folder.glob("*.nc"))
    print(f"Found {len(fps)} time-series")

    # Create h5 file
    dataset_fp = READY_PATH / f"dataset_{extraction_strategy}.h5"
    if dataset_fp.exists() and not force_recreate:
        print(f"Dataset {dataset_fp} already exists")
        return

    with h5py.File(dataset_fp, "w") as f:
        groups = {f"fold{i}": f.create_group(f"fold{i}") for i in range(1, 6)}
        groups["test"] = f.create_group("test")

        for fp in tqdm(fps):
            ts = xr.open_dataarray(fp)
            if extraction_strategy == "pixel-wise-3x3":
                # need to stack all bands to get a 2D array
                ts = ts.stack(z=("x", "y", "band"))

            group_key = "test" if ts.aoi in AOIS_TEST else f"fold{ts.bin}"
            grp = groups[group_key]

            ds_name = f"{ts.aoi}_orbit_{ts.orbit}_{ts.unosat_id}"
            ds = grp.create_dataset(ds_name, data=ts.values)

            # closest date before the war and after the date of analysis
            # eg if id_left_bound = 10, and id_right_bound = 20, then the destuction happened
            # after T10 and before T20
            ds.attrs["id_left_bound"] = np.max(np.where(ts.date < np.datetime64("2022-02-24")))
            ds.attrs["id_right_bound"] = np.min(np.where(ts.date > np.datetime64(ts.date_of_analysis)))

            for attr in ts.attrs:
                ds.attrs[attr] = ts.attrs[attr]

        print(f"Created h5 file in {print_sec(time.time() - start_time)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction_strategy", type=str, default="3x3")
    parser.add_argument("--force_recreate", action="store_true")
    args = parser.parse_args()

    create_h5_dataset(args.extraction_strategy, force_recreate=args.force_recreate)
