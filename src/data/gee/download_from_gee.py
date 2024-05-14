"""Download pixel-wise time series from GEE for all AOIs and orbits."""

import ee
import geemap
import multiprocessing as mp
from pathlib import Path
from src.utils.gee import init_gee, custom_mosaic_same_day
from src.gee.data.unosat import get_unosat_geo
from src.data import load_unosat_labels
from src.gee.data.collections import get_s1_collection
from src.constants import S1_PATH

init_gee()


def extract_ts(
    s1: ee.ImageCollection,
    geometry: ee.Geometry,
    filepath: Path,
    verbose: bool = True,
    download_local: bool = True,
):
    """Extract a time series pixel-wise given a Sentinel-1 collection and a point."""

    def extract_point(img):
        date = img.date().format("YYYY-MM-dd")

        # sample
        value = img.sample(geometry, scale=10).first()

        # Test if null (can happen fo some AOI at the limit of the tile)
        value = ee.Feature(
            ee.Algorithms.If(
                ee.Algorithms.IsEqual(value, None),
                ee.Feature(None, {"VV": None, "VH": None}),
                value,
            )
        )
        return value.set("date", date)  # add date

    fc = ee.FeatureCollection(s1.map(extract_point))
    if download_local:
        geemap.ee_export_vector(fc, filename=filepath, verbose=verbose, selectors=["date", "VV", "VH"])
    return fc


def download_all_ts_aoi_orbit(aoi, orbit, base_folder, overwrite: bool = False, use_multiprocessing: bool = True):
    # Get Sentinel-1 collection
    geometry_ee = get_unosat_geo(aoi)
    start, end = get_date_range(aoi, orbit)
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD")
    s1 = get_s1_collection(geometry_ee, start=start, end=ee.Date(end).advance(1, "day"))
    s1 = s1.filterMetadata("relativeOrbitNumber_start", "equals", orbit)
    s1 = s1.map(lambda img: img.set("date", ee.Date(img.date()).format("YYYY-MM-dd")))

    # If number of dates != number of images -> need mosaicking
    s1 = ee.ImageCollection(
        ee.Algorithms.If(
            s1.aggregate_array("date").distinct().size().eq(s1.size()),
            s1,
            custom_mosaic_same_day(s1),
        )
    )

    # Get UNOSAT labels
    labels = load_unosat_labels(aoi)

    # Download time series
    folder = base_folder / aoi / f"orbit_{orbit}"
    folder.mkdir(parents=True, exist_ok=True)

    if use_multiprocessing:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            args_list = [
                (
                    s1,
                    ee.Geometry.Point(row.geometry.x, row.geometry.y),
                    folder / f"{id_}.csv",
                    False,
                )
                for id_, row in labels.iterrows()
            ]

            if not overwrite:
                # Only keep files that do not exist
                args_list = [args for args in args_list if not args[2].exists()]

            if args_list:
                print(f"Downloading {len(args_list)} time series for {aoi} orbit {orbit}...")
                pool.starmap(extract_ts, args_list)
                print(f"All time series downloaded for {aoi} orbit {orbit}")
            else:
                print(f"All time series already downloaded for {aoi} orbit {orbit}")
    else:
        n_download = 0
        for id_, row in labels.iterrows():
            filepath = folder / f"{id_}.csv"
            if not overwrite and filepath.exists():
                continue
            extract_ts(
                s1,
                ee.Geometry.Point(row.geometry.x, row.geometry.y),
                filepath,
                verbose=False,
            )
            n_download += 1
        print(f"Downloaded {n_download} tiles for {aoi} orbit {orbit}")


def get_date_range(aoi, orbit):
    """Based on previously downloaded data"""
    folder = S1_PATH / aoi / f"orbit_{orbit}"
    fps = sorted(folder.glob("*tif"))
    return fps[0].stem, fps[-1].stem


if __name__ == "__main__":
    import time
    from src.utils.time import print_sec
    from src.data.utils import aoi_orbit_iterator
    from src.gee.constants import RAW_TS_GEE_PATH

    # Download time series for all AOIs
    RAW_TS_GEE_PATH.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    for aoi, orbit in aoi_orbit_iterator():
        if not aoi.startswith("UKR"):
            continue
        download_all_ts_aoi_orbit(aoi, orbit, RAW_TS_GEE_PATH, overwrite=False, use_multiprocessing=True)
    print(f"Downloading all time series took {print_sec(time.time() - start_time)}")
