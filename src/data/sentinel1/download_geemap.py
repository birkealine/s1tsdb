"""Download Sentinel-1 tiles using geemap"""

import datetime as dt
import ee
import geemap
import json
from pathlib import Path
from shapely import Polygon
import time
from typing import Union

from src.constants import CRS_GLOBAL, CRS_UKRAINE, RAW_S1_PATH
from src.data import load_unosat_aois
from src.utils.gee import init_gee, shapely_to_gee, custom_mosaic_same_day
from src.utils.time import print_sec

# init_gee()


def download_all_s1(
    start_date: Union[dt.date, str],
    end_date: Union[dt.date, str],
    local_folder: Path = RAW_S1_PATH,
    add_extra_aois: bool = False,
    only_ukraine: bool = True,
):
    """
    Download all Sentinel-1 tiles for a given time range using geemap.

    /!\ This function download the same dates for all AOIs, which is not optimal. # noqa

    Args:
        start_date (Union[dt.date, str]): The start date.
        end_date (Union[dt.date, str]): The end date.
        local_folder (Path, optional): Path to the folder. Defaults to RAW_S1_PATH.
        add_extra_aois (bool, optional): Whether to add extra AOIs. Defaults to False.
        only_ukraine (bool, optional): Whether to download only for Ukraine. Defaults to True.
    """

    start_time = time.time()

    # Prepare dataframe with geometries, ...
    aois = load_unosat_aois(add_extra_aois=add_extra_aois)
    aois.set_index("aoi", inplace=True)

    # Transform all shapes into convex hulls, since GEE returns anyway the smallest rectangle
    # around the polygon
    aois.geometry = aois.apply(lambda x: x.geometry.convex_hull, axis=1)

    # Add buffer around aoi (in meters)
    aois.geometry = aois.to_crs(CRS_UKRAINE).buffer(2000).to_crs(CRS_GLOBAL)

    d_error = {}
    for aoi, row in aois.iterrows():
        if only_ukraine and not aoi.startswith("UKR"):
            continue
        print(f"Downloading {aoi}...")
        d_error[aoi] = download_s1(
            geometry=row.geometry,
            start_date=start_date,
            end_date=end_date,
            local_folder=local_folder / aoi,
        )

    print(f"Downloaded all tiles in {print_sec(time.time() - start_time)}")

    # Save errors in json file
    with open(local_folder / "errors.json", "w") as f:
        json.dump(d_error, f)


def download_s1(
    geometry: Union[Polygon, ee.Geometry],
    start_date: Union[dt.date, str],
    end_date: Union[dt.date, str],
    local_folder: Union[str, Path],
    orbit: int = None,
    crs: Union[str, int] = None,
):
    """
    Download Sentinel-1 tiles for a given time range and geometry.

    Store each orbit in a separate folder.

    Args:
        geometry (Union[Polygon, ee.Geometry]): The geometry. (in EPSG:4326)
        start_date (Union[dt.date, str]): The start date.
        end_date (Union[dt.date, str]): The end date.
        local_folder (Union[str, Path]): Path to the folder.
        orbit (int, optional): The orbit number. Defaults to None (download all orbits)
        crs (Union[str, int], optional): The CRS. Defaults to None.
    """

    # Check input
    geometry_ee = geometry if isinstance(geometry, ee.Geometry) else shapely_to_gee(geometry)
    local_folder = Path(local_folder) if not isinstance(local_folder, Path) else local_folder
    if crs is not None and isinstance(crs, int):
        crs = f"EPSG:{crs}"

    s1 = get_s1_collection(geometry_ee, start_date, end_date)

    # Get all orbits
    if orbit is not None:
        orbits = [orbit]
    else:
        orbits = s1.aggregate_array("orbit_number").distinct().getInfo()

    # Download each orbit in a separate folder
    d_error = {}
    for orbit in orbits:
        # Create folder
        folder = local_folder / f"orbit_{orbit}"
        folder.mkdir(parents=True, exist_ok=True)

        # Filter by orbit
        s1_orbit = s1.filter(ee.Filter.eq("orbit_number", orbit))

        # Mosaic same day (needed at least for UKR13)
        mosaic_s1 = custom_mosaic_same_day(s1_orbit)
        n_tiles_total = mosaic_s1.size().getInfo()

        # Filter out existing ones
        existing_ids = ee.List([f.stem for f in folder.glob("*.tif")])
        s1_ready = mosaic_s1.filter(ee.Filter.Not(ee.Filter.inList("id_", existing_ids)))

        n_tiles = s1_ready.size().getInfo()
        print(f"Orbit: {orbit}: Total number of tiles: {n_tiles_total}, new: {n_tiles}")

        # Download in the folder
        if n_tiles:
            # The issue with download_ee_image_collection is that a single error causes the whole
            # code to crash
            # geemap.download_ee_image_collection(
            #     s1_ready, out_dir=folder, region=geometry_ee, scale=10
            # )
            d_error[orbit] = custom_download_ee_image_collection(
                s1_ready, out_dir=folder, region=geometry_ee, scale=10, crs=crs
            )

    return d_error


def custom_download_ee_image_collection(
    collection: ee.ImageCollection, out_dir: Path, region: ee.Geometry, scale: int = None, crs=None
):
    """Same as geemap.download_ee_image_collection, but with a try/except to avoid crashing"""

    if not isinstance(collection, ee.ImageCollection):
        raise ValueError("ee_object must be an ee.ImageCollection.")

    if out_dir is None:
        out_dir = Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    count = int(collection.size().getInfo())
    print(f"Total number of images: {count}\n")

    col_list = collection.toList(count)
    list_files_error = []
    for i in range(count):
        image = ee.Image(col_list.get(i))
        name = image.get("system:index").getInfo() + ".tif"
        filename = out_dir / name
        print(f"Downloading {i + 1}/{count}: {name}")
        try:
            geemap.download_ee_image(image, filename, region=region, scale=scale, crs=crs)
        except Exception as e:
            print(f"Error downloading {name}: {e}")
            list_files_error.append(name)
    return list_files_error


def get_s1_collection(geometry_ee: ee.Geometry, start_date: Union[dt.date, str], end_date: Union[dt.date, str]):
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD")  # decibel, orthorectified
    s1_target = (
        s1.filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("platform_number", "A"))  # only Sentinel-1A (tosatellite,  keep 12 days revisiting time)
        .filterDate(start_date, end_date)
        .filterBounds(geometry_ee)
        .map(lambda img: img.select(["VV", "VH"]))
    )

    # add metadata as attributes
    s1_target = (
        s1_target.map(lambda img: img.set("id_", img.id()))
        .map(lambda img: img.set("date", ee.Date(img.date()).format("YYYY-MM-dd")))
        .map(lambda img: img.set("orbit_number", ee.Number(img.get("relativeOrbitNumber_start"))))
        .map(lambda img: img.set("orbit_direction", ee.String(img.get("orbitProperties_pass"))))
    )
    return s1_target


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", type=str, default="2019-07-01")
    parser.add_argument("--end_date", type=str, default="2023-06-30")
    parser.add_argument("--local_folder", type=str, default=RAW_S1_PATH)
    parser.add_argument("--add_extra_aois", action="store_true")

    args = parser.parse_args()

    download_all_s1(
        start_date=args.start_date,
        end_date=args.end_date,
        local_folder=Path(args.local_folder),
        add_extra_aois=args.add_extra_aois,
    )
