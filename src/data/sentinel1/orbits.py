""" Script to create dataframe with valid/best Sentinel-1 orbit for each AOI"""

import ee
import geemap
import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, shape
from typing import List, Tuple

from src.constants import CRS_GLOBAL, PROCESSED_PATH
from src.data.sentinel1.download_geemap import get_s1_collection
from src.data import load_unosat_aois
from src.utils.geometry import load_country_boundaries
from src.utils.gee import shapely_to_gee, init_gee

d_country_year = {"Iraq": 2015, "Palestine": 2021, "Syria": 2015, "Ukraine": 2022}


def get_valid_orbits(aoi: str) -> List[int]:
    """
    Get the valid orbits for a given AOI.

    Args:
        aoi (str): The AOI name

    Returns:
        List[int]: The list of valid orbits
    """
    df_orbits = load_df_orbits()
    return df_orbits.loc[aoi, "valid_orbits"]


def get_best_orbit(aoi: str) -> int:
    """
    Get the best orbit for a given AOI.

    It is defined as the orbit with the largest intersection with the AOI. (the first one when multiple are valid)

    Args:
        aoi (str): The AOI name

    Returns:
        List[int]: The best orbit
    """
    df_orbits = load_df_orbits()
    return df_orbits.loc[aoi, "best_orbit"]


def load_df_orbits() -> gpd.GeoDataFrame:
    """
    Load the GeoDataFrame with the valid orbits (and the best) for each AOI.

    If the file does not exist, it is created.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame
    """

    fp = PROCESSED_PATH / "orbits" / "aoi_orbits.csv"
    if not fp.exists():
        df_orbits = create_df_orbits()
    else:
        df_orbits = pd.read_csv(fp)

    # Cast back into list and set index
    df_orbits.valid_orbits = df_orbits.valid_orbits.apply(lambda x: [int(i) for i in x.split(",")])
    df_orbits.set_index("aoi", inplace=True)

    # orbit 109 for UKR8 causes some bugs (both locally and in GEE)
    df_orbits.at["UKR8", "valid_orbits"] = [i for i in df_orbits.loc["UKR8", "valid_orbits"] if i != 109]
    return df_orbits


def create_df_orbits() -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame with the valid orbits for each AOI.

    Do so by computing the intersection between the AOI and the footprints of the orbits.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame
    """

    orbits_folder = PROCESSED_PATH / "orbits"
    orbits_folder.mkdir(exist_ok=True, parents=True)

    aois = load_unosat_aois(add_country=True)

    # Iterate over AOIs and get valid orbits and best orbit
    df_orbits = pd.DataFrame(columns=["valid_orbits", "best_orbit", "aoi"])
    for country, aois_country in aois.groupby("country"):
        year = d_country_year[country]
        gdf = load_gee_output_as_geopandas(country, year)
        gdf_12days = gdf[(gdf.date >= f"{year}-01-01") & (gdf.date < f"{year}-01-13")]
        valid_best_orbits = aois_country.apply(lambda row: get_valid_orbits_for_geo(row.geometry, gdf_12days), axis=1)
        _df_orbits = pd.DataFrame(valid_best_orbits.to_list(), columns=["valid_orbits", "best_orbit"])
        _df_orbits["aoi"] = aois_country.aoi.to_list()
        df_orbits = pd.concat([df_orbits, _df_orbits])

    # Post-process and save
    df_orbits.valid_orbits = df_orbits.valid_orbits.apply(lambda x: sorted([int(i) for i in x]))
    df_orbits.reset_index(drop=True, inplace=True)
    df_orbits.valid_orbits = df_orbits.valid_orbits.apply(lambda x: ",".join(map(str, x)))
    df_orbits.to_csv(orbits_folder / "aoi_orbits.csv", index=False)
    print("GeoDataFrame with orbit numbers per AOI saved")
    return df_orbits


def get_valid_orbits_for_geo(geo: Polygon, gdf_orbits: gpd.GeoDataFrame) -> Tuple[List[int], int]:
    """
    Get valid orbits for a given AOI. Also returns the best one. (the one with the largest intersection)

    Args:
        geo (Polygon): The AOI
        gdf_orbits (gpd.GeoDataFrame): The GeoDataFrame with the orbits

    Returns:
        Tuple[List[int], int]: The list of valid orbits and the best orbit
    """
    gdf_aoi = gdf_orbits[gdf_orbits.intersects(geo)].dissolve(by="orbit_number").reset_index()
    gdf_aoi.orbit_number = gdf_aoi.orbit_number.astype(int)
    gdf_aoi["intersection"] = gdf_aoi.apply(lambda row: row.geometry.intersection(geo).area / geo.area, axis=1)
    gdf_aoi = gdf_aoi[["orbit_number", "intersection"]].sort_values(by="intersection", ascending=False)
    valid_orbits = gdf_aoi[gdf_aoi.intersection > 0.95].orbit_number.to_list()
    best_orbit = valid_orbits[0]
    return valid_orbits, best_orbit


def load_gee_output_as_geopandas(country: str, year: int) -> gpd.GeoDataFrame:
    """
    Load the footprint file of Sentinel-1 orbits as a GeoDataFrame.

    If the file does not exist, it is created with GEE.

    Args:
        country (str): the country name (eg Ukraine, Iraq, ...)
        year (int): The year

    Returns:
        gpd.GeoDataFrame: The footprints
    """
    fp = PROCESSED_PATH / "orbits" / f"s1_orbits_{country}_{year}.csv"
    if not fp.exists():
        print(f"The file {fp.stem} with Sentinel-1 orbits does not exist. Downloading it now from GEE...")
        get_s1_footprints_from_gee(country, year)
    df = pd.read_csv(fp)
    gdf = gpd.GeoDataFrame(df)
    gdf.geometry = gdf.apply(lambda row: shape(json.loads(row.geometry)), axis=1)
    gdf.set_crs(CRS_GLOBAL, inplace=True)
    return gdf


def get_s1_footprints_from_gee(country: str, year: int) -> None:
    """
    Create a CSV file with the footprints of all Sentinel-1 orbits for a given country and year.

    Args:
        country (str): the country name (eg Ukraine, Iraq, ...)
        year (int): The year
    """

    init_gee()

    print(f"Creating footprint file for {country}, {year} of Sentinel-1 orbits with GEE...")

    fp = PROCESSED_PATH / "orbits" / f"s1_orbits_{country}_{year}.csv"
    assert (
        not fp.exists()
    ), f"The file {fp.stem} with Sentinel-1 orbits already exists. Delete it if you want to re-download it."

    # Load country boundaries
    boundary = load_country_boundaries(country)

    # Load S1 collection
    s1 = get_s1_collection(shapely_to_gee(boundary), f"{year}-01-01", f"{year}-12-31")

    # Transform into feature collection of metadata
    def _image_to_feature(image):
        return ee.Feature(
            None,
            {
                "id_": image.get("id_"),
                "date": image.get("date"),
                "orbit_direction": image.get("orbit_direction"),
                "orbit_number": image.get("orbit_number"),
                "geometry": image.geometry(),  # This won't work straight ahead, as geoomtry is exported as string...
            },
        )

    fc = ee.FeatureCollection(s1.map(_image_to_feature))

    # Export to CSV
    geemap.ee_export_vector(fc, fp, selectors=["id_", "date", "orbit_direction", "orbit_number", "geometry"])
    print("Done")
