""" Script to handle Microsoft's GlobalMLBuildingFootprints"""

# from geocube.api.core import make_geocube
import geopandas as gpd
import mercantile
import pandas as pd
from shapely.geometry import shape, Polygon
from tqdm import tqdm
from typing import Dict, List, Union

from src.constants import RAW_PATH, CRS_GLOBAL

MICROSOFT_BUILDINGS_RAW_PATH = RAW_PATH / "microsoft_buildings"
MICROSOFT_BUILDINGS_RAW_PATH.mkdir(exist_ok=True, parents=True)


def load_buildings_geo(geo: Polygon) -> gpd.GeoDataFrame:
    """Load all buildings within the given shape as a GeoDataFrame"""

    # Find all quadkeys
    quadkeys = quadkeys_in_shape(geo)

    df_list = []
    for qk in quadkeys:
        fp = MICROSOFT_BUILDINGS_RAW_PATH / f"{qk}.geojson"
        assert fp.exists(), f"Need to download microsoft buildings (quadkey={qk})"
        gdf_qk = gpd.read_file(fp)

        # keep track of quadkey for unique index
        gdf_qk.index.name = "building_id"
        gdf_qk.reset_index(inplace=True)
        gdf_qk.building_id = gdf_qk.building_id.apply(lambda id_: f"{qk}_{id_}")

        df_list.append(gdf_qk[gdf_qk.intersects(geo)])
    gdf_buildings = gpd.GeoDataFrame(pd.concat(df_list, ignore_index=True))
    return gdf_buildings[["geometry", "building_id"]]


def quadkeys_in_shape(geo: Polygon) -> List[str]:
    """Find all quadkeys that intersects the given shape"""
    quadkeys = [fp.stem for fp in MICROSOFT_BUILDINGS_RAW_PATH.glob("*.geojson")]

    # Check that the buildings ahve been downloaded before
    if not len(quadkeys):
        print("Need to download the microsoft buildigns first")
        print("Downloading Microsoft footprints from Ukraine")
        download_microsoft_footprint("Ukraine")
        quadkeys = [fp.stem for fp in MICROSOFT_BUILDINGS_RAW_PATH.glob("*.geojson")]

    gdf_qk = gpd.GeoDataFrame([quadkey_to_polygon(qk) for qk in quadkeys], crs=CRS_GLOBAL)
    gdf_qk_in = gdf_qk[gdf_qk.intersects(geo)]
    return gdf_qk_in.qk


def quadkey_to_polygon(qk) -> Dict[str, Union[str, Polygon]]:
    """Quadkey notations to polygon"""
    tile = mercantile.quadkey_to_tile(qk)
    lgnlat = mercantile.bounds(tile)
    geo = lgnlat_to_geo(lgnlat)
    return {"qk": qk, "geometry": geo}


def lgnlat_to_geo(lgnlat) -> Polygon:
    """Longitude Latitude Box to list of coords"""
    coords = [
        (lgnlat.west, lgnlat.south),
        (lgnlat.east, lgnlat.south),
        (lgnlat.east, lgnlat.north),
        (lgnlat.west, lgnlat.north),
    ]
    return Polygon(coords)


def download_microsoft_footprint(location="Ukraine"):
    """
    This snippet demonstrates how to access and convert the buildings
    data from .csv.gz to geojson for use in common GIS tools. You will
    need to install pandas, geopandas, and shapely.

    # From https://github.com/microsoft/GlobalMLBuildingFootprints
    """

    MICROSOFT_BUILDINGS_RAW_PATH.mkdir(exist_ok=True)

    dataset_links = pd.read_csv(
        "https://minedbuildings.blob.core.windows.net/global-buildings/dataset-links.csv"  # noqa E501
    )
    links = dataset_links[dataset_links.Location == location]

    # Check if already donwloaded:
    existing_quadkeys = [int(f.stem) for f in MICROSOFT_BUILDINGS_RAW_PATH.glob("*.geojson")]
    quadkeys_to_download = [q for q in links.QuadKey.unique() if q not in existing_quadkeys]
    if not len(quadkeys_to_download):
        print(f"Microsoft GlobalMLBuildingFootprints already downloaded for {location}.")
        return

    links = links[links.QuadKey.isin(quadkeys_to_download)]
    print(f"Downloading {len(links)} Microsoft GlobalMLBuildingFootprints")
    for _, row in tqdm(links.iterrows(), total=len(links)):
        df = pd.read_json(row.Url, lines=True)
        df["geometry"] = df["geometry"].apply(shape)
        gdf = gpd.GeoDataFrame(df, crs=CRS_GLOBAL)
        gdf.to_file(MICROSOFT_BUILDINGS_RAW_PATH / f"{row.QuadKey}.geojson", driver="GeoJSON")
    print("Microsoft GlobalMLBuildingFootprints downloaded")


if __name__ == "__main__":
    download_microsoft_footprint("Ukraine")
