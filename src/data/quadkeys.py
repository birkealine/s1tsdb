"""Script to generate grids of quadkeys in Ukraine"""

import geopandas as gpd
import pandas as pd
import mercantile
from shapely.geometry import Polygon
from typing import Dict, Union
from src.data.buildings.microsoft import MICROSOFT_BUILDINGS_RAW_PATH
from src.utils.geometry import load_country_boundaries
from src.constants import PROCESSED_PATH
from src.utils.time import timeit

QUADKEYS_PATH = PROCESSED_PATH / "quadkeys_grid"
QUADKEYS_PATH.mkdir(exist_ok=True, parents=True)


def load_ukraine_quadkeys_grid(zoom=9, clip_to_ukraine=True):
    """Load the grid of quadkeys in Ukraine."""

    fp = QUADKEYS_PATH / f"ukraine_qk_grid_zoom{zoom}.geojson"
    if not fp.exists():
        create_ukraine_quadkeys_grid(zoom=zoom)

    gdf_qk = gpd.read_file(fp)

    if clip_to_ukraine:
        # final geometry along ukrainian borders
        ukr_geo = load_country_boundaries("Ukraine")
        gdf_qk_border = gdf_qk[gdf_qk.area_in_ukraine < 1].copy()
        gdf_qk_border.geometry = gdf_qk_border.geometry.apply(lambda geo: geo.intersection(ukr_geo))
        gdf_qk.loc[gdf_qk_border.index, "geometry"] = gdf_qk_border.geometry
    return gdf_qk


def load_country_quadkeys_grid(zoom=9, country = "Ukraine", clip_to_country=True):
    """
    Load the grid of quadkeys in specified country.
    [function by birke]
    """

    fp = QUADKEYS_PATH / f"{country.lower()}_qk_grid_zoom{zoom}.geojson"
    if not fp.exists():
        create_country_quadkeys_grid(zoom=zoom)

    gdf_qk = gpd.read_file(fp)

    if clip_to_country:
        # final geometry along country borders
        cnt_geo = load_country_boundaries(country)
        gdf_qk_border = gdf_qk[gdf_qk[f"area_in_{country.lower()}"] < 1].copy() 
        gdf_qk_border.geometry = gdf_qk_border.geometry.apply(lambda geo: geo.intersection(cnt_geo))
        gdf_qk.loc[gdf_qk_border.index, "geometry"] = gdf_qk_border.geometry
    return gdf_qk


@timeit
def create_ukraine_quadkeys_grid(zoom=9):

    print(f"Creating quadkey grid with zoom = {zoom}...")

    # Original quadkeys from Microsoft have zoom 9 (len(qk) = 9)
    quadkeys = [fp.stem for fp in MICROSOFT_BUILDINGS_RAW_PATH.glob("*.geojson")]
    original_zoom = len(quadkeys[0])
    if zoom > original_zoom:
        quadkeys = zoom_quadkeys(quadkeys, zoom=zoom - original_zoom)
    elif zoom < original_zoom:
        quadkeys = sorted(set([qk[:zoom] for qk in quadkeys]))
    gdf_qk = gpd.GeoDataFrame([quadkey_to_polygon(qk) for qk in quadkeys], crs="EPSG:4326")

    # clip all fully outside ukraine
    ukr_geo = load_country_boundaries("Ukraine")
    gdf_qk = gdf_qk[gdf_qk.intersects(ukr_geo)]

    # Flag those not entirely in country
    gdf_qk["area_in_ukraine"] = gdf_qk.geometry.apply(lambda geo: geo.intersection(ukr_geo).area / geo.area)
    gdf_qk.to_file(QUADKEYS_PATH / f"ukraine_qk_grid_zoom{zoom}.geojson", driver="GeoJSON")
    print(f"Quadkey grid with zoom = {zoom} saved.")

@timeit
def create_country_quadkeys_grid(zoom=9, country = "Ukraine"):
    '''
    [function by birke]
    '''
    print(f"Creating quadkey grid with zoom = {zoom}...")

    # Original quadkeys from Microsoft have zoom 9 (len(qk) = 9)
    quadkeys = create_quadkey_list(country) #function creates quadkey list independent of downloaded microsoft building files / simply based on csv
    original_zoom = len(quadkeys[0])
    if zoom > original_zoom:
        quadkeys = zoom_quadkeys(quadkeys, zoom=zoom - original_zoom)
    elif zoom < original_zoom:
        quadkeys = sorted(set([qk[:zoom] for qk in quadkeys]))
    gdf_qk = gpd.GeoDataFrame([quadkey_to_polygon(qk) for qk in quadkeys], crs="EPSG:4326")

    # clip all fully outside ukraine
    cnt_geo = load_country_boundaries(country)
    gdf_qk = gdf_qk[gdf_qk.intersects(cnt_geo)]

    # Flag those not entirely in country
    gdf_qk[f"area_in_{country.lower()}"] = gdf_qk.geometry.apply(lambda geo: geo.intersection(cnt_geo).area / geo.area)
    gdf_qk.to_file(QUADKEYS_PATH / f"{country.lower()}_qk_grid_zoom{zoom}.geojson", driver="GeoJSON")
    print(f"Quadkey grid with zoom = {zoom} saved.")

def zoom_quadkeys(quadkeys, zoom):

    def _zoom_quadkeys(qks):
        return [qk + str(i) for qk in qks for i in range(4)]

    for _ in range(zoom):
        quadkeys = _zoom_quadkeys(quadkeys)
    return quadkeys


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

def create_quadkey_list(location="Ukraine"):
    """
    Creates list of quadkeys (str) for specified country
    [function by birke]
    """
    dataset_links = pd.read_csv(
        "https://minedbuildings.blob.core.windows.net/global-buildings/dataset-links.csv"  # noqa E501
    )
    links = dataset_links[dataset_links.Location == location]

    quadkeys = [str(q) for q in links.QuadKey.unique()]
    
    return quadkeys