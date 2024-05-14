"""Script to generate grids of quadkeys in Ukraine"""

import geopandas as gpd
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


if __name__ == "__main__":
    load_ukraine_quadkeys_grid(zoom=9)
    load_ukraine_quadkeys_grid(zoom=10)
    load_ukraine_quadkeys_grid(zoom=11)
    load_ukraine_quadkeys_grid(zoom=8)
    load_ukraine_quadkeys_grid(zoom=7)
