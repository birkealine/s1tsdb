import geopandas as gpd
import osmnx as ox
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform
from typing import List, Tuple, Union
import xarray as xr

from src.constants import RAW_PATH, EXTERNAL_PATH


def load_country_boundaries(country: str) -> Tuple[Polygon, MultiPolygon]:
    """
    Load shapefile with country boundaries.

    If file does not exist, download it from OSM.

    Args:
        country (str): Name of the country (eg Ukraine, Iraq, ...)

    Returns:
        Tuple[Polygon, MultiPolygon]: The boundaries
    """

    folder = RAW_PATH / "countries"
    folder.mkdir(exist_ok=True)

    fp = folder / f"{country}.shp"

    if not fp.exists():
        print(f"The file with {country} boundaries does not exist. Downloading it now...")
        gdf = ox.geocode_to_gdf(country)
        gdf[["geometry"]].to_file(fp)
        print("Done")

    return gpd.read_file(fp).iloc[0].geometry


def load_ukraine_admin_polygons(adm_level=4):
    assert adm_level in [1, 2, 3, 4]
    ukraine_admin_path = sorted((EXTERNAL_PATH / "UKR_admin_boundaries").glob(f"*_adm{adm_level}*.shp"))[0]
    columns = [f"ADM{i}_EN" for i in range(1, adm_level + 1)] + ["geometry"]
    ukr_admin = gpd.read_file(ukraine_admin_path)[columns]
    ukr_admin.index.name = "admin_id"
    ukr_admin.reset_index(inplace=True)
    ukr_admin["admin_id"] = ukr_admin["admin_id"].apply(lambda x: f"{adm_level}_{x}")
    return ukr_admin

def load_country_admin_polygons(adm_level=4, country = "Ukraine"):
    assert adm_level in [1, 2, 3, 4]
    cnt_admin_path = sorted((EXTERNAL_PATH / f"{country}_admin_boundaries").glob(f"*_adm{adm_level}*.shp"))[0]
    columns = [f"ADM{i}_EN" for i in range(1, adm_level + 1)] + ["geometry"]
    cnt_admin = gpd.read_file(cnt_admin_path)[columns]
    cnt_admin.index.name = "admin_id"
    cnt_admin.reset_index(inplace=True)
    cnt_admin["admin_id"] = cnt_admin["admin_id"].apply(lambda x: f"{adm_level}_{x}")
    return cnt_admin

def reproject_geo(geo, current_crs, target_crs):
    """Reprojects a Shapely geometrz from the current CRS to a new CRS."""
    transformer = Transformer.from_crs(current_crs, target_crs, always_xy=True)
    return transform(transformer.transform, geo)


def get_best_utm_crs(gdf: gpd.GeoDataFrame) -> str:
    """Get the best UTM CRS for the given GeoDataFrame."""
    mean_lon = gdf.geometry.unary_union.centroid.x
    utm_zone = int(((mean_lon + 180) / 6) % 60) + 1
    utm_crs = f"EPSG:326{utm_zone}" if gdf.geometry.unary_union.centroid.y > 0 else f"EPSG:327{utm_zone}"
    return utm_crs


class BBox:
    """Custom BBox class to easily store everything we need."""

    def __init__(
        self,
        ids: Tuple[int, int, int, int],
        xar: xr.DataArray,
    ):
        """
        Initialize the bounding box class

        Args:
            ids (Tuple[int, int, int, int]): IDs of xmin, ymin, xmax and ymax
            xar (xr.DataArray): The target DataArray, to get coordinates, crs, res...
        """
        self.ids = ids
        self._xmin = xar.x[ids[0]]
        self._xmax = xar.x[ids[2] - 1]
        self._ymin = xar.y[ids[1]]
        self._ymax = xar.y[ids[3] - 1]

        self.crs = xar.rio.crs
        self.res = xar.rio.resolution()
        self._geometry = self.get_geometry()

    def get_geometry(self) -> Polygon:
        """
        Returns the Polygon spanned by x and y coordinates.

        As the coordinates indicates the center of the pixel, we must use the resolution
        to surround fully the pixels at the boundaries

        Returns:
            Polygon: The georeferenced Polygon
        """

        # The pixel coordinates are in the center of the pixel,
        xmin = self._xmin - self.res[0] / 2
        ymin = self._ymin - self.res[1] / 2
        xmax = self._xmax + self.res[0] / 2
        ymax = self._ymax + self.res[1] / 2
        polygon = (
            (xmin, ymin),
            (xmin, ymax),
            (xmax, ymax),
            (xmax, ymin),
            (xmin, ymin),
        )
        return Polygon(polygon)

    @property
    def geometry(self) -> Polygon:
        return self._geometry


def percentage_intersection(g1: Polygon, g2: Polygon) -> float:
    """ratio of g1 that intersects with g2 (in %)"""
    return 100 * g1.intersection(g2).area / g1.area


def convert_3D_2D(geometry: gpd.GeoSeries) -> List[Union[Polygon, MultiPolygon]]:
    """
    Copnvert a GeoSeries of 2D/3D Multi/Polygons and returns a list of 2D Multi/Polygons

    From: https://gist.github.com/rmania/8c88377a5c902dfbc134795a7af538d8

    Args:
        geometry (gpd.GeoSeries): The series of shape to convert

    Returns:
        List[Union[Polygon, MultiPolygon]]: The list of geometries flattened
    """
    new_geo = []
    for p in geometry:
        if p.has_z:
            if p.geom_type == "Polygon":
                lines = [xy[:2] for xy in list(p.exterior.coords)]
                new_p = Polygon(lines)
                new_geo.append(new_p)
            elif p.geom_type == "MultiPolygon":
                new_multi_p = []
                for ap in list(p.geoms):
                    lines = [xy[:2] for xy in list(ap.exterior.coords)]
                    new_p = Polygon(lines)
                    new_multi_p.append(new_p)
                new_geo.append(MultiPolygon(new_multi_p))
        else:
            new_geo.append(p)
    return new_geo
