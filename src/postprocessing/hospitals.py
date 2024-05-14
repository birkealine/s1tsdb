import geopandas as gpd
import multiprocessing as mp
import osmnx as ox
import pandas as pd
from pathlib import Path
from shapely.geometry import Polygon

from src.constants import RAW_PATH
from src.postprocessing.utils import get_prediction_for_geo
from src.utils.time import timeit

OSM_RAW_PATH = RAW_PATH / "osm"
OSM_RAW_PATH.mkdir(exist_ok=True, parents=True)

COLUMNS_TO_KEEP = [
    "element_type",
    "osmid",
    "amenity",
    "name",
    "name:en",
    "source_ref",
    "geometry",
    "healthcare",
    "healthcare:speciality",
    "building",
    "health_facility:type",
    "official_name",
    "website",
    "official_name:en",
    "hospital",
    "addr:city",
]


def add_predictions_to_hospitals(gdf_hospitals: gpd.GeoDataFrame, fp_preds: Path) -> gpd.GeoDataFrame:
    """Add columns 'weighted_mean' to the dataframe."""
    args = [(row.osmid, fp_preds, row.geometry) for _, row in gdf_hospitals.iterrows()]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        preds = pool.map(_process_hospital, args)
    return gdf_hospitals.merge(pd.DataFrame(preds), on="osmid")


def _process_hospital(args):
    id_, fp, geo = args
    return dict(osmid=id_, weighted_mean=get_prediction_for_geo(fp, geo))


def load_gdf_hospitals(place="Ukraine"):

    fp = OSM_RAW_PATH / f"{place}_hospitals.geojson"
    if fp.exists():
        gdf_hospitals = gpd.read_file(fp)
    else:
        gdf_hospitals = create_gdf_hospitals(place)
        gdf_hospitals.to_file(fp, driver="GeoJSON")
    return gdf_hospitals


@timeit
def create_gdf_hospitals(place="Ukraine"):
    """
    Create a GeoDataFrame with hospitals from OSM for a given place.

    Args:
        place (str, optional): Name of the place. Defaults to "Ukraine".

    Returns:
        _type_: Datframe with all hospitals in that place, and columns COLUMNS_TO_KEEP.
    """

    print(f"Creating gdf_hospitals for {place}")
    # Create the file
    tags = {"amenity": "hospital"}  # might have to check this
    gdf_hospitals = ox.features_from_place(place, tags=tags).reset_index()
    gdf_hospitals = gdf_hospitals[COLUMNS_TO_KEEP]

    # filter geometries that are fully within another one
    joined_gdf = gdf_hospitals.sjoin(gdf_hospitals, how="inner", predicate="within")
    indices_within_others = joined_gdf[joined_gdf.osmid_left != joined_gdf.osmid_right].index
    gdf_hospitals = gdf_hospitals[~gdf_hospitals.index.isin(indices_within_others)]

    # Keep only polygons (not points) -> TODO: VERIFY THIS ASSUMPTION
    gdf_hospitals = gdf_hospitals[gdf_hospitals.geometry.apply(lambda x: isinstance(x, Polygon))].copy()
    print("Done.")
    return gdf_hospitals
