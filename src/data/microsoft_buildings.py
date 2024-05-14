import geopandas as gpd
from src.constants import PROCESSED_PATH


MICROSOFT_BUILDINGS_PATH = PROCESSED_PATH / "microsoft_buildings"


def load_microsoft_unosat_buildings(aoi):
    fp = MICROSOFT_BUILDINGS_PATH / f"microsoft_unosat_{aoi}.geojson"
    if fp.exists():
        gdf = gpd.read_file(fp)
        try:
            gdf.set_index("building_id", inplace=True)
        except KeyError:
            pass  # UKR16 does not have Microsoft footprints. Returns empty dataframe.
    else:
        raise FileNotFoundError(f"File {fp} does not exist, and implemented in another repo")

    return gdf
