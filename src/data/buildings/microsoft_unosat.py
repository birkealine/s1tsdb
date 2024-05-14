import geopandas as gpd

from src.data.buildings.microsoft import load_buildings_geo
from src.data.unosat import get_unosat_geometry, load_unosat_labels
from src.constants import PROCESSED_PATH, CRS_GLOBAL
from src.utils.geometry import get_best_utm_crs

MICROSOFT_UNOSAT_PATH = PROCESSED_PATH / "microsoft_unosat"
MICROSOFT_UNOSAT_PATH.mkdir(exist_ok=True, parents=True)

BUFFERS = [0, 2, 5]  # Buffer in meters


def load_buildings_aoi(aoi):
    """Load all buildings within the given AOI as a GeoDataFrame, with the corresponding unosat labels"""

    fp = MICROSOFT_UNOSAT_PATH / f"msft_unosat_{aoi}.geojson"
    if fp.exists():
        return gpd.read_file(fp)
    else:
        return create_buildings_unosat(aoi)


def create_buildings_unosat(aoi):

    # Load Microsoft buildings
    geo = get_unosat_geometry(aoi)
    buildings = load_buildings_geo(geo)

    if not len(buildings):
        print(f"No Microsoft buildings found for {aoi}")
        return

    # Load UNOAST labels and add them to the GeoDataFrame
    points = load_unosat_labels(aoi, labels_to_keep=None)

    # Get CRS for projection (to take buffer in meters)
    crs_proj = get_best_utm_crs(points)

    for b in BUFFERS:
        buildings_ = buildings.copy()
        if b:
            buildings_.geometry = buildings.to_crs(crs_proj).buffer(b).to_crs(CRS_GLOBAL)
        pts_in = gpd.overlay(points, buildings_[["geometry", "building_id"]], how="intersection")
        buildings_with_labels = pts_in.groupby("building_id").agg({"damage": "min"})
        if b:
            buildings_with_labels.rename(columns={"damage": f"damage_{b}m"}, inplace=True)
        buildings = buildings.merge(buildings_with_labels, on="building_id", how="left")
    buildings.fillna(6, inplace=True)  # no damage

    # Save to file
    buildings.to_file(MICROSOFT_UNOSAT_PATH / f"msft_unosat_{aoi}.geojson", driver="GeoJSON")
    print(f"File with Microsoft buildings and UNOSAT labels for {aoi} saved.")

    return buildings


if __name__ == "__main__":
    from src.data import get_all_aois

    for aoi in get_all_aois():
        create_buildings_unosat(aoi)
