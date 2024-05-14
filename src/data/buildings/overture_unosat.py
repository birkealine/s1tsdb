import duckdb
import geopandas as gpd
import warnings
from src.utils.geometry import get_best_utm_crs
from src.data import get_all_aois, get_unosat_geometry, load_unosat_labels
from src.constants import CRS_GLOBAL, OVERTURE_BUILDINGS_RAW_PATH, OVERTURE_AOI_PATH
from src.utils.time import timeit


OVERTURE_AOI_PATH.mkdir(exist_ok=True, parents=True)

FP_RAW_PARQUET = OVERTURE_BUILDINGS_RAW_PATH / "ukraine_buildings.parquet"


def load_overture_buildings_aoi(aoi):

    fp = OVERTURE_AOI_PATH / f"overture_unosat_{aoi}.geojson"
    if not fp.exists():
        print(f"Creating file with Overture buildings and UNOSAT labels for {aoi}...")
        create_overture_aoi_unosat(aoi)
    return gpd.read_file(fp)


def create_all_overture_aoi_unosat(force_recreate=False):

    db = duckdb.connect()
    db.execute("INSTALL spatial; INSTALL httpfs; LOAD spatial; LOAD httpfs; SET s3_region='us-west-2';")

    for aoi in get_all_aois():
        create_overture_aoi_unosat(aoi, db, force_recreate=force_recreate)


def create_overture_aoi_unosat(aoi, db=None, force_recreate=False):

    if db is None:
        db = duckdb.connect()
        db.execute("INSTALL spatial; INSTALL httpfs; LOAD spatial; LOAD httpfs; SET s3_region='us-west-2';")

    geo = get_unosat_geometry(aoi)
    minx, miny, maxx, maxy = geo.bounds
    fp = OVERTURE_AOI_PATH / f"overture_unosat_{aoi}.geojson"
    if fp.exists() and not force_recreate:
        print(f"File with Overture buildings and UNOSAT labels for {aoi} already exists.")
        return gpd.read_file(fp)

    query = f"""
    COPY (
        SELECT
            JSON(sources)[0].dataset as dataset,
            ST_GeomFromWKB(geometry) as geometry
        FROM
            read_parquet('{FP_RAW_PARQUET}', hive_partitioning=1)
        WHERE
            bbox.minX + (bbox.maxX - bbox.minX)/2 >= {minx}
        AND bbox.minY + (bbox.maxY - bbox.minY)/2 >= {miny}
        AND bbox.minX + (bbox.maxX - bbox.minX)/2 <= {maxx}
        AND bbox.minY + (bbox.maxY - bbox.minY)/2 <= {maxy}
    ) TO '{fp}'
    WITH (FORMAT GDAL, Driver 'GeoJSON')
    """
    db.execute(query)

    # Postprocess directly (add unosat labels, clip to aoi polygon) and overwrite the file
    gdf = gpd.read_file(fp)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        gdf = gdf[gdf.centroid.intersects(geo)]  # seems faster than duckdb in this case
    gdf.index.name = "building_id"
    gdf.reset_index(inplace=True)

    points = load_unosat_labels(aoi, labels_to_keep=None).reset_index()
    crs_proj = get_best_utm_crs(points)

    BUFFERS = [0, 2, 5]  # Buffer in meters
    for b in BUFFERS:
        gdf_ = gdf.copy()
        if b:
            gdf_.geometry = gdf.to_crs(crs_proj).buffer(b).to_crs(CRS_GLOBAL)
        pts_in = gpd.overlay(points, gdf_[["geometry", "building_id"]], how="intersection")
        pts_sorted = pts_in.sort_values(by=["building_id", "damage"])  # easy trick to keep unosat_id with lowest damage
        gdf_with_labels = pts_sorted.groupby("building_id").agg({"damage": "first", "unosat_id": "first"})
        if b:
            gdf_with_labels.rename(columns={"damage": f"damage_{b}m", "unosat_id": f"unosat_id_{b}m"}, inplace=True)

        gdf = gdf.merge(gdf_with_labels, on="building_id", how="left")
    gdf.unosat_id.fillna(-1, inplace=True)  # no unosat label
    gdf.fillna(6, inplace=True)  # no damage

    gdf.to_file(fp, driver="GeoJSON")
    print(f"File with Overture buildings and UNOSAT labels for {aoi} saved.")
    return gdf


@timeit
def main():
    create_all_overture_aoi_unosat(force_recreate=True)


if __name__ == "__main__":
    main()
