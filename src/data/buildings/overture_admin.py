import duckdb
import geopandas as gpd
import multiprocessing as mp
from tqdm import tqdm
import warnings

from src.utils.geometry import load_ukraine_admin_polygons
from src.constants import OVERTURE_ADMIN_PATH, OVERTURE_BUILDINGS_RAW_PATH
from src.utils.time import timeit

FP_RAW_PARQUET = OVERTURE_BUILDINGS_RAW_PATH / "ukraine_buildings.parquet"


def load_overture_buildings_admins(admin_id):

    fp = OVERTURE_ADMIN_PATH / f"overture_admin_{admin_id}.geojson"
    if not fp.exists():
        print(f"Creating file with Overture buildings for admin {admin_id}...")
        create_overture_admin(admin_id)
    return gpd.read_file(fp)


def create_overture_admin(admin_id, admin_geo=None, db=None, force_recreate=False):

    if db is None:
        db = duckdb.connect()
        db.execute("INSTALL spatial; INSTALL httpfs; LOAD spatial; LOAD httpfs; SET s3_region='us-west-2';")

    if admin_geo is None:
        gdf_admin = load_ukraine_admin_polygons(adm_level=int(admin_id.split("_")[0]))
        gdf_admin.set_index("admin_id", inplace=True)
        admin_geo = gdf_admin.loc[admin_id].geometry

    minx, miny, maxx, maxy = admin_geo.bounds
    OVERTURE_ADMIN_PATH.mkdir(exist_ok=True)
    fp = OVERTURE_ADMIN_PATH / f"overture_admin_{admin_id}.geojson"
    if fp.exists() and not force_recreate:
        print(f"File with Overture building for admin {admin_id} already exists.")
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

    # postprocess
    gdf = gpd.read_file(fp)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        gdf = gdf[gdf.centroid.intersects(admin_geo)]  # seems faster than duckdb in this case
    gdf.index.name = "building_id"
    gdf.reset_index(inplace=True)

    gdf.to_file(fp, driver="GeoJSON")
    print(f"File with Overture buildings for admin {admin_id} saved.")
    return gdf


def create_overture_admin_(id_row):
    admin_id, row = id_row
    create_overture_admin(admin_id, row.geometry)


@timeit
def create_all_overture_admin_multiprocessing(adm_level=3):

    OVERTURE_ADMIN_PATH.mkdir(exist_ok=True)

    gdf_admin = load_ukraine_admin_polygons(adm_level=adm_level).set_index("admin_id")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        list(tqdm(pool.imap(create_overture_admin_, gdf_admin.iterrows()), total=gdf_admin.shape[0]))


if __name__ == "__main__":
    create_all_overture_admin_multiprocessing()
    print("Done.")
