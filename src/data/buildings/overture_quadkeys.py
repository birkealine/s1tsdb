"""Download all buildings from Overture Maps in Ukraine using DuckDB"""

import duckdb
import geopandas as gpd
import multiprocessing as mp
import pandas as pd
from shapely.geometry import box
from tqdm import tqdm
from src.utils.geometry import load_country_boundaries
from src.constants import OVERTURE_BUILDINGS_RAW_PATH, OVERTURE_QK_PATH
from src.data.quadkeys import load_ukraine_quadkeys_grid
from src.utils.time import timeit

OVERTURE_QK_PATH.mkdir(exist_ok=True, parents=True)
FP_RAW_PARQUET = OVERTURE_BUILDINGS_RAW_PATH / "ukraine_buildings.parquet"

db = duckdb.connect()
db.execute("INSTALL spatial; INSTALL httpfs; LOAD spatial; LOAD httpfs; SET s3_region='us-west-2';")


def load_gdf_overture_qk(zoom=9):
    """Geodataframe with quadkeys and buildings infos (bbox and number). Ready for predictions"""

    fp = OVERTURE_QK_PATH / f"grid_overture_qk_zoom{zoom}.geojson"
    if not fp.exists():
        create_gdf_overture_qk(zoom=zoom)

    return gpd.read_file(fp)


@timeit
def create_gdf_overture_qk(zoom=9):

    print(f"Creating geodataframe with quadkeys and buildings infos for zoom {zoom}...")

    folder_individual_qks = OVERTURE_QK_PATH / f"zoom{zoom}"
    if not folder_individual_qks.exists():
        print(f"Creating individual quadkeys for zoom {zoom}...")
        create_all_buildings_quadkeys_multiprocessing(zoom=zoom)

    fps = sorted(folder_individual_qks.glob("*.geojson"))
    with mp.Pool(mp.cpu_count()) as p:
        d_results = p.map(get_info_from_quad, fps)
    df = pd.DataFrame(d_results)

    # postprocess
    df = df[df.n_buildings > 0]
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[box(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in zip(df.xmin, df.ymin, df.xmax, df.ymax)],
        crs="EPSG:4326",
    )
    gdf.to_file(OVERTURE_QK_PATH / f"grid_overture_qk_zoom{zoom}.geojson")
    print(f"Grid of overture buildigns within quadkeys for zoom {zoom} saved")


def get_info_from_quad(fp):
    """Get the number of buildings and the bbox of the buildings in the quadkey file."""
    qk = fp.stem
    n_buildings, xmin, ymin, xmax, ymax = db.execute(
        f"""
              SELECT
                     count(*),
                     min(ST_XMin(geom)),
                     min(ST_YMin(geom)),
                     max(ST_XMax(geom)),
                     max(ST_YMax(geom)),

              FROM
                     ST_Read('{fp}')
       """
    ).fetchall()[0]
    return {"qk": qk, "n_buildings": n_buildings, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}


def process_quadkey(qk_row):
    """Keep buildings whose centroids are within the quadkey and within Ukraine"""
    qk, row = qk_row
    zoom = len(qk)

    db = duckdb.connect()
    db.execute("INSTALL spatial; INSTALL httpfs; LOAD spatial; LOAD httpfs; SET s3_region='us-west-2';")

    geo_ukraine = load_country_boundaries("Ukraine").simplify(0.001)  # important to simplify the geometry

    minx, miny, maxx, maxy = row["geometry"].bounds
    fp = OVERTURE_QK_PATH / f"zoom{zoom}" / f"{qk}.geojson"
    if fp.exists():
        return

    # Centroid of building is within the quadkey. Using bbox is faster than ST_Centroid
    condition = f"""
            bbox.minX + (bbox.maxX - bbox.minX)/2 >= {minx}
        AND bbox.minY + (bbox.maxY - bbox.minY)/2 >= {miny}
        AND bbox.minX + (bbox.maxX - bbox.minX)/2 <= {maxx}
        AND bbox.minY + (bbox.maxY - bbox.minY)/2 <= {maxy}
    """

    db.execute(
        f"""
        COPY (
            SELECT
                type,
                version,
                CAST(updatetime as varchar) as updateTime,
                height,
                numfloors as numFloors,
                level,
                class,
                JSON(names) as names,
                JSON(sources)[0].dataset as dataset,
                ST_GeomFromWKB(geometry) as geometry
            FROM
                read_parquet('{FP_RAW_PARQUET}', hive_partitioning=1)
            WHERE
                {condition}
        ) TO '{fp}'
        WITH (FORMAT GDAL, Driver 'GeoJSON')
    """
    )

    if row.area_in_ukraine != 1:
        # Clip the buildings outside of Ukraine (do it with geopandas, not with DuckDB, as it's faster)
        gdf = gpd.read_file(fp)
        gdf = gdf[gdf.covered_by(geo_ukraine)]
        gdf.to_file(fp, driver="GeoJSON")


@timeit
def create_all_buildings_quadkeys(zoom=9):
    gdf_grid = load_ukraine_quadkeys_grid(zoom=zoom).set_index("qk")
    folder = OVERTURE_QK_PATH / f"zoom{zoom}"
    folder.mkdir(exist_ok=True)

    for qk, row in (pbar := tqdm(gdf_grid.iterrows(), total=gdf_grid.shape[0])):
        pbar.set_description(f"Processing {qk}")
        process_quadkey((qk, row))


@timeit
def create_all_buildings_quadkeys_multiprocessing(zoom=9):
    gdf_grid = load_ukraine_quadkeys_grid(zoom=zoom).set_index("qk")
    folder = OVERTURE_QK_PATH / f"zoom{zoom}"
    folder.mkdir(exist_ok=True)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        list(tqdm(pool.imap(process_quadkey, gdf_grid.iterrows()), total=gdf_grid.shape[0]))


if __name__ == "__main__":

    # create_all_buildings_quadkeys_multiprocessing(zoom=10)

    load_gdf_overture_qk(zoom=7)
