"""Download all buildings from Overture Maps in Ukraine using DuckDB"""

import duckdb
from src.utils.geometry import load_country_boundaries
from src.constants import RAW_PATH
from src.utils.time import timeit

OVERTURE_BUILDINGS_RAW_PATH = RAW_PATH / "overture_buildings"
OVERTURE_BUILDINGS_RAW_PATH.mkdir(exist_ok=True, parents=True)

OVERTURE_RELEASE = "s3://overturemaps-us-west-2/release/2024-02-15-alpha.0"


def download_overture_buildings(bbox, filepath="buildings.parquet"):
    """Download all buildings from Overture Maps in the given bounding box using DuckDB. Save as parquet"""

    minx, miny, maxx, maxy = bbox

    db = duckdb.connect()
    db.execute("INSTALL spatial; INSTALL httpfs; LOAD spatial; LOAD httpfs; SET s3_region='us-west-2';")

    db.execute(
        f"""
            COPY (
                SELECT
                    *
                FROM
                    read_parquet('{OVERTURE_RELEASE}/theme=buildings/type=building/*', hive_partitioning=1)
                WHERE
                    bbox.minX >= {minx}
                AND bbox.minY >= {miny}
                AND bbox.maxX <= {maxx}
                AND bbox.maxY <= {maxy}
            ) TO '{filepath}'
            WITH (FORMAT 'Parquet');
        """
    )

    # DuckDB can't export geoparquet directly, so we use gpq to convert it to geoparquet
    # (nb: gpq downloaded from https://github.com/planetlabs/gpq/releases (linux-adm64)
    # and manually put into the conda env:
    # tar -xzf gpq-linux-amd64.tar.gz
    # mv gpq ./s1tsdd-env/bin/)
    # print("converting parquet file to geoparquet...")
    # subprocess.run(["gpq", "convert", str(OVERTURE_BUILDINGS_RAW_PATH / "ukraine_buildings.parquet")])
    # print("Done.")


@timeit
def main():

    ukraine_geo = load_country_boundaries("Ukraine")
    bbox = ukraine_geo.bounds
    filepath = OVERTURE_BUILDINGS_RAW_PATH / "ukraine_buildings.parquet"

    print("Downloading Overture buildings in Ukraine...")
    download_overture_buildings(bbox, filepath)
    print("Overture buildings in Ukraine downloaded.")


if __name__ == "__main__":
    main()
