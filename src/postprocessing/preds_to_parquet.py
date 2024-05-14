import duckdb
import geopandas as gpd
import pandas
from threading import Thread

from src.constants import PREDS_PATH
from src.utils.geometry import load_ukraine_admin_polygons, get_best_utm_crs


def preds_to_parquet(run_name):

    adm1 = load_ukraine_admin_polygons(adm_level=1).set_index("admin_id")
    adm2 = load_ukraine_admin_polygons(adm_level=2).set_index("admin_id")
    adm3 = load_ukraine_admin_polygons(adm_level=3).set_index("admin_id")
    adm4 = load_ukraine_admin_polygons(adm_level=4).set_index("admin_id")

    def prepare_gdf_preds(gdf, adm3_id):

        gdf_pivot = gdf.reset_index().pivot_table(index="building_id", columns="post_date", values="weighted_mean")
        gdf = gpd.GeoDataFrame(
            gdf_pivot.join(gdf.groupby("building_id").agg({"geometry": "first", "dataset": "first"})), crs=gdf.crs
        )
        gdf["area"] = gdf.to_crs(get_best_utm_crs(gdf)).area

        # Add admin names and ids
        d_admins = {k: v for k, v in adm3.loc[adm3_id].to_dict().items() if k.startswith("ADM")}
        for k, v in d_admins.items():
            gdf[k] = v

        adm1_id = adm1[adm1["ADM1_EN"] == d_admins["ADM1_EN"]].index[0]
        adm2_id = adm2[(adm2["ADM1_EN"] == d_admins["ADM1_EN"]) & (adm2["ADM2_EN"] == d_admins["ADM2_EN"])].index[0]
        gdf["adm1_id"] = adm1_id
        gdf["adm2_id"] = adm2_id
        gdf["adm3_id"] = adm3_id

        # For adm4, we need to cross reference with the building polygons
        adm4_ = adm4[
            (adm4.ADM1_EN == d_admins["ADM1_EN"])
            & (adm4.ADM2_EN == d_admins["ADM2_EN"])
            & (adm4.ADM3_EN == d_admins["ADM3_EN"])
        ]
        gdf["ADM4_EN"] = None
        gdf["adm4_id"] = None
        for adm4_id, adm4_row in adm4_.iterrows():
            gdf_ = gdf[gdf.within(adm4_row.geometry)]
            gdf.loc[gdf_.index, "ADM4_EN"] = adm4_row.ADM4_EN
            gdf.loc[gdf_.index, "adm4_id"] = adm4_id

        # geomtry as wkt
        gdf["geometry_wkt"] = gdf["geometry"].apply(lambda x: x.wkt)

        # reset index
        gdf = gdf.reset_index()
        return gdf[sorted(gdf.columns)]

    db_name = PREDS_PATH / run_name / "buildings_preds.db"

    db = duckdb.connect(f"{db_name}")
    db.execute("INSTALL spatial; INSTALL httpfs; LOAD spatial; LOAD httpfs; SET s3_region='us-west-2';")

    admin_preds_folder = PREDS_PATH / run_name / "admin_preds"
    _fps = sorted(admin_preds_folder.glob("*.geojson"))

    # check if table building_preds exists
    try:
        db.execute("SELECT COUNT(*) FROM buildings_preds").fetchall()

        # filter out existing adm3_ids
        existing_adm3_ids = db.execute("SELECT DISTINCT adm3_id FROM buildings_preds").fetchdf().adm3_id.values
        fps = [fp for fp in _fps if fp.stem not in existing_adm3_ids]
        print(f"Found {len(fps)}/{len(_fps)} new admin areas to process.")
    except Exception:
        fp = _fps[0]
        gdf = prepare_gdf_preds(gpd.read_file(fp).set_index(["building_id", "post_date"]), fp.stem)
        df = gdf.drop(columns=["geometry"])
        df.fillna(999, inplace=True)
        db.execute("CREATE TABLE buildings_preds AS SELECT * FROM df")
        print('Table "buildings_preds" created.')
        print(f"{len(fps)} files to process.")

    def write_from_thread(adm3_id, db):

        # Create a DuckDB connection specifically for this thread
        local_db = db.cursor()

        # insert into db
        fp = admin_preds_folder / f"{adm3_id}.geojson"
        gdf = prepare_gdf_preds(gpd.read_file(fp).set_index(["building_id", "post_date"]), fp.stem)
        df = gdf.drop(columns=["geometry"])
        df.fillna(999, inplace=True)
        local_db.execute("INSERT INTO buildings_preds SELECT * FROM df")

    write_thread_count = 20

    fps_to_compute = fps[:write_thread_count]
    fps = fps[write_thread_count:]
    while fps_to_compute:
        threads = []
        print(f"Processing {len(fps_to_compute)} files, {len(fps)} files left to process.")
        for i in range(min(write_thread_count, len(fps_to_compute))):
            print(f"Starting thread {i}. {fps_to_compute[i].stem}")
            threads.append(Thread(target=write_from_thread, args=(fps_to_compute[i].stem, db)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        fps_to_compute = fps[:write_thread_count]
        if fps_to_compute:
            fps = fps[write_thread_count:]


if __name__ == "__main__":
    run_name = "240307"
    preds_to_parquet(run_name)
