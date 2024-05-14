import geopandas as gpd
import multiprocessing as mp
from osgeo import gdal
import pandas as pd
from pathlib import Path
from shapely.geometry import box
import tempfile
import xarray as xr
import warnings

from src.constants import PREDS_PATH, OVERTURE_ADMIN_PATH
from src.postprocessing.utils import read_fp_within_geo, vectorize_xarray_3d
from src.utils.gdrive import drive_to_local, get_files_in_folder
from src.utils.geometry import load_ukraine_admin_polygons
from src.utils.time import timeit


@timeit
def drive_to_result(run_name, post_dates=None):

    # 1. download to local
    # download_and_merge_all_dates(run_name, post_dates)

    # 2. Prediction per building
    create_all_gdf_overture_with_preds(run_name)
    # create_all_gdf_overture_with_preds_mp(run_name, cpu=3)

    # 3. Aggregate per oblasts
    aggregate_preds_for_all_oblast(run_name, "oblasts_with_preds_agg", postprocess=True)
    aggregate_preds_for_all_oblast(run_name, "oblasts_with_preds_agg_no_postprocessing", postprocess=False)


# ====================== 1. Download and merge ======================
def download_and_merge_all_dates(run_name, post_dates=None):

    local_folder = PREDS_PATH / run_name
    local_folder.mkdir(exist_ok=True, parents=True)
    drive_folders = get_files_in_folder(f"{run_name}_quadkeys_predictions", return_names=True)

    if post_dates is not None:
        # filter folders to download
        post_dates_ = [f"{p[0]}_{p[1]}" for p in post_dates]
        drive_folders = [f for f in drive_folders if f in post_dates_]

    print(f"Downloading {len(drive_folders)} folders")

    for drive_folder in drive_folders:
        if drive_folder == "cfg.yaml":
            continue

        name_file = f"ukraine_{drive_folder}.tif"
        donwload_and_merge(drive_folder, local_folder, name_file, save_individual_files=False)


def donwload_and_merge(drive_folder, local_folder, name_file, save_individual_files=False):

    if (local_folder / name_file).exists():
        print(f"{name_file} already exists")
        return

    print(f"Downloading {drive_folder}")

    with tempfile.TemporaryDirectory() as tmp:

        if save_individual_files:
            local_folder_indiv = local_folder / drive_folder.split("/")[-1]
        else:
            local_folder_indiv = Path(tmp)

        drive_to_local(drive_folder, local_folder_indiv, delete_in_drive=False, verbose=0)
        print(f"Finished downloading {drive_folder}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tif_files = [str(fp) for fp in local_folder_indiv.glob("*.tif")]
            output_file = str(local_folder / name_file)
            print(f"Merging {len(tif_files)} files into {name_file}")
            gdal.Warp(output_file, tif_files, format="GTiff")
            print(f"Finished merging {name_file}")


# ====================== 2. Preds per building ======================
@timeit
def create_all_gdf_overture_with_preds_mp(run_name, cpu=5):

    gdf_admin = load_ukraine_admin_polygons(adm_level=3)
    folder_preds = PREDS_PATH / run_name / "admin_preds"
    folder_preds.mkdir(exist_ok=True, parents=True)

    args = [
        (admin_id, run_name, folder_preds, 0)
        for admin_id in gdf_admin.admin_id
        if not (folder_preds / f"{admin_id}.geojson").exists()
    ]
    print(f"Processing {len(args)} admin units...")
    with mp.Pool(cpu) as pool:
        pool.starmap(create_gdf_overture_with_preds, args)


@timeit
def create_all_gdf_overture_with_preds(run_name):

    gdf_admin = load_ukraine_admin_polygons(adm_level=3)
    folder_preds = PREDS_PATH / run_name / "admin_preds"
    folder_preds.mkdir(exist_ok=True, parents=True)

    for admin_id in gdf_admin.admin_id:
        if (folder_preds / f"{admin_id}.geojson").exists():
            continue
        print(f"Processing {admin_id}...")
        create_gdf_overture_with_preds(admin_id, run_name, folder_preds, verbose=1)


def create_gdf_overture_with_preds(admin_id, run_name, folder_preds, verbose):

    fp = folder_preds / f"{admin_id}.geojson"
    if fp.exists():
        print(f"{fp.name} already exists")
        return

    gdf_buildings = gpd.read_file(OVERTURE_ADMIN_PATH / f"overture_admin_{admin_id}.geojson").set_index("building_id")
    if verbose:
        print(f"{gdf_buildings.shape[0]} buildings for admin {admin_id} loaded")
    post_dates = find_post_dates(run_name)
    post_dates_ = [p[0] for p in post_dates]  # keep only first date for reference

    # Read and stack preds for each date
    fp_preds = [PREDS_PATH / run_name / f'ukraine_{"_".join(post_date)}.tif' for post_date in post_dates]
    dates = xr.Variable("date", pd.to_datetime(post_dates_))
    preds = xr.concat(
        [read_fp_within_geo(fp, box(*gdf_buildings.total_bounds)) for fp in fp_preds], dim=dates
    ).squeeze()
    if verbose:
        print(f"Preds TIF files read and stacked ({preds.shape})")

    # Vectorize pixels
    gdf_pixels = vectorize_xarray_3d(preds, post_dates_)
    if verbose:
        print(f"Pixels vectorized ({gdf_pixels.shape})")

    # Overlap with buildings
    overlap = gpd.overlay(gdf_buildings.reset_index(), gdf_pixels, how="intersection").set_index("building_id")
    if verbose:
        print(f"Overlap computed ({overlap.shape})")

    # Add area of overlap
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        overlap["polygon_area"] = overlap.area

    # Compute weighted mean for each building and date
    overlap[[f"{d}_weighted_value" for d in post_dates_]] = overlap[post_dates_].multiply(
        overlap["polygon_area"], axis=0
    )
    grps = overlap.groupby("building_id")
    gdf_weighted_mean = (
        grps[[f"{d}_weighted_value" for d in post_dates_]].sum().divide(grps["polygon_area"].sum(), axis=0)
    )
    gdf_weighted_mean = gdf_weighted_mean.stack().reset_index(level=1)
    gdf_weighted_mean.columns = ["post_date", "weighted_mean"]
    gdf_weighted_mean["post_date"] = gdf_weighted_mean["post_date"].apply(lambda x: x.split("_")[0])
    gdf_weighted_mean.set_index("post_date", append=True, inplace=True)

    # Compute max value for each building and date
    gdf_max = overlap.groupby("building_id")[post_dates_].max().stack().to_frame(name="max")
    gdf_max.index.names = ["building_id", "post_date"]

    # Merge with original buildings
    gdf_buildings_with_preds = gdf_buildings.join(gdf_weighted_mean).join(gdf_max).sort_index()
    if verbose:
        print("Weighted mean and max extracted for each building.")

    # Save to file
    gdf_buildings_with_preds.to_file(fp, driver="GeoJSON")
    print(f"Finished creating gdf with preds for admin {admin_id}")


from src.postprocessing.preds_buildings import vectorize_xarray_with_gdf


def old_create_gdf_overture_with_preds(admin_id, run_name, folder_preds, verbose=0):

    fp = folder_preds / f"{admin_id}.geojson"
    if fp.exists():
        print(f"{fp.name} already exists")
        return

    gdf_buildings = gpd.read_file(OVERTURE_ADMIN_PATH / f"overture_admin_{admin_id}.geojson")
    gdfs = []
    for post_date in find_post_dates(run_name):
        fp_preds = PREDS_PATH / run_name / f'ukraine_{"_".join(post_date)}.tif'
        assert fp_preds.exists()
        preds = read_fp_within_geo(fp_preds, box(*gdf_buildings.total_bounds))
        preds_vectorized = vectorize_xarray_with_gdf(preds, gdf_buildings, name_id="building_id", verbose=verbose)
        gdf_buildings_with_preds = gdf_buildings.merge(preds_vectorized, on="building_id")
        gdf_buildings_with_preds["post_date"] = post_date[0]  # first date only
        gdfs.append(gdf_buildings_with_preds)

    gdf_buildings_with_all_preds = pd.concat(gdfs).set_index(["building_id", "post_date"]).sort_index()
    gdf_buildings_with_all_preds.to_file(fp, driver="GeoJSON")
    print(f"Finished creating gdf with preds for admin {admin_id}")


def find_post_dates(run_name):
    local_folder = PREDS_PATH / run_name
    post_dates = []
    for file in local_folder.glob("ukraine_*.tif"):
        post_date = file.stem.split("_")[-2:]
        post_dates.append((post_date[0], post_date[1]))
    return post_dates


# ====================== 2.5 Postprocess ======================
def postprocess_preds(df, threshold=0.5, agg_method="weighted_mean"):
    """
    1. Set to 0 all buildings that were predicted as destroyed before the war

    2. Rolling max, 'a building that is destroyed stays destroyed'
    """

    df_before_war = df.loc[df.index.get_level_values("post_date") < "2022-01-01"]
    df_before_war_max = df_before_war.groupby("building_id")[agg_method].max()
    buildings_already_destroyed = df_before_war_max[df_before_war_max > 255 * threshold].index
    # print(f"Buildings already destroyed: {len(buildings_already_destroyed)}/{len(df_before_war_max)}")

    df.loc[buildings_already_destroyed, agg_method] = 0

    n_dates = len(set(df.index.get_level_values("post_date")))
    df = df.groupby("building_id")[agg_method].rolling(n_dates, min_periods=1).max().reset_index(level=0, drop=True)
    return df


# ====================== 3. Oblasts Aggregation ======================
def aggregate_preds_for_all_oblast(run_name, folder_preds_agg="oblasts_with_preds_agg", postprocess=False):

    folder = PREDS_PATH / run_name / folder_preds_agg
    folder.mkdir(exist_ok=True, parents=True)

    adm1 = load_ukraine_admin_polygons(adm_level=1)
    for i, o in enumerate(adm1.ADM1_EN.unique()):

        fp = folder / f"preds_agg_{o}.geojson"
        if fp.exists():
            print(f"Skipping {o}...")
            continue

        print(f"Processing {o} ({i+1}/{len(adm1.ADM1_EN.unique())})...")
        gdf = aggregate_all_admins(run_name, oblasts=[o], postprocess=postprocess)
        if gdf is not None:
            gdf.to_file(folder / f"preds_agg_{o}.geojson", driver="GeoJSON")
            print(f"Saved {len(gdf)//len(find_post_dates(run_name))} admin regions for {o}")
    print("Finished")


def aggregate_all_admins(run_name, oblasts=None, postprocess=True):

    gdf_admin = load_ukraine_admin_polygons(adm_level=3)
    if oblasts is not None:
        gdf_admin = gdf_admin[gdf_admin["ADM1_EN"].isin(oblasts)]

    # Only use the ones that hve been predicted
    adm_processed = [fp.stem for fp in (PREDS_PATH / run_name / "admin_preds").glob("*.geojson")]
    gdf_admin = gdf_admin[gdf_admin.admin_id.isin(adm_processed)]
    if gdf_admin.empty:
        print("No admin units to process")
        return

    print(f"Processing {len(gdf_admin)} administative units...")

    args = [(admin_id, row, postprocess) for admin_id, row in gdf_admin.set_index("admin_id").iterrows()]
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(aggregate_admin_, args)
    results = [r for r in results if r is not None]  # remove settlements without buildings
    gdf = gpd.GeoDataFrame(pd.concat(results), crs=gdf_admin.crs).reset_index().set_index(["admin_id", "post_date"])

    return gdf


def load_gdf_overture_with_preds(admin_id, run_name):

    fp = PREDS_PATH / run_name / "admin_preds" / f"{admin_id}.geojson"
    assert fp.exists(), f"{fp.name} does not exist"
    # print(f"{fp.name} does not exist, creating it")
    # create_gdf_overture_with_preds(admin_id, run_name, PREDS_PATH / run_name / "admin_preds", verbose=1)

    return gpd.read_file(fp).set_index(["building_id", "post_date"])


def aggregate_admin(admin_id, postprocess=True):

    gdf_preds = load_gdf_overture_with_preds(admin_id, run_name)
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    gdf_count = pd.DataFrame()
    for col in ["weighted_mean", "max"]:
        for t in thresholds:
            column_name = f'count_{col.split("_")[-1]}_{t:.2f}'

            gdf_preds_col = (
                postprocess_preds(gdf_preds.copy(), threshold=t, agg_method=col) if postprocess else gdf_preds[col]
            )
            counts = gdf_preds_col.groupby("post_date").apply(count_above_x, t).to_frame(name=column_name)
            # If gdf_count is empty, initialize it with the counts; otherwise, join the new counts
            if gdf_count.empty:
                gdf_count = counts
            else:
                gdf_count = gdf_count.join(counts)

    gdf_count["n_buildings"] = len(set(gdf_preds.index.get_level_values("building_id")))
    return gdf_count


def aggregate_admin_(admin_id, row, postprocess=True):

    gdf_count = aggregate_admin(admin_id, postprocess=postprocess)
    gdf_count["admin_id"] = admin_id
    gdf_count["geometry"] = row.geometry

    for k, v in row.items():
        if k.startswith("ADM"):
            gdf_count[k] = v
    return gdf_count


def count_above_x(group, threshold):
    return (group > 255 * threshold).sum()


if __name__ == "__main__":
    run_name = "240307"
    drive_to_result(run_name)
