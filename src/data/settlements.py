import geopandas as gpd
import multiprocessing as mp
import pandas as pd
from rasterstats import zonal_stats
import rioxarray as rxr
from rioxarray.merge import merge_arrays
from shapely.geometry import box

from src.constants import PROCESSED_PATH, EXTERNAL_PATH, RAW_PATH
from src.data.buildings.microsoft import load_buildings_geo
from src.utils.geometry import load_country_boundaries, reproject_geo

SETTLEMENTS_PATH = PROCESSED_PATH / "ukraine_settlements.geojson"

MSFT_SETTLEMENTS_PATH = PROCESSED_PATH / "msft_settlements"
MSFT_SETTLEMENTS_PATH.mkdir(exist_ok=True, parents=True)


def load_gdf_settlements():

    if not SETTLEMENTS_PATH.exists():
        create_gdf_settlements()

    gdf = gpd.read_file(SETTLEMENTS_PATH)
    gdf.index.name = "settlement_id"
    return gdf


def create_gdf_settlements(threshold_urban=0.1, min_smod_level=13):
    """See notebook settlements.ipynb for details."""

    # Load Ukraine regions (level 4 is the finest one)
    ukraine_regions_path = sorted((EXTERNAL_PATH / "UKR_admin_boundaries").glob("*_adm4*.shp"))[0]
    ukr_regions = gpd.read_file(ukraine_regions_path)[["ADM4_EN", "ADM3_EN", "ADM2_EN", "ADM1_EN", "geometry"]]

    # Load settlements model (SMOD) from GSHL and binarize it
    GSHL_SMOD_PATH = RAW_PATH / "ghsl_smod"
    geo_ukr = load_country_boundaries("Ukraine")
    smod_paths = sorted(GSHL_SMOD_PATH.glob("*.tif"))
    smods = [rxr.open_rasterio(p) for p in smod_paths]
    geo_ukr_reproj = reproject_geo(geo_ukr, "EPSG:4326", smods[0].rio.crs)
    smods_merged = merge_arrays(smods)
    smod_ukr = smods_merged.rio.clip([geo_ukr_reproj])
    smod_bin = smod_ukr.where((smod_ukr >= min_smod_level) | (smod_ukr == smod_ukr.rio.nodata), 0).where(
        smod_ukr < min_smod_level, 1
    )

    # Add percentage of build-up area
    df = pd.DataFrame(
        zonal_stats(
            vectors=ukr_regions.to_crs(smod_bin.rio.crs),
            raster=smod_bin.squeeze().values,
            affine=smod_bin.rio.transform(),
            stats=["mean"],
            nodata=smod_bin.rio.nodata,
        )
    )
    ukr_regions["perc_build_up"] = df["mean"]
    print("Settlements data added.")

    # add population data
    ukr_pop = rxr.open_rasterio(EXTERNAL_PATH / "ukr_population_density_2020_1km.tif")
    df = pd.DataFrame(
        zonal_stats(
            vectors=ukr_regions.to_crs(ukr_pop.rio.crs),
            raster=ukr_pop.squeeze().values,
            affine=ukr_pop.rio.transform(),
            stats=["sum"],
            nodata=ukr_pop.rio.nodata,
        ),
    )
    ukr_regions["population"] = df["sum"]
    print("Population data added.")

    # Filter settlements (only those with more than 10% of build-up area)
    poly_settlements = ukr_regions[ukr_regions["perc_build_up"] >= threshold_urban].copy()
    poly_settlements["geometry_box"] = poly_settlements.geometry.apply(lambda g: box(*g.bounds))
    poly_settlements.drop("geometry_box", axis=1).to_file(SETTLEMENTS_PATH, driver="GeoJSON")
    print("Settlements saved.")


def _save_buildings_settlements(args):

    index, geometry, folder = args
    fp = folder / f"{index}.geojson"
    if not fp.exists():
        buildings = load_buildings_geo(geometry)
        buildings.to_file(fp, driver="GeoJSON")
        print(f"Saved {fp.name}")
    else:
        print(f"File {fp.name} already exists")


def create_all_buildings_settlements():
    """
    Create a GeoJSON file for each settlement with the Microsoft buildings within it.
    """
    ukr_settlements = load_gdf_settlements()
    args = [(i, row.geometry, MSFT_SETTLEMENTS_PATH) for i, row in ukr_settlements.iterrows()]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(_save_buildings_settlements, args)


if __name__ == "__main__":
    create_all_buildings_settlements()
