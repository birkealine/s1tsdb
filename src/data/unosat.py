import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from typing import List, Union
from src.constants import PROCESSED_PATH


def load_unosat_labels(
    aoi: Union[str, List[str]] = None,
    labels_to_keep: List[int] = [1, 2],
    country: Union[str, List[str]] = "Ukraine",
    combine_epoch: bool = "last",
) -> gpd.GeoDataFrame:
    """
    Load UNOSAT labels processed.

    Args:
        aoi (Union[str, List[str]]): The AOI(s) to load. If None, loads everything
        labels_to_keep (List[int]): Which labels to keep. Default to [1,2]
        country (Union[str, List[str]]): The country/ies to load. If None, loads everything
        combine_epoch (bool): For points that have multiple observations, we keep only one label.
            Either the 'last' one or the 'min' one (eg the strongest label). Default to 'last'

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with all UNOSAT labels
    """

    labels_fp = PROCESSED_PATH / "unosat_labels.feather"
    assert labels_fp.exists(), "The GeoDataFrame has not been created yet."

    gdf = gpd.read_feather(labels_fp).reset_index(drop=True)

    if "bin" not in gdf.columns:
        gdf = assign_bins_to_labels(gdf)
        # gdf.to_feather(labels_fp)
        # print("Saved with new column bin")

    if combine_epoch is not None:
        if combine_epoch == "last":
            # Only keep most recent epoch for each point
            gdf = gdf.loc[gdf.groupby(gdf.geometry.to_wkt())["ep"].idxmax()]
        elif combine_epoch == "min":
            # Only keep strongest label for each point
            gdf = gdf.loc[gdf.groupby(gdf.geometry.to_wkt())["damage"].idxmin()]
        else:
            raise ValueError("combine_epoch must be 'last' or 'min'")

    if labels_to_keep is not None:
        # Only keep some labels
        gdf = gdf[gdf.damage.isin(labels_to_keep)]

    if country is not None:
        # Only keep some countries
        country = [country] if isinstance(country, str) else country
        gdf = gdf[gdf.country.isin(country)]

    if aoi is not None:
        # Only keep some AOIs
        aoi = [aoi] if isinstance(aoi, str) else aoi
        gdf = gdf[gdf.aoi.isin(aoi)]

    return gdf


def load_unosat_aois(
    force_recreate: bool = False, add_country: bool = False, only_ukraine: bool = True
) -> gpd.GeoDataFrame:
    """
    Load GeoDataFrame with all AOIs

    Args:
        force_recreate (bool): If True, recreates the dataframe from the shapefiles
        add_country (bool): If True, adds a column with the country name. Default to False
        only_ukraine (bool): If True, only keeps AOIs in Ukraine. Default to True

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with column 'aoi' and 'geometry'
    """

    aoi_fp = PROCESSED_PATH / "unosat_aoi.feather"

    if not aoi_fp.exists() or force_recreate:
        raise NotImplementedError("Not implemented in this repo")
    else:
        gdf_aoi = gpd.read_feather(aoi_fp)

    if only_ukraine:
        gdf_aoi = gdf_aoi[gdf_aoi.aoi.str.startswith("UKR")]

    if add_country:
        d_code_to_country = {"IRQ": "Iraq", "PSE": "Palestine", "SYR": "Syria", "UKR": "Ukraine"}
        gdf_aoi["country"] = gdf_aoi.apply(lambda row: d_code_to_country[row.aoi[:3]], axis=1)

    return gdf_aoi


def get_unosat_geometry(aoi: str) -> Polygon:
    """Load AOI geometry"""
    aois = load_unosat_aois().set_index("aoi")
    geo = aois.loc[aoi].geometry
    return geo


def assign_bins_to_labels(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    # Keep track of labels
    gdf.index.name = "unosat_id"
    gdf.reset_index(inplace=True)
    aois = load_unosat_aois().set_index("aoi")

    dfs = []
    for aoi, row in gdf.groupby("aoi"):
        geo = aois.loc[aoi, "geometry"]
        row = assign_bins(row, geo)
        row["aoi"] = aoi
        dfs.append(row)
    points_with_bins = pd.concat(dfs, ignore_index=True).set_index("unosat_id")
    return points_with_bins


def assign_bins(gdf, geo):
    """Assign bins according to the geometry of the AOI."""
    xmin, _, xmax, _ = geo.bounds
    bin_width = (xmax - xmin) / 5

    def _assign_bin(point):
        bin_index = int((point.x - xmin) / bin_width) + 1  # (from 1 to 5)
        return bin_index

    gdf["bin"] = gdf.geometry.apply(_assign_bin)
    return gdf
