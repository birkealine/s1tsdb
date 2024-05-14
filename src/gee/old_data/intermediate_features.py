"""Save intermediate features collections (takes forever)"""

import datetime as dt
from dateutil.relativedelta import relativedelta
import ee

from src.gee.constants import D_AOI_ORBITS, TRAIN_AOIS, TEST_AOIS, ASSETS_PATH
from src.gee.utils import init_gee
from src.gee.classification.features_extractor import manual_stats_from_s1
from src.gee.data.unosat import get_unosat_geo, get_unosat_labels
from src.gee.data.persistent_scaterrers import compute_ps, assign_ps
from src.gee.data.collections import get_s1_collection
from src.gee.data.utils import get_first_and_last_date, fill_nan_with_mean

init_gee()


def create_all_features(first_start_date="2020-06-01", last_start_date="2022-06-01", n_tiles=32, extract_window=30):
    # Generate monthly start dates
    start_date = dt.datetime.strptime(first_start_date, "%Y-%m-%d")
    last_start_date = dt.datetime.strptime(last_start_date, "%Y-%m-%d")
    start_dates = []
    while start_date <= last_start_date:
        start_dates.append(start_date.strftime("%Y-%m-%d"))
        start_date += relativedelta(months=1)

    for aoi in TRAIN_AOIS + TEST_AOIS:
        for orbit in D_AOI_ORBITS[aoi]:
            # Monthly slider
            for start_date in start_dates:
                ASSET_ID = f"s1tsdd_Ukraine/{n_tiles}d_{extract_window}m/{aoi}_orbit_{orbit}_{start_date}"
                labels_with_stats = get_labels_with_s1_stats(aoi, orbit, start_date, n_tiles, extract_window)

                # Export as big asset
                task = ee.batch.Export.table.toAsset(
                    collection=labels_with_stats,
                    description=f"AllStats__{aoi}_orbit_{orbit}_{start_date}",
                    assetId=ASSETS_PATH + ASSET_ID,
                )
                task.start()


def get_labels_with_s1_stats(aoi, orbit, start_date, n_tiles, extract_window):
    # UNOSAT labels
    labels = get_labels_prepared(aoi, orbit, all_labels=True)

    # Get S1 collection for the given aoi, orbit, start_date and n_tiles
    geo = get_unosat_geo(aoi)
    s1 = get_s1_collection(geo, start_date).filterMetadata("relativeOrbitNumber_start", "equals", orbit).limit(n_tiles)
    dates = ee.Dictionary(get_first_and_last_date(s1))
    first_date = ee.Date(dates.get("first"))
    last_date = ee.Date(dates.get("last"))
    s1 = fill_nan_with_mean(s1)

    # Extract stats (mean, std, ...)
    manual_features = manual_stats_from_s1(s1)

    def assign_features_to_label(f):
        date_of_analysis = ee.Date(f.get("date"))
        source = ee.String(f.get("source"))
        label = ee.Algorithms.If(
            source.eq("random"), 0, determine_label(date_of_analysis, first_date, last_date)
        )  # Determine label according to dates
        stats = manual_features.reduceRegion(
            reducer=ee.Reducer.first(),  # Only need the single value from each band since we've reduced the temporal dim.
            geometry=f.geometry(),
            scale=extract_window,  # 10m or 30m
        )
        return f.set(stats).set("label", label).set("startDate", start_date)

    labels_with_stats = labels.map(lambda f: assign_features_to_label(f))
    return labels_with_stats


def determine_label(date_of_analysis, first_date_ts, last_date_ts):
    # Define the date of invasion
    date_of_invasion = ee.Date("2022-02-24")

    # Determine the label based on the provided logic
    label = ee.Algorithms.If(
        last_date_ts.millis().lt(date_of_invasion.millis()),
        0,  # Not destroyed
        ee.Algorithms.If(
            last_date_ts.millis()
            .gt(date_of_analysis.millis())
            .And(first_date_ts.millis().lt(date_of_invasion.millis())),
            1,  # Destroyed
            -1,  # Unknown
        ),
    )
    return label


def get_labels_prepared(aoi, orbit, all_labels=False):
    # Get labels for the area of interest (AOI)
    geo = get_unosat_geo(aoi)
    labels = get_unosat_labels(aoi, all_labels)
    labels = labels.map(lambda f: f.set("source", "unosat"))

    # Get random locations to use as negative samples
    random_points = ee.FeatureCollection(ASSETS_PATH + "randomPoints/" + aoi)
    random_points = random_points.map(lambda f: f.set("source", "random"))

    # Combine UNOSAT and random labels
    labels = ee.FeatureCollection(labels.toList(labels.size()).cat(random_points.toList(random_points.size())))

    # Assign bins and PS (Persistent Scatterers)
    labels = assign_bins(labels, geo)
    ps_mask = compute_ps(aoi, orbit)
    labels = assign_ps(labels, ps_mask)

    # Assign additional parameters
    labels = labels.map(lambda f: f.set("aoi", aoi).set("orbit", orbit))
    return labels


def assign_bins(points, geo, n_bins=5):
    # Get coordinates and calculate bin size
    coords = ee.List(geo.bounds().coordinates().get(0))
    min_lon = ee.Number(ee.List(coords.get(0)).get(0))
    max_lon = ee.Number(ee.List(coords.get(1)).get(0))

    delta = max_lon.subtract(min_lon)
    bin_size = delta.divide(n_bins)

    # Function to assign a bin based on longitude
    def assign_bin(feature):
        longitude = ee.Number(feature.geometry().coordinates().get(0))
        bin = (
            longitude.subtract(min_lon).divide(bin_size).floor().add(1).min(n_bins)
        )  # Ensure the maximum value is n_bins
        return feature.set("bin", bin)

    binned_points = points.map(assign_bin)
    return binned_points
