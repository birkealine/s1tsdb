import ee
import geemap
from omegaconf import OmegaConf

from src.gee.constants import ASSETS_PATH
from src.gee.data.datasets import load_dataset
from src.gee.classification.features_extractor import manual_stats_from_s1
from src.gee.utils import init_gee

init_gee()

FEATURES_NAMES = [
    "VV_mean",
    "VV_stdDev",
    "VV_median",
    "VV_max",
    "VV_min",
    "VV_skew",
    "VV_kurtosis",
    "VV_variance",
    "VH_mean",
    "VH_stdDev",
    "VH_median",
    "VH_max",
    "VH_min",
    "VH_skew",
    "VH_kurtosis",
    "VH_variance",
    "VV_ptp",
    "VH_ptp",
    "VV_mean_slice0",
    "VV_stdDev_slice0",
    "VV_mean_slice1",
    "VV_stdDev_slice1",
    "VV_mean_slice2",
    "VV_stdDev_slice2",
    "VV_mean_slice3",
    "VV_stdDev_slice3",
    "VV_mean_slice4",
    "VV_stdDev_slice4",
    "VV_mean_slice5",
    "VV_stdDev_slice5",
    "VV_mean_slice6",
    "VV_stdDev_slice6",
    "VV_mean_slice7",
    "VV_stdDev_slice7",
    "VH_mean_slice0",
    "VH_stdDev_slice0",
    "VH_mean_slice1",
    "VH_stdDev_slice1",
    "VH_mean_slice2",
    "VH_stdDev_slice2",
    "VH_mean_slice3",
    "VH_stdDev_slice3",
    "VH_mean_slice4",
    "VH_stdDev_slice4",
    "VH_mean_slice5",
    "VH_stdDev_slice5",
    "VH_mean_slice6",
    "VH_stdDev_slice6",
    "VH_mean_slice7",
    "VH_stdDev_slice7",
]


def preds_and_export(start_dates_training, random_loc, n_tiles, start_date_inference, folder, n_limit=None):

    settlements = ee.FeatureCollection(ASSETS_PATH + "s1tsdd_Ukraine/ukraine_settlements")
    if n_limit:
        settlements = settlements.limit(n_limit)

    for id_ in settlements.aggregate_array("settlement_id").getInfo():

        settlement = settlements.filterMetadata("settlement_id", "equals", id_)
        pred = infer_settlements(settlement, start_dates_training, random_loc, n_tiles, start_date_inference)
        name = f"settlement_{id_}"
        description = f"Ukraine_settlement_{id_}_{start_date_inference}_2dates_32d"
        task = ee.batch.Export.image.toDrive(
            image=pred.multiply(2**8 - 1).toUint8(),  # multiply by 255 and convert to uint8
            description=description,
            folder=folder,
            fileNamePrefix=name,
            region=settlement.first().geometry(),
            scale=10,
        )
        task.start()
        if id_ % 5 == 0:
            print(f"Exporting id_ {id_}.")


def infer_settlements(settlement, start_dates_training, random_loc, n_tiles, start_date_inference):
    geo = settlement.geometry()
    preds = preds_full_pipeline(
        start_dates_training, random_loc, n_tiles, start_date_inference, geo, orbits_inference=None, verbose=0
    )
    preds = preds.set("settlement_id", settlement.get("settlement_id"))
    return preds


# def infer_for_all_settlements(settlements, start_dates_training, random_loc, n_tiles, start_date_inference):
#     def _infer_one_settlements(f):
#         geo = f.geometry()
#         preds = preds_full_pipeline(
#             start_dates_training, random_loc, n_tiles, start_date_inference, geo, orbits_inference=None, verbose=0
#         )
#         preds = preds.set("settlement_id", f.get("settlement_id"))
#         return preds

#     preds = settlements.map(_infer_one_settlements)
#     return preds


def preds_full_pipeline(
    start_dates_training, random_loc, n_tiles, start_date_inference, geo_inference, orbits_inference=None, verbose=1
):
    # training dataset
    cfg_train = OmegaConf.create(
        dict(
            split="train",
            fold=None,
            random_loc=random_loc,
            keep_damage=[1, 2],
            n_tiles=n_tiles,
            extract_window=30,
            start_dates=start_dates_training,
            save_if_doesnt_exist=True,
            verbose=verbose,
        )
    )
    ds_train = load_dataset(**cfg_train)
    if verbose:
        print(f"start_dates_training: {start_dates_training} - random_loc: {random_loc} - n_tiles: {n_tiles}")

    # train classifier
    classifier = ee.Classifier.smileRandomForest(50)

    trained_clf = classifier.train(features=ds_train, classProperty="label", inputProperties=ee.List(FEATURES_NAMES))
    if verbose:
        print("Classifier trained.")

    # Sentinel-1 data
    start_date_ee = ee.Date(start_date_inference)
    end_date_ee = start_date_ee.advance(12 * n_tiles, "day")
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("platform_number", "A"))
        .filterDate(start_date_ee, end_date_ee)
        .filterBounds(geo_inference)
        .select(["VV", "VH"])
    )
    if verbose:
        print(f"Sentinel-1 data loaded from {start_date_inference}")

    # Inference for each orbit and mean
    trained_clf = trained_clf.setOutputMode("PROBABILITY")

    def infer_orbit(orbit):
        s1_orbit = s1.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))
        stats_orbit = manual_stats_from_s1(s1_orbit, start_date_inference)
        preds_proba_orbit = stats_orbit.classify(trained_clf)
        return preds_proba_orbit

    if orbits_inference is None:
        orbits_counts = s1.aggregate_histogram("relativeOrbitNumber_start")
        # only keep orbits with all the images (-2 to not penalize orbits with a couple of missing images)
        orbits_counts = orbits_counts.map(lambda k, v: ee.Algorithms.If(ee.Number(v).gte(n_tiles - 2), k, None))
        orbits_inference = orbits_counts.keys().map(lambda k: ee.Number.parse(k))  # cast keys back to number
        if verbose:
            print(f"Orbits to infer: {orbits_inference.getInfo()}")

    preds = ee.ImageCollection(ee.List(orbits_inference).map(infer_orbit)).mean()
    if verbose:
        print("Inference done.")
    return preds


if __name__ == "__main__":
    cfg_inference = {
        "start_dates_training": ["2020-10-01", "2021-10-01"],
        "random_loc": 0,
        "n_tiles": 32,
        "start_date_inference": "2021-10-01",
    }

    folder_name = f"settlements_preds_{cfg_inference['start_date_inference']}_2dates_32d"
    preds_and_export(**cfg_inference, folder=folder_name, n_limit=None)
