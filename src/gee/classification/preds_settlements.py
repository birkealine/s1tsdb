import ee
from omegaconf import OmegaConf

from src.gee.constants import ASSETS_PATH
from src.gee.utils import init_gee
from src.gee.classification.utils import sklearn_to_gee_kwds
from src.utils.gdrive import create_drive_folder, create_yaml_file_in_drive_from_config_dict

init_gee()

FEATURES_NAMES = [
    "VH_pre_3x3_stdDev",
    "VV_pre_3x3_min",
    "VV_post_3x3_stdDev",
    "VH_post_3x3_min",
    "VH_post_3x3_stdDev",
    "VH_pre_3x3_max",
    "VV_pre_3x3_max",
    "VH_post_3x3_kurtosis",
    "VH_pre_3x3_median",
    "VH_pre_3x3_mean",
    "VH_pre_3x3_min",
    "VH_pre_3x3_skew",
    "VV_pre_3x3_median",
    "VV_pre_3x3_kurtosis",
    "VH_post_3x3_median",
    "VV_post_3x3_min",
    "VH_post_3x3_max",
    "VH_post_3x3_mean",
    "VH_post_3x3_skew",
    "VV_post_3x3_median",
    "VV_pre_3x3_mean",
    "VV_pre_3x3_skew",
    "VV_post_3x3_max",
    "VV_post_3x3_mean",
    "VV_post_3x3_kurtosis",
    "VH_pre_3x3_kurtosis",
    "VV_pre_3x3_stdDev",
    "VV_post_3x3_skew",
]


def get_classifier_trained(cfg, verbose=1):

    # Load precomputed features
    run_name_ = cfg.run_name.split("_")[0]
    fc = ee.FeatureCollection(ASSETS_PATH + f"s1tsdd_Ukraine/{run_name_}_features_ready_train")
    if cfg.data.train_on_all:
        fc_test = ee.FeatureCollection(ASSETS_PATH + f"s1tsdd_Ukraine/{run_name_}_features_ready_test")
        if verbose:
            print("Training with all data")
        fc = ee.FeatureCollection([fc, fc_test]).flatten()

    # Train classifier
    classifier = ee.Classifier.smileRandomForest(**sklearn_to_gee_kwds(cfg.model_kwargs, verbose=verbose))
    classifier = classifier.setOutputMode("CLASSIFICATION")
    trained_classifier = classifier.train(features=fc, classProperty="label", inputProperties=FEATURES_NAMES)
    return trained_classifier


def get_s1(geo, orbit=None):
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("platform_number", "A"))
        .filterBounds(geo)
        .select(["VV", "VH"])
    )

    if orbit is not None:
        s1 = s1.filterMetadata("relativeOrbitNumber_start", "equals", orbit)
    return s1


ORIGINAL_NAMES = [
    "VV_mean",
    "VV_stdDev",
    "VV_median",
    "VV_max",
    "VV_min",
    "VV_skew",
    "VV_kurtosis",
    "VH_mean",
    "VH_stdDev",
    "VH_median",
    "VH_max",
    "VH_min",
    "VH_skew",
    "VH_kurtosis",
]


def get_new_names(bands, prefix):
    new_bands = []
    for b in bands:
        b_, stat = b.split("_")
        new_bands.append(f"{b_}_{prefix}_{stat}")
    return new_bands


def convolve_mean(img, radius):
    return img.focalMean(radius, "square", units="meters").set("system:time_start", img.get("system:time_start"))


def infer_geo(geo, cfg, stats_reducers, orbits=None, verbose=0):

    classifier = get_classifier_trained(cfg, verbose=verbose)
    classifier = classifier.setOutputMode("PROBABILITY")

    s1 = get_s1(geo)
    if orbits is None:
        # Choose orbits that at least 5 images per time period
        list_orbits = []
        for _, (start, end) in cfg.data.time_periods.items():
            s1_ = s1.filterDate(start, end)
            orbits_counts = s1_.aggregate_histogram("relativeOrbitNumber_start")
            # At least 5 images per orbit (two months of data)
            orbits_counts = orbits_counts.map(lambda k, v: ee.Algorithms.If(ee.Number(v).gte(5), k, None))
            orbits_inference = orbits_counts.keys().map(lambda k: ee.Number.parse(k))  # cast keys back to number
            list_orbits.append(orbits_inference)
        orbits_inference = list_orbits[0].filter(ee.Filter.inList("item", list_orbits[1]))
        if verbose:
            print(f"Orbits to infer: {orbits_inference.getInfo()}")

    def infer_s1_orbit(orbit):

        s1_orbit = s1.filterMetadata("relativeOrbitNumber_start", "equals", orbit)
        s1_features = None
        for window in cfg.data.extract_winds:

            if int(window[0]) > 1:
                # convolve with a focal mean
                radius = int(window[0]) // 2

                def _convolve_mean(img):
                    return img.focalMean(radius, "square", units="meters").set(
                        "system:time_start", img.get("system:time_start")
                    )

                s1_orbit = s1_orbit.map(_convolve_mean)

            for name_period, (start, end) in cfg.data.time_periods.items():

                s1_dates = s1_orbit.filterDate(start, end)
                prefix = f"{name_period}_{window}"

                _s1_features = s1_dates.reduce(stats_reducers)
                _s1_features = _s1_features.select(ORIGINAL_NAMES, get_new_names(ORIGINAL_NAMES, prefix))
                s1_features = _s1_features if s1_features is None else s1_features.addBands(_s1_features)

        return s1_features.classify(classifier)

    return ee.ImageCollection(orbits_inference.map(infer_s1_orbit)).mean()


def preds_and_export(cfg, stats_reducers, folder, n_limit=None):

    # Get all settlements
    settlements = ee.FeatureCollection(ASSETS_PATH + "s1tsdd_Ukraine/ukraine_settlements")
    if n_limit:
        settlements = settlements.limit(n_limit)

    ids = settlements.aggregate_array("settlement_id").getInfo()
    # only settlements that do not exist yet
    from src.constants import PREDS_PATH

    folder_already_exist = PREDS_PATH / "240224" / "2021-02-24_2022-02-23" / "settlements_preds"
    ids_ = [i for i in ids if not (folder_already_exist / f"settlement_{i}.tif").exists()]
    print(f"{len(ids_)}/{len(ids)}")

    for id_ in ids_:

        settlement = settlements.filter(ee.Filter.eq("settlement_id", id_))

        geo = settlement.geometry()
        preds = infer_geo(geo, cfg, stats_reducers, orbits=None)
        preds = preds.set("settlement_id", id_)

        name = f"settlement_{id_}"
        description = f"Ukraine_settlement_{id_}_{'_'.join(cfg.data.time_periods.post)}"
        task = ee.batch.Export.image.toDrive(
            image=preds.multiply(2**8 - 1).toUint8(),  # multiply by 255 and convert to uint8
            description=description,
            folder=folder,
            fileNamePrefix=name,
            region=settlement.first().geometry(),
            scale=10,
        )
        task.start()
        if id_ % 5 == 0:
            print(f"Exporting id_ {id_}.")


if __name__ == "__main__":

    import datetime as dt

    cfg = OmegaConf.create(
        dict(
            aggregation_method="mean",
            model_name="random_forest",
            model_kwargs=dict(
                n_estimators=100,  # 200,
                # min_samples_leaf=2,
                n_jobs=12,
            ),
            data=dict(
                aois_test=[f"UKR{i}" for i in range(1, 19) if i not in [1, 2, 3, 4]],
                train_on_all=True,
                damages_to_keep=[1, 2],
                extract_winds=["3x3"],  # ['1x1', '3x3', '5x5']
                # random_neg_labels=0.1,  # percentage of negative labels to add in training set (eg 0.1 for 10%)
                time_periods=None,
            ),
            seed=123,
            run_name=None,
        )
    )

    stats_reducers = (
        ee.Reducer.mean()
        .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.median(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.max(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.min(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.skew(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.kurtosis(), sharedInputs=True)
    )

    # folder_name = "Prediction_All_Settlements_OneYearBefore_1202024"

    # dict(pre=("2020-02-24", "2021-02-23"), post=("2021-02-24", "2022-02-23"))
    # dict(pre=("2020-02-24", "2021-02-23"), post=("2023-02-24", "2024-02-23"))
    for time_periods in [dict(pre=("2020-02-24", "2021-02-23"), post=("2021-02-24", "2022-02-23"))]:

        cfg.data.time_periods = time_periods
        cfg.run_name = "240224_2021-02-24_2022-02-23"
        folder = f"{cfg.run_name}_settlements_predictions"

        # create folder in Drive (raises error if already exists)
        # create_drive_folder(folder)
        # create_yaml_file_in_drive_from_config_dict(cfg, folder)
        preds_and_export(cfg, stats_reducers=stats_reducers, folder=folder, n_limit=None)
