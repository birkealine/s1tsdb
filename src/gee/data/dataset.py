import ee
from typing import List, Tuple

from src.gee.classification.reducers import get_reducers
from src.gee.constants import ASSETS_PATH
from src.gee.utils import fc_to_list
from src.gee.data.utils import asset_exists
from src.gee.data.unosat import get_unosat_labels
from src.data.utils import get_all_aois, aoi_orbit_iterator
from src.constants import UKRAINE_WAR_START


def get_dataset_ready(run_name, split="train", post_dates: List[Tuple[str, str]] = None):

    if run_name < "240225":
        # Asset paths were saved without post_dates
        asset_path = ASSETS_PATH + f"s1tsdd_Ukraine/{run_name}/features_ready_{split}"
        if not asset_exists(asset_path):
            raise ValueError(f"Asset {asset_path} does not exist. Run create_datasets_ready functions first.")
        fc = ee.FeatureCollection(asset_path)

    elif run_name < "240301":
        # datasets were saved by post_dates
        assert post_dates is not None, "post_dates should be provided for models trained after 240225"

        fcs = []
        for post in post_dates:
            asset_path = ASSETS_PATH + f"s1tsdd_Ukraine/{run_name}/features_ready_{split}_{'_'.join(post)}"
            if not asset_exists(asset_path):
                raise ValueError(f"Asset {asset_path} does not exist. Run create_datasets_ready functions first.")
            fcs.append(ee.FeatureCollection(asset_path))
        fc = ee.FeatureCollection(fcs).flatten()
    else:
        # datasets were saved either by 3 months or by 1year
        period = "1year" if len(post_dates) == 2 else "3months"
        asset_path = ASSETS_PATH + f"s1tsdd_Ukraine/{run_name}/features_ready_{split}_{period}"
        fc = ee.FeatureCollection(asset_path)
    return fc


def create_dataset_ready_all_dates_from_cfg(cfg, split, export=False):
    if split == "train":
        aois = [a for a in get_all_aois() if a not in cfg.data.aois_test]
    else:
        aois = cfg.data.aois_test

    return create_dataset_ready_all_dates(
        run_name=cfg.run_name,
        split=split,
        aois=aois,
        damages_to_keep=cfg.data.damages_to_keep,
        d_periods=cfg.data.time_periods,
        extract_winds=cfg.data.extract_winds,
        reducer_names=cfg.reducer_names,
        export=export,
    )


def create_dataset_ready_all_dates(
    run_name, split, aois, damages_to_keep, d_periods, extract_winds, reducer_names, export=False
):

    fs = []
    for pre in d_periods["pre"]:
        for post in d_periods["post"]:
            d_periods_ = dict(pre=pre, post=post)
            fs += create_dataset(aois, damages_to_keep, d_periods_, extract_winds, reducer_names)
    fc = ee.FeatureCollection(fs).flatten()

    n_post_dates = len(d_periods["post"])
    task = ee.batch.Export.table.toAsset(
        collection=fc,
        description=f"{run_name} {split} data all {n_post_dates} dates",
        assetId=ASSETS_PATH + f"s1tsdd_Ukraine/{run_name}/features_ready_{split}_all_{n_post_dates}_dates",
    )
    if export:
        task.start()
        print(f"Exporting {run_name} {split} data all {n_post_dates} dates")

    return fc


def create_dataset_ready_one_per_date_from_cfg(cfg, split, export=False):
    if split == "train":
        aois = [a for a in get_all_aois() if a not in cfg.data.aois_test]
    else:
        aois = cfg.data.aois_test

    return create_dataset_ready_one_per_date(
        run_name=cfg.run_name,
        split=split,
        aois=aois,
        damages_to_keep=cfg.data.damages_to_keep,
        d_periods=cfg.data.time_periods,
        extract_winds=cfg.data.extract_winds,
        reducer_names=cfg.reducer_names,
        export=export,
    )


def create_dataset_ready_one_per_date(
    run_name, split, aois, damages_to_keep, d_periods, extract_winds, reducer_names, export=False
):
    d = {}
    for pre in d_periods["pre"]:
        for post in d_periods["post"]:
            d_periods_ = dict(pre=pre, post=post)
            fs = create_dataset(aois, damages_to_keep, d_periods_, extract_winds, reducer_names)
            fc = ee.FeatureCollection(fs).flatten()
            task = ee.batch.Export.table.toAsset(
                collection=fc,
                description=f"{run_name} {split} data {'_'.join(post)}",
                assetId=ASSETS_PATH + f"s1tsdd_Ukraine/{run_name}/features_ready_{split}_{'_'.join(post)}",
            )
            if export:
                task.start()
                print(f"Exporting {run_name} {split} data {'_'.join(post)}")

            d["_".join(post)] = fc
    return d


def create_dataset(aois, damages_to_keep, d_periods, extract_winds, reducer_names):

    # Function to extract features (mean, std, ...) from the collection
    reducer_names = list(reducer_names)  # GEE does not like ListConfig
    reducer = get_reducers(reducer_names)

    end_post_period = d_periods["post"][1]
    if end_post_period <= UKRAINE_WAR_START:
        label = 0
    else:
        label = 1  # but need extra filtering to remove unknown ones

    features = []
    for aoi, orbit in aoi_orbit_iterator():
        if aoi not in aois:
            continue

        points = get_unosat_labels(aoi, True)
        points = points.filter(ee.Filter.inList("damage", list(damages_to_keep)))
        points = points.map(
            lambda f: f.set(
                {
                    "label": label,
                    "aoi": aoi,
                    "orbit": orbit,
                    "start_pre": d_periods["pre"][0],
                    "end_pre": d_periods["pre"][1],
                    "start_post": d_periods["post"][0],
                    "end_post": d_periods["post"][1],
                }
            )
        )
        if label == 1:
            # Only keep rows for which we know the label for sure
            # (the analysis was done before the end of the post period
            points = points.filter(ee.Filter.lte("date", end_post_period))

        for window in extract_winds:

            fc = get_fc_ts(aoi, orbit, window, damages_to_keep)
            for name_period, (start, end) in d_periods.items():

                fc_dates = fc.filterDate(start, end)
                prefix = f"{name_period}_{window}"

                def extract_features_per_point(point):
                    point = ee.Feature(point)
                    stats_vv = fc_dates.filter(ee.Filter.eq("unosat_id", point.get("unosat_id"))).reduceColumns(
                        reducer, ["VV"]
                    )
                    stats_vv = stats_vv.rename(reducer_names, [f"VV_{prefix}_{c}" for c in reducer_names])
                    stats_vh = fc_dates.filter(ee.Filter.eq("unosat_id", point.get("unosat_id"))).reduceColumns(
                        reducer, ["VH"]
                    )
                    stats_vh = stats_vh.rename(reducer_names, [f"VH_{prefix}_{c}" for c in reducer_names])
                    point = point.set(stats_vv).set(stats_vh)
                    return point

                points = points.map(extract_features_per_point)
        features.append(points)
    return features


def get_fc_ts(aoi, orbit, extract, damages_to_keep=[1, 2]):
    """Load precomputed features for training or testing given the aoi and orbit."""

    if any([d not in [1, 2, 3] for d in damages_to_keep]):
        raise ValueError("damages_to_keep should be in [1,2,3], other values not precomputed yet.")

    fc_path = ASSETS_PATH + f"s1tsdd_Ukraine/ts{extract}_20200301_20230301/{aoi}_orbit{orbit}"  # label 1 and 2
    fc = ee.FeatureCollection(fc_path)
    if 3 in damages_to_keep:
        fc3_path = ASSETS_PATH + f"s1tsdd_Ukraine/ts{extract}_20200301_20230301/{aoi}_orbit{orbit}_label_3"
        fc3 = ee.FeatureCollection(fc3_path)
        fc = ee.FeatureCollection(fc_to_list(fc).cat(fc_to_list(fc3)))
    return fc


if __name__ == "__main__":

    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        dict(
            aggregation_method="mean",
            model_name="random_forest",
            model_kwargs=dict(
                n_estimators=200,
                min_samples_leaf=2,
                n_jobs=12,
            ),
            data=dict(
                aois_test=[f"UKR{i}" for i in range(1, 19) if i not in [1, 2, 3, 4]],
                damages_to_keep=[1, 2],
                extract_winds=["3x3"],  # ['1x1', '3x3', '5x5']
                time_periods={
                    "pre": [("2020-02-24", "2021-02-23")],  # always only one
                    "post": [("2021-02-24", "2022-02-23"), ("2022-02-24", "2023-02-23")],
                    # "post": [
                    #     ("2021-02-24", "2021-05-23"),
                    #     ("2021-05-24", "2021-08-23"),
                    #     ("2021-08-24", "2021-11-23"),
                    #     ("2021-11-24", "2022-02-23"),
                    #     ("2022-02-24", "2022-05-23"),
                    #     ("2022-05-24", "2022-08-23"),
                    #     ("2022-08-24", "2022-11-23"),
                    #     ("2022-11-24", "2023-02-23"),
                    # ],
                },
                time_periods_inference=None,
                # time_periods_inference={
                #     'pre': ("2020-02-24", "2021-02-23"),
                #     'post': [("2022-02-24", "2023-02-23")] # need to be a list
                # }
            ),
            reducer_names=["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"],
            train_on_all=False,  # train on all damages (train + test split)
            verbose=1,
            seed=123,
            run_name="240229",
        )
    )

    create_dataset_ready_one_per_date_from_cfg(cfg, "train", export=True)
    create_dataset_ready_one_per_date_from_cfg(cfg, "test", export=True)
