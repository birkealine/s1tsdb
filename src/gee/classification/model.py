"""Script to run the classification pipeline on GEE"""

import ee

from src.utils.gee import init_gee
from src.gee.data.utils import asset_exists
from src.gee.classification.utils import sklearn_to_gee_kwds
from src.gee.data.dataset import get_dataset_ready
from src.gee.constants import ASSETS_PATH

init_gee()


def get_classifier_trained(cfg, verbose=1, train_once=True, seed=0):

    if cfg.model_name == "random_forest":
        classifier = ee.Classifier.smileRandomForest(**cfg.model_kwargs, seed=seed)
    else:
        raise NotImplementedError(f"Model {cfg.model_name} not implemented.")
    classifier = classifier.setOutputMode("CLASSIFICATION")

    # Get dataset (run_name is the model name (only first part if it contains _)
    run_name = cfg.run_name.split("_")[0]

    classifier_trained = train_classifier(classifier, run_name, cfg, "train", verbose=verbose, train_once=train_once)

    # If train_on_all, we also train with the test set
    if cfg.train_on_all:
        if verbose:
            print("Training on all")
        classifier_trained = train_classifier(
            classifier_trained, run_name, cfg, "test", verbose=verbose, train_once=train_once
        )

    return classifier_trained


def train_classifier(
    classifier,
    run_name,
    cfg,
    split,
    verbose=1,
    train_once=True,  # not sure what is the most efficient way to train the classifier
):
    """Train the classifier on the dataset"""

    # features_names = get_features_names(cfg)  # name of features to train on
    # new_features_names = [str(i) for i in range(len(features_names))]  # make classifier smaller

    if run_name < "240225":
        fc = get_dataset_ready(run_name, split=split)
        # fc = fc.select(features_names + ["label"], new_features_names + ["label"])
        # classifier = classifier.train(features=fc, classProperty="label", inputProperties=new_features_names)
        classifier = classifier.train(features=fc, classProperty="label", inputProperties=get_features_names(cfg))
        if verbose:
            print(f"Trained on {run_name} ({split} set) (size = {fc.size().getInfo()}).")
    elif run_name < "240301":

        if train_once:
            # Option 1: Train on all post dates at once
            fc = get_dataset_ready(run_name, split=split, post_dates=cfg.data.time_periods.post)
            # fc = fc.select(features_names + ["label"], new_features_names + ["label"])
            # classifier = classifier.train(fc, "label", new_features_names)
            classifier = classifier.train(features=fc, classProperty="label", inputProperties=get_features_names(cfg))
            if verbose:
                print(
                    f"Trained on {run_name} ({split} set) - all post dates: {cfg.data.time_periods.post} (size = {fc.size().getInfo()})."  # noqa E501
                )
        else:
            # Option 2: Train on all post dates, one by one
            n = 0
            for post in cfg.data.time_periods.post:
                print(post)

                fc = get_dataset_ready(run_name, split=split, post_dates=[post])
                # fc = fc.select(features_names + ["label"], new_features_names + ["label"])
                # classifier = classifier.train(fc, "label", new_features_names)
                classifier = classifier.train(
                    features=fc, classProperty="label", inputProperties=get_features_names(cfg)
                )
                if verbose:
                    n_ = fc.size().getInfo()
                    n += n_
                    print(f"Trained on {run_name} ({split} set) - {post} (size = {n_}).")
            if verbose:
                print(f"Total size of {split} set: {n}.")
    else:
        period = "1year" if len(cfg.data.time_periods.post) == 2 else "3months"
        asset_path = ASSETS_PATH + f"s1tsdd_Ukraine/{run_name}/features_ready_{split}_{period}"

        if not asset_exists(asset_path):
            asset_path = ASSETS_PATH + f"s1tsdd_Ukraine/240301/features_ready_{split}_{period}"
            print("using dataset from 240301")
        fc = ee.FeatureCollection(asset_path)
        classifier = classifier.train(features=fc, classProperty="label", inputProperties=get_features_names(cfg))
        if verbose:
            n = fc.size().getInfo()
            print(f"Total size of {split} set: {n}.")

    return classifier


def get_features_names(cfg):
    """Get features names for Classifier from the config"""
    bands = ["VV", "VH"]
    names = cfg.data.time_periods.keys()
    winds = cfg.data.extract_winds
    reducers = cfg.reducer_names
    return [f"{b}_{n}_{w}_{r}" for b in bands for n in names for w in winds for r in reducers]


# ============== LOAD / EXPORT UTILS ==============
# For some reasons, the export classifier function do not work properly...


# def export_classifier(classifier, asset_id):
#     classifier_serialized = ee.serializer.toJSON(classifier)
#     ee.batch.Export.table.toAsset(
#         collection=ee.FeatureCollection(ee.Feature(ee.Geometry.Point((0, 0))).set("classifier", classifier_serialized)),
#         description=asset_id.split("/")[-1],
#         assetId=asset_id,
#     ).start()
#     print("Starting export of classifier to asset ", asset_id)


# def load_classifier(asset_id):
#     assert asset_exists(asset_id), f"Asset {asset_id} does not exist."
#     json = ee.Feature(ee.FeatureCollection(asset_id).first()).get("classifier").getInfo()
#     return ee.deserializer.fromJSON(json)


def export_classifier(classifier, asset_id, export_as_trees=False):
    if export_as_trees:
        trees = ee.List(ee.Dictionary(classifier.explain()).get("trees"))
        col = ee.FeatureCollection(trees.map(lambda x: ee.Feature(ee.Geometry.Point([0, 0])).set("tree", x)))
    else:
        classifier_serialized = ee.serializer.toJSON(classifier)
        col = ee.FeatureCollection(ee.Feature(ee.Geometry.Point((0, 0))).set("classifier", classifier_serialized))

    ee.batch.Export.table.toAsset(
        collection=col,
        description=asset_id.split("/")[-1],
        assetId=asset_id,
    ).start()
    print("Starting export of classifier to asset ", asset_id)


def load_classifier(asset_id, export_as_trees=False):
    assert asset_exists(asset_id), f"Asset {asset_id} does not exist."

    if export_as_trees:
        trees = ee.FeatureCollection(asset_id).aggregate_array("tree")
        return ee.Classifier.decisionTreeEnsemble(trees)
    else:
        json = ee.Feature(ee.FeatureCollection(asset_id).first()).get("classifier").getInfo()
        return ee.deserializer.fromJSON(json)
