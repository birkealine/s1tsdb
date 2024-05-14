import ee
from omegaconf import DictConfig, OmegaConf
import re
from typing import List
from tqdm import tqdm

from src.gee.constants import ASSETS_PATH
from src.gee.utils import init_gee
from src.gee.classification.inference import predict_geo
from src.gee.classification.model import get_classifier_trained
from src.utils.gdrive import get_files_in_folder

init_gee()


def predict_and_export_all_grids(
    classifier: ee.Classifier,
    cfg: DictConfig,
    folder: str,
    ids: List[str] = None,
    n_limit: int = None,
    verbose: int = 0,
):
    """
    Predict and export for all grids (quadkeys) in Ukraine.

    If ids is not None, predict only these grids. If n_limit is given, only predict on n_limit grids.
    """

    # Get all grids
    print(f"Predicting for quadkey grid with zoom {cfg.inference.quadkey_zoom}")
    grids = ee.FeatureCollection(ASSETS_PATH + f"s1tsdd_Ukraine/quadkeys_grid_zoom{cfg.inference.quadkey_zoom}")

    if ids is None:
        # No IDs were given, we predict on all (or n_limit if given)
        if n_limit:
            # For debugging
            grids = grids.limit(n_limit)
        ids = grids.aggregate_array("qk").getInfo()
    else:
        # make sure ids are strings
        ids = [str(id_) for id_ in ids]

    # Filter IDs that have already been predicted (names are qk_12345678.tif for instance)
    files = get_files_in_folder(folder, return_names=True)
    existing_names = [f.split(".")[0] for f in files if f.startswith("qk_")]
    ids = [id_ for id_ in ids if id_ not in existing_names]

    # get operations still running
    def get_description(id_):
        return f"{cfg.run_name}_qk{id_}_{'_'.join(cfg.inference.time_periods.post)}"

    ops = [o for o in ee.data.listOperations() if o["metadata"]["state"] in ["PENDING", "RUNNING"]]
    ids_running = [o["metadata"]["description"] for o in ops]
    ids = [id_ for id_ in ids if get_description(id_) not in ids_running]

    print(f"Predicting and exporting {len(ids)} grids")
    for id_ in tqdm(ids):

        grid = grids.filter(ee.Filter.eq("qk", id_))
        preds = predict_geo(
            grid.geometry(),
            classifier,
            cfg.inference.time_periods,
            cfg.data.extract_winds,
            cfg.reducer_names,
            orbits=None,
            verbose=verbose,
        )
        preds = preds.set("qk", id_)

        name = f"qk_{id_}"
        task = ee.batch.Export.image.toDrive(
            image=preds.multiply(2**8 - 1).toUint8(),  # multiply by 255 and convert to uint8
            description=get_description(id_),
            folder=folder,
            fileNamePrefix=name,
            region=grid.geometry(),
            scale=10,
            maxPixels=1e13,
        )
        task.start()


if __name__ == "__main__":

    from src.utils.gdrive import create_drive_folder, create_yaml_file_in_drive_from_config_dict

    cfg = OmegaConf.create(
        dict(
            aggregation_method="mean",
            model_name="random_forest",
            model_kwargs=dict(
                numberOfTrees=100,
                minLeafPopulation=3,
                maxNodes=1e4,
            ),
            data=dict(
                aois_test=[f"UKR{i}" for i in range(1, 19) if i not in [1, 2, 3, 4]],
                damages_to_keep=[1, 2],
                extract_winds=["3x3"],  # ['1x1', '3x3', '5x5']
                time_periods={  # to train
                    "pre": ("2020-02-24", "2021-02-23"),  # always only one
                    "post": "3months",
                },
            ),
            inference=dict(
                time_periods={
                    "pre": ("2020-02-24", "2021-02-23"),  # always only one
                    "post": [
                        ("2021-02-24", "2021-05-23"),
                        ("2021-05-24", "2021-08-23"),
                        ("2021-08-24", "2021-11-23"),
                        ("2021-11-24", "2022-02-23"),
                        ("2022-02-24", "2022-05-23"),
                        ("2022-05-24", "2022-08-23"),
                        ("2022-08-24", "2022-11-23"),
                        ("2022-11-24", "2023-02-23"),
                    ],
                },
                quadkey_zoom=8,
            ),
            reducer_names=["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"],
            train_on_all=False,  # train on all damages (train + test split)
            verbose=0,
            export_as_trees=False,
            seed=123,
            run_name="240307",  # must be string
        )
    )

    # Load classifier
    def get_classifier_id(cfg):
        n_trees = cfg.model_kwargs.numberOfTrees
        all_data = "_all_data" if cfg.train_on_all else ""
        export = "_export_tree" if cfg.export_as_trees else ""
        asset_id = f"{ASSETS_PATH}s1tsdd_Ukraine/{cfg.run_name}/classifier_{cfg.data.time_periods.post}_{n_trees}trees{all_data}{export}"
        return asset_id

    asset_id = get_classifier_id(cfg)
    print(asset_id)
    from src.gee.classification.model import load_classifier
    from src.gee.data.utils import asset_exists

    if not asset_exists(asset_id):
        print("Classifier does not exist")
        classifier = get_classifier_trained(cfg, verbose=1)
    else:
        classifier = load_classifier(asset_id)

    # Create folder in Drive and save config
    base_folder_name = f"{cfg.run_name}_quadkeys_predictions"
    try:
        # Create drive folder and save config
        create_drive_folder(base_folder_name)
        create_yaml_file_in_drive_from_config_dict(cfg, base_folder_name)
    except Exception:
        # get input from user to be sure they want to continue
        print("Folder already exists. Continue? (y/n)")
        user_input = input()
        if user_input != "y":
            raise ValueError("Interrupted")

    # Iterate for all post periods
    if isinstance(cfg.inference.time_periods.post[0], str):
        cfg.inference.time_periods.post = [cfg.inference.time_periods.post]

    post_periods = cfg.inference.time_periods.post
    for post_period in post_periods:

        folder_name = f"{base_folder_name}/{'_'.join(post_period)}"
        cfg.inference.time_periods.post = post_period
        try:
            # Create drive folder and save config
            create_drive_folder(folder_name)
        except Exception:
            # get input from user to be sure they want to continue
            print("Folder already exists. Continue? (y/n)")
            user_input = input()
            if user_input != "y":
                raise ValueError("Interrupted")

        # Launch predictions
        predict_and_export_all_grids(
            classifier=classifier,
            cfg=cfg,
            folder=folder_name.split("/")[-1],
            ids=None,
            n_limit=None,
            verbose=cfg.verbose,
        )
