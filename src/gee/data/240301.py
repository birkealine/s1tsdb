## CREATE FEATURES 240301
import ee
from src.gee.constants import ASSETS_PATH
from src.gee.data.dataset import get_dataset_ready

post = [
    ("2021-02-24", "2021-05-23"),
    ("2021-05-24", "2021-08-23"),
    ("2021-08-24", "2021-11-23"),
    ("2021-11-24", "2022-02-23"),
    ("2022-02-24", "2022-05-23"),
    ("2022-05-24", "2022-08-23"),
    ("2022-08-24", "2022-11-23"),
    ("2022-11-24", "2023-02-23"),
]
fc = get_dataset_ready("240229", split="train", post_dates=post)
fc_test = get_dataset_ready("240229", split="test", post_dates=post)
task = ee.batch.Export.table.toAsset(
    collection=fc,
    description="240301 train data 3months",
    assetId=ASSETS_PATH + "s1tsdd_Ukraine/240301/features_ready_train_3months",
).start()
task = ee.batch.Export.table.toAsset(
    collection=fc_test,
    description="240301 test data 3months",
    assetId=ASSETS_PATH + "s1tsdd_Ukraine/240301/features_ready_test_3months",
).start()

post = [("2021-02-24", "2022-02-23"), ("2022-02-24", "2023-02-23")]
fc = get_dataset_ready("240229", split="train", post_dates=post)
fc_test = get_dataset_ready("240229", split="test", post_dates=post)
task = ee.batch.Export.table.toAsset(
    collection=fc,
    description="240301 train data 1year",
    assetId=ASSETS_PATH + "s1tsdd_Ukraine/240301/features_ready_train_1year",
).start()
task = ee.batch.Export.table.toAsset(
    collection=fc_test,
    description="240301 test data 1year",
    assetId=ASSETS_PATH + "s1tsdd_Ukraine/240301/features_ready_test_1year",
).start()
