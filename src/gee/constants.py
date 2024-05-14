import ee

from src.utils.gee import init_gee
from src.constants import RAW_PATH, PROCESSED_PATH

init_gee()

RAW_TS_GEE_PATH = RAW_PATH / "time_series_gee"
TS_GEE_PATH = PROCESSED_PATH / "time_series" / "gee"
ASSETS_PATH = "projects/rmac-ethz/assets/"
UNOSAT_COLOR = "#289946"
PROBS_PALETTE = ["yellow", "red", "purple"]
UNOSAT_PALETTE = ee.Dictionary(
    {
        "1": "8b0000",
        "2": "ff8c00",
        "3": "ffd700",
        "4": "808000",
        "5": "808080",
        "6": "008000",
        "7": "8b4513",
        "11": "bdb76b",
    }
)
SEED = 123


D_AOI_ORBITS = {
    "UKR1": [43, 94],
    "UKR2": [87, 160, 36],
    "UKR3": [43, 145, 21, 94],
    "UKR4": [43, 94],
    "UKR5": [43, 94],
    "UKR6": [14, 87, 36, 138],
    "UKR7": [43, 116],
    "UKR8": [87, 160, 36],  # [87, 160, 36, 109] // 109 seems to be causing a bug...
    "UKR9": [87, 160, 36, 109],
    "UKR10": [87, 65, 138],
    "UKR11": [145, 21, 94],
    "UKR12": [116, 65],
    "UKR13": [116, 65],
    "UKR14": [43, 94, 167],
    "UKR15": [116, 65],
    "UKR16": [14, 87, 65, 138],
    "UKR17": [14, 65, 138],
    "UKR18": [14, 116, 167],
}
TEST_AOIS = ["UKR6", "UKR8", "UKR12", "UKR15"]
TRAIN_AOIS = [k for k in D_AOI_ORBITS.keys() if k not in TEST_AOIS]


# Some folds can be empty (UKR17's fold 1 and 5 are always empty (?)). We precompute them here to avoid
# lengthy filtering later (found with find_empty_fc below)
EMPTY_FC = {
    "always": ["UKR17_1", "UKR17_5"],
    "12": ["UKR12_5", "UKR15_1", "UKR17_4", "UKR18_1", "UKR18_5"],
    "123": ["UKR15_1", "UKR17_4", "UKR18_5"],
}

# def find_empty_fc(keep_labels=[1, 2], only_unosat=True):
#     """Find empty FCs for each AOI, orbit and fold"""
#     print(f"Empty FC for keep_labels={keep_labels}, only_unosat={only_unosat}:")
#     date = '2020-06-01 # random date'
#     for aoi, orbits in D_AOI_ORBITS.items():
#         for orbit in orbits:
#             asset_path = ASSETS_PATH + f"s1tsdd_Ukraine/{N_TILES}d_{EXTRACT_WINDOW}m/{aoi}_orbit_{orbit}_{date}"
#             fc = ee.FeatureCollection(asset_path)
#             if only_unosat:
#                 fc = fc.filterMetadata("source", "equals", "unosat")
#             if keep_labels:
#                 fc = fc.filter(ee.Filter.inList("damage", keep_labels + [None]))  # None for random points
#             for fold in range(1, 6):
#                 fc_fold = fc.filter(ee.Filter.eq("bin", fold))
#                 size = fc_fold.size().getInfo()
#                 if not size:
#                     print(f"aoi={aoi}, fold={fold}")
#             break
