from pathlib import Path

# ------------------- PATH CONSTANTS -------------------
constants_path = Path(__file__)
SRC_PATH = constants_path.parent
PROJECT_PATH = SRC_PATH.parent

LOGS_PATH = PROJECT_PATH / "logs"
DATA_PATH = PROJECT_PATH / "data"
SECRETS_PATH = PROJECT_PATH / "secrets"

RAW_PATH = DATA_PATH / "raw"
PROCESSED_PATH = DATA_PATH / "processed"
READY_PATH = DATA_PATH / "ready"
EXTERNAL_PATH = DATA_PATH / "external"
PREDS_PATH = DATA_PATH / "predictions"

RAW_S1_PATH = RAW_PATH / "sentinel_1"
S1_PATH = PROCESSED_PATH / "sentinel_1"

TS_PATH = PROCESSED_PATH / "time_series"

OVERTURE_QK_PATH = PROCESSED_PATH / "overture_buildings" / "quadkeys"
OVERTURE_AOI_PATH = PROCESSED_PATH / "overture_buildings" / "aoi"
OVERTURE_ADMIN_PATH = PROCESSED_PATH / "overture_buildings" / "admin"
OVERTURE_BUILDINGS_RAW_PATH = RAW_PATH / "overture_buildings"


# ---------------- CONFIGS CONSTANTS ----------------
HYDRA_CONFIG_PATH = SRC_PATH / "configs"
HYDRA_CONFIG_NAME = "config"

# ------------------- PROJECT CONSTANTS -------------------
AOIS_TEST = ["UKR6", "UKR7", "UKR8", "UKR12", "UKR15", "UKR16"]  # ["UKR6", "UKR8", "UKR12", "UKR15"]
SENTINEL_1_LAUNCH = "2014-04-03"
UKRAINE_WAR_START = "2022-02-24"
CRS_GLOBAL = 4326
CRS_UKRAINE = 32636  # /!\ only perfectly valid for middle of country
NO_DATA_VALUE = -999

# ------------------- VISUALIZATION CONSTANTS -------------------
DAMAGE_COLORS = {
    1: "darkred",  # Destroyed
    2: "darkorange",  # Severe Damage
    3: "gold",  # Moderate Damage
    4: "olive",  # Possible Damage
    5: "grey",  # Impact Crater (Damage to Road)
    6: "green",  # No Visible Damage
    7: "saddlebrown",  # Impact Crater (Damage to Field)
    11: "darkkhaki",  # Possible Damage from adjacent impact, debris
}

DAMAGE_SCHEME = {
    1: "Destroyed",
    2: "Severe Damage",
    3: "Moderate Damage",
    4: "Possible Damage",
    5: "Impact Crater (Damage to Road)",
    6: "No Visible Damage",
    7: "Impact Crater (Damage to Field)",
    11: "Possible Damage from adjacent impact, debris",
}

# ------------------- LOGGING -------------------
WANDB_USERNAME = "odietrich"
WANDB_PROJECT = "s1-time-series-damage-detection"
