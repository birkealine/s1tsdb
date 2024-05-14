from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import random
import rasterio
import re
import rioxarray as rxr
import shapely
from typing import List
import xarray as xr

from src.data.sentinel1.orbits import get_valid_orbits, get_best_orbit
from src.data import load_unosat_labels, load_unosat_aois
from src.constants import TS_PATH, PROCESSED_PATH, EXTERNAL_PATH

VALID_EXTRACTION_STRATEGIES = ["pixel-wise", "3x3", "gee", "pixel-wise-3x3", "5x5"]


def get_random_ts(aoi=None, orbit=None, extraction_strategy: str = "3x3", seed=123):
    id_, aoi, orbit = get_random_id(aoi, orbit, return_aoi=True, return_orbit=True, seed=seed)
    return read_ts(aoi, orbit, id_, extraction_strategy)


def get_random_id(aoi=None, orbit=None, return_aoi=False, return_orbit=False, seed=123):
    random.seed(seed)
    if aoi is None:
        aoi = random.choice([f"UKR{i}" for i in range(1, 19)])
    elif isinstance(aoi, list):
        aoi = random.choice(aoi)

    labels = load_unosat_labels(aoi)
    if orbit is None:
        orbit = random.choice(get_valid_orbits(aoi))
    id_ = random.choice(labels.index)
    if return_aoi and return_orbit:
        return id_, aoi, orbit
    elif return_aoi:
        return id_, aoi
    elif return_orbit:
        return id_, orbit
    else:
        return id_


def read_ts(aoi, orbit, id_, extraction_strategy: str = "3x3"):
    fp = get_folder_ts(extraction_strategy) / aoi / f"orbit_{orbit}" / f"{id_}.nc"
    return read_ts_from_fp(fp)


def read_ts_from_fp(fp):
    assert fp.exists(), f"File {fp} does not exist"
    xa = xr.open_dataarray(fp)
    return xa


def get_all_ts_orbit(aoi, id_, extraction_strategy: str = "3x3"):
    """Get all time series for a given point"""
    orbits = get_valid_orbits(aoi)
    d_ts = {o: read_ts(aoi, o, id_, extraction_strategy) for o in orbits}
    return d_ts


def get_folder_ts(extraction_strategy: str = "3x3"):
    """Folder in PROCESSED_PATH where to store raw time series"""
    if extraction_strategy not in VALID_EXTRACTION_STRATEGIES:
        raise ValueError(f"extraction_strategy must be in {VALID_EXTRACTION_STRATEGIES}, got {extraction_strategy}")

    folder = TS_PATH / extraction_strategy.replace("-", "_")
    folder.mkdir(exist_ok=True, parents=True)
    return folder


def load_ps_mask(aoi, orbit=None):
    """Load PS mask"""
    orbit = orbit or get_best_orbit(aoi)
    fp = PROCESSED_PATH / "persistent_scatterers" / f"ps_{aoi}_orbit{orbit}.tif"
    if not fp.exists():
        raise FileNotFoundError(f"File {fp} does not exist, supposed to be computed in other repo")
    _ps_mask = rxr.open_rasterio(fp).rio.write_nodata(0)
    return _ps_mask.squeeze()


def get_all_aois(only_ukraine: bool = True) -> List[str]:
    """
    Return all AOIs.

    Args:
        add_extra_aois (bool): Whether to add extra AOIs or not

    Returns:
        List[str]: List of aoi names
    """
    aois = load_unosat_aois().aoi
    if only_ukraine:
        aois = [aoi for aoi in aois if aoi.startswith("UKR")]

    # sort to have UKR1, UKR2, UKR3, ... instead of UKR1, UKR10, UKR11, ...
    return sorted(aois, key=lambda x: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", x)])


def aoi_orbit_iterator(only_ukraine: bool = True):
    """Iterator over all AOIs and valid orbits"""
    for aoi in get_all_aois(only_ukraine=only_ukraine):
        orbits = get_valid_orbits(aoi)
        for orbit in orbits:
            yield aoi, orbit


def load_aois_config() -> DictConfig:
    """Load the YAML file with the instructions to build AOIs"""

    fp = EXTERNAL_PATH / "aois_conf.yaml"
    assert fp.exists(), "The YAML file with AOIs instructions does not exist."
    return OmegaConf.load(fp)


def get_map_aoi_city():
    """Get a dictionary with the city name for each AOI"""
    aois_cfg = load_aois_config()
    d_aois_city = {}
    for aoi, cfg in aois_cfg.items():
        cities = cfg.city
        if isinstance(cities, str):
            city = cities
        elif aoi == "UKR2":
            city = "NW Kyiv"
        elif aoi == "UKR16":
            city = cities[-1]  # Kherson
        else:
            city = cities[0]
        d_aois_city[aoi] = city
    return d_aois_city


def aoi_to_city(aoi: str) -> str:
    """Get the city name for an AOI"""
    d_aois_city = get_map_aoi_city()
    return d_aois_city[aoi]
