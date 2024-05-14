"""Load datasets from Google Earth Engine for training and testing."""

import datetime as dt
from dateutil.relativedelta import relativedelta
import ee

from src.gee.constants import D_AOI_ORBITS, EMPTY_FC, TRAIN_AOIS, TEST_AOIS, SEED
from src.gee.data.utils import asset_exists, get_base_asset_folder
from src.gee.utils import fc_to_list


def load_dataset(
    split,
    fold=None,
    random_loc=0.1,  # proportion of random locations to keep
    keep_damage=[1, 2],
    n_tiles=32,
    extract_window=30,
    start_dates=["2020-10-01", "2021-10-01"],
    save_if_doesnt_exist=True,
    verbose=1,
):
    if split == "valid":
        assert fold is not None, 'fold should be specified for split "valid"'

    # if asset is saved, load it, otherwise create it
    base_path = get_base_asset_folder(n_tiles, extract_window)
    asset_name = get_name_asset(split, fold, keep_damage, random_loc, start_dates)
    asset_path = base_path + f"Final/{asset_name}"
    if asset_exists(asset_path):
        if verbose:
            print(f"Loading dataset from {asset_path}")
        fc = ee.FeatureCollection(asset_path)
    else:
        fc = create_dataset(split, fold, random_loc, keep_damage, n_tiles, extract_window, start_dates, verbose=verbose)

        if save_if_doesnt_exist:
            task = ee.batch.Export.table.toAsset(
                collection=fc,
                description=f"{asset_name}",
                assetId=asset_path,
            )
            task.start()
            if verbose:
                print(f"Task {asset_name} started")
    if verbose:
        print(f"Dataset {split} loaded (size={fc.size().getInfo()})")
    return fc


def create_dataset(
    split,
    fold=None,
    random_loc=0.1,  # proportion of random locations to keep
    keep_damage=[1, 2],
    n_tiles=32,
    extract_window=30,
    start_dates=None,
    verbose=1,
):
    if verbose:
        print(
            f"Creating dataset for split={split}, fold={fold}, random_loc={random_loc},\
            keep_damage={keep_damage}, n_tiles={n_tiles}, extract_window={extract_window}, start_dates={start_dates}"
        )

    # make sure keep_damage is a list (can be a omegaconf.listconfig.ListConfig)
    keep_damage = list(keep_damage)

    # some fc can be empty, we discard them directly here
    empty_fcs = get_empty_fcs(keep_damage, fold)

    # Get AOIs
    aois = TEST_AOIS if split == "test" else TRAIN_AOIS

    # Loop over aoi, orbits and start_dates and stack all the featureCollections
    list_fc = ee.List([])
    base_path = get_base_asset_folder(n_tiles, extract_window)
    for aoi, orbits in D_AOI_ORBITS.items():
        # Checks
        if empty_fcs and f"{aoi}_{fold}" in empty_fcs:
            continue
        if aoi not in aois:
            continue

        for orbit in orbits:
            for start_date in start_dates:
                if is_invalid_combo(aoi, orbit, start_date):
                    continue

                asset_path = base_path + f"{aoi}_orbit_{orbit}_{start_date}"
                try:
                    fc = ee.FeatureCollection(asset_path)
                except Exception:
                    print(f"Error loading asset {asset_path}. Need to compute intermediate steps first")
                    raise FileNotFoundError

                # Keep only desired type of damage
                fc_unosat = fc.filterMetadata("source", "equals", "unosat")
                fc_unosat = fc_unosat.filter(ee.Filter.inList("damage", keep_damage))

                if random_loc:
                    # Sample a certain percentage of random locations (with fixed seed!)
                    fc_random = fc.filterMetadata("source", "equals", "random")
                    sampled_random = fc_random.randomColumn("random", SEED).filterMetadata(
                        "random", "less_than", random_loc
                    )
                    fc = fc_unosat.merge(sampled_random)
                else:
                    # Keep only UNOSAT labels
                    fc = fc_unosat

                if fold:
                    if split == "valid":
                        fc = fc.filter(ee.Filter.eq("bin", fold))
                    elif split == "train":
                        bins = [i for i in range(1, 6) if i != fold]
                        fc = fc.filter(ee.Filter.inList("bin", bins))
                    else:
                        assert 0, 'fold should not be specified for split "test"'

                list_fc = list_fc.cat(fc_to_list(fc))

    fc = ee.FeatureCollection(list_fc)

    # Remove invalid labels
    fc = fc.filter(ee.Filter.neq("label", -1))

    # Correct bug when creating intermediate collections: random locations should all have labels 0 !!!
    fc = fc.map(
        lambda f: f.set("label", ee.Algorithms.If(ee.String(f.get("source")).equals("random"), 0, f.get("label")))
    )

    return fc


def get_all_start_dates(first_start_date, last_start_date, every_n_months=1):
    # Generate monthly start dates
    start_date = dt.datetime.strptime(first_start_date, "%Y-%m-%d")
    last_start_date = dt.datetime.strptime(last_start_date, "%Y-%m-%d")
    start_dates = []
    while start_date <= last_start_date:
        start_dates.append(start_date.strftime("%Y-%m-%d"))
        start_date += relativedelta(months=every_n_months)
    return start_dates


def get_empty_fcs(keep_damage, fold):
    """Avoid empty featureCollections before stacking..."""
    if fold:
        empty_fcs = EMPTY_FC["always"]
        key = "".join([str(i) for i in keep_damage])
        if key in EMPTY_FC:
            empty_fcs += EMPTY_FC[key]
        return empty_fcs
    else:
        return None


def get_name_asset(split, fold, keep_damage, random_perc, start_dates, **kwargs):
    assert split in ["train", "valid", "test", "rf"]
    keep_damage = "_".join([str(i) for i in keep_damage])
    fold = f"fold{fold}_" if fold is not None else ""
    random_perc = f"random_{int(100*random_perc)}perc_" if random_perc else ""
    start_dates = "_".join(sorted(start_dates)) if len(start_dates) < 6 else "all_dates"
    return f"{split}_{fold}labels_{keep_damage}_{random_perc}{start_dates}"


def is_invalid_combo(aoi, orbit, start_date):
    """Combination of aoi, orbit and start_date that are invalid for some reason (cause errors)"""
    INVALID = ["UKR8_36_2020-08-01"]
    return f"{aoi}_{orbit}_{start_date}" in INVALID
