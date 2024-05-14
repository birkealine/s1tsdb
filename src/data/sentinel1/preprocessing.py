"""Preprocess raw Sentinel-1 tiles and create time-series for HDF5 file"""
import datetime as dt
import os
from pathlib import Path
from rasterio.enums import Resampling
import re
import time
from tqdm import tqdm
from typing import List, Union

from src.constants import RAW_S1_PATH, S1_PATH
from src.data.sentinel1.utils import read_s1, crop_multiple, save_s1
from src.data.sentinel1.orbits import get_valid_orbits, get_best_orbit
from src.data.utils import get_all_aois
from src.utils.time import print_sec


def preprocess_s1(
    which_aois: Union[str, List[str]] = "all",
    raw_folder: Union[str, Path] = RAW_S1_PATH,
    processed_folder: Union[str, Path] = S1_PATH,
    force_recreate: bool = False,
) -> None:
    """
    Preprocess raw Sentinel-1 tiles.

    Reproject all tiles of the same AOI against one "target", so that they can be stacked
    on a later stage. Stacking all tiles will inevitably modify slightly the data, we use bilinear
    resampling to diminish the impact of this.

    Args:
        which_aois (Union[str, List[str]): Which aoi to preprocess. Can be either a list of aoi
            names or 'all', in which cases all existing AOIs will be preprocessed. Defaults to all.
        raw_folder (Union[str, Path], optional): Path to raw s1 files. Defaults to RAW_S1_PATH.
        processed_folder (Union[str, Path], optional): Folder where to save processed s1 tiles.
            Defaults to S1_PATH.
        force_recreate (bool, optional): If True, will force the recreation of all tiles.
    """
    print("Processing all Sentinel-1 tiles.")
    start_time = time.time()

    raw_folder = Path(raw_folder) if not isinstance(raw_folder, Path) else raw_folder
    processed_folder = Path(processed_folder) if not isinstance(processed_folder, Path) else processed_folder

    # Get all aois
    if which_aois == "all":
        aois = get_all_aois()  # ignoring extra AOIs anyway...
    else:
        aois = [which_aois] if isinstance(which_aois, str) else which_aois

    n_tiles = 0

    for aoi in aois:  # noqa E203
        # Only keep orbits that are 'valid' (ie cover fully the aoi -> computed before)
        orbits = get_valid_orbits(aoi)

        with tqdm(total=len(list((raw_folder / aoi).glob("**/*.tif")))) as pbar:
            for orbit in orbits:
                pbar.set_description(f"{aoi} - {orbit}")

                # All tiles for the given aoi
                fps = sorted((raw_folder / aoi / f"orbit_{orbit}").glob("*.tif"))
                if len(fps) == 0:
                    print(f"No tiles for {aoi} - {orbit}")
                    continue

                # folder for the given aoi
                folder = processed_folder / aoi / f"orbit_{orbit}"
                folder.mkdir(exist_ok=True, parents=True)

                # Take first file as target for all other tiles
                fp_target = fps[0]
                target = read_s1(fp_target, chunks=None)

                new_fp = folder / f"{s1_id_to_date(fp_target.stem)}.tif"
                if not new_fp.exists() or force_recreate:
                    # Only save if does not exist or force_recreate
                    # Crop before saving to make life easier later
                    # Tiles were downloaded with buffer so all good
                    save_s1(crop_multiple(target, 128), new_fp)
                    n_tiles += 1
                pbar.update(1)

                for fp in fps[1:]:
                    new_fp = folder / f"{s1_id_to_date(fp.stem)}.tif"
                    if not new_fp.exists() or force_recreate:
                        # Skip if already exists and not force_recreate
                        # Reproject and match all files with target
                        original_s1 = read_s1(fp, chunks=None)
                        # reproject to target using bilinear (nearest can add slight shifts)
                        s1 = original_s1.rio.reproject_match(target, resampling=Resampling.bilinear)

                        # Crop like target
                        save_s1(crop_multiple(s1, 128), new_fp)
                        n_tiles += 1
                    pbar.update(1)

            # Create a symlink to the 'best' orbit to avoid specifying the orbit everytime
            best_orbit = get_best_orbit(aoi)

            # delete symlink if exists
            if (processed_folder / aoi / "best").exists():
                os.remove(processed_folder / aoi / "best")
            os.symlink(processed_folder / aoi / f"orbit_{best_orbit}", processed_folder / aoi / "best")

    print(f"{n_tiles} processed and saved (in {print_sec(time.time()-start_time)}).")


def s1_id_to_date(s1_id: str) -> dt.date:
    """Get date from Sentinel-1 name"""
    timestamp = re.findall(r"(\d{8}T\d{6})", s1_id)[0]
    return dt.datetime.strptime(timestamp, "%Y%m%dT%H%M%S").date()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Sentinel-1 tiles.")
    parser.add_argument(
        "--aoi",
        type=str,
        default="all",
        nargs="+",
        help="Which AOI to preprocess. Can be either a list of aoi names or 'all', in which cases all existing AOIs will be preprocessed.",
    )
    args = parser.parse_args()
    preprocess_s1(which_aois=args.aoi)
