import ee
from typing import Dict, List, Tuple

from src.gee.data.collections import get_s1_collection
from src.gee.classification.reducers import get_reducers


def predict_geo(
    geo: ee.Geometry,
    classifier: ee.Classifier,
    time_periods: Dict[str, Tuple[str, str]],
    extract_windows: List[str],
    reducer_names: List[str],
    orbits: List[int] = None,
    verbose: int = 0,
) -> ee.image:
    """
    Predict the damage probability per pixel within the given geometry for the given classifier.

    The prediction is averaged over orbits.
    """

    # Make sure the classifier is in the correct mode
    classifier = classifier.setOutputMode("PROBABILITY")

    # Get Sentinel-1 collection for the given orbit
    s1 = get_s1_collection(geo)
    if orbits is None:
        orbits = find_orbits(s1, time_periods)
        if verbose:
            print(f"Orbits to infer: {orbits.getInfo()}")

    def predict_s1_orbit(orbit):

        s1_orbit = s1.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))

        # Collection to features
        s1_features = col_to_features(s1_orbit, reducer_names, time_periods, extract_windows)

        # Predict
        return s1_features.classify(classifier)

    # Predict for each orbit and aggregate with mean
    preds = ee.ImageCollection(orbits.map(predict_s1_orbit)).mean()
    return preds


def col_to_features(
    col: ee.ImageCollection,
    reducer_names: List[str],
    time_periods: Dict[str, Tuple[str, str]],
    extract_windows: List[str],
) -> ee.Image:
    """Convert a collection of images to a single image with features as bands."""
    s1_features = None

    reducer_names = list(reducer_names)  # GEE does not like ListConfig
    reducer = get_reducers(reducer_names)
    original_col_names = [f"{b}_{r}" for b in ["VV", "VH"] for r in reducer_names]

    for window in extract_windows:

        if int(window[0]) > 1:
            # convolve (similar to looking at a larger window) with radius (eg 15m for 3x3 window)
            col = convolve_collection(col, 10 * int(window[0]) // 2, "square", "meters")

        for name_period, (start, end) in time_periods.items():

            s1_dates = col.filterDate(start, end)
            prefix = f"{name_period}_{window}"

            # Get features
            _s1_features = s1_dates.reduce(reducer)
            _s1_features = _s1_features.select(original_col_names, get_new_names(original_col_names, prefix))
            s1_features = _s1_features if s1_features is None else s1_features.addBands(_s1_features)

    return s1_features


def convolve_collection(
    img_col: ee.imagecollection, radius: int, kernel_type: str = "square", units: str = "meters"
) -> ee.imagecollection:
    """Convolve each image in the collection with a focal mean of radius `radius`"""

    def _convolve_mean(img):
        return img.focalMean(radius, kernel_type, units=units).set("system:time_start", img.get("system:time_start"))

    return img_col.map(_convolve_mean)


def find_orbits(s1: ee.featurecollection, d_time_periods: dict, min_number: int = 5) -> ee.List:
    """Find all orbtis that appear at least min_number in each time period."""
    list_orbits = []
    for _, (start, end) in d_time_periods.items():
        s1_ = s1.filterDate(start, end)
        orbits_counts = s1_.aggregate_histogram("relativeOrbitNumber_start")
        # At least 5 images per orbit (two months of data)
        orbits_counts = orbits_counts.map(lambda k, v: ee.Algorithms.If(ee.Number(v).gte(min_number), k, None))
        orbits_inference = orbits_counts.keys().map(lambda k: ee.Number.parse(k))  # cast keys back to number
        list_orbits.append(orbits_inference)
    return list_orbits[0].filter(ee.Filter.inList("item", list_orbits[1]))


def get_new_names(bands, prefix):
    new_bands = []
    for b in bands:
        b_, r = b.split("_")
        new_bands.append(f"{b_}_{prefix}_{r}")
    return new_bands
