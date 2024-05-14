import ee
from typing import List


def init_gee(project="rmac-ethz"):
    """Initialize GEE. Works also when working through ssh"""
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate(auth_mode="localhost")
        ee.Initialize(project=project)


def compute_metrics(preds):
    # Get confusion matrix ('classification' is the property added by RF)
    conf_matrix = preds.errorMatrix("label", "classification").getInfo()
    ((tn, fp), (fn, tp)) = conf_matrix

    # Compute metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def concat_fcs(fcs: List[ee.FeatureCollection]):
    list_fc = fc_to_list(fcs[0])
    for fc in fcs[1:]:
        list_fc = list_fc.cat(fc_to_list(fc))
    return ee.FeatureCollection(list_fc)


def fc_to_list(fc):
    return fc.toList(fc.size())


def mask_and_smoothen(img):
    # Should be called right before displaying the image

    # Urban layer only
    urban = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterDate("2020-02-24", "2022-02-24").mean().select("built")

    # Smoothen with Gaussian Kernel
    gaussian_kernel = ee.Kernel.gaussian(radius=30, sigma=10, units="meters")

    return img.convolve(gaussian_kernel).updateMask(urban.gt(0.1))


def draw_polygon_edges(poly, map, width=5, color="red", name="AOI"):
    """Draw AOI edges on map"""
    map.addLayer(ee.Image().paint(poly, 0, width), {"palette": [color]}, name)
    return map
