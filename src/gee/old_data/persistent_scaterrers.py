import ee

from src.gee.data.unosat import get_unosat_geo
from src.gee.data.collections import get_s1_collection

THRESHOLD_PS_COV = 0.65
THRESHOLD_PS_SIGMA = -2


def compute_ps(aoi, orbit, start="2019-02-24", end="2022-02-24"):
    # Filter for orbit and AOI
    geo = get_unosat_geo(aoi)
    s1_ps = get_s1_collection(geo, start, end).filterMetadata("relativeOrbitNumber_start", "equals", orbit).select("VV")

    # Convert to Persistent Scatterers
    ps = s1_to_ps(s1_ps)
    return ps


def s1_to_ps(col):
    # Function to convert from dB to natural units
    def to_natural(img):
        return ee.Image(10.0).pow(img.select(0).divide(10.0))

    # Calculate pixel-wise covariance of amplitude values
    col_nat = col.map(to_natural)
    cov = col_nat.reduce(ee.Reducer.stdDev()).divide(col_nat.mean())

    # Select Persistent Scatterer (PS) candidates
    ps = cov.lt(THRESHOLD_PS_COV).And(col.mean().gt(THRESHOLD_PS_SIGMA))
    return ps.rename("ps")


def assign_ps(features, ps_mask):
    def _assign_ps(feature):
        point = feature.geometry()
        mask_value = ps_mask.reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=10).get("ps")
        return feature.set("ps", mask_value)

    return features.map(_assign_ps)
