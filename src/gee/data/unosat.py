import ee

from src.gee.constants import ASSETS_PATH, UNOSAT_PALETTE


def get_unosat_geo(aoi):
    return ee.FeatureCollection(ASSETS_PATH + f"AOIs/{aoi}").geometry()


def get_unosat_labels(aoi, all_labels=False):
    if all_labels:
        return ee.FeatureCollection(ASSETS_PATH + f"UNOSAT_labels/{aoi}_full")
    else:
        return ee.FeatureCollection(ASSETS_PATH + f"UNOSAT_labels/{aoi}")


def display_unosat_labels(labels, Map):
    # Add style with the correct color

    def color_label(feature):
        # need to convert key in string with format
        color = UNOSAT_PALETTE.get(ee.Number(feature.get("damage")).format("%d"))
        return feature.set("style", {"color": color})

    labels_colored = labels.map(color_label)

    # Add to map
    vis_params = {"styleProperty": "style"}
    Map.addLayer(labels_colored.style(**vis_params), {}, "UNOSAT labels")
    return Map
