import ee

from src.gee.constants import ASSETS_PATH
from src.gee.data.utils import fill_nan_with_mean, create_folder, asset_exists
from src.gee.data.unosat import get_unosat_geo, get_unosat_labels
from src.gee.data.collections import get_s1_collection
from src.gee.utils import init_gee


init_gee()


def create_fc_aoi_orbit(aoi, orbit, scale=10, start_date="2020-03-01", end_date="2023-03-01", labels_to_keep=[1, 2]):
    """Creates a feature collection with V and VH value for each date and each point."""

    # Check if asset id exists
    extract = f"{scale//10}x{scale//10}"
    folder = ASSETS_PATH + "s1tsdd_Ukraine/"
    folder += f'ts{extract}_{start_date.replace("-","")}_{end_date.replace("-","")}'
    if labels_to_keep != [1, 2]:
        folder += f'_label_{"".join([str(l) for l in labels_to_keep])}'
    create_folder(folder, verbose=0)
    asset_id = folder + f"/{aoi}_orbit{orbit}"
    if asset_exists(asset_id):
        print(f"Asset {asset_id} already exists.")
        return

    labels = get_unosat_labels(aoi, True)
    labels = labels.filter(ee.Filter.inList("damage", labels_to_keep))

    geo = get_unosat_geo(aoi)
    s1 = get_s1_collection(geo, start_date, end_date).filterMetadata("relativeOrbitNumber_start", "equals", orbit)
    s1 = fill_nan_with_mean(s1)

    def extract_all_imgs(img_col, feat_col, scale, img_bands=["VV", "VH"]):
        def extract_img(img):
            def extract_point(point):
                bands = img.select(img_bands).reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=point.geometry(), scale=scale
                )
                return point.set(bands).set("system:time_start", img.get("system:time_start"))

            return feat_col.map(extract_point)

        return img_col.map(extract_img).flatten()

    fc_extracted = extract_all_imgs(s1, labels, scale)

    ee.batch.Export.table.toAsset(
        collection=fc_extracted, description=f"{aoi}_orbit{orbit}_{scale}m", assetId=asset_id
    ).start()
    print(f"Exporting {aoi}_orbit{orbit}_{scale}m")


if __name__ == "__main__":

    from src.data.utils import aoi_orbit_iterator

    start_date = "2020-03-01"
    end_date = "2023-03-01"
    labels_to_keep = [3]

    for aoi, orbit in aoi_orbit_iterator():
        for scale in [30]:
            create_fc_aoi_orbit(aoi, orbit, scale, start_date, end_date, labels_to_keep=labels_to_keep)
