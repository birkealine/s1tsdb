import ee

from src.gee.constants import ASSETS_PATH


def get_base_asset_folder(n_tiles, extract_window):
    return ASSETS_PATH + f"s1tsdd_Ukraine/{n_tiles}d_{extract_window}m/"


def get_first_and_last_date(col):
    first_date = ee.Date(ee.Image(col.first()).get("system:time_start"))
    last_date = ee.Date(ee.Image(col.sort("system:time_start", False).first()).get("system:time_start"))
    return {"first": first_date, "last": last_date}


def fill_nan_with_mean(col):
    col_mean = col.reduce(ee.Reducer.mean())

    def _fill_nan_with_mean(img):
        mask = img.mask().Not()
        filled_img = img.unmask().add(col_mean.multiply(mask))
        filled_img = filled_img.copyProperties(img, img.propertyNames())
        return filled_img

    return col.map(_fill_nan_with_mean)


# ======= ASSET MANAGEMENT =======
def asset_exists(asset_id):
    try:
        ee.data.getAsset(asset_id)
        return True
    except ee.ee_exception.EEException:
        return False


def delete_asset(asset_id):
    try:
        ee.data.deleteAsset(asset_id)
        print(f"{asset_id} deleted")
        return True
    except ee.ee_exception.EEException:
        return False


def rename_asset(original_path, new_path):
    try:
        ee.data.renameAsset(original_path, new_path)
        print(f"Asset renamed from {original_path} to {new_path}")
    except Exception as e:
        print(f"Error renaming asset: {e}")


def create_folder(folder_path, verbose=0):
    try:
        ee.data.createAsset({"type": "FOLDER"}, folder_path)
        if verbose:
            print(f"Folder created at {folder_path}")
    except Exception as e:
        if verbose:
            print(f"Error creating folder: {e}")


def list_assets(folder_path, print_list=False):
    try:
        asset_list = [a["id"] for a in ee.data.getList({"id": folder_path})]
        if print_list:
            print(f"Assets in {folder_path}: {asset_list}")
        return asset_list
    except Exception as e:
        print(f"Error listing assets: {e}")
