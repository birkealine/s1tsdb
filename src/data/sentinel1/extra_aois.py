from shapely.geometry import box
import geopandas as gpd
from src.constants import CRS_GLOBAL, CRS_UKRAINE, RAW_S1_PATH
from src.data.sentinel1.download_geemap import download_s1
from src.utils.gee import init_gee

init_gee()

# cf notebook new_aois.ipynb
zapo_geo = box(34.96, 47.74, 35.3, 47.94)
dnipro_geo = box(34.8, 48.6043, 35.30186, 48.3243)
data = {"city": ["Zaporizhzhya", "Dnipro"], "geometry": [zapo_geo, dnipro_geo]}
gdf = gpd.GeoDataFrame(data, crs=CRS_GLOBAL)
start_date = "2019-07-01"
end_date = "2023-06-30"
folder = RAW_S1_PATH

gdf.geometry = gdf.to_crs(CRS_UKRAINE).buffer(2000).to_crs(CRS_GLOBAL)
for i, row in gdf.iterrows():
    # if row.city != "Zaporizhzhya":
    #     print("ignoring", row.city)
    #     continue
    download_s1(
        geometry=row.geometry,
        start_date=start_date,
        end_date=end_date,
        local_folder=folder / row.city,
    )
