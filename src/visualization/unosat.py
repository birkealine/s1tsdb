import contextily as ctx
import matplotlib.pyplot as plt
from src.constants import DAMAGE_COLORS, DAMAGE_SCHEME
from src.data.sentinel1.utils import get_target_s1
from src.data import load_unosat_labels
from src.data.utils import aoi_to_city


# TODO: Improve (made in a hurry)
def plot_unosat_labels(aoi, ax=None, color=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    target = get_target_s1(aoi)
    labels = load_unosat_labels(aoi).to_crs(target.rio.crs)
    for damage, group in labels.groupby("damage"):
        color_ = color if color else DAMAGE_COLORS[damage]
        group.plot(ax=ax, color=color_, legend=True, label=DAMAGE_SCHEME[damage])
    ctx.add_basemap(ax, crs=target.rio.crs, zoom=14, source=ctx.providers.CartoDB.VoyagerNoLabels, zorder=-1)
    ax.set_title(f"UNOSAT Labels for {aoi} ({aoi_to_city(aoi)})")
    if color is None:
        plt.legend(loc="lower right")
    plt.show()
