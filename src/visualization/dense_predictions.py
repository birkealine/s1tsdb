import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray as rxr

from src.constants import CRS_GLOBAL
from src.data import load_unosat_labels
from src.data.unosat import get_unosat_geometry
from src.visualization.utils import add_text_box

from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_dense_predictions(
    preds,
    aoi,
    start,
    end,
    threshold=0.5,
    damages_to_keep=[1, 2],
    ax=None,
    clip_to_aoi=True,
    compute_metrics=True,
    show=True,
):

    if clip_to_aoi:
        geo = get_unosat_geometry(aoi)
        geo_crs = gpd.GeoDataFrame(None, geometry=[geo], crs=CRS_GLOBAL).to_crs(preds.rio.crs).iloc[0].geometry
        preds = preds.rio.clip([geo_crs])

    if compute_metrics:
        assert start and end, "Start and end dates must be provided to compute metrics"
        # Read labels to compute metrics
        labels = load_unosat_labels(aoi, damages_to_keep).to_crs(preds.rio.crs)
        dates = labels.date.dt.strftime("%Y-%m-%d")
        labels_within_dates = labels[(start <= dates) & (dates <= end)]
        if len(labels_within_dates):
            labels_within_dates["preds"] = labels_within_dates.apply(
                lambda x: extract_raster_value(x.geometry, preds), axis=1
            )
            labels_predicted = len(labels_within_dates[labels_within_dates.preds > threshold])
            n_labels = len(labels_within_dates)
            recall = 100 * labels_predicted / n_labels
            metrics_txt = f"""With threshold {threshold}
            Recall = {recall:.2f}%
            ({labels_predicted}/{n_labels} labels predicted)"""
        else:
            metrics_txt = "No labels yet"

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.get_figure()
    im = preds.where(preds > threshold).plot.imshow(cmap="YlOrRd", ax=ax, add_colorbar=False, vmin=0.5, vmax=1)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1], label="Damage Probability")
    cbar.ax.axhline(threshold, c="k")  # horizontal line to indicate threshold
    ax.set_title(f"Predictions for {aoi}. Period = {start} - {end}")
    if compute_metrics:
        add_text_box(ax, metrics_txt, y=0.05, va="bottom")

    # Add basemap under imshow, which has zorder=0
    ctx.add_basemap(ax, crs=preds.rio.crs, zoom=14, source=ctx.providers.CartoDB.VoyagerNoLabels, zorder=-1)

    if show:
        plt.show()
    else:
        return fig


def plot_dense_predictions_from_fp(fp, **kwargs):
    assert fp.exists(), f"File {fp} does not exist"
    preds = rxr.open_rasterio(fp)
    return plot_dense_predictions(preds, **kwargs)


def extract_raster_value(point, raster):
    value = raster.sel(x=point.x, y=point.y, method="nearest").item()
    return value
