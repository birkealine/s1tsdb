import datetime as dt
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import xarray as xr

from src.data.utils import read_ts
from src.data.sentinel1.orbits import get_valid_orbits
from src.visualization.utils import add_text_box
from src.constants import UKRAINE_WAR_START


def plot_ts(
    ts: xr.DataArray,
    ax=None,
    title=None,
    add_legend=True,
    add_invasion_date=True,
    add_analysis_date=True,
    loc_legend="lower left",
):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    # Plot vertical line at 0
    ax.axhline(0, color="k", linewidth=0.5)

    d_color = {"VV": "C1", "VH": "C0"}

    # Plot each band with correct color
    for band in ts.band.values:
        if band not in d_color:
            # ignore additional bands if any
            continue
        ts.sel(band=band).plot(x="date", color=d_color[band], label=band, ax=ax)

    invasion_date = UKRAINE_WAR_START
    if add_invasion_date and ts.date[0].dt.strftime("%Y-%m-%d") < invasion_date < ts.date[-1].dt.strftime("%Y-%m-%d"):
        ax.axvline(
            dt.date.fromisoformat(invasion_date),
            color="r",
            linestyle="--",
            label="date of invasion",
        )

    analysis_date = ts.date_of_analysis
    if add_analysis_date and ts.date[0].dt.strftime("%Y-%m-%d") < analysis_date < ts.date[-1].dt.strftime("%Y-%m-%d"):
        ax.axvline(
            dt.date.fromisoformat(analysis_date),
            color="g",
            linestyle="--",
            label="date of analysis",
        )

    # Add legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Backscatter (dB)")
    ax.set_ylim([-25, 10])
    ax.grid(axis="x")
    if add_legend:
        ax.legend(loc=loc_legend)
    if title is None:
        title = f"{ts.aoi} - orbit {ts.orbit} - ID {ts.unosat_id}"
    ax.set_title(title)
    return ax


def plot_ts_from_id(
    aoi, orbit, id_, start_date=None, end_date=None, n_dates=None, extraction_strategy="pixel-wise", **kwargs
):
    # Read time-series
    xa = read_ts(aoi, orbit, id_, extraction_strategy)

    # Select date range
    xa = xa.sel(date=slice(start_date, end_date))
    if n_dates is not None:
        if n_dates > xa.shape[0]:
            print(f"Warning: {n_dates} dates requested but only {xa.shape[0]} available")
        else:
            xa = xa.isel(date=slice(0, n_dates))
    return plot_ts(xa, **kwargs)


def plot_all_ts_from_id(
    aoi, id_, start_date=None, end_date=None, n_dates=None, extraction_strategy="pixel-wise", **kwargs
):
    orbits = get_valid_orbits(aoi)
    fig, axs = plt.subplots(len(orbits), 1, figsize=(10, 3 * len(orbits)))
    for i, orbit in enumerate(orbits):
        # Read time-series
        xa = read_ts(aoi, orbit, id_, extraction_strategy)

        # Select date range
        xa = xa.sel(date=slice(start_date, end_date))
        if n_dates is not None:
            if n_dates > xa.shape[0]:
                print(f"Warning: {n_dates} dates requested but only {xa.shape[0]} available")
            else:
                xa = xa.isel(date=slice(0, n_dates))
        plot_ts(xa, ax=axs[i], **kwargs)
    plt.tight_layout()


def plot_distribution_prediction(df, aggregated=False, ax=None, add_metrics=True, threshold=0.5):
    """Plot distribution of predictions on test set"""

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    df[df.label == 0].preds_proba.plot.hist(ax=ax, bins=50, label="Intact", alpha=0.8)
    df[df.label == 1].preds_proba.plot.hist(ax=ax, bins=50, label="Destroyed", alpha=0.8)

    ax.axvline(threshold, color="r", linestyle="--", label=f"threshold={threshold}")
    ax.set_xlim(0, 1)
    title = "Prediction on Test Set" + (" (aggregated)" if aggregated else "") + f" (N={len(df)})"
    ax.set_title(title)
    ax.set_xlabel("Probability of Destruction")
    ax.set_ylabel("Count")
    ax.legend(loc="lower left")
    if add_metrics:
        y_true = df.label.values
        y_preds = (df.preds_proba > threshold).astype(int)
        ax = add_box_with_metrics(y_true, y_preds, ax)
    return ax


def add_box_with_metrics(y_true, y_preds, ax):
    acc = accuracy_score(y_true, y_preds)
    prec = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)

    metrics_txt = f"Accuracy: {acc:.2f}\n" f"Precision: {prec:.2f}\n" f"Recall: {recall:.2f}\n" f"F1: {f1:.2f}"
    add_text_box(ax, metrics_txt, x=0.02, y=0.97, ha="left", va="top")
    return ax
