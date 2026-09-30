"""Plotting functions for exploratory analysis and model results.

Every function returns a matplotlib `Figure` and, when `save_path` is given,
also writes it to disk. All figures share one visual style.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src import config
from src.utils import format_currency

PRIMARY = "#1F4E79"
ACCENT = "#E07A1F"
MUTED = "#9AA5B1"
FIGURE_DPI = 150
DEFAULT_SIZE = (9, 5)


def set_plot_style() -> None:
    """Apply the shared seaborn/matplotlib style used by every figure."""
    sns.set_theme(style="whitegrid", context="notebook", palette=[PRIMARY, ACCENT])
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.dpi": FIGURE_DPI,
        }
    )


def _currency_axis(axis: plt.Axes, which: str = "y") -> None:
    """Format an axis with compact currency labels."""
    formatter = mticker.FuncFormatter(lambda value, _: format_currency(value))
    target = axis.yaxis if which == "y" else axis.xaxis
    target.set_major_formatter(formatter)


def _finish(fig: Figure, save_path: Path | None) -> Figure:
    """Tighten the layout and optionally save the figure."""
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_weekly_revenue(weekly: pd.Series, save_path: Path | None = None) -> Figure:
    """Line chart of weekly revenue with the average as a reference line."""
    fig, ax = plt.subplots(figsize=DEFAULT_SIZE)
    ax.plot(weekly.index, weekly.values, marker="o", color=PRIMARY)
    ax.axhline(weekly.mean(), color=MUTED, linestyle="--", label="Average week")
    ax.set(title="Weekly revenue", xlabel="Week starting", ylabel="Revenue")
    _currency_axis(ax)
    ax.legend()
    return _finish(fig, save_path)


def plot_revenue_by_segment(
    summary: pd.DataFrame, title: str, save_path: Path | None = None
) -> Figure:
    """Horizontal bar chart of revenue per segment, labelled with its share."""
    fig, ax = plt.subplots(figsize=DEFAULT_SIZE)
    ordered = summary.sort_values("Revenue")
    bars = ax.barh(ordered.index, ordered["Revenue"], color=PRIMARY)
    labels = [
        f"{format_currency(rev)}  ({share:.0%})"
        for rev, share in zip(ordered["Revenue"], ordered["Revenue_Share"], strict=True)
    ]
    ax.bar_label(bars, labels=labels, padding=4)
    ax.set(title=title, xlabel="Revenue", ylabel="")
    ax.set_xlim(right=ordered["Revenue"].max() * 1.25)
    _currency_axis(ax, which="x")
    return _finish(fig, save_path)


def plot_deal_size_by_region(df: pd.DataFrame, save_path: Path | None = None) -> Figure:
    """Box plot of deal size per region, ordered by median deal size."""
    fig, ax = plt.subplots(figsize=DEFAULT_SIZE)
    order = (
        df.groupby(config.REGION_COL)[config.TARGET_COL].median().sort_values().index
    )
    sns.boxplot(
        data=df,
        x=config.REGION_COL,
        y=config.TARGET_COL,
        order=order,
        color=PRIMARY,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x=config.REGION_COL,
        y=config.TARGET_COL,
        order=order,
        color=ACCENT,
        size=4,
        alpha=0.7,
        ax=ax,
    )
    ax.set(title="Deal size by region", xlabel="", ylabel="Deal value")
    _currency_axis(ax)
    return _finish(fig, save_path)


def plot_quantity_and_price_by_region(
    summary: pd.DataFrame, save_path: Path | None = None
) -> Figure:
    """Side-by-side bars of average quantity and average price per region."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.5))
    ordered = summary.sort_values("Avg_Quantity")
    left.bar(ordered.index, ordered["Avg_Quantity"], color=PRIMARY)
    left.set(title="Average units per deal", ylabel="Units")
    ordered = summary.sort_values("Avg_Price")
    right.bar(ordered.index, ordered["Avg_Price"], color=ACCENT)
    right.set(title="Average unit price", ylabel="Price")
    _currency_axis(right)
    return _finish(fig, save_path)


def plot_region_product_heatmap(
    matrix: pd.DataFrame, save_path: Path | None = None
) -> Figure:
    """Heat map of revenue for each region and product combination."""
    fig, ax = plt.subplots(figsize=DEFAULT_SIZE)
    annotations = matrix.map(format_currency)
    sns.heatmap(
        matrix, annot=annotations, fmt="", cmap="Blues", cbar=False, linewidths=1, ax=ax
    )
    ax.set(title="Revenue by region and product", xlabel="", ylabel="")
    return _finish(fig, save_path)


def plot_backtest_errors(
    metrics: pd.DataFrame, save_path: Path | None = None
) -> Figure:
    """Bar chart of back-test MAE for each forecasting method."""
    fig, ax = plt.subplots(figsize=DEFAULT_SIZE)
    ordered = metrics.sort_values("MAE", ascending=False)
    ax.barh(ordered.index, ordered["MAE"], color=PRIMARY)
    ax.set(title="Weekly forecast error by method (lower is better)", xlabel="MAE")
    _currency_axis(ax, which="x")
    return _finish(fig, save_path)


def plot_forecast(
    weekly: pd.Series, forecast: pd.DataFrame, save_path: Path | None = None
) -> Figure:
    """History plus a shaded forecast band for the coming weeks."""
    fig, ax = plt.subplots(figsize=DEFAULT_SIZE)
    ax.plot(weekly.index, weekly.values, marker="o", color=PRIMARY, label="Actual")
    ax.plot(
        forecast.index, forecast["Forecast"], marker="o", color=ACCENT, label="Forecast"
    )
    ax.fill_between(
        forecast.index,
        forecast["Lower"],
        forecast["Upper"],
        color=ACCENT,
        alpha=0.2,
        label="80% interval",
    )
    ax.set(title="Weekly revenue forecast", xlabel="Week starting", ylabel="Revenue")
    _currency_axis(ax)
    ax.legend()
    return _finish(fig, save_path)


def plot_model_comparison(
    results: pd.DataFrame, save_path: Path | None = None
) -> Figure:
    """Bar chart of cross-validated RMSE for each candidate model."""
    fig, ax = plt.subplots(figsize=DEFAULT_SIZE)
    ordered = results.sort_values("RMSE", ascending=False)
    bars = ax.barh(ordered.index, ordered["RMSE"], color=PRIMARY)
    ax.bar_label(bars, labels=[f"R² {r2:.2f}" for r2 in ordered["R2"]], padding=4)
    ax.set(title="Deal-value model comparison (lower RMSE is better)", xlabel="RMSE")
    ax.set_xlim(right=ordered["RMSE"].max() * 1.2)
    _currency_axis(ax, which="x")
    return _finish(fig, save_path)


def plot_actual_vs_predicted(
    actual: pd.Series, predicted: np.ndarray, save_path: Path | None = None
) -> Figure:
    """Scatter of cross-validated predictions against actual deal values."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, color=PRIMARY, alpha=0.7)
    limit = max(actual.max(), predicted.max()) * 1.05
    ax.plot(
        [0, limit], [0, limit], color=MUTED, linestyle="--", label="Perfect prediction"
    )
    ax.set(
        title="Predicted vs actual deal value",
        xlabel="Actual",
        ylabel="Predicted (cross-validated)",
    )
    _currency_axis(ax, which="x")
    _currency_axis(ax, which="y")
    ax.legend()
    return _finish(fig, save_path)
