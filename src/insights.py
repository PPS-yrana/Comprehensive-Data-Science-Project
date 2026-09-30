"""Business analytics: segment summaries and statistical tests."""

import pandas as pd
from scipy import stats

from src import config


def summarise_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Summarise revenue, deal count, and deal size for each segment.

    Args:
        df: Sales data.
        column: Column to group by (for example `Region` or `Product`).

    Returns:
        One row per segment, sorted by revenue, with revenue share.
    """
    summary = (
        df.groupby(column)
        .agg(
            Revenue=(config.TARGET_COL, "sum"),
            Deals=(config.TARGET_COL, "size"),
            Avg_Deal=(config.TARGET_COL, "mean"),
            Avg_Quantity=(config.QUANTITY_COL, "mean"),
            Avg_Price=(config.PRICE_COL, "mean"),
        )
        .sort_values("Revenue", ascending=False)
    )
    summary["Revenue_Share"] = summary["Revenue"] / summary["Revenue"].sum()
    return summary.round(2)


def region_product_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build a Region x Product revenue matrix.

    Args:
        df: Sales data.

    Returns:
        Revenue with regions as rows and products as columns.
    """
    return df.pivot_table(
        index=config.REGION_COL,
        columns=config.PRODUCT_COL,
        values=config.TARGET_COL,
        aggfunc="sum",
        fill_value=0,
    )


def kruskal_test(df: pd.DataFrame, column: str) -> dict[str, float]:
    """Test whether deal size differs across the segments of `column`.

    Kruskal-Wallis is used because deal sizes are right-skewed and the
    sample is small, so normality cannot be assumed.

    Args:
        df: Sales data.
        column: Segment column to compare.

    Returns:
        The test statistic and p-value.
    """
    groups = [g[config.TARGET_COL] for _, g in df.groupby(column)]
    statistic, p_value = stats.kruskal(*groups)
    return {"statistic": float(statistic), "p_value": float(p_value)}


def compare_region_to_rest(df: pd.DataFrame, region: str) -> dict[str, float]:
    """Compare one region's deal sizes with all other regions combined.

    Args:
        df: Sales data.
        region: Region to isolate.

    Returns:
        Average deal sizes for both groups and the Mann-Whitney p-value.
    """
    in_region = df[df[config.REGION_COL] == region][config.TARGET_COL]
    others = df[df[config.REGION_COL] != region][config.TARGET_COL]
    _, p_value = stats.mannwhitneyu(in_region, others)
    return {
        "region_avg_deal": float(in_region.mean()),
        "others_avg_deal": float(others.mean()),
        "p_value": float(p_value),
    }
