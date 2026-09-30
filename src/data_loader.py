"""Load the raw sales data and run basic data-quality checks."""

from pathlib import Path

import pandas as pd

from src import config


def load_sales_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the sales CSV, parse dates, and sort chronologically.

    Args:
        path: Location of the raw CSV file.

    Returns:
        A DataFrame sorted by date with a fresh integer index.
    """
    df = pd.read_csv(path, parse_dates=[config.DATE_COL])
    return df.sort_values(config.DATE_COL).reset_index(drop=True)


def validate_sales_data(df: pd.DataFrame) -> dict[str, object]:
    """Run data-quality checks and return the results as a dictionary.

    Args:
        df: Sales data as returned by `load_sales_data`.

    Returns:
        A mapping of check name to result, suitable for JSON serialisation.
    """
    expected_total = df[config.QUANTITY_COL] * df[config.PRICE_COL]
    full_range = pd.date_range(df[config.DATE_COL].min(), df[config.DATE_COL].max())
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "date_start": df[config.DATE_COL].min().strftime("%Y-%m-%d"),
        "date_end": df[config.DATE_COL].max().strftime("%Y-%m-%d"),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df[config.CUSTOMER_COL].duplicated().sum()),
        "non_positive_quantity_or_price": int(
            ((df[config.QUANTITY_COL] <= 0) | (df[config.PRICE_COL] <= 0)).sum()
        ),
        "total_sales_mismatches": int((expected_total != df[config.TARGET_COL]).sum()),
        "missing_calendar_days": int(len(full_range.difference(df[config.DATE_COL]))),
    }
