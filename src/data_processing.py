"""Functions to load, clean, and prepare the sales data for analysis."""

import pandas as pd


def load_data(path="data/raw/sales_data.csv"):
    """Load the sales CSV and parse the Date column.

    Args:
        path: Path to the CSV file.

    Returns:
        A DataFrame sorted by date.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def check_data_quality(df):
    """Print basic data quality checks (missing values, duplicates, etc).

    Args:
        df: Sales DataFrame.
    """
    print("Rows, columns:", df.shape)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    # Total_Sales should always equal Quantity * Price
    mismatches = (df["Quantity"] * df["Price"] != df["Total_Sales"]).sum()
    print("Rows where Quantity * Price != Total_Sales:", mismatches)


def add_features(df):
    """Add simple calendar-based features used later in the analysis.

    Args:
        df: Sales DataFrame with a Date column.

    Returns:
        A copy of df with Month, Weekday and Week columns added.
    """
    df = df.copy()
    df["Month"] = df["Date"].dt.month_name()
    df["Weekday"] = df["Date"].dt.day_name()
    df["Week"] = df["Date"].dt.isocalendar().week
    return df


def get_weekly_revenue(df):
    """Aggregate total revenue per week.

    Args:
        df: Sales DataFrame with a Date column.

    Returns:
        A Series of total revenue indexed by week start date.
    """
    weekly = df.set_index("Date")["Total_Sales"].resample("W").sum()
    return weekly
