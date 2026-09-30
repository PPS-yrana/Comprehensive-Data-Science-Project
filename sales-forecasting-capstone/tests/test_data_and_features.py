import pandas as pd

from src import config
from src.data_loader import validate_sales_data
from src.features import add_calendar_features, aggregate_weekly_revenue


def test_data_is_clean(sales_df):
    report = validate_sales_data(sales_df)
    assert report["rows"] == 100
    assert report["missing_values"] == 0
    assert report["total_sales_mismatches"] == 0


def test_data_sorted_by_date(sales_df):
    assert sales_df[config.DATE_COL].is_monotonic_increasing


def test_calendar_features_added(sales_df):
    result = add_calendar_features(sales_df)
    assert {"Month", "Weekday", "Week_Start"} <= set(result.columns)
    assert (
        result["Week_Start"].dt.dayofweek.nunique() == 1
    )  # every week starts on the same weekday


def test_weekly_aggregation_drops_partial_week(sales_df):
    weekly = aggregate_weekly_revenue(sales_df)
    assert len(weekly) == 14
    assert weekly.index.max() == pd.Timestamp("2024-04-01")
    partial_week_revenue = sales_df[sales_df[config.DATE_COL] >= "2024-04-08"][
        config.TARGET_COL
    ].sum()
    assert weekly.sum() == sales_df[config.TARGET_COL].sum() - partial_week_revenue
