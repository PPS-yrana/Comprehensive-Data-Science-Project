"""Feature engineering: calendar features, weekly aggregation, preprocessing."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month, weekday, and week-start columns derived from the sale date.

    Args:
        df: Sales data containing the date column.

    Returns:
        A copy of `df` with `Month`, `Weekday`, and `Week_Start` columns.
    """
    result = df.copy()
    dates = result[config.DATE_COL]
    result["Month"] = dates.dt.strftime("%b")
    result["Weekday"] = dates.dt.day_name()
    result["Week_Start"] = dates.dt.to_period(config.WEEK_FREQ).dt.start_time
    return result


def aggregate_weekly_revenue(df: pd.DataFrame) -> pd.Series:
    """Sum revenue per week and drop the trailing partial week.

    A partial final week would look like a sudden revenue collapse and
    distort any forecast, so it is excluded.

    Args:
        df: Sales data sorted by date.

    Returns:
        Weekly revenue indexed by week-start date.
    """
    weekly = (
        df.set_index(config.DATE_COL)[config.TARGET_COL]
        .resample(config.WEEK_FREQ, label="left", closed="left")
        .sum()
    )
    last_day = df[config.DATE_COL].max()
    last_week_end = weekly.index[-1] + pd.Timedelta(days=config.DAYS_PER_WEEK - 1)
    if last_day < last_week_end:
        weekly = weekly.iloc[:-1]
    weekly.index.name = "Week_Start"
    return weekly.rename("Weekly_Revenue")


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing step for the deal-value model.

    Returns:
        A transformer that one-hot encodes categoricals and scales numerics.
    """
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                config.CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), config.NUMERIC_FEATURES),
        ]
    )
