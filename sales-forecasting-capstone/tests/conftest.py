"""Shared fixtures."""

import pandas as pd
import pytest

from src import config
from src.data_loader import load_sales_data


@pytest.fixture(scope="session")
def sales_df() -> pd.DataFrame:
    """The real sales dataset."""
    return load_sales_data()


@pytest.fixture()
def weekly_series() -> pd.Series:
    """A small synthetic weekly series with a known mean."""
    index = pd.date_range("2024-01-01", periods=8, freq=config.WEEK_FREQ)
    return pd.Series([100, 120, 90, 110, 100, 130, 95, 105], index=index, dtype=float)
