import pytest

from src import config, models
from src.predict import estimate_deal_value


def test_model_beats_baseline(sales_df):
    features, target = sales_df[config.MODEL_FEATURES], sales_df[config.TARGET_COL]
    results = models.cross_validate_models(features, target)
    baseline_rmse = results.loc[models.BASELINE_NAME, "RMSE"]
    assert results.loc[models.FINAL_MODEL_NAME, "RMSE"] < 0.8 * baseline_rmse


def test_prediction_grows_with_quantity():
    small = estimate_deal_value("Laptop", "North", 1).estimated_value
    large = estimate_deal_value("Laptop", "North", 9).estimated_value
    assert large > small


def test_prediction_never_negative():
    assert estimate_deal_value("Monitor", "West", 1).estimated_value >= 0


@pytest.mark.parametrize(
    ("product", "region", "quantity"),
    [("Toaster", "North", 2), ("Laptop", "Mars", 2), ("Laptop", "North", 0)],
)
def test_invalid_inputs_rejected(product, region, quantity):
    with pytest.raises(ValueError):
        estimate_deal_value(product, region, quantity)
