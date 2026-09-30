"""Inference helpers shared by the API and the dashboard."""

import json
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.models import clip_prediction, load_model


@dataclass(frozen=True)
class DealEstimate:
    """A deal-value estimate and how far off such estimates typically are."""

    estimated_value: float
    typical_error: float


@lru_cache(maxsize=1)
def get_model() -> Pipeline:
    """Load the trained model once and reuse it."""
    return load_model()


@lru_cache(maxsize=1)
def get_metadata() -> dict:
    """Load model metadata (cross-validated error, valid categories)."""
    return json.loads(config.MODEL_METADATA_PATH.read_text())


def estimate_deal_value(product: str, region: str, quantity: int) -> DealEstimate:
    """Estimate the value of a deal before it is priced.

    Args:
        product: Product name; must be one the model was trained on.
        region: Region name; must be one the model was trained on.
        quantity: Number of units in the deal (positive integer).

    Returns:
        The estimated deal value and the model's typical absolute error.

    Raises:
        ValueError: If an input is unknown or out of range.
    """
    metadata = get_metadata()
    if product not in metadata["products"]:
        raise ValueError(f"Unknown product: {product!r}")
    if region not in metadata["regions"]:
        raise ValueError(f"Unknown region: {region!r}")
    if quantity < 1:
        raise ValueError("Quantity must be at least 1")

    row = pd.DataFrame(
        [
            {
                config.PRODUCT_COL: product,
                config.REGION_COL: region,
                config.QUANTITY_COL: quantity,
            }
        ]
    )
    prediction = get_model().predict(row)[0]
    return DealEstimate(
        estimated_value=round(clip_prediction(prediction)),
        typical_error=round(metadata["cv_mae"]),
    )


def load_forecast() -> pd.DataFrame:
    """Load the saved weekly forecast table."""
    return pd.read_csv(config.FORECAST_PATH, index_col=0, parse_dates=True)
