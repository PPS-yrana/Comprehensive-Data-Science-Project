"""FastAPI service exposing the deal-value model and the weekly forecast.

Run from the project root with:

    uvicorn deployment.api:app --reload
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import estimate_deal_value, get_metadata, load_forecast

app = FastAPI(
    title="Sales Forecasting API",
    description="Deal-value estimates and weekly revenue forecasts.",
    version="1.0.0",
)


class DealRequest(BaseModel):
    """Inputs describing a prospective deal."""

    product: str = Field(examples=["Laptop"])
    region: str = Field(examples=["North"])
    quantity: int = Field(ge=1, le=100, examples=[5])


class DealResponse(BaseModel):
    """Estimated deal value and the model's typical error."""

    estimated_value: float
    typical_error: float


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the service is running."""
    return {"status": "ok"}


@app.get("/options")
def options() -> dict[str, list[str]]:
    """List the products and regions the model accepts."""
    metadata = get_metadata()
    return {"products": metadata["products"], "regions": metadata["regions"]}


@app.post("/predict", response_model=DealResponse)
def predict(request: DealRequest) -> DealResponse:
    """Estimate the value of a deal from its product, region, and quantity."""
    try:
        estimate = estimate_deal_value(
            request.product, request.region, request.quantity
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return DealResponse(
        estimated_value=estimate.estimated_value, typical_error=estimate.typical_error
    )


@app.get("/forecast")
def forecast() -> list[dict]:
    """Return the weekly revenue forecast with its 80% interval."""
    table = load_forecast().reset_index()
    table["Week_Start"] = table["Week_Start"].dt.strftime("%Y-%m-%d")
    return table.to_dict("records")
