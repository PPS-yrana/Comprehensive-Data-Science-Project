"""Project-wide configuration: paths, column names, and modelling constants."""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sales_data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
MODEL_PATH = PROJECT_ROOT / "deployment" / "model" / "deal_value_model.joblib"
MODEL_METADATA_PATH = PROJECT_ROOT / "deployment" / "model" / "model_metadata.json"
FORECAST_PATH = PROCESSED_DIR / "weekly_forecast.csv"

# --- Column names ------------------------------------------------------------
DATE_COL = "Date"
PRODUCT_COL = "Product"
QUANTITY_COL = "Quantity"
PRICE_COL = "Price"
CUSTOMER_COL = "Customer_ID"
REGION_COL = "Region"
TARGET_COL = "Total_Sales"

# --- Feature groups for the deal-value model ---------------------------------
# Price is deliberately excluded: Total_Sales = Quantity * Price exactly, so
# using it would leak the target and the model would be useless before a deal
# is priced.
CATEGORICAL_FEATURES = [PRODUCT_COL, REGION_COL]
NUMERIC_FEATURES = [QUANTITY_COL]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# --- Modelling constants -----------------------------------------------------
RANDOM_STATE = 42
CV_FOLDS = 5
CV_REPEATS = 5
RIDGE_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
FOREST_TREES = 300
FOREST_MIN_LEAF = 3
BOOSTING_TREES = 100
BOOSTING_DEPTH = 2
BOOSTING_LEARNING_RATE = 0.05
BOOSTING_SUBSAMPLE = 0.8

# --- Forecasting constants ---------------------------------------------------
WEEK_FREQ = "W-MON"
MIN_TRAIN_WEEKS = 5
FORECAST_HORIZON_WEEKS = 4
INTERVAL_QUANTILES = (0.10, 0.90)  # 80% empirical prediction interval
DAYS_PER_WEEK = 7
MOVING_AVERAGE_WINDOW = 3
SMOOTHING_ALPHA = 0.3
BASELINE_METHOD = "Historical mean"
SELECTION_TOLERANCE = 0.05  # A rival must beat the baseline MAE by 5% to win.

# --- Statistics and presentation ---------------------------------------------
SIGNIFICANCE_LEVEL = 0.05
CURRENCY_SYMBOL = "₹"  # Assumed; the dataset does not state its currency.
