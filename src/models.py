"""Feature preparation, model training/comparison, and revenue forecasting."""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder


def prepare_features(df):
    """One-hot encode Product and Region, and combine with Quantity.

    Price is left out on purpose: Total_Sales = Quantity * Price, so
    including Price would let the model "cheat" by just multiplying it back.

    Args:
        df: Sales DataFrame.

    Returns:
        A tuple (X, y, encoder) with the feature matrix, target, and the
        fitted encoder (needed later to encode new inputs for prediction).
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[["Product", "Region"]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    X = pd.concat([encoded_df, df[["Quantity"]].reset_index(drop=True)], axis=1)
    y = df["Total_Sales"].reset_index(drop=True)
    return X, y, encoder


def compare_models(X_train, y_train):
    """Compare a few regression models using 5-fold cross-validation.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        A DataFrame with the mean cross-validated R2 for each model,
        sorted best first.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }

    results = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        results.append({"Model": name, "R2": scores.mean()})

    return pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """Fit a model and evaluate it with three metrics on the test set.

    Args:
        model: A scikit-learn regressor.
        X_train, X_test, y_train, y_test: Train/test split.

    Returns:
        A tuple (fitted model, metrics dict, y_pred).
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }
    return model, metrics, y_pred


def save_model(model, encoder, path="deployment/model/sales_model.joblib"):
    """Save the trained model and encoder together so both can be reloaded.

    Args:
        model: Fitted scikit-learn model.
        encoder: Fitted OneHotEncoder used during training.
        path: Output file path.
    """
    joblib.dump({"model": model, "encoder": encoder}, path)


def load_model(path="deployment/model/sales_model.joblib"):
    """Load a model and encoder saved by save_model.

    Args:
        path: File path.

    Returns:
        A dict with keys 'model' and 'encoder'.
    """
    return joblib.load(path)


def predict_deal_value(product, region, quantity, model, encoder):
    """Predict the value of a single deal.

    Args:
        product: Product name.
        region: Region name.
        quantity: Number of units.
        model: Fitted regression model.
        encoder: Fitted OneHotEncoder matching the model's training features.

    Returns:
        Predicted deal value (float, never negative).
    """
    input_df = pd.DataFrame([{"Product": product, "Region": region}])
    encoded = encoder.transform(input_df)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
    encoded_df["Quantity"] = quantity

    prediction = model.predict(encoded_df)[0]
    return max(prediction, 0)


def forecast_next_weeks(weekly_revenue, n_weeks=4):
    """Forecast future weekly revenue using a simple linear trend.

    A train/test split on the last 3 weeks is used to sanity-check the
    trend line against a plain average before trusting it.

    Args:
        weekly_revenue: Series of past weekly revenue.
        n_weeks: Number of future weeks to forecast.

    Returns:
        A tuple (forecast Series, dict comparing average-based vs
        trend-based MAE on the held-out weeks).
    """
    values = weekly_revenue.values
    weeks = np.arange(len(values)).reshape(-1, 1)

    # Hold out the last 3 weeks to compare a flat average vs a trend line.
    test_size = 3
    train_weeks, test_weeks = weeks[:-test_size], weeks[-test_size:]
    train_values, test_values = values[:-test_size], values[-test_size:]

    average_forecast = np.repeat(train_values.mean(), test_size)
    average_mae = mean_absolute_error(test_values, average_forecast)

    trend_model = LinearRegression().fit(train_weeks, train_values)
    trend_forecast = trend_model.predict(test_weeks)
    trend_mae = mean_absolute_error(test_values, trend_forecast)

    comparison = {"Average MAE": average_mae, "Trend MAE": trend_mae}

    # Use whichever method did better on the held-out weeks for the real forecast.
    final_model = LinearRegression().fit(weeks, values)
    if average_mae <= trend_mae:
        future_values = np.repeat(values.mean(), n_weeks)
    else:
        future_weeks = np.arange(len(values), len(values) + n_weeks).reshape(-1, 1)
        future_values = final_model.predict(future_weeks)
    future_values = np.maximum(future_values, 0)

    future_dates = pd.date_range(
        weekly_revenue.index[-1] + pd.Timedelta(weeks=1), periods=n_weeks, freq="W"
    )
    forecast = pd.Series(future_values, index=future_dates, name="Forecast")
    return forecast, comparison
