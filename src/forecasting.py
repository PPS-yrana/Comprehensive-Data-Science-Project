"""Weekly revenue forecasting with rolling-origin evaluation.

With only a few months of data, complex models cannot be trusted. Instead we
compare a small set of transparent forecasters and let a rolling-origin
back-test decide which one, if any, is worth using.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from src import config

Forecaster = Callable[[np.ndarray, int], np.ndarray]


def naive_last(history: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the most recent observation."""
    return np.repeat(history[-1], horizon)


def historical_mean(history: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the mean of all past observations."""
    return np.repeat(history.mean(), horizon)


def moving_average(history: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the mean of the most recent observations."""
    return np.repeat(history[-config.MOVING_AVERAGE_WINDOW :].mean(), horizon)


def exponential_smoothing(history: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast a flat level using simple exponential smoothing."""
    level = history[0]
    for value in history[1:]:
        level = config.SMOOTHING_ALPHA * value + (1 - config.SMOOTHING_ALPHA) * level
    return np.repeat(level, horizon)


def linear_trend(history: np.ndarray, horizon: int) -> np.ndarray:
    """Extrapolate a straight line fitted to the history."""
    steps = np.arange(len(history))
    slope, intercept = np.polyfit(steps, history, deg=1)
    future_steps = np.arange(len(history), len(history) + horizon)
    return intercept + slope * future_steps


FORECASTERS: dict[str, Forecaster] = {
    "Naive (last week)": naive_last,
    "Historical mean": historical_mean,
    "Moving average (3 wk)": moving_average,
    "Exponential smoothing": exponential_smoothing,
    "Linear trend": linear_trend,
}


def rolling_origin_evaluation(
    series: pd.Series, min_train: int = config.MIN_TRAIN_WEEKS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Back-test every forecaster one week ahead, expanding the training window.

    Args:
        series: Weekly revenue indexed by week start.
        min_train: Number of weeks used before the first forecast.

    Returns:
        A tuple of (metrics per method, forecast errors per method and week).
        Errors are defined as actual minus forecast.
    """
    values = series.to_numpy(dtype=float)
    eval_weeks = series.index[min_train:]
    errors = {}
    for name, forecaster in FORECASTERS.items():
        errors[name] = [
            values[t] - forecaster(values[:t], 1)[0]
            for t in range(min_train, len(values))
        ]
    errors_df = pd.DataFrame(errors, index=eval_weeks)
    metrics = pd.DataFrame(
        {
            "MAE": errors_df.abs().mean(),
            "RMSE": np.sqrt((errors_df**2).mean()),
        }
    ).sort_values("MAE")
    metrics["MAE_pct_of_mean_week"] = 100 * metrics["MAE"] / series.mean()
    return metrics.round(1), errors_df


def select_method(metrics: pd.DataFrame) -> str:
    """Pick the forecasting method using a parsimony rule.

    With so few weeks, tiny differences in back-test error are noise. The
    simple historical mean is kept unless another method beats it by more
    than `SELECTION_TOLERANCE`.

    Args:
        metrics: Back-test metrics from `rolling_origin_evaluation`.

    Returns:
        The name of the selected method.
    """
    best_method = metrics["MAE"].idxmin()
    baseline_mae = metrics.loc[config.BASELINE_METHOD, "MAE"]
    required_mae = baseline_mae * (1 - config.SELECTION_TOLERANCE)
    if metrics.loc[best_method, "MAE"] < required_mae:
        return best_method
    return config.BASELINE_METHOD


def forecast_next_weeks(
    series: pd.Series,
    method: str,
    errors: pd.Series,
    horizon: int = config.FORECAST_HORIZON_WEEKS,
) -> pd.DataFrame:
    """Forecast future weeks with an empirical prediction interval.

    The interval adds quantiles of the method's back-test errors to the
    point forecast. It is approximate, but honest about the noise level.

    Args:
        series: Weekly revenue indexed by week start.
        method: Key of `FORECASTERS` to use.
        errors: Back-test errors (actual minus forecast) for `method`.
        horizon: Number of weeks to forecast.

    Returns:
        A DataFrame with `Forecast`, `Lower`, and `Upper` per future week.
    """
    point = FORECASTERS[method](series.to_numpy(dtype=float), horizon)
    low_q, high_q = errors.quantile(config.INTERVAL_QUANTILES)
    step = pd.Timedelta(days=config.DAYS_PER_WEEK)
    future_index = pd.DatetimeIndex(
        [series.index[-1] + step * (i + 1) for i in range(horizon)],
        name=series.index.name,
    )
    return pd.DataFrame(
        {
            "Forecast": point,
            "Lower": np.maximum(point + low_q, 0),
            "Upper": point + high_q,
        },
        index=future_index,
    ).round(0)
