import numpy as np

from src import config, forecasting


def test_historical_mean_forecast(weekly_series):
    forecast = forecasting.historical_mean(weekly_series.to_numpy(), 3)
    assert np.allclose(forecast, weekly_series.mean())


def test_linear_trend_extrapolates():
    history = np.array([10.0, 20.0, 30.0, 40.0])
    assert np.allclose(forecasting.linear_trend(history, 2), [50.0, 60.0])


def test_rolling_origin_shapes(weekly_series):
    metrics, errors = forecasting.rolling_origin_evaluation(weekly_series, min_train=4)
    assert len(errors) == len(weekly_series) - 4
    assert set(metrics.index) == set(forecasting.FORECASTERS)


def test_parsimony_rule_prefers_baseline_when_gain_is_small(weekly_series):
    metrics, _ = forecasting.rolling_origin_evaluation(weekly_series, min_train=4)
    metrics.loc[:, "MAE"] = 100.0
    metrics.loc["Linear trend", "MAE"] = 99.0  # only a 1% improvement
    assert forecasting.select_method(metrics) == config.BASELINE_METHOD


def test_forecast_interval_ordering(weekly_series):
    _, errors = forecasting.rolling_origin_evaluation(weekly_series, min_train=4)
    result = forecasting.forecast_next_weeks(
        weekly_series, config.BASELINE_METHOD, errors[config.BASELINE_METHOD], horizon=3
    )
    assert len(result) == 3
    assert (result["Lower"] <= result["Forecast"]).all()
    assert (result["Forecast"] <= result["Upper"]).all()
    assert result.index[0] == weekly_series.index[-1] + np.timedelta64(7, "D")
