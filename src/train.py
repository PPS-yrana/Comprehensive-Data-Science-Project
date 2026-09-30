"""End-to-end training pipeline.

Run from the project root with:

    python -m src.train

The pipeline loads and validates the data, produces every figure and table
used in the reports, evaluates and fits the models, and writes the artefacts
consumed by the API and dashboard.
"""

import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_val_predict

from src import config, eda, forecasting, insights, models
from src.data_loader import load_sales_data, validate_sales_data
from src.features import add_calendar_features, aggregate_weekly_revenue
from src.utils import save_json


def run_business_analysis(df: pd.DataFrame, weekly: pd.Series) -> dict:
    """Create segment summaries, tests, and the descriptive figures."""
    region_summary = insights.summarise_by(df, config.REGION_COL)
    product_summary = insights.summarise_by(df, config.PRODUCT_COL)
    matrix = insights.region_product_matrix(df)
    weakest_region = region_summary["Avg_Deal"].idxmin()

    region_summary.to_csv(config.PROCESSED_DIR / "region_summary.csv")
    product_summary.to_csv(config.PROCESSED_DIR / "product_summary.csv")
    matrix.to_csv(config.PROCESSED_DIR / "region_product_revenue.csv")

    figures = config.FIGURES_DIR
    eda.plot_weekly_revenue(weekly, figures / "01_weekly_revenue.png")
    eda.plot_revenue_by_segment(
        region_summary, "Revenue by region", figures / "02_revenue_by_region.png"
    )
    eda.plot_revenue_by_segment(
        product_summary, "Revenue by product", figures / "03_revenue_by_product.png"
    )
    eda.plot_deal_size_by_region(df, figures / "04_deal_size_by_region.png")
    eda.plot_quantity_and_price_by_region(
        region_summary, figures / "05_quantity_price_by_region.png"
    )
    eda.plot_region_product_heatmap(matrix, figures / "06_region_product_heatmap.png")

    return {
        "total_revenue": float(df[config.TARGET_COL].sum()),
        "total_deals": int(len(df)),
        "average_deal": float(df[config.TARGET_COL].mean()),
        "region_summary": region_summary.reset_index().to_dict("records"),
        "product_summary": product_summary.reset_index().to_dict("records"),
        "region_test": insights.kruskal_test(df, config.REGION_COL),
        "product_test": insights.kruskal_test(df, config.PRODUCT_COL),
        "weakest_region": weakest_region,
        "weakest_region_vs_rest": insights.compare_region_to_rest(df, weakest_region),
    }


def run_forecasting(weekly: pd.Series) -> dict:
    """Back-test forecasters, pick the best, and save the forecast."""
    metrics, errors = forecasting.rolling_origin_evaluation(weekly)
    best_method = forecasting.select_method(metrics)
    forecast = forecasting.forecast_next_weeks(weekly, best_method, errors[best_method])

    weekly.to_csv(config.PROCESSED_DIR / "weekly_revenue.csv")
    forecast.to_csv(config.FORECAST_PATH)
    eda.plot_backtest_errors(metrics, config.FIGURES_DIR / "07_forecast_backtest.png")
    eda.plot_forecast(weekly, forecast, config.FIGURES_DIR / "08_weekly_forecast.png")

    return {
        "weeks_used": int(len(weekly)),
        "average_week": float(weekly.mean()),
        "backtest": metrics.reset_index(names="Method").to_dict("records"),
        "best_method": best_method,
        "forecast": forecast.reset_index()
        .astype({"Week_Start": str})
        .to_dict("records"),
    }


def run_deal_value_model(df: pd.DataFrame) -> dict:
    """Cross-validate candidates, fit the winner, and save it for deployment."""
    features, target = df[config.MODEL_FEATURES], df[config.TARGET_COL]
    results = models.cross_validate_models(features, target)
    final_model = models.fit_final_model(features, target)

    splitter = RepeatedKFold(
        n_splits=config.CV_FOLDS, n_repeats=1, random_state=config.RANDOM_STATE
    )
    predictions = cross_val_predict(
        models.build_tuned_ridge(), features, target, cv=splitter
    )
    eda.plot_model_comparison(results, config.FIGURES_DIR / "09_model_comparison.png")
    eda.plot_actual_vs_predicted(
        target, predictions, config.FIGURES_DIR / "10_actual_vs_predicted.png"
    )

    models.save_model(final_model)
    best = results.loc[models.FINAL_MODEL_NAME]
    save_json(
        {
            "model": models.FINAL_MODEL_NAME,
            "trained_on_rows": int(len(df)),
            "cv_r2": float(best["R2"]),
            "cv_mae": float(best["MAE"]),
            "cv_rmse": float(best["RMSE"]),
            "products": sorted(df[config.PRODUCT_COL].unique()),
            "regions": sorted(df[config.REGION_COL].unique()),
        },
        config.MODEL_METADATA_PATH,
    )
    baseline = results.loc[models.BASELINE_NAME]
    return {
        "cv_results": results.reset_index(names="Model").to_dict("records"),
        "chosen_model": models.FINAL_MODEL_NAME,
        "best_alpha": float(final_model.named_steps["model"].alpha),
        "rmse_improvement_vs_baseline": float(1 - best["RMSE"] / baseline["RMSE"]),
        "mae_improvement_vs_baseline": float(1 - best["MAE"] / baseline["MAE"]),
        "drivers": models.extract_drivers(final_model).to_dict("records"),
    }


def main() -> None:
    """Run the full pipeline and write all artefacts."""
    eda.set_plot_style()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = add_calendar_features(load_sales_data())
    weekly = aggregate_weekly_revenue(df)

    report = {
        "data_quality": validate_sales_data(df),
        "business": run_business_analysis(df, weekly),
        "forecasting": run_forecasting(weekly),
        "deal_value_model": run_deal_value_model(df),
    }
    save_json(report, config.METRICS_PATH)
    print(f"Pipeline complete. Metrics written to {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
