# Technical Documentation

## Methodology

1. **Data loading & cleaning** (`src/data_processing.py`): loaded the CSV,
   checked for missing values, duplicate rows, and whether
   `Quantity * Price == Total_Sales` for every row. All checks passed — no
   cleaning was needed.
2. **Feature engineering:** added `Month`, `Weekday`, and `Week` from the
   `Date` column for EDA. For the prediction model, `Product` and `Region`
   were one-hot encoded and combined with `Quantity`. `Price` was
   deliberately excluded, since `Total_Sales = Quantity * Price` exactly —
   including it would let the model leak the answer instead of learning a
   real relationship.
3. **Exploratory analysis:** revenue and average deal size by region and
   product, a correlation heatmap of the numeric columns, and a region ×
   product revenue heatmap to check whether West's shortfall was about
   product mix or something else.
4. **Statistical test:** a one-way ANOVA (`scipy.stats.f_oneway`) tested
   whether average deal size differs significantly across the 4 regions.
   Result: F ≈ 2.16, p ≈ 0.10 (not significant at the 5% level, though
   close, likely due to the small sample).
5. **Forecasting:** weekly revenue was aggregated, and two simple approaches
   — a flat historical average and a linear trend line — were compared by
   holding out the last 3 weeks and measuring MAE on them. The historical
   average performed better and was used for the final 4-week forecast.
6. **Deal-value model:** three regression models (Linear Regression, Ridge
   Regression, Random Forest) were compared using **5-fold cross-validation**
   on the training set (80/20 train/test split). Ridge Regression had the
   best average cross-validated R² and was selected as the final model,
   evaluated on the held-out test set with three metrics (MAE, RMSE, R²).
7. **Deployment:** the trained model and its encoder are saved together with
   `joblib` and loaded by a Streamlit dashboard that lets a user pick a
   product, region, and quantity to get a predicted deal value.

## Model Evaluation

| Model | Cross-Val R² (5-fold) |
|---|---|
| Ridge Regression | 0.228 |
| Linear Regression | 0.216 |
| Random Forest | 0.191 |

**Final model (Ridge, alpha=1.0), test set:**

| Metric | Value |
|---|---|
| MAE | ₹60,563 |
| RMSE | ₹76,215 |
| R² | 0.456 |

## Code Structure

```
src/data_processing.py   load_data, check_data_quality, add_features, get_weekly_revenue
src/visualization.py       6 plotting functions, one shared color scheme
src/models.py                prepare_features, compare_models, train_and_evaluate,
                              forecast_next_weeks, save_model/load_model, predict_deal_value
```

Every function has a short docstring (purpose, Args, Returns) and the same
style is used throughout: simple function-based modules (no classes needed
for a project this size), consistent naming, and no repeated logic between
the notebook and the dashboard — both import from `src/`.

## Known Limitations

- 100 rows / 14 weeks is a small sample; the regional ANOVA result (p ≈ 0.10)
  did not reach the standard 5% significance level, and both the forecast and
  model should be re-evaluated as more data is collected.
- The forecast is a simple average-vs-trend comparison, not a dedicated
  time-series model — appropriate given how little history is available, but
  worth revisiting with more weeks of data.
- The deal-value model explains under half of the variance in deal value
  (R² 0.46); it's a planning aid, not a precise pricing tool.
