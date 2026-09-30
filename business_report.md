# Business Report: Sales Forecasting & Regional Performance

## Executive Summary

This project analyzed 100 sales transactions (₹12.4M total revenue, Jan–Apr
2024) to answer three business questions: where is performance strong or
weak, can revenue be forecast, and can a deal's value be estimated in
advance. Key finding: the West region lags noticeably behind the other three
in average deal size, revenue should be budgeted around the historical
average rather than an assumed trend, and a simple regression model can give
a useful (though not precise) early estimate of deal value.

## 1. Regional & Product Performance

| Region | Total Revenue | Avg. Deal Size |
|---|---|---|
| North | ₹3.98M | ₹142,273 |
| South | ₹3.74M | ₹138,439 |
| East | ₹2.52M | ₹132,613 |
| West | ₹2.12M | **₹81,689** |

West's average deal size is noticeably lower than the other three regions. A
one-way ANOVA test across all four regions gives **p ≈ 0.10** — just short of
the conventional 5% significance threshold, likely because the sample (100
transactions) is relatively small. The gap is not statistically airtight, but
it's descriptively consistent and comes from West selling fewer units per
deal, not from a different product mix (the region × product revenue
breakdown looks broadly similar in shape across regions).

**Recommendation:** Look into West's sales process — bundling, upselling, or
account-size targets — rather than changing its product catalog. Re-run this
comparison as more transactions accumulate to see if the gap becomes
statistically clearer.

## 2. Revenue Forecast

With ~14 weeks of history, two forecasting approaches were compared on the
final 3 weeks before committing to one:

| Method | MAE on held-out weeks |
|---|---|
| **Historical average (selected)** | ₹430,714 |
| Linear trend | ₹545,908 |

The historical average performed better, meaning there's no clear upward or
downward trend yet in this short a history. The forecast for the next 4 weeks
carries forward the average of **~₹824K/week**.

**Recommendation:** Budget to this average rather than a projected trend, and
revisit the forecast monthly as more data comes in — a real trend may become
visible with a longer history.

## 3. Deal-Value Estimation

A regression model was built to estimate a deal's value from `Product`,
`Region`, and `Quantity` (not `Price`, which is excluded because it's
definitionally part of the target). Three models were compared with 5-fold
cross-validation:

| Model | Cross-Val R² |
|---|---|
| **Ridge Regression (selected)** | 0.23 |
| Linear Regression | 0.22 |
| Random Forest | 0.19 |

On the held-out test set, the final Ridge model achieved:

| Metric | Value |
|---|---|
| R² | 0.46 |
| MAE | ₹60,563 |
| RMSE | ₹76,215 |

The model explains under half of the variation in deal value — useful as a
rough, early estimate for sizing a sales pipeline, but not precise enough to
use as a final quote.

**Recommendation:** Use the model (available in the deployed dashboard) for
early-stage pipeline planning only.

## Caveats

- Sample size is small (100 transactions, 14 weeks) — all results, especially
  the regional significance test, should be revisited as more data arrives.
- Currency is assumed to be ₹ (INR); the source data does not state a currency.
- The deal-value model uses only 3 input fields; more features (customer
  history, season, discounts) would likely improve accuracy.
