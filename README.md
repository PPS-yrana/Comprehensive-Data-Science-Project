# Comprehensive Data Science Project: Sales Forecasting & Regional Performance

Capstone project — a complete data science workflow from raw sales data to
a deployed prediction dashboard, with business recommendations.

## Business Problem

Using 100 sales transactions from Jan–Apr 2024 (4 regions, 5 products):
1. Where is performance strong or weak across regions/products?
2. Can next month's revenue be forecast from this history?
3. Can a deal's value be estimated from Product, Region, and Quantity?

## Key Results

- **Regional gap:** West's average deal (₹81.7K) is clearly the lowest of the
  4 regions (vs ₹132K–142K elsewhere). An ANOVA test gives p ≈ 0.10 — just
  short of the usual 5% significance cutoff, likely because the sample is
  small — but the descriptive gap is consistent and worth a business-side look.
- **Forecast:** no clear trend in 14 weeks of data; forecasting the historical
  average (~₹824K/week) beat a linear trend line on the held-out weeks.
- **Deal-value model:** Ridge Regression, chosen via 5-fold cross-validation
  comparison against Linear Regression and Random Forest, achieves R² ≈ 0.46
  on the test set (MAE ≈ ₹60.6K).

Full analysis: [`capstone_project.ipynb`](capstone_project.ipynb).
Business framing: [`reports/business_report.md`](reports/business_report.md).

## Project Structure

```
├── capstone_project.ipynb      # Main analysis notebook
├── src/
│   ├── data_processing.py      # Load, clean, and add basic features
│   ├── visualization.py        # All plotting functions
│   └── models.py                # Feature prep, model comparison, forecasting
├── data/
│   └── raw/sales_data.csv
├── reports/
│   ├── business_report.md
│   ├── technical_documentation.md
│   ├── metrics.json             # Numbers behind both reports (generated)
│   └── figures/                 # Charts (generated)
├── deployment/
│   ├── app.py                    # Streamlit dashboard
│   └── model/                    # Saved model (generated)
├── presentation/
│   └── capstone_presentation.pptx
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the Notebook

Open `capstone_project.ipynb` in Jupyter — it reads `data/raw/sales_data.csv`
and runs the full analysis, including saving the trained model used by the
dashboard.

## Running the Dashboard

```bash
streamlit run deployment/app.py
```

The dashboard has three tabs: sales overview, weekly forecast, and a
deal-value estimator (product + region + quantity → predicted value).

## Notes on Approach

- `Price` is deliberately excluded from the deal-value model's features,
  since `Total_Sales = Quantity × Price` exactly — including it would let the
  model just multiply the target back out instead of learning anything.
- Model choice (Ridge Regression) is based on 5-fold cross-validation, not
  just the test-set score, to reduce the chance of picking a model that got
  lucky on one split.
- The forecast compares a simple average against a linear trend on held-out
  weeks before picking one, rather than assuming a trend exists.
