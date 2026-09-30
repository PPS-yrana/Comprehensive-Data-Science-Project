"""Streamlit dashboard: sales overview, forecast, and a deal-value estimator.

Run with:
    streamlit run deployment/app.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import streamlit as st

from src.data_processing import add_features, load_data, get_weekly_revenue
from src.models import forecast_next_weeks, load_model, predict_deal_value
from src.visualization import (
    plot_region_product_heatmap,
    plot_revenue_by_column,
    plot_weekly_trend,
)

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
st.title("Sales Forecasting & Regional Performance")


@st.cache_data
def get_data():
    return add_features(load_data())


df = get_data()
tab1, tab2, tab3 = st.tabs(["Overview", "Forecast", "Deal-Value Estimator"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"₹{df['Total_Sales'].sum():,.0f}")
    col2.metric("Total Deals", f"{len(df)}")
    col3.metric("Average Deal", f"₹{df['Total_Sales'].mean():,.0f}")

    left, right = st.columns(2)
    with left:
        st.pyplot(plot_revenue_by_column(df, "Region", "Revenue by Region"))
    with right:
        st.pyplot(plot_revenue_by_column(df, "Product", "Revenue by Product"))

    st.pyplot(plot_region_product_heatmap(df))

with tab2:
    weekly = get_weekly_revenue(df)
    forecast, comparison = forecast_next_weeks(weekly)

    st.pyplot(plot_weekly_trend(weekly))
    st.write("Next 4 weeks forecast:")
    st.dataframe(forecast.rename("Forecasted Revenue"))
    st.caption(
        f"A flat historical average (MAE ₹{comparison['Average MAE']:,.0f} on the "
        f"last 3 weeks) beat a linear trend (MAE ₹{comparison['Trend MAE']:,.0f}), "
        "so the forecast is the historical average rather than a trend line."
    )

with tab3:
    saved = load_model()
    model, encoder = saved["model"], saved["encoder"]

    col1, col2, col3 = st.columns(3)
    product = col1.selectbox("Product", sorted(df["Product"].unique()))
    region = col2.selectbox("Region", sorted(df["Region"].unique()))
    quantity = col3.number_input("Quantity", min_value=1, max_value=50, value=5)

    prediction = predict_deal_value(product, region, quantity, model, encoder)
    st.metric("Estimated Deal Value", f"₹{prediction:,.0f}")
    st.caption(
        "Model: Ridge Regression, test R² ≈ 0.46, typical error ≈ ₹60,600. "
        "Use this as a rough planning estimate, not a final quote."
    )
