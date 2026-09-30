"""Plotting functions used in the notebook and the dashboard.

All charts share the same colour scheme (seaborn's default palette with one
primary colour) so the report and the dashboard look consistent.
"""

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
PRIMARY_COLOR = "#2E5090"
SECONDARY_COLOR = "#E8871E"


def plot_revenue_by_column(df, column, title):
    """Bar chart of total revenue grouped by a column (e.g. Region, Product).

    Args:
        df: Sales DataFrame.
        column: Column name to group by.
        title: Chart title.

    Returns:
        The matplotlib Figure.
    """
    revenue = df.groupby(column)["Total_Sales"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=revenue.index, y=revenue.values, color=PRIMARY_COLOR, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Total Revenue")
    ax.set_xlabel(column)
    plt.tight_layout()
    return fig


def plot_weekly_trend(weekly_revenue):
    """Line chart of weekly revenue over time.

    Args:
        weekly_revenue: Series of revenue indexed by week.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(weekly_revenue.index, weekly_revenue.values, marker="o", color=PRIMARY_COLOR)
    ax.set_title("Weekly Revenue Trend")
    ax.set_xlabel("Week")
    ax.set_ylabel("Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    """Heatmap of correlation between numeric columns.

    Args:
        df: Sales DataFrame.

    Returns:
        The matplotlib Figure.
    """
    numeric_cols = df[["Quantity", "Price", "Total_Sales"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap="Blues", ax=ax)
    ax.set_title("Correlation Between Numeric Features")
    plt.tight_layout()
    return fig


def plot_region_product_heatmap(df):
    """Heatmap of revenue by region and product.

    Args:
        df: Sales DataFrame.

    Returns:
        The matplotlib Figure.
    """
    pivot = df.pivot_table(
        index="Region", columns="Product", values="Total_Sales", aggfunc="sum", fill_value=0
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", ax=ax)
    ax.set_title("Revenue by Region and Product")
    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(y_test, y_pred):
    """Scatter plot comparing actual and predicted values.

    Args:
        y_test: True target values.
        y_pred: Predicted target values.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, color=PRIMARY_COLOR, alpha=0.7)
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], "--", color="gray", label="Perfect Prediction")
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title("Actual vs Predicted Sales")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_model_comparison(results_df):
    """Bar chart comparing R2 scores of different models.

    Args:
        results_df: DataFrame with columns 'Model' and 'R2'.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=results_df, x="Model", y="R2", color=PRIMARY_COLOR, ax=ax)
    ax.set_title("Model Comparison (R2 Score)")
    ax.set_ylabel("R2 Score")
    plt.tight_layout()
    return fig
