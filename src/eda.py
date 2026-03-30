# =============================================================================
# src/eda.py — Exploratory Data Analysis plots
# All plots save to outputs/plots/ automatically.
# =============================================================================

import logging
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from config import PLOTS_DIR, PLOT_DPI, PLOT_STYLE, HOLIDAY_COL, SALES_COL

logger = logging.getLogger(__name__)
sns.set_style(PLOT_STYLE)
plt.rcParams["figure.dpi"] = PLOT_DPI

FMT_K = mticker.FuncFormatter(lambda x, _: f"${int(x/1e3):,}K")
FMT_M = mticker.FuncFormatter(lambda x, _: f"${int(x/1e6):.1f}M")


def _save(filename: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {path}")


# -----------------------------------------------------------------------------
# 1. Aggregated weekly sales over time
# -----------------------------------------------------------------------------

def plot_weekly_sales(weekly_sales: pd.Series) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(weekly_sales, linewidth=2, color="darkblue",
             label="Aggregated Weekly Sales")
    plt.gca().yaxis.set_major_formatter(FMT_M)
    plt.title("Walmart Aggregated Weekly Sales — All 45 Stores",
              fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Total Sales ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save("weekly_sales_over_time.png")


# -----------------------------------------------------------------------------
# 2. Holiday vs non-holiday sales
# -----------------------------------------------------------------------------

def plot_holiday_impact(df: pd.DataFrame) -> None:
    holiday     = df[df[HOLIDAY_COL] == 1][SALES_COL]
    non_holiday = df[df[HOLIDAY_COL] == 0][SALES_COL]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].boxplot([non_holiday, holiday], labels=["Non-Holiday", "Holiday"])
    axes[0].set_ylabel("Weekly Sales ($)")
    axes[0].set_title("Sales Distribution: Holiday vs Non-Holiday",
                      fontweight="bold")
    axes[0].yaxis.set_major_formatter(FMT_K)
    axes[0].grid(axis="y", alpha=0.3)

    avgs  = [non_holiday.mean(), holiday.mean()]
    bars  = axes[1].bar(["Non-Holiday", "Holiday"], avgs,
                         color=["steelblue", "coral"], alpha=0.8)
    axes[1].set_ylabel("Average Weekly Sales ($)")
    axes[1].set_title("Average Sales Comparison", fontweight="bold")
    axes[1].yaxis.set_major_formatter(FMT_K)
    for bar in bars:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2., h,
                     f"${h/1e3:.0f}K", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save("holiday_impact.png")


# -----------------------------------------------------------------------------
# 3. Sales distribution
# -----------------------------------------------------------------------------

def plot_sales_distribution(df: pd.DataFrame) -> None:
    df_copy = df.copy()
    df_copy["Year"]    = df_copy["Date"].dt.year
    df_copy["Quarter"] = df_copy["Date"].dt.quarter

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].hist(df_copy[SALES_COL], bins=50, alpha=0.7,
                    color="steelblue", edgecolor="black")
    axes[0, 0].set_xlabel("Weekly Sales ($)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Distribution of Weekly Sales", fontweight="bold")
    axes[0, 0].xaxis.set_major_formatter(FMT_K)
    axes[0, 0].grid(alpha=0.3)

    stats.probplot(df_copy[SALES_COL], dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Q-Q Plot — Normality Test", fontweight="bold")
    axes[0, 1].grid(alpha=0.3)

    years = sorted(df_copy["Year"].unique())
    axes[1, 0].boxplot(
        [df_copy[df_copy["Year"] == y][SALES_COL] for y in years],
        labels=years
    )
    axes[1, 0].set_xlabel("Year")
    axes[1, 0].set_ylabel("Weekly Sales ($)")
    axes[1, 0].set_title("Sales Distribution by Year", fontweight="bold")
    axes[1, 0].yaxis.set_major_formatter(FMT_K)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].violinplot(
        [df_copy[df_copy["Quarter"] == q][SALES_COL].values for q in [1, 2, 3, 4]],
        positions=[1, 2, 3, 4],
        showmeans=True, showmedians=True
    )
    axes[1, 1].set_xticks([1, 2, 3, 4])
    axes[1, 1].set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    axes[1, 1].set_ylabel("Weekly Sales ($)")
    axes[1, 1].set_title("Sales Distribution by Quarter", fontweight="bold")
    axes[1, 1].yaxis.set_major_formatter(FMT_K)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    _save("sales_distribution.png")


# -----------------------------------------------------------------------------
# 4. Correlation heatmap
# -----------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    features = [SALES_COL, "Temperature", "Fuel_Price",
                "CPI", "Unemployment", HOLIDAY_COL]
    corr = df[features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm",
                center=0, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix — Sales and External Factors",
              fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    _save("correlation_heatmap.png")


# -----------------------------------------------------------------------------
# 5. Time series decomposition
# -----------------------------------------------------------------------------

def plot_decomposition(weekly_sales: pd.Series) -> None:
    decomposition = sm.tsa.seasonal_decompose(
        weekly_sales, model="additive", period=52
    )
    fig = decomposition.plot()
    fig.set_size_inches(14, 8)
    plt.suptitle("Time Series Decomposition (52-week Annual Period)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("decomposition.png")


# -----------------------------------------------------------------------------
# 6. ACF / PACF
# -----------------------------------------------------------------------------

def plot_acf_pacf(weekly_sales: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(weekly_sales,  lags=20, ax=axes[0])
    plot_pacf(weekly_sales, lags=15, ax=axes[1])
    axes[0].set_title("Autocorrelation Function (ACF)", fontweight="bold")
    axes[1].set_title("Partial Autocorrelation Function (PACF)", fontweight="bold")
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    _save("acf_pacf.png")


# -----------------------------------------------------------------------------
# 7. Top stores
# -----------------------------------------------------------------------------

def plot_top_stores(df: pd.DataFrame, top_n: int = 10) -> None:
    store_totals = (df.groupby("Store")[SALES_COL]
                      .sum()
                      .sort_values(ascending=False)
                      .head(top_n))

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(store_totals)), store_totals.values,
                   color="steelblue", alpha=0.8)
    plt.xticks(range(len(store_totals)),
               [f"Store {s}" for s in store_totals.index], rotation=45)
    plt.ylabel("Total Sales ($)")
    plt.title(f"Top {top_n} Walmart Stores by Total Sales",
              fontsize=14, fontweight="bold")
    plt.gca().yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${int(x/1e6):.1f}M")
    )
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h,
                 f"${h/1e6:.1f}M", ha="center", va="bottom", fontsize=9)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save("top_stores.png")


# -----------------------------------------------------------------------------
# Master EDA runner
# -----------------------------------------------------------------------------

def run_eda(df: pd.DataFrame, weekly_sales: pd.Series) -> None:
    """
    Run the full EDA suite and save all plots to outputs/plots/.
    """
    logger.info("Running EDA...")
    plot_weekly_sales(weekly_sales)
    plot_holiday_impact(df)
    plot_sales_distribution(df)
    plot_correlation_heatmap(df)
    plot_decomposition(weekly_sales)
    plot_acf_pacf(weekly_sales)
    plot_top_stores(df)
    logger.info("EDA complete. All plots saved.")
