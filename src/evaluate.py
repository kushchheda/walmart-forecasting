# =============================================================================
# src/evaluate.py — Model evaluation, metrics, and comparison
# =============================================================================

import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from config import PLOTS_DIR, RESULTS_DIR, PLOT_DPI, PLOT_STYLE

logger = logging.getLogger(__name__)
sns.set_style(PLOT_STYLE)


# -----------------------------------------------------------------------------
# Metric functions
# -----------------------------------------------------------------------------

def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    predicted = predicted.reindex(actual.index)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    predicted = predicted.reindex(actual.index)
    return float(np.mean(np.abs(actual - predicted)))


def mape(actual: pd.Series, predicted: pd.Series) -> float:
    predicted = predicted.reindex(actual.index)
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def compute_metrics(actual: pd.Series, predicted: pd.Series,
                    model_name: str) -> dict:
    """
    Compute MAE, RMSE, and MAPE for a single model.

    Parameters
    ----------
    actual     : pd.Series  Ground truth test values
    predicted  : pd.Series  Model forecast values
    model_name : str

    Returns
    -------
    dict with keys: Model, MAE, RMSE, MAPE
    """
    return {
        "Model": model_name,
        "MAE":   mae(actual, predicted),
        "RMSE":  rmse(actual, predicted),
        "MAPE":  mape(actual, predicted)
    }


# -----------------------------------------------------------------------------
# Comparison table
# -----------------------------------------------------------------------------

def compare_models(actual: pd.Series,
                   baseline: pd.Series,
                   arima: pd.Series,
                   sarima: pd.Series,
                   prophet: pd.Series) -> pd.DataFrame:
    """
    Build a sorted comparison table across all four models.

    Saves results/model_comparison.csv automatically.

    Returns
    -------
    pd.DataFrame sorted by MAPE ascending (best model first).
    """
    results = [
        compute_metrics(actual, baseline, "Baseline MA(4)"),
        compute_metrics(actual, arima,    "ARIMA(5,1,0)"),
        compute_metrics(actual, sarima,   f"SARIMA(3,1,3)x(0,1,2,13)"),
        compute_metrics(actual, prophet,  "Prophet"),
    ]
    df = pd.DataFrame(results).sort_values("MAPE").reset_index(drop=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Model comparison saved → {out_path}")

    # Log summary
    best = df.iloc[0]
    baseline_mape = df[df["Model"] == "Baseline MA(4)"]["MAPE"].values[0]
    improvement = (baseline_mape - best["MAPE"]) / baseline_mape * 100

    logger.info("=" * 60)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("=" * 60)
    logger.info(f"\n{df.to_string(index=False)}")
    logger.info(f"\nBest model : {best['Model']}")
    logger.info(f"Best MAPE  : {best['MAPE']:.2f}%")
    logger.info(f"Improvement over baseline: {improvement:.1f}%")
    logger.info("=" * 60)

    return df


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def plot_forecast_comparison(train: pd.Series, test: pd.Series,
                              forecasts: dict) -> None:
    """
    Plot all model forecasts against actual test values on one chart.

    Parameters
    ----------
    train     : pd.Series  Training data
    test      : pd.Series  Actual test values
    forecasts : dict       {model_name: forecast_series}
    """
    colors = {
        "Baseline MA(4)":           "gray",
        "ARIMA(5,1,0)":             "orange",
        "SARIMA(3,1,3)x(0,1,2,13)": "red",
        "Prophet":                  "purple",
    }

    plt.figure(figsize=(14, 6))
    plt.plot(train, label="Training Data", color="blue",
             linewidth=1.5, alpha=0.7)
    plt.plot(test,  label="Actual (Test)", color="green",
             linewidth=2, marker="o", markersize=4)

    for name, forecast in forecasts.items():
        plt.plot(test.index, forecast, label=name,
                 color=colors.get(name, "black"),
                 linewidth=2, linestyle="--")

    plt.gca().yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${int(x/1e6):.1f}M")
    )
    plt.title("All Models: Forecast vs Actual (Last 13 Weeks)",
              fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Total Weekly Sales ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "forecast_comparison.png")
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    logger.info(f"Forecast comparison plot saved → {path}")


def plot_metrics_bar(comparison_df: pd.DataFrame) -> None:
    """
    Save a 3-panel bar chart of MAE, RMSE, and MAPE across all models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["MAE", "RMSE", "MAPE"]
    colors  = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        bars = ax.bar(comparison_df["Model"], comparison_df[metric],
                      color=colors[idx], alpha=0.7)
        ax.set_title(f"{metric} Comparison", fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "metrics_comparison.png")
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    logger.info(f"Metrics bar chart saved → {path}")


def save_forecasts(test: pd.Series, forecasts: dict) -> None:
    """
    Save all model forecasts alongside actuals to a single CSV.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame({"Actual": test})
    for name, series in forecasts.items():
        df[name] = series.values
    path = os.path.join(RESULTS_DIR, "forecasts.csv")
    df.to_csv(path)
    logger.info(f"Forecast values saved → {path}")
