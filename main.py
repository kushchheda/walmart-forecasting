# =============================================================================
# main.py — Walmart Sales Forecasting Pipeline
#
# Usage:
#   python main.py
#
# What it does:
#   1. Loads and cleans raw Walmart sales data
#   2. Runs exploratory data analysis (saves plots to outputs/plots/)
#   3. Trains four models: MA(4) baseline, ARIMA, SARIMA, Prophet
#   4. Evaluates all models on the last 13 weeks
#   5. Saves forecast values and comparison metrics to outputs/results/
# =============================================================================

import logging
import os

from config import PLOTS_DIR, RESULTS_DIR
from src.clean    import run_cleaning_pipeline
from src.eda      import run_eda
from src.features import split_train_test, get_baseline_forecast
from src.models   import run_arima, run_sarima, run_prophet
from src.evaluate import (
    compare_models,
    plot_forecast_comparison,
    plot_metrics_bar,
    save_forecasts,
)

# -----------------------------------------------------------------------------
# Logging setup — logs to console with timestamps
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("WALMART SALES FORECASTING PIPELINE — START")
    logger.info("=" * 60)

    # Create output directories
    os.makedirs(PLOTS_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load, clean, and aggregate data
    # ------------------------------------------------------------------
    logger.info("STEP 1: Data loading and cleaning")
    df, weekly_sales, weekly_exog = run_cleaning_pipeline()

    # ------------------------------------------------------------------
    # Step 2: Exploratory data analysis
    # ------------------------------------------------------------------
    logger.info("STEP 2: Exploratory data analysis")
    run_eda(df, weekly_sales)

    # ------------------------------------------------------------------
    # Step 3: Train / test split
    # ------------------------------------------------------------------
    logger.info("STEP 3: Splitting train / test sets")
    train, test = split_train_test(weekly_sales)

    # ------------------------------------------------------------------
    # Step 4: Fit models and generate forecasts
    # ------------------------------------------------------------------
    logger.info("STEP 4: Fitting models")

    baseline_forecast = get_baseline_forecast(train, test)
    arima_forecast    = run_arima(train, test)
    sarima_forecast   = run_sarima(train, test)
    prophet_forecast  = run_prophet(train, test)

    forecasts = {
        "Baseline MA(4)":            baseline_forecast,
        "ARIMA(5,1,0)":              arima_forecast,
        "SARIMA(3,1,3)x(0,1,2,13)": sarima_forecast,
        "Prophet":                   prophet_forecast,
    }

    # ------------------------------------------------------------------
    # Step 5: Evaluate and save results
    # ------------------------------------------------------------------
    logger.info("STEP 5: Evaluating models")

    comparison_df = compare_models(
        test,
        baseline_forecast,
        arima_forecast,
        sarima_forecast,
        prophet_forecast,
    )

    plot_forecast_comparison(train, test, forecasts)
    plot_metrics_bar(comparison_df)
    save_forecasts(test, forecasts)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    best = comparison_df.iloc[0]
    baseline_mape = comparison_df[
        comparison_df["Model"] == "Baseline MA(4)"
    ]["MAPE"].values[0]
    improvement = (baseline_mape - best["MAPE"]) / baseline_mape * 100

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Best model : {best['Model']}")
    logger.info(f"MAPE       : {best['MAPE']:.2f}%")
    logger.info(f"Improvement over baseline: {improvement:.1f}%")
    logger.info(f"Plots   → {PLOTS_DIR}/")
    logger.info(f"Results → {RESULTS_DIR}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
