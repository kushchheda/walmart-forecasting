# =============================================================================
# config.py — Central configuration for Walmart Sales Forecasting Pipeline
# All constants, paths, and model parameters live here.
# To change behavior, edit this file only — do not touch model code.
# =============================================================================

import os

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
DATA_PATH       = os.path.join("data", "Walmart.csv")
OUTPUT_DIR      = "outputs"
PLOTS_DIR       = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR     = os.path.join(OUTPUT_DIR, "results")

# -----------------------------------------------------------------------------
# Data constants
# -----------------------------------------------------------------------------
DATE_FORMAT     = "%d-%m-%Y"       # Format used in raw Walmart.csv
DATE_COL        = "Date"
SALES_COL       = "Weekly_Sales"
STORE_COL       = "Store"
HOLIDAY_COL     = "Holiday_Flag"

EXOG_FEATURES   = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Holiday_Flag"]

# -----------------------------------------------------------------------------
# Modeling constants
# -----------------------------------------------------------------------------
TEST_SIZE        = 13              # Last 13 weeks held out for testing
SEASONAL_PERIOD  = 13              # Quarterly seasonality (13 weeks)
MA_WINDOW        = 4               # Moving average window (4 weeks)

# ARIMA order — determined from ACF/PACF analysis in EDA
ARIMA_ORDER      = (5, 1, 0)

# SARIMA order — determined by grid search (AIC-optimized)
SARIMA_ORDER         = (3, 1, 3)
SARIMA_SEASONAL_ORDER = (0, 1, 2, SEASONAL_PERIOD)

# Prophet settings
PROPHET_SEASONALITY_MODE = "multiplicative"
PROPHET_INTERVAL_WIDTH   = 0.95

# -----------------------------------------------------------------------------
# Plot settings
# -----------------------------------------------------------------------------
PLOT_DPI     = 100
PLOT_STYLE   = "whitegrid"
