# =============================================================================
# src/features.py — Feature engineering and train/test splitting
# =============================================================================

import logging
import pandas as pd

from config import TEST_SIZE, MA_WINDOW

logger = logging.getLogger(__name__)


def add_moving_average(weekly_sales: pd.Series, window: int = MA_WINDOW) -> pd.Series:
    """
    Compute a rolling moving average baseline forecast.

    Uses shift(1) before rolling to prevent data leakage —
    the average only uses data from BEFORE the current week.

    Parameters
    ----------
    weekly_sales : pd.Series
        Aggregated weekly sales indexed by Date.
    window : int
        Number of weeks to average (default: 4).

    Returns
    -------
    pd.Series
        Rolling moving average series (same index as input).
    """
    ma = weekly_sales.shift(1).rolling(window=window).mean()
    logger.info(f"Computed {window}-week moving average (leak-free).")
    return ma


def split_train_test(weekly_sales: pd.Series, test_size: int = TEST_SIZE):
    """
    Split the weekly series into train and test sets.

    The last `test_size` weeks are held out as the test set,
    matching the 13-week holdout used in the original analysis.

    Parameters
    ----------
    weekly_sales : pd.Series
        Full aggregated weekly sales series.
    test_size : int
        Number of weeks to hold out for testing.

    Returns
    -------
    train : pd.Series
    test  : pd.Series
    """
    train = weekly_sales.iloc[:-test_size]
    test  = weekly_sales.iloc[-test_size:]

    logger.info(f"Train: {len(train)} weeks "
                f"({train.index.min().date()} → {train.index.max().date()})")
    logger.info(f"Test:  {len(test)} weeks "
                f"({test.index.min().date()} → {test.index.max().date()})")
    return train, test


def get_baseline_forecast(train: pd.Series, test: pd.Series,
                           window: int = MA_WINDOW) -> pd.Series:
    """
    Generate the MA(4) baseline forecast for the test period.

    Takes the last moving-average value from the training set
    and projects it flat across all test weeks.

    Parameters
    ----------
    train : pd.Series
        Training portion of weekly sales.
    test : pd.Series
        Test portion (used only for its index).
    window : int
        Moving average window.

    Returns
    -------
    pd.Series
        Flat forecast indexed to the test period.
    """
    ma_train = add_moving_average(train, window=window)
    last_ma_value = ma_train.iloc[-1]
    baseline_forecast = pd.Series([last_ma_value] * len(test), index=test.index)
    logger.info(f"Baseline MA({window}) forecast value: ${last_ma_value:,.2f}")
    return baseline_forecast
