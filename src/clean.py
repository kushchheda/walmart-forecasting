# =============================================================================
# src/clean.py — Data loading, cleaning, and aggregation
# =============================================================================

import logging
import pandas as pd

from config import (
    DATA_PATH, DATE_COL, DATE_FORMAT, SALES_COL,
    STORE_COL, EXOG_FEATURES, HOLIDAY_COL
)

logger = logging.getLogger(__name__)


def load_raw(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Walmart CSV and parse dates.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with Date parsed as datetime.
    """
    logger.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format=DATE_FORMAT)
    logger.info(f"Loaded {len(df):,} records | "
                f"{df[STORE_COL].nunique()} stores | "
                f"{df[DATE_COL].nunique()} weeks")
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run basic data quality checks and log findings.

    Checks
    ------
    - Missing values
    - Negative sales (known issue in Walmart dataset)
    - Duplicate rows
    """
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"Missing values found:\n{missing[missing > 0]}")
    else:
        logger.info("No missing values found.")

    neg_sales = (df[SALES_COL] < 0).sum()
    if neg_sales > 0:
        logger.warning(f"{neg_sales} rows with negative Weekly_Sales — will be removed.")

    dupes = df.duplicated().sum()
    if dupes > 0:
        logger.warning(f"{dupes} duplicate rows found — will be removed.")

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning rules to raw dataframe.

    Steps
    -----
    1. Remove negative sales
    2. Drop duplicates
    """
    initial_len = len(df)
    df = df[df[SALES_COL] >= 0].drop_duplicates().reset_index(drop=True)
    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} rows during cleaning.")
    return df


def aggregate_weekly(df: pd.DataFrame):
    """
    Aggregate all 45 stores into a single weekly time series.

    - Weekly_Sales  → summed across all stores
    - Exogenous features → averaged across stores (Holiday_Flag takes max)

    Returns
    -------
    weekly_sales : pd.Series
        Total weekly sales indexed by Date.
    weekly_exog : pd.DataFrame
        Weekly exogenous features indexed by Date.
    """
    df_indexed = df.set_index(DATE_COL)

    weekly_sales = df_indexed[SALES_COL].resample("W").sum()

    weekly_exog = df_indexed[EXOG_FEATURES].resample("W").mean()
    weekly_exog[HOLIDAY_COL] = df_indexed[HOLIDAY_COL].resample("W").max().fillna(0)

    logger.info(f"Aggregated to {len(weekly_sales)} weekly observations.")
    return weekly_sales, weekly_exog


def run_cleaning_pipeline(filepath: str = DATA_PATH):
    """
    Full cleaning pipeline: load → validate → clean → aggregate.

    Returns
    -------
    df : pd.DataFrame
        Cleaned raw dataframe (store-level).
    weekly_sales : pd.Series
        Aggregated weekly sales.
    weekly_exog : pd.DataFrame
        Aggregated weekly exogenous features.
    """
    df = load_raw(filepath)
    df = validate(df)
    df = clean(df)
    weekly_sales, weekly_exog = aggregate_weekly(df)
    return df, weekly_sales, weekly_exog
