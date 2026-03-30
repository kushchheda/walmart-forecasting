# =============================================================================
# src/models.py — ARIMA, SARIMA, and Prophet forecasting models
# =============================================================================

import logging
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import (
    ARIMA_ORDER,
    SARIMA_ORDER, SARIMA_SEASONAL_ORDER,
    PROPHET_SEASONALITY_MODE, PROPHET_INTERVAL_WIDTH,
    TEST_SIZE
)

logger = logging.getLogger(__name__)


def run_arima(train: pd.Series, test: pd.Series,
              order: tuple = ARIMA_ORDER) -> pd.Series:
    """
    Fit ARIMA model on training data and forecast test period.

    Order (5,1,0) was selected based on ACF/PACF analysis:
    - p=5: 5 autoregressive lags (from PACF cutoff)
    - d=1: one differencing pass to remove trend
    - q=0: no moving average terms needed

    Parameters
    ----------
    train : pd.Series
    test  : pd.Series  (used only for its index)
    order : tuple      (p, d, q)

    Returns
    -------
    pd.Series
        ARIMA forecast indexed to test period.
    """
    logger.info(f"Fitting ARIMA{order}...")
    model = ARIMA(train, order=order)
    fit   = model.fit()
    forecast = fit.forecast(steps=len(test))
    forecast.index = test.index
    logger.info(f"ARIMA{order} fitted. AIC: {fit.aic:.2f}")
    return forecast


def run_sarima(train: pd.Series, test: pd.Series,
               order: tuple = SARIMA_ORDER,
               seasonal_order: tuple = SARIMA_SEASONAL_ORDER) -> pd.Series:
    """
    Fit SARIMA model on training data and forecast test period.

    Parameters (3,1,3)x(0,1,2,13) were selected via AIC-minimizing
    grid search over p∈[0-3], d∈[0-1], q∈[0-3],
    P∈[0-2], D∈[0-1], Q∈[0-2] with seasonal period m=13.

    Parameters
    ----------
    train          : pd.Series
    test           : pd.Series  (used only for its index)
    order          : tuple (p, d, q)
    seasonal_order : tuple (P, D, Q, m)

    Returns
    -------
    pd.Series
        SARIMA forecast indexed to test period.
    """
    logger.info(f"Fitting SARIMA{order}x{seasonal_order}...")
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fit = model.fit(disp=False, maxiter=200)
    forecast_obj = fit.get_forecast(steps=len(test))
    forecast = forecast_obj.predicted_mean
    forecast.index = test.index
    logger.info(f"SARIMA{order}x{seasonal_order} fitted. AIC: {fit.aic:.2f}")
    return forecast


def run_prophet(train: pd.Series, test: pd.Series,
                seasonality_mode: str = PROPHET_SEASONALITY_MODE,
                interval_width: float = PROPHET_INTERVAL_WIDTH) -> pd.Series:
    """
    Fit a Prophet model on training data and forecast test period.

    Configuration:
    - weekly_seasonality=True  : captures within-quarter patterns
    - yearly_seasonality=True  : captures annual holiday spikes
    - daily_seasonality=False  : not applicable to weekly data
    - seasonality_mode='multiplicative' : handles holiday spike scaling

    Parameters
    ----------
    train            : pd.Series
    test             : pd.Series  (used only for its index)
    seasonality_mode : str
    interval_width   : float

    Returns
    -------
    pd.Series
        Prophet forecast indexed to test period.
    """
    try:
        from prophet import Prophet
    except ImportError:
        try:
            from fbprophet import Prophet
        except ImportError:
            logger.error("Prophet is not installed. Run: pip install prophet")
            raise

    logger.info("Fitting Prophet model...")

    df_train = train.reset_index()
    df_train.columns = ["ds", "y"]

    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode=seasonality_mode,
        interval_width=interval_width
    )
    model.fit(df_train)

    future = model.make_future_dataframe(periods=len(test), freq="W")
    forecast_df = model.predict(future)
    forecast = (
        forecast_df
        .set_index("ds")["yhat"]
        .iloc[-len(test):]
        .reindex(test.index)
    )
    logger.info("Prophet model fitted successfully.")
    return forecast
