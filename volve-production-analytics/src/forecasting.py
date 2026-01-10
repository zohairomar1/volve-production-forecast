"""
Forecasting Module
==================

Time-series forecasting models for production prediction.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Literal
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

from .config import FORECAST_HORIZON, MIN_HISTORY_MONTHS


def seasonal_naive_forecast(
    series: pd.Series,
    horizon: int = FORECAST_HORIZON,
    seasonal_period: int = 12,
) -> pd.DataFrame:
    """
    Seasonal naive forecast: predict using value from same month last year.

    Parameters
    ----------
    series : pd.Series
        Historical time series indexed by date.
    horizon : int
        Number of periods to forecast.
    seasonal_period : int
        Seasonal period (12 for monthly data).

    Returns
    -------
    pd.DataFrame
        Forecast dataframe with columns: date, yhat.
    """
    series = series.sort_index()

    # Get last date
    last_date = series.index[-1]

    # Generate forecast dates
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS"
    )

    # Get seasonal values
    forecasts = []
    for date in forecast_dates:
        # Find same month from previous year(s)
        same_month_values = series[series.index.month == date.month]
        if len(same_month_values) > 0:
            # Use most recent same-month value
            forecasts.append(same_month_values.iloc[-1])
        else:
            # Fallback to last known value
            forecasts.append(series.iloc[-1])

    return pd.DataFrame({
        "date": forecast_dates,
        "yhat": forecasts,
    })


def naive_forecast(
    series: pd.Series,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """
    Simple naive forecast: predict using last known value.

    Parameters
    ----------
    series : pd.Series
        Historical time series indexed by date.
    horizon : int
        Number of periods to forecast.

    Returns
    -------
    pd.DataFrame
        Forecast dataframe with columns: date, yhat.
    """
    series = series.sort_index()
    last_date = series.index[-1]
    last_value = series.iloc[-1]

    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS"
    )

    return pd.DataFrame({
        "date": forecast_dates,
        "yhat": [last_value] * horizon,
    })


def exponential_smoothing_forecast(
    series: pd.Series,
    horizon: int = FORECAST_HORIZON,
    seasonal_periods: int = 12,
    trend: Optional[str] = "add",
    seasonal: Optional[str] = "add",
) -> pd.DataFrame:
    """
    Holt-Winters Exponential Smoothing forecast.

    Parameters
    ----------
    series : pd.Series
        Historical time series indexed by date.
    horizon : int
        Number of periods to forecast.
    seasonal_periods : int
        Number of periods in a seasonal cycle.
    trend : str, optional
        Type of trend component ('add', 'mul', or None).
    seasonal : str, optional
        Type of seasonal component ('add', 'mul', or None).

    Returns
    -------
    pd.DataFrame
        Forecast dataframe with columns: date, yhat, yhat_lower, yhat_upper.
    """
    series = series.sort_index()

    # Ensure no negative or zero values for multiplicative models
    if seasonal == "mul" or trend == "mul":
        series = series.clip(lower=1)

    # Check if we have enough data for seasonality
    if len(series) < 2 * seasonal_periods:
        # Fall back to non-seasonal model
        seasonal = None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = ExponentialSmoothing(
                series,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods if seasonal else None,
                initialization_method="estimated",
            )
            fitted = model.fit(optimized=True)

            # Generate forecast
            forecast = fitted.forecast(horizon)

            # Generate prediction intervals (approximate using residual std)
            residuals = fitted.resid
            std_resid = residuals.std()

            forecast_dates = pd.date_range(
                start=series.index[-1] + pd.DateOffset(months=1),
                periods=horizon,
                freq="MS"
            )

            return pd.DataFrame({
                "date": forecast_dates,
                "yhat": forecast.values,
                "yhat_lower": forecast.values - 1.96 * std_resid,
                "yhat_upper": forecast.values + 1.96 * std_resid,
            })

    except Exception:
        # Fallback to seasonal naive if ETS fails
        return seasonal_naive_forecast(series, horizon)


def forecast_series(
    df: pd.DataFrame,
    target_col: str = "oil",
    series_id: Optional[str] = None,
    model: Literal["baseline", "ets"] = "ets",
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """
    Generate forecast for a specific series.

    Parameters
    ----------
    df : pd.DataFrame
        Production data with 'date', 'wellbore', and target column.
    target_col : str
        Column to forecast.
    series_id : str, optional
        Wellbore to forecast. If None, forecasts total production.
    model : str
        Model type: 'baseline' (seasonal naive) or 'ets' (exponential smoothing).
    horizon : int
        Forecast horizon in months.

    Returns
    -------
    pd.DataFrame
        Forecast dataframe with columns: date, yhat, series_id.
    """
    df = df.copy()

    # Filter to specific wellbore or aggregate
    if series_id is None or series_id == "TOTAL":
        # Aggregate total production
        series_df = df.groupby("date")[target_col].sum().reset_index()
        series_id = "TOTAL"
    else:
        series_df = df[df["wellbore"] == series_id][["date", target_col]]

    if len(series_df) < MIN_HISTORY_MONTHS:
        raise ValueError(
            f"Insufficient history for forecasting. Need {MIN_HISTORY_MONTHS} months, "
            f"have {len(series_df)}."
        )

    # Create series with date index
    series = series_df.set_index("date")[target_col].sort_index()

    # Generate forecast
    if model == "baseline":
        forecast_df = seasonal_naive_forecast(series, horizon)
    else:
        forecast_df = exponential_smoothing_forecast(series, horizon)

    # Add series identifier
    forecast_df["series_id"] = series_id
    forecast_df["model"] = model

    # Ensure non-negative forecasts
    forecast_df["yhat"] = forecast_df["yhat"].clip(lower=0)
    if "yhat_lower" in forecast_df.columns:
        forecast_df["yhat_lower"] = forecast_df["yhat_lower"].clip(lower=0)
    if "yhat_upper" in forecast_df.columns:
        forecast_df["yhat_upper"] = forecast_df["yhat_upper"].clip(lower=0)

    return forecast_df


def forecast_all_wellbores(
    df: pd.DataFrame,
    target_col: str = "oil",
    model: Literal["baseline", "ets"] = "ets",
    horizon: int = FORECAST_HORIZON,
    include_total: bool = True,
) -> pd.DataFrame:
    """
    Generate forecasts for all wellbores and optionally total production.

    Parameters
    ----------
    df : pd.DataFrame
        Production data.
    target_col : str
        Column to forecast.
    model : str
        Model type.
    horizon : int
        Forecast horizon.
    include_total : bool
        Whether to include total field forecast.

    Returns
    -------
    pd.DataFrame
        Combined forecast dataframe for all series.
    """
    forecasts = []

    # Forecast total
    if include_total:
        try:
            total_forecast = forecast_series(df, target_col, None, model, horizon)
            forecasts.append(total_forecast)
        except ValueError:
            pass

    # Forecast each wellbore
    for wellbore in df["wellbore"].unique():
        try:
            wb_forecast = forecast_series(df, target_col, wellbore, model, horizon)
            forecasts.append(wb_forecast)
        except ValueError:
            continue

    if not forecasts:
        return pd.DataFrame()

    return pd.concat(forecasts, ignore_index=True)


def get_historical_with_forecast(
    df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    series_id: str = "TOTAL",
    target_col: str = "oil",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get aligned historical and forecast data for plotting.

    Parameters
    ----------
    df : pd.DataFrame
        Historical production data.
    forecast_df : pd.DataFrame
        Forecast data.
    series_id : str
        Series to extract.
    target_col : str
        Target column name.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Historical and forecast dataframes.
    """
    # Get historical
    if series_id == "TOTAL":
        historical = df.groupby("date")[target_col].sum().reset_index()
    else:
        historical = df[df["wellbore"] == series_id][["date", target_col]]

    historical = historical.sort_values("date")

    # Get forecast for this series
    forecast = forecast_df[forecast_df["series_id"] == series_id].copy()

    return historical, forecast
