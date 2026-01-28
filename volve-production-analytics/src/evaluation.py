"""
Evaluation Module
=================

Backtesting and model evaluation metrics for forecasting.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Literal, Optional
from pathlib import Path

from .config import BACKTEST_PERIODS, PROCESSED_DATA_DIR, MIN_HISTORY_MONTHS
from .forecasting import forecast_series


def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(actual - predicted))


def mean_absolute_percentage_error(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Excludes observations where actual is zero or near-zero to avoid
    division artifacts. Returns NaN if no valid observations remain.
    """
    # Exclude zero/near-zero actuals (standard MAPE practice)
    mask = np.abs(actual) > 1e-3
    if not np.any(mask):
        return np.nan
    actual_valid = actual[mask]
    predicted_valid = predicted[mask]
    return np.mean(np.abs((actual_valid - predicted_valid) / actual_valid)) * 100


def root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((actual - predicted) ** 2))


def weighted_absolute_percentage_error(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Calculate Weighted Absolute Percentage Error (WAPE).

    More robust than MAPE for series with varying magnitudes or near-zero values.
    WAPE = sum(|actual - predicted|) / sum(|actual|) * 100

    Returns NaN only if total actual is zero (undefined denominator).
    """
    # Remove NaN values from both arrays (paired removal)
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]

    if len(actual_clean) == 0:
        return np.nan

    total_actual = np.sum(np.abs(actual_clean))
    if total_actual < 1e-3:
        # Denominator is zero - WAPE undefined
        return np.nan

    return np.sum(np.abs(actual_clean - predicted_clean)) / total_actual * 100


def rolling_origin_backtest(
    df: pd.DataFrame,
    target_col: str = "oil",
    series_id: Optional[str] = None,
    model: Literal["baseline", "ets"] = "ets",
    test_periods: int = BACKTEST_PERIODS,
    forecast_horizon: int = 1,
) -> pd.DataFrame:
    """
    Perform rolling-origin backtesting.

    For each test period, train on all data before that period and
    forecast the next `forecast_horizon` months.

    Parameters
    ----------
    df : pd.DataFrame
        Production data.
    target_col : str
        Column to forecast.
    series_id : str, optional
        Wellbore to evaluate. If None, evaluates total production.
    model : str
        Model type: 'baseline' or 'ets'.
    test_periods : int
        Number of periods to use for testing.
    forecast_horizon : int
        Forecast horizon for each origin.

    Returns
    -------
    pd.DataFrame
        Backtest results with actual and predicted values.
    """
    df = df.copy()

    # Get series data
    if series_id is None or series_id == "TOTAL":
        series_df = df.groupby("date")[target_col].sum().reset_index()
        series_id = "TOTAL"
    else:
        series_df = df[df["wellbore"] == series_id][["date", target_col]]

    series_df = series_df.sort_values("date").reset_index(drop=True)

    # Determine test period start
    n_periods = len(series_df)
    if n_periods < MIN_HISTORY_MONTHS + test_periods:
        # Reduce test periods if not enough data
        test_periods = max(1, n_periods - MIN_HISTORY_MONTHS)

    test_start_idx = n_periods - test_periods

    results = []

    for i in range(test_periods):
        # Training data: everything before current test point
        train_end_idx = test_start_idx + i
        train_df = series_df.iloc[:train_end_idx]

        if len(train_df) < MIN_HISTORY_MONTHS:
            continue

        # Actual values for this forecast horizon
        actual_end_idx = min(train_end_idx + forecast_horizon, n_periods)
        actual_values = series_df.iloc[train_end_idx:actual_end_idx]

        if len(actual_values) == 0:
            continue

        try:
            # Create temporary df for forecast function
            temp_df = train_df.copy()
            temp_df["wellbore"] = series_id

            # Generate forecast
            forecast_df = forecast_series(
                temp_df,
                target_col=target_col,
                series_id=series_id,
                model=model,
                horizon=forecast_horizon,
            )

            # Match forecasts to actuals
            for j, (_, actual_row) in enumerate(actual_values.iterrows()):
                if j < len(forecast_df):
                    results.append({
                        "date": actual_row["date"],
                        "actual": actual_row[target_col],
                        "predicted": forecast_df.iloc[j]["yhat"],
                        "horizon": j + 1,
                        "model": model,
                        "series_id": series_id,
                    })

        except Exception:
            continue

    return pd.DataFrame(results)


def compute_backtest_metrics(backtest_results: pd.DataFrame) -> Dict:
    """
    Compute evaluation metrics from backtest results.

    Parameters
    ----------
    backtest_results : pd.DataFrame
        Results from rolling_origin_backtest.

    Returns
    -------
    Dict
        Dictionary of metrics including MAE, MAPE, WAPE, RMSE.
    """
    if len(backtest_results) == 0:
        return {"mae": np.nan, "mape": np.nan, "wape": np.nan, "rmse": np.nan, "n_observations": 0}

    # Convert to float64 arrays explicitly (fixes object dtype issues with np.isnan)
    actual = np.asarray(backtest_results["actual"], dtype=np.float64)
    predicted = np.asarray(backtest_results["predicted"], dtype=np.float64)

    # Remove any NaN pairs - all metrics use the same cleaned arrays
    valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual_clean = actual[valid_mask]
    predicted_clean = predicted[valid_mask]

    if len(actual_clean) == 0:
        return {"mae": np.nan, "mape": np.nan, "wape": np.nan, "rmse": np.nan, "n_observations": 0}

    # Compute all metrics on the same cleaned arrays
    mae_val = float(np.mean(np.abs(actual_clean - predicted_clean)))
    rmse_val = float(np.sqrt(np.mean((actual_clean - predicted_clean) ** 2)))

    # MAPE: exclude zero actuals
    nonzero_mask = np.abs(actual_clean) > 1e-3
    if np.any(nonzero_mask):
        mape_val = float(np.mean(np.abs((actual_clean[nonzero_mask] - predicted_clean[nonzero_mask]) / actual_clean[nonzero_mask])) * 100)
    else:
        mape_val = np.nan

    # WAPE: sum(|error|) / sum(|actual|)
    total_actual = np.sum(np.abs(actual_clean))
    if total_actual > 1e-3:
        wape_val = float(np.sum(np.abs(actual_clean - predicted_clean)) / total_actual * 100)
    else:
        wape_val = np.nan  # Denominator is zero

    return {
        "mae": mae_val,
        "mape": mape_val,
        "wape": wape_val,
        "rmse": rmse_val,
        "n_observations": len(actual_clean),
    }


def evaluate_models(
    df: pd.DataFrame,
    target_col: str = "oil",
    series_ids: Optional[List[str]] = None,
    test_periods: int = BACKTEST_PERIODS,
    forecast_horizon: int = 1,
) -> pd.DataFrame:
    """
    Evaluate both baseline and ETS models for multiple series.

    Parameters
    ----------
    df : pd.DataFrame
        Production data.
    target_col : str
        Column to forecast.
    series_ids : list, optional
        List of series to evaluate. If None, evaluates TOTAL and all wellbores.
    test_periods : int
        Number of test periods.
    forecast_horizon : int
        Forecast horizon.

    Returns
    -------
    pd.DataFrame
        Metrics for each model and series combination.
    """
    if series_ids is None:
        series_ids = ["TOTAL"] + list(df["wellbore"].unique())

    results = []
    models = ["baseline", "ets"]

    for series_id in series_ids:
        for model in models:
            try:
                backtest_df = rolling_origin_backtest(
                    df,
                    target_col=target_col,
                    series_id=series_id,
                    model=model,
                    test_periods=test_periods,
                    forecast_horizon=forecast_horizon,
                )

                metrics = compute_backtest_metrics(backtest_df)
                metrics["series_id"] = series_id
                metrics["model"] = model
                metrics["horizon"] = forecast_horizon
                results.append(metrics)

            except Exception:
                continue

    return pd.DataFrame(results)


def save_metrics(
    metrics_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> None:
    """
    Save evaluation metrics to JSON file.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics dataframe.
    output_path : Path, optional
        Output file path. Defaults to standard location.
    """
    if output_path is None:
        output_path = PROCESSED_DATA_DIR / "metrics.json"

    # Convert to nested dictionary structure
    metrics_dict = {
        "evaluation_results": metrics_df.to_dict(orient="records"),
        "summary": {
            "total_series_evaluated": len(metrics_df["series_id"].unique()),
            "models_compared": list(metrics_df["model"].unique()),
        }
    }

    # Add model comparison summary
    if len(metrics_df) > 0:
        total_metrics = metrics_df[metrics_df["series_id"] == "TOTAL"]
        if len(total_metrics) > 0:
            metrics_dict["summary"]["total_field_metrics"] = (
                total_metrics[["model", "mae", "mape", "rmse"]]
                .to_dict(orient="records")
            )

    with open(output_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)


def load_metrics(filepath: Optional[Path] = None) -> Dict:
    """
    Load metrics from JSON file.

    Parameters
    ----------
    filepath : Path, optional
        Path to metrics file.

    Returns
    -------
    Dict
        Metrics dictionary.
    """
    if filepath is None:
        filepath = PROCESSED_DATA_DIR / "metrics.json"

    with open(filepath, "r") as f:
        return json.load(f)
