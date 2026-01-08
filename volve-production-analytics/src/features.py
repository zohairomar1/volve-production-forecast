"""
Feature Engineering Module
==========================

Create additional features for analysis and forecasting.
"""

import pandas as pd
import numpy as np
from typing import Optional


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features from the date column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'date' column.

    Returns
    -------
    pd.DataFrame
        Dataframe with added time features.
    """
    df = df.copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # Month index (months since first date in dataset)
    min_date = df["date"].min()
    df["month_index"] = (
        (df["date"].dt.year - min_date.year) * 12 +
        (df["date"].dt.month - min_date.month)
    )

    return df


def add_rolling_features(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    windows: Optional[list] = None,
    group_col: str = "wellbore",
) -> pd.DataFrame:
    """
    Add rolling average features for specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list, optional
        Columns to compute rolling averages for. Defaults to ['oil', 'gas', 'water'].
    windows : list, optional
        Rolling window sizes in months. Defaults to [3, 6].
    group_col : str
        Column to group by for rolling calculations.

    Returns
    -------
    pd.DataFrame
        Dataframe with rolling average columns.
    """
    df = df.copy()
    columns = columns or ["oil", "gas", "water"]
    windows = windows or [3, 6]

    # Filter to columns that exist
    columns = [col for col in columns if col in df.columns]

    # Sort for rolling calculation
    df = df.sort_values([group_col, "date"])

    for col in columns:
        for window in windows:
            col_name = f"{col}_rolling_{window}m"
            df[col_name] = (
                df.groupby(group_col)[col]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )

    return df


def add_yoy_change(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    group_col: str = "wellbore",
) -> pd.DataFrame:
    """
    Add year-over-year percentage change features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list, optional
        Columns to compute YoY change for. Defaults to ['oil'].
    group_col : str
        Column to group by.

    Returns
    -------
    pd.DataFrame
        Dataframe with YoY change columns.
    """
    df = df.copy()
    columns = columns or ["oil"]
    columns = [col for col in columns if col in df.columns]

    df = df.sort_values([group_col, "date"])

    for col in columns:
        col_name = f"{col}_yoy_pct"
        # Shift by 12 months within each group
        shifted = df.groupby(group_col)[col].shift(12)
        # Calculate percentage change, avoiding division by zero
        df[col_name] = np.where(
            shifted > 0,
            ((df[col] - shifted) / shifted) * 100,
            np.nan
        )

    return df


def add_mom_change(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    group_col: str = "wellbore",
) -> pd.DataFrame:
    """
    Add month-over-month percentage change features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list, optional
        Columns to compute MoM change for. Defaults to ['oil'].
    group_col : str
        Column to group by.

    Returns
    -------
    pd.DataFrame
        Dataframe with MoM change columns.
    """
    df = df.copy()
    columns = columns or ["oil"]
    columns = [col for col in columns if col in df.columns]

    df = df.sort_values([group_col, "date"])

    for col in columns:
        col_name = f"{col}_mom_pct"
        shifted = df.groupby(group_col)[col].shift(1)
        df[col_name] = np.where(
            shifted > 0,
            ((df[col] - shifted) / shifted) * 100,
            np.nan
        )

    return df


def add_uptime_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate uptime rate from on_stream hours.

    Assumes on_stream is in hours. Uptime rate = on_stream / total_hours_in_month.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'on_stream' and 'date' columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with 'uptime_rate' column.
    """
    df = df.copy()

    if "on_stream" not in df.columns:
        return df

    # Calculate total hours in each month
    df["days_in_month"] = df["date"].dt.days_in_month
    df["hours_in_month"] = df["days_in_month"] * 24

    # Calculate uptime rate (0 to 1)
    df["uptime_rate"] = np.where(
        df["hours_in_month"] > 0,
        df["on_stream"] / df["hours_in_month"],
        np.nan
    )

    # Cap at 1.0 (can't have more than 100% uptime)
    df["uptime_rate"] = df["uptime_rate"].clip(upper=1.0)

    # Clean up temporary columns
    df = df.drop(columns=["days_in_month", "hours_in_month"])

    return df


def engineer_features(
    df: pd.DataFrame,
    add_rolling: bool = True,
    add_yoy: bool = True,
    add_mom: bool = True,
    add_uptime: bool = True,
) -> pd.DataFrame:
    """
    Apply all feature engineering steps.

    Parameters
    ----------
    df : pd.DataFrame
        Input production dataframe.
    add_rolling : bool
        Whether to add rolling average features.
    add_yoy : bool
        Whether to add year-over-year change features.
    add_mom : bool
        Whether to add month-over-month change features.
    add_uptime : bool
        Whether to add uptime rate feature.

    Returns
    -------
    pd.DataFrame
        Dataframe with all engineered features.
    """
    df = add_time_features(df)

    if add_rolling:
        df = add_rolling_features(df)

    if add_yoy:
        df = add_yoy_change(df)

    if add_mom:
        df = add_mom_change(df)

    if add_uptime:
        df = add_uptime_rate(df)

    return df
