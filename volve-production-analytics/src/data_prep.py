"""
Data Preparation Module
=======================

Handles loading, cleaning, and standardizing Volve production data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union

from .config import COLUMN_MAPPING, OUTPUT_COLUMNS, PROCESSED_DATA_DIR


def load_raw_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load raw production data from CSV, Excel, or text file.

    Parameters
    ----------
    filepath : str or Path
        Path to the raw data file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe as loaded from file.
    """
    filepath = Path(filepath)

    if filepath.suffix.lower() == ".csv":
        return pd.read_csv(filepath)
    elif filepath.suffix.lower() == ".xlsx":
        return pd.read_excel(filepath)
    elif filepath.suffix.lower() == ".txt":
        # Try tab-separated first, then comma
        try:
            return pd.read_csv(filepath, sep="\t")
        except Exception:
            return pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to standard schema using COLUMN_MAPPING.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with original column names.

    Returns
    -------
    pd.DataFrame
        Dataframe with standardized column names.
    """
    # Create reverse mapping for columns that exist in the dataframe
    rename_map = {}
    for original, standard in COLUMN_MAPPING.items():
        if original in df.columns:
            rename_map[original] = standard

    return df.rename(columns=rename_map)


def create_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a proper datetime date column from Year and Month.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'year' and 'month' columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with 'date' column added.
    """
    df = df.copy()

    if "year" in df.columns and "month" in df.columns:
        # Create date as first day of each month
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str) + "-01"
        )
    elif "DATEPRD" in df.columns:
        # Alternative: parse existing date column
        df["date"] = pd.to_datetime(df["DATEPRD"])
        # Normalize to first of month
        df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    else:
        raise ValueError("Cannot create date column: missing year/month or date columns")

    return df


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert production columns to numeric, handling invalid values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with cleaned numeric columns.
    """
    df = df.copy()
    numeric_cols = ["oil", "gas", "water", "on_stream", "gas_injection", "water_injection"]

    for col in numeric_cols:
        if col in df.columns:
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Replace negative values with NaN (invalid for production)
            df.loc[df[col] < 0, col] = np.nan

    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in production data.

    Strategy:
    - Production volumes: keep NaN (don't assume zero production)
    - Injection volumes: fill with 0 (missing often means no injection)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with handled missing values.
    """
    df = df.copy()

    # Fill injection columns with 0 (no injection is common)
    for col in ["gas_injection", "water_injection"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and order output columns per schema.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with only output columns.
    """
    # Only select columns that exist
    available_cols = [col for col in OUTPUT_COLUMNS if col in df.columns]
    return df[available_cols]


def prepare_data(
    filepath: Union[str, Path],
    save_output: bool = True,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Full data preparation pipeline.

    Parameters
    ----------
    filepath : str or Path
        Path to raw data file.
    save_output : bool
        Whether to save processed data to files.
    output_dir : Path, optional
        Directory for output files. Defaults to PROCESSED_DATA_DIR.

    Returns
    -------
    pd.DataFrame
        Cleaned, standardized production dataframe.
    """
    output_dir = output_dir or PROCESSED_DATA_DIR

    # Load raw data
    df = load_raw_data(filepath)

    # Standardize column names
    df = standardize_columns(df)

    # Create date column
    df = create_date_column(df)

    # Clean numeric columns
    df = clean_numeric_columns(df)

    # Handle missing values
    df = fill_missing_values(df)

    # Select output columns
    df = select_output_columns(df)

    # Sort by wellbore and date
    df = df.sort_values(["wellbore", "date"]).reset_index(drop=True)

    # Save outputs
    if save_output:
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_dir / "volve_monthly.parquet", index=False)
        df.to_csv(output_dir / "volve_monthly.csv", index=False)

    return df


def load_processed_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load processed production data.

    Parameters
    ----------
    filepath : str or Path, optional
        Path to processed data. Defaults to standard location.

    Returns
    -------
    pd.DataFrame
        Processed production dataframe.
    """
    if filepath is None:
        filepath = PROCESSED_DATA_DIR / "volve_monthly.parquet"

    filepath = Path(filepath)

    if filepath.suffix == ".parquet":
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath, parse_dates=["date"])

    return df


def aggregate_total_production(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate production across all wellbores to get total field production.

    Parameters
    ----------
    df : pd.DataFrame
        Wellbore-level production data.

    Returns
    -------
    pd.DataFrame
        Monthly total production for the field.
    """
    numeric_cols = ["oil", "gas", "water", "on_stream", "gas_injection", "water_injection"]
    available_cols = [col for col in numeric_cols if col in df.columns]

    total = df.groupby("date")[available_cols].sum().reset_index()
    total["wellbore"] = "TOTAL"

    return total
