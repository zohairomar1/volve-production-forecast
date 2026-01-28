"""
Reporting Module
================

Generate email-ready summaries and reports.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict
from pathlib import Path

from .config import PROCESSED_DATA_DIR, DASHBOARD_URL
from .features import add_rolling_features


def get_last_month_summary(df: pd.DataFrame) -> Dict:
    """
    Get summary statistics for the most recent month with actual production.

    Parameters
    ----------
    df : pd.DataFrame
        Production data with features.

    Returns
    -------
    Dict
        Summary statistics.
    """
    # Find the last month with actual production (oil > 0)
    monthly_totals = df.groupby("date")["oil"].sum()
    months_with_production = monthly_totals[monthly_totals > 0]

    if len(months_with_production) == 0:
        latest_date = df["date"].max()
    else:
        latest_date = months_with_production.index.max()

    prev_month = latest_date - pd.DateOffset(months=1)
    year_ago = latest_date - pd.DateOffset(months=12)

    # Latest month total
    latest_data = df[df["date"] == latest_date]
    total_oil = latest_data["oil"].sum()
    total_gas = latest_data["gas"].sum()
    total_water = latest_data["water"].sum()

    # Previous month total
    prev_data = df[df["date"] == prev_month]
    prev_oil = prev_data["oil"].sum() if len(prev_data) > 0 else np.nan
    prev_gas = prev_data["gas"].sum() if len(prev_data) > 0 else np.nan
    prev_water = prev_data["water"].sum() if len(prev_data) > 0 else np.nan

    # Year ago total
    yago_data = df[df["date"] == year_ago]
    yago_oil = yago_data["oil"].sum() if len(yago_data) > 0 else np.nan

    # Calculate changes
    mom_change = ((total_oil - prev_oil) / prev_oil * 100) if prev_oil > 0 else np.nan
    gas_mom_change = ((total_gas - prev_gas) / prev_gas * 100) if prev_gas > 0 else np.nan
    water_mom_change = ((total_water - prev_water) / prev_water * 100) if prev_water > 0 else np.nan
    yoy_change = ((total_oil - yago_oil) / yago_oil * 100) if yago_oil > 0 else np.nan

    return {
        "report_date": latest_date.strftime("%Y-%m-%d"),
        "report_month": latest_date.strftime("%B %Y"),
        "total_oil": round(total_oil, 2),
        "total_gas": round(total_gas, 2),
        "total_water": round(total_water, 2),
        "prev_month_oil": round(prev_oil, 2) if not np.isnan(prev_oil) else None,
        "mom_change_pct": round(mom_change, 1) if not np.isnan(mom_change) else None,
        "gas_mom_change_pct": round(gas_mom_change, 1) if not np.isnan(gas_mom_change) else None,
        "water_mom_change_pct": round(water_mom_change, 1) if not np.isnan(water_mom_change) else None,
        "yago_oil": round(yago_oil, 2) if not np.isnan(yago_oil) else None,
        "yoy_change_pct": round(yoy_change, 1) if not np.isnan(yoy_change) else None,
    }


def get_top_wellbores(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Get top N wellbores by oil production in the latest month with data.

    Parameters
    ----------
    df : pd.DataFrame
        Production data.
    n : int
        Number of top wellbores to return.

    Returns
    -------
    pd.DataFrame
        Top wellbores with their production.
    """
    # Find the last month with actual production
    monthly_totals = df.groupby("date")["oil"].sum()
    months_with_production = monthly_totals[monthly_totals > 0]

    if len(months_with_production) == 0:
        latest_date = df["date"].max()
    else:
        latest_date = months_with_production.index.max()

    latest_data = df[df["date"] == latest_date]

    # Filter out rows with NaN oil values
    latest_data = latest_data.dropna(subset=["oil"])

    top = (
        latest_data.nlargest(n, "oil")[["wellbore", "oil", "gas", "water"]]
        .reset_index(drop=True)
    )

    return top


def detect_anomalies(
    df: pd.DataFrame,
    threshold_pct: float = 30.0,
) -> pd.DataFrame:
    """
    Detect anomalies: sudden drops in production vs rolling average.

    Parameters
    ----------
    df : pd.DataFrame
        Production data with rolling features.
    threshold_pct : float
        Percentage drop threshold for flagging anomalies.

    Returns
    -------
    pd.DataFrame
        Dataframe of detected anomalies.
    """
    # Ensure rolling features exist
    if "oil_rolling_3m" not in df.columns:
        df = add_rolling_features(df, columns=["oil"], windows=[3])

    latest_date = df["date"].max()
    latest_data = df[df["date"] == latest_date].copy()

    anomalies = []
    for _, row in latest_data.iterrows():
        if row["oil_rolling_3m"] > 0:
            pct_diff = ((row["oil"] - row["oil_rolling_3m"]) / row["oil_rolling_3m"]) * 100
            if pct_diff < -threshold_pct:
                anomalies.append({
                    "wellbore": row["wellbore"],
                    "current_oil": row["oil"],
                    "rolling_avg": row["oil_rolling_3m"],
                    "pct_change": round(pct_diff, 1),
                })

    return pd.DataFrame(anomalies)


def format_number(value: float, decimals: int = 0) -> str:
    """Format number with thousand separators."""
    if value is None or np.isnan(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def format_change(value: float) -> str:
    """Format percentage change with sign."""
    if value is None or np.isnan(value):
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def generate_email_summary(
    df: pd.DataFrame,
    forecast_df: Optional[pd.DataFrame] = None,
    dashboard_url: str = DASHBOARD_URL,
    sharepoint_url: str = "",
) -> str:
    """
    Generate email-ready summary text.

    Parameters
    ----------
    df : pd.DataFrame
        Production data.
    forecast_df : pd.DataFrame, optional
        Forecast data.
    dashboard_url : str
        URL to Streamlit dashboard.
    sharepoint_url : str
        URL to SharePoint folder.

    Returns
    -------
    str
        Formatted email summary text.
    """
    summary = get_last_month_summary(df)
    top_wellbores = get_top_wellbores(df)
    anomalies = detect_anomalies(df)

    # Build summary text
    lines = [
        f"VOLVE PRODUCTION REPORT - {summary['report_month']}",
        "=" * 50,
        "",
        "KEY METRICS",
        "-" * 20,
        f"Total Oil Production: {format_number(summary['total_oil'])} Sm3",
        f"Month-over-Month Change: {format_change(summary['mom_change_pct'])}",
        f"Year-over-Year Change: {format_change(summary['yoy_change_pct'])}",
        "",
        f"Total Gas Production: {format_number(summary['total_gas'])} Sm3",
        f"Total Water Production: {format_number(summary['total_water'])} Sm3",
        "",
        "TOP PRODUCING WELLBORES",
        "-" * 20,
    ]

    for i, row in top_wellbores.iterrows():
        lines.append(f"{i+1}. {row['wellbore']}: {format_number(row['oil'])} Sm3 oil")

    # Add forecast if available
    if forecast_df is not None and len(forecast_df) > 0:
        total_forecast = forecast_df[forecast_df["series_id"] == "TOTAL"]
        if len(total_forecast) > 0:
            next_month = total_forecast.iloc[0]
            lines.extend([
                "",
                "FORECAST",
                "-" * 20,
                f"Next Month Forecast: {format_number(next_month['yhat'])} Sm3 oil",
            ])
            if "yhat_lower" in next_month and "yhat_upper" in next_month:
                lines.append(
                    f"Range: {format_number(next_month['yhat_lower'])} - "
                    f"{format_number(next_month['yhat_upper'])} Sm3"
                )

    # Add anomalies
    if len(anomalies) > 0:
        lines.extend([
            "",
            "ALERTS",
            "-" * 20,
        ])
        for _, row in anomalies.iterrows():
            lines.append(
                f"- {row['wellbore']}: {row['pct_change']}% vs 3-month average"
            )

    # Add links
    lines.extend([
        "",
        "LINKS",
        "-" * 20,
        f"Dashboard: {dashboard_url}",
    ])
    if sharepoint_url:
        lines.append(f"SharePoint: {sharepoint_url}")

    lines.extend([
        "",
        "-" * 50,
        f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ])

    return "\n".join(lines)


def save_email_summary(
    df: pd.DataFrame,
    forecast_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate and save email summary to file.

    Parameters
    ----------
    df : pd.DataFrame
        Production data.
    forecast_df : pd.DataFrame, optional
        Forecast data.
    output_path : Path, optional
        Output file path.

    Returns
    -------
    str
        The generated summary text.
    """
    if output_path is None:
        output_path = PROCESSED_DATA_DIR / "email_summary.txt"

    summary_text = generate_email_summary(df, forecast_df)

    with open(output_path, "w") as f:
        f.write(summary_text)

    return summary_text


def generate_summary_dict(
    df: pd.DataFrame,
    forecast_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Generate summary as dictionary (for JSON/API use).

    Parameters
    ----------
    df : pd.DataFrame
        Production data.
    forecast_df : pd.DataFrame, optional
        Forecast data.

    Returns
    -------
    Dict
        Summary dictionary.
    """
    summary = get_last_month_summary(df)
    top_wellbores = get_top_wellbores(df)
    anomalies = detect_anomalies(df)

    result = {
        "report_date": summary["report_date"],
        "metrics": summary,
        "top_wellbores": top_wellbores.to_dict(orient="records"),
        "anomalies": anomalies.to_dict(orient="records") if len(anomalies) > 0 else [],
    }

    if forecast_df is not None and len(forecast_df) > 0:
        total_forecast = forecast_df[forecast_df["series_id"] == "TOTAL"]
        if len(total_forecast) > 0:
            result["forecast"] = total_forecast[["date", "yhat"]].head(3).to_dict(orient="records")

    return result
