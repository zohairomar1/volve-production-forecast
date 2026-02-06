"""
Automation Status and Test Harness
===================================

Tracks pipeline run history and provides a test harness for
simulating report generation without sending emails.
"""

import json
import time
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from src.config import PROCESSED_DATA_DIR, SHAREPOINT_SITE_URL


def get_current_status() -> Dict:
    """
    Return current automation configuration status.

    Returns dict with keys: sharepoint_configured, storage_mode,
    last_pipeline_output_exists, processed_files.
    """
    processed_exists = PROCESSED_DATA_DIR.exists()
    processed_files = []
    if processed_exists:
        processed_files = [f.name for f in PROCESSED_DATA_DIR.iterdir() if f.is_file()]

    return {
        "sharepoint_configured": bool(SHAREPOINT_SITE_URL),
        "storage_mode": "sharepoint" if SHAREPOINT_SITE_URL else "local",
        "last_pipeline_output_exists": (PROCESSED_DATA_DIR / "email_summary.txt").exists(),
        "processed_files": processed_files,
    }


def run_test_report(
    df: pd.DataFrame,
    forecast_df: Optional[pd.DataFrame] = None,
    simulate_failure: Optional[str] = None,
) -> Dict:
    """
    Execute a test report generation (no email sent).

    Parameters
    ----------
    df : pd.DataFrame
        Production data.
    forecast_df : pd.DataFrame, optional
        Forecast data.
    simulate_failure : str, optional
        "missing_credentials" or "sharepoint_io" to simulate failures.

    Returns dict with run log entry.
    """
    start = time.time()
    run_entry = {
        "run_id": datetime.now().isoformat(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "is_test_run": True,
        "summary_generated": False,
        "error_message": None,
        "storage_mode": "local",
        "status": "pending",
    }

    if simulate_failure == "missing_credentials":
        run_entry["status"] = "failure"
        run_entry["storage_mode"] = "sharepoint"
        run_entry["error_message"] = "Simulated: Azure AD credentials not configured"
        run_entry["duration_seconds"] = round(time.time() - start, 2)
        return run_entry

    if simulate_failure == "sharepoint_io":
        run_entry["status"] = "failure"
        run_entry["storage_mode"] = "sharepoint"
        run_entry["error_message"] = "Simulated: SharePoint I/O timeout (network unreachable)"
        run_entry["duration_seconds"] = round(time.time() - start, 2)
        return run_entry

    try:
        from src.reporting import generate_summary_dict
        summary = generate_summary_dict(df, forecast_df)
        run_entry["status"] = "success"
        run_entry["summary_generated"] = True
        run_entry["summary_preview"] = json.dumps(summary, indent=2, default=str)[:500]
    except Exception as e:
        run_entry["status"] = "failure"
        run_entry["error_message"] = str(e)

    run_entry["duration_seconds"] = round(time.time() - start, 2)
    return run_entry
