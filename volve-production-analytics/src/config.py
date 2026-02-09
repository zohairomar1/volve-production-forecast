"""
Configuration settings for the Volve Production Analytics pipeline.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Column mapping: standardize various possible column names to our schema
COLUMN_MAPPING = {
    # Wellbore identifier
    "Wellbore name": "wellbore",
    "WELLBORE_NAME": "wellbore",
    "wellbore_name": "wellbore",
    "Well": "wellbore",

    # Date components
    "Year": "year",
    "YEAR": "year",
    "Month": "month",
    "MONTH": "month",

    # Production volumes
    "Oil": "oil",
    "OIL": "oil",
    "BORE_OIL_VOL": "oil",
    "oil_vol": "oil",

    "Gas": "gas",
    "GAS": "gas",
    "BORE_GAS_VOL": "gas",
    "gas_vol": "gas",

    "Water": "water",
    "WATER": "water",
    "BORE_WAT_VOL": "water",
    "water_vol": "water",

    # Uptime/On-stream hours
    "On Stream": "on_stream",
    "ON_STREAM": "on_stream",
    "ON_STREAM_HRS": "on_stream",
    "operating_hours": "on_stream",

    # Injection volumes (optional)
    "GI": "gas_injection",
    "WI": "water_injection",
}

# Standard output schema
OUTPUT_COLUMNS = [
    "date",
    "wellbore",
    "oil",
    "gas",
    "water",
    "on_stream",
    "gas_injection",
    "water_injection",
]

# Forecasting settings
FORECAST_HORIZON = 6  # months ahead
BACKTEST_PERIODS = 12  # months for backtesting
MIN_HISTORY_MONTHS = 12  # minimum history for forecasting

# SharePoint settings (from environment)
SHAREPOINT_SITE_URL = os.getenv("SHAREPOINT_SITE_URL", "")
SHAREPOINT_RAW_FOLDER = os.getenv("SHAREPOINT_RAW_FOLDER", "")
SHAREPOINT_PROCESSED_FOLDER = os.getenv("SHAREPOINT_PROCESSED_FOLDER", "")

# Azure AD settings (from environment)
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID", "")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID", "")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET", "")

# Gemini AI settings (from environment)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Dashboard
DASHBOARD_URL = os.getenv("DASHBOARD_URL", "http://localhost:8501")
