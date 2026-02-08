"""Tests for the JSON export helper."""

import json
import pytest
import pandas as pd
import sys
from pathlib import Path

# Add project root so we can import the helper from the streamlit app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.streamlit_app import _build_export_json


@pytest.fixture
def sample_summary():
    return {
        "report_month": "September 2015",
        "total_oil": 45000.0,
        "total_gas": 3200000.0,
        "total_water": 12000.0,
        "mom_change_pct": -5.2,
        "yoy_change_pct": -18.3,
    }


@pytest.fixture
def sample_forecast_df():
    return pd.DataFrame({
        "date": pd.date_range("2015-10-01", periods=3, freq="MS"),
        "yhat": [42000.0, 40000.0, 38000.0],
    })


@pytest.fixture
def sample_anomaly_df():
    return pd.DataFrame({
        "date": pd.date_range("2015-01-01", periods=5, freq="MS"),
        "oil": [1000, 2000, 500, 1800, 1500],
        "is_anomaly": [False, False, True, False, False],
    })


class TestBuildExportJson:
    def test_valid_json(self, sample_summary, sample_forecast_df, sample_anomaly_df):
        result = _build_export_json(
            sample_summary, None, sample_anomaly_df, sample_forecast_df,
            {"wape": 10.8, "mae": 4500}, {"mode": "Total Field"}, 2.5,
        )
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_has_expected_keys(self, sample_summary, sample_forecast_df, sample_anomaly_df):
        result = _build_export_json(
            sample_summary, None, sample_anomaly_df, sample_forecast_df,
            {"wape": 10.8}, {"mode": "Total Field"}, 2.5,
        )
        parsed = json.loads(result)
        for key in ["export_timestamp", "filters", "kpis", "anomaly_count",
                     "forecast", "forecast_deltas", "model_metrics"]:
            assert key in parsed, f"Missing key: {key}"

    def test_forecast_deltas(self, sample_summary, sample_forecast_df, sample_anomaly_df):
        result = _build_export_json(
            sample_summary, None, sample_anomaly_df, sample_forecast_df,
            {}, {"mode": "Total Field"}, 2.5,
        )
        parsed = json.loads(result)
        deltas = parsed["forecast_deltas"]
        assert deltas["last_actual"] == 45000.0
        assert deltas["next_forecast"] == 42000.0
        assert deltas["delta"] == -3000.0

    def test_anomaly_count(self, sample_summary, sample_anomaly_df):
        result = _build_export_json(
            sample_summary, None, sample_anomaly_df, None,
            {}, {"mode": "Total Field"}, 2.5,
        )
        parsed = json.loads(result)
        assert parsed["anomaly_count"] == 1

    def test_empty_forecast_handled(self, sample_summary, sample_anomaly_df):
        result = _build_export_json(
            sample_summary, None, sample_anomaly_df, None,
            {}, {"mode": "Total Field"}, 2.5,
        )
        parsed = json.loads(result)
        assert parsed["forecast"] == []
        assert parsed["forecast_deltas"] == {}
