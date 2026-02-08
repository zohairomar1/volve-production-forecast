"""Tests for the automation status module."""

import pytest
import pandas as pd
from unittest.mock import patch
from pathlib import Path

from src.automation.status import get_current_status, run_test_report


# ---------------------------------------------------------------------------
# Current status
# ---------------------------------------------------------------------------

class TestGetCurrentStatus:
    def test_returns_expected_keys(self):
        status = get_current_status()
        assert "sharepoint_configured" in status
        assert "storage_mode" in status
        assert "last_pipeline_output_exists" in status
        assert "processed_files" in status

    def test_storage_mode_local_when_no_url(self):
        with patch("src.automation.status.SHAREPOINT_SITE_URL", ""):
            status = get_current_status()
            assert status["storage_mode"] == "local"
            assert status["sharepoint_configured"] is False

    def test_storage_mode_sharepoint_when_url_set(self):
        with patch("src.automation.status.SHAREPOINT_SITE_URL", "https://example.sharepoint.com"):
            status = get_current_status()
            assert status["storage_mode"] == "sharepoint"
            assert status["sharepoint_configured"] is True


# ---------------------------------------------------------------------------
# Test report
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Minimal DataFrame matching production schema."""
    return pd.DataFrame({
        "date": pd.date_range("2015-01-01", periods=12, freq="MS"),
        "wellbore": ["15/9-F-11"] * 12,
        "oil": [1000 + i * 100 for i in range(12)],
        "gas": [50000] * 12,
        "water": [500] * 12,
    })


class TestRunTestReport:
    def test_success(self, sample_df):
        result = run_test_report(sample_df)
        assert result["status"] == "success"
        assert result["summary_generated"] is True
        assert result["is_test_run"] is True
        assert result["duration_seconds"] >= 0

    def test_simulated_credential_failure(self, sample_df):
        result = run_test_report(sample_df, simulate_failure="missing_credentials")
        assert result["status"] == "failure"
        assert "credentials" in result["error_message"].lower()
        assert result["storage_mode"] == "sharepoint"

    def test_simulated_io_failure(self, sample_df):
        result = run_test_report(sample_df, simulate_failure="sharepoint_io")
        assert result["status"] == "failure"
        assert "SharePoint" in result["error_message"]
        assert result["storage_mode"] == "sharepoint"

    def test_has_timestamp(self, sample_df):
        result = run_test_report(sample_df)
        assert "timestamp" in result
        assert len(result["timestamp"]) > 0
