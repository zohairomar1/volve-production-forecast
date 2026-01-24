"""
Tests for forecasting module - shape and structure validation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.forecasting import (
    seasonal_naive_forecast,
    naive_forecast,
    exponential_smoothing_forecast,
    forecast_series,
    forecast_all_wellbores,
)


@pytest.fixture
def sample_series():
    """Create a sample time series for testing."""
    dates = pd.date_range(start="2013-01-01", periods=36, freq="MS")
    # Create realistic production pattern with trend and seasonality
    values = 10000 + np.sin(np.arange(36) * 2 * np.pi / 12) * 2000 + np.random.randn(36) * 500
    values = np.maximum(values, 0)  # Ensure positive
    return pd.Series(values, index=dates)


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe with multiple wellbores."""
    dates = pd.date_range(start="2013-01-01", periods=24, freq="MS")

    data = []
    for wellbore in ["Well-A", "Well-B"]:
        for date in dates:
            data.append({
                "date": date,
                "wellbore": wellbore,
                "oil": np.random.uniform(5000, 15000),
                "gas": np.random.uniform(100000, 500000),
                "water": np.random.uniform(0, 5000),
            })

    return pd.DataFrame(data)


class TestSeasonalNaiveForecast:
    """Tests for seasonal naive forecasting."""

    def test_returns_correct_horizon(self, sample_series):
        """Forecast should have requested number of periods."""
        horizon = 6
        result = seasonal_naive_forecast(sample_series, horizon=horizon)

        assert len(result) == horizon

    def test_returns_required_columns(self, sample_series):
        """Forecast should have date and yhat columns."""
        result = seasonal_naive_forecast(sample_series, horizon=3)

        assert "date" in result.columns
        assert "yhat" in result.columns

    def test_forecast_dates_are_future(self, sample_series):
        """Forecast dates should be after last historical date."""
        result = seasonal_naive_forecast(sample_series, horizon=3)

        assert result["date"].min() > sample_series.index.max()

    def test_forecast_dates_are_monthly(self, sample_series):
        """Forecast dates should be monthly intervals."""
        result = seasonal_naive_forecast(sample_series, horizon=6)

        date_diffs = result["date"].diff().dropna()
        # All diffs should be approximately 1 month (28-31 days)
        for diff in date_diffs:
            assert 28 <= diff.days <= 31


class TestNaiveForecast:
    """Tests for simple naive forecasting."""

    def test_returns_correct_horizon(self, sample_series):
        """Forecast should have requested number of periods."""
        horizon = 4
        result = naive_forecast(sample_series, horizon=horizon)

        assert len(result) == horizon

    def test_uses_last_value(self, sample_series):
        """Naive forecast should use last observed value."""
        result = naive_forecast(sample_series, horizon=3)
        last_value = sample_series.iloc[-1]

        assert all(result["yhat"] == last_value)


class TestExponentialSmoothingForecast:
    """Tests for ETS forecasting."""

    def test_returns_correct_horizon(self, sample_series):
        """Forecast should have requested number of periods."""
        horizon = 6
        result = exponential_smoothing_forecast(sample_series, horizon=horizon)

        assert len(result) == horizon

    def test_returns_required_columns(self, sample_series):
        """Forecast should have date, yhat, and confidence bounds."""
        result = exponential_smoothing_forecast(sample_series, horizon=3)

        assert "date" in result.columns
        assert "yhat" in result.columns
        # Confidence bounds are optional but expected
        assert "yhat_lower" in result.columns
        assert "yhat_upper" in result.columns

    def test_confidence_bounds_order(self, sample_series):
        """Lower bound should be <= yhat <= upper bound."""
        result = exponential_smoothing_forecast(sample_series, horizon=3)

        assert all(result["yhat_lower"] <= result["yhat"])
        assert all(result["yhat"] <= result["yhat_upper"])

    def test_forecasts_are_positive(self, sample_series):
        """Production forecasts should be non-negative."""
        result = exponential_smoothing_forecast(sample_series, horizon=6)

        assert all(result["yhat"] >= 0)
        assert all(result["yhat_lower"] >= 0)


class TestForecastSeries:
    """Tests for forecast_series function."""

    def test_returns_expected_shape(self, sample_dataframe):
        """Forecast should have expected number of rows and columns."""
        horizon = 3
        result = forecast_series(
            sample_dataframe,
            target_col="oil",
            series_id="TOTAL",
            model="baseline",
            horizon=horizon
        )

        assert len(result) == horizon
        assert "date" in result.columns
        assert "yhat" in result.columns
        assert "series_id" in result.columns
        assert "model" in result.columns

    def test_series_id_is_set(self, sample_dataframe):
        """Series ID should be set in output."""
        result = forecast_series(
            sample_dataframe,
            target_col="oil",
            series_id="Well-A",
            model="ets",
            horizon=3
        )

        assert all(result["series_id"] == "Well-A")

    def test_model_is_recorded(self, sample_dataframe):
        """Model name should be recorded in output."""
        result = forecast_series(
            sample_dataframe,
            target_col="oil",
            series_id="TOTAL",
            model="baseline",
            horizon=3
        )

        assert all(result["model"] == "baseline")


class TestForecastAllWellbores:
    """Tests for forecast_all_wellbores function."""

    def test_includes_total(self, sample_dataframe):
        """Should include TOTAL series by default."""
        result = forecast_all_wellbores(
            sample_dataframe,
            target_col="oil",
            model="baseline",
            horizon=3,
            include_total=True
        )

        assert "TOTAL" in result["series_id"].values

    def test_includes_all_wellbores(self, sample_dataframe):
        """Should include forecasts for all wellbores."""
        result = forecast_all_wellbores(
            sample_dataframe,
            target_col="oil",
            model="baseline",
            horizon=3
        )

        wellbores = sample_dataframe["wellbore"].unique()
        for wb in wellbores:
            assert wb in result["series_id"].values

    def test_excludes_total_when_requested(self, sample_dataframe):
        """Should exclude TOTAL when include_total=False."""
        result = forecast_all_wellbores(
            sample_dataframe,
            target_col="oil",
            model="baseline",
            horizon=3,
            include_total=False
        )

        assert "TOTAL" not in result["series_id"].values


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_short_series(self):
        """Should handle series with minimal data."""
        dates = pd.date_range(start="2020-01-01", periods=15, freq="MS")
        values = np.random.uniform(1000, 5000, size=15)
        series = pd.Series(values, index=dates)

        # Should not raise error
        result = exponential_smoothing_forecast(series, horizon=3)
        assert len(result) == 3

    def test_handles_zeros(self):
        """Should handle series with zero values."""
        dates = pd.date_range(start="2020-01-01", periods=24, freq="MS")
        values = np.random.uniform(0, 5000, size=24)
        values[5:8] = 0  # Some zeros
        series = pd.Series(values, index=dates)

        result = exponential_smoothing_forecast(series, horizon=3)
        assert len(result) == 3
        assert all(result["yhat"] >= 0)
