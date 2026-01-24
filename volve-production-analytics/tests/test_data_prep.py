"""
Tests for data preparation module.
"""

import pytest
import pandas as pd
import numpy as np
from io import StringIO
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_prep import (
    standardize_columns,
    create_date_column,
    clean_numeric_columns,
    prepare_data,
    aggregate_total_production,
)


@pytest.fixture
def sample_raw_data():
    """Create sample raw data matching Volve format."""
    data = """Wellbore name,NPDCode,Year,Month,On Stream,Oil,Gas,Water,GI,WI
15/9-F-11,7078,2013,7,113,3923.00,590505.00,0.00,,
15/9-F-11,7078,2013,8,587,25496.00,3871942.00,520.00,,
15/9-F-11,7078,2013,9,539,23775.00,3661574.00,1574.00,,
15/9-F-1 C,7405,2014,4,228,11142.00,1597937.00,0.00,,
15/9-F-1 C,7405,2014,5,734,24902.00,3496230.00,783.00,,
"""
    return pd.read_csv(StringIO(data))


@pytest.fixture
def sample_csv_file(sample_raw_data):
    """Create a temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_raw_data.to_csv(f, index=False)
        return Path(f.name)


class TestStandardizeColumns:
    """Tests for column standardization."""

    def test_renames_known_columns(self, sample_raw_data):
        """Known columns should be renamed to standard names."""
        result = standardize_columns(sample_raw_data)

        assert "wellbore" in result.columns
        assert "year" in result.columns
        assert "month" in result.columns
        assert "oil" in result.columns
        assert "gas" in result.columns
        assert "water" in result.columns
        assert "on_stream" in result.columns

    def test_preserves_unknown_columns(self, sample_raw_data):
        """Unknown columns should be preserved."""
        sample_raw_data["custom_column"] = 1
        result = standardize_columns(sample_raw_data)

        assert "custom_column" in result.columns


class TestCreateDateColumn:
    """Tests for date column creation."""

    def test_creates_date_from_year_month(self, sample_raw_data):
        """Date column should be created from year and month."""
        df = standardize_columns(sample_raw_data)
        result = create_date_column(df)

        assert "date" in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_date_is_first_of_month(self, sample_raw_data):
        """Dates should be first day of month."""
        df = standardize_columns(sample_raw_data)
        result = create_date_column(df)

        assert all(result["date"].dt.day == 1)

    def test_date_values_correct(self, sample_raw_data):
        """Date values should match year/month."""
        df = standardize_columns(sample_raw_data)
        result = create_date_column(df)

        # First row should be July 2013
        assert result.iloc[0]["date"] == pd.Timestamp("2013-07-01")


class TestCleanNumericColumns:
    """Tests for numeric column cleaning."""

    def test_converts_to_numeric(self, sample_raw_data):
        """Production columns should be numeric."""
        df = standardize_columns(sample_raw_data)
        result = clean_numeric_columns(df)

        assert pd.api.types.is_numeric_dtype(result["oil"])
        assert pd.api.types.is_numeric_dtype(result["gas"])
        assert pd.api.types.is_numeric_dtype(result["water"])

    def test_handles_invalid_values(self):
        """Invalid values should become NaN."""
        df = pd.DataFrame({
            "oil": ["100", "invalid", "300"],
            "gas": [1000, 2000, 3000],
        })
        result = clean_numeric_columns(df)

        assert pd.isna(result.loc[1, "oil"])
        assert result.loc[0, "oil"] == 100

    def test_handles_negative_values(self):
        """Negative values should become NaN."""
        df = pd.DataFrame({
            "oil": [100, -50, 300],
        })
        result = clean_numeric_columns(df)

        assert pd.isna(result.loc[1, "oil"])


class TestPrepareData:
    """Tests for full data preparation pipeline."""

    def test_returns_required_columns(self, sample_csv_file):
        """Prepared data should have required columns."""
        result = prepare_data(sample_csv_file, save_output=False)

        required = ["date", "wellbore", "oil", "gas", "water"]
        for col in required:
            assert col in result.columns

    def test_date_is_monotonic_per_wellbore(self, sample_csv_file):
        """Dates should be monotonically increasing within each wellbore."""
        result = prepare_data(sample_csv_file, save_output=False)

        for wellbore in result["wellbore"].unique():
            wb_data = result[result["wellbore"] == wellbore]
            assert wb_data["date"].is_monotonic_increasing

    def test_no_duplicate_dates_per_wellbore(self, sample_csv_file):
        """Should not have duplicate dates within a wellbore."""
        result = prepare_data(sample_csv_file, save_output=False)

        for wellbore in result["wellbore"].unique():
            wb_data = result[result["wellbore"] == wellbore]
            assert not wb_data["date"].duplicated().any()


class TestAggregateTotalProduction:
    """Tests for production aggregation."""

    def test_aggregates_by_date(self, sample_csv_file):
        """Should aggregate production by date."""
        df = prepare_data(sample_csv_file, save_output=False)
        result = aggregate_total_production(df)

        assert "date" in result.columns
        assert "oil" in result.columns
        assert result["wellbore"].unique()[0] == "TOTAL"

    def test_sums_production_values(self, sample_csv_file):
        """Aggregated values should be sum of wellbores."""
        df = prepare_data(sample_csv_file, save_output=False)
        result = aggregate_total_production(df)

        # For dates with multiple wellbores, total should be sum
        for date in result["date"]:
            expected = df[df["date"] == date]["oil"].sum()
            actual = result[result["date"] == date]["oil"].values[0]
            assert np.isclose(expected, actual)
