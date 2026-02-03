"""
Tests for SharePoint integration module.

Uses unittest.mock to test both local fallback and SharePoint API paths
without requiring Azure credentials or network access.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io_sharepoint import SharePointClient, sync_from_sharepoint, sync_to_sharepoint


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def tmp_dirs(tmp_path):
    """Create temporary raw and processed directories with sample files."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()

    # Create sample files in raw
    (raw_dir / "data.csv").write_text("col1,col2\n1,2\n")
    (raw_dir / "data.txt").write_text("not a csv")

    # Create sample files in processed
    (processed_dir / "output.csv").write_text("result,value\na,1\n")
    (processed_dir / "metrics.json").write_text("{}")

    return raw_dir, processed_dir


# =============================================================================
# LOCAL MODE TESTS
# =============================================================================

class TestLocalMode:
    """Test SharePointClient in local (fallback) mode."""

    def test_init_local_mode(self):
        client = SharePointClient(use_sharepoint=False)
        assert client.use_sharepoint is False
        assert client._access_token is None

    @patch("src.io_sharepoint.RAW_DATA_DIR")
    def test_list_files_local(self, mock_raw_dir, tmp_dirs):
        raw_dir, _ = tmp_dirs
        mock_raw_dir.__truediv__ = lambda self, x: raw_dir / x
        # Point to actual raw dir for iteration
        mock_raw_dir.exists.return_value = True
        mock_raw_dir.iterdir = raw_dir.iterdir

        client = SharePointClient(use_sharepoint=False)
        # Patch the local path resolution
        with patch("src.io_sharepoint.RAW_DATA_DIR", raw_dir):
            files = client.list_files()

        assert len(files) == 2
        names = [f["name"] for f in files]
        assert "data.csv" in names
        assert "data.txt" in names

    @patch("src.io_sharepoint.RAW_DATA_DIR")
    def test_list_files_filter_extension(self, mock_raw_dir, tmp_dirs):
        raw_dir, _ = tmp_dirs

        client = SharePointClient(use_sharepoint=False)
        with patch("src.io_sharepoint.RAW_DATA_DIR", raw_dir):
            files = client.list_files(file_extension=".csv")

        assert len(files) == 1
        assert files[0]["name"] == "data.csv"

    @patch("src.io_sharepoint.RAW_DATA_DIR")
    def test_list_files_empty_dir(self, mock_raw_dir, tmp_dirs):
        raw_dir, _ = tmp_dirs
        empty = raw_dir.parent / "empty"
        empty.mkdir()

        client = SharePointClient(use_sharepoint=False)
        with patch("src.io_sharepoint.RAW_DATA_DIR", empty):
            files = client.list_files()

        assert files == []

    @patch("src.io_sharepoint.RAW_DATA_DIR")
    def test_list_files_nonexistent_dir(self, mock_raw_dir, tmp_dirs):
        raw_dir, _ = tmp_dirs

        client = SharePointClient(use_sharepoint=False)
        with patch("src.io_sharepoint.RAW_DATA_DIR", raw_dir):
            files = client.list_files("nonexistent_subfolder")

        assert files == []

    @patch("src.io_sharepoint.RAW_DATA_DIR")
    def test_download_file_local(self, mock_raw_dir, tmp_dirs):
        raw_dir, _ = tmp_dirs
        dest = tmp_dirs[1] / "downloaded.csv"

        client = SharePointClient(use_sharepoint=False)
        with patch("src.io_sharepoint.RAW_DATA_DIR", raw_dir):
            result = client.download_file("data.csv", str(dest))

        assert result is True
        assert dest.exists()
        assert dest.read_text() == "col1,col2\n1,2\n"

    @patch("src.io_sharepoint.RAW_DATA_DIR")
    def test_download_file_not_found(self, mock_raw_dir, tmp_dirs):
        raw_dir, _ = tmp_dirs
        dest = tmp_dirs[1] / "missing.csv"

        client = SharePointClient(use_sharepoint=False)
        with patch("src.io_sharepoint.RAW_DATA_DIR", raw_dir):
            result = client.download_file("nonexistent.csv", str(dest))

        assert result is False

    @patch("src.io_sharepoint.PROCESSED_DATA_DIR")
    def test_upload_file_local(self, mock_processed_dir, tmp_dirs):
        raw_dir, processed_dir = tmp_dirs
        source = raw_dir / "data.csv"

        client = SharePointClient(use_sharepoint=False)
        with patch("src.io_sharepoint.PROCESSED_DATA_DIR", processed_dir):
            result = client.upload_file(str(source), "uploaded.csv")

        assert result is True
        assert (processed_dir / "uploaded.csv").exists()

    @patch("src.io_sharepoint.RAW_DATA_DIR")
    def test_get_latest_file(self, mock_raw_dir, tmp_dirs):
        raw_dir, _ = tmp_dirs

        client = SharePointClient(use_sharepoint=False)
        with patch("src.io_sharepoint.RAW_DATA_DIR", raw_dir):
            latest = client.get_latest_file(file_extension=".csv")

        assert latest is not None
        assert latest["name"] == "data.csv"

    @patch("src.io_sharepoint.RAW_DATA_DIR")
    def test_get_latest_file_no_match(self, mock_raw_dir, tmp_dirs):
        raw_dir, _ = tmp_dirs

        client = SharePointClient(use_sharepoint=False)
        with patch("src.io_sharepoint.RAW_DATA_DIR", raw_dir):
            latest = client.get_latest_file(file_extension=".parquet")

        assert latest is None


# =============================================================================
# CREDENTIAL VALIDATION TESTS
# =============================================================================

class TestCredentialValidation:
    """Test credential checking for SharePoint mode."""

    @patch("src.io_sharepoint.SHAREPOINT_SITE_URL", "https://company.sharepoint.com/sites/Test")
    @patch("src.io_sharepoint.AZURE_CLIENT_SECRET", "")
    @patch("src.io_sharepoint.AZURE_CLIENT_ID", "test-id")
    @patch("src.io_sharepoint.AZURE_TENANT_ID", "test-tenant")
    def test_missing_client_secret_raises(self):
        with pytest.raises(ValueError, match="credentials not configured"):
            SharePointClient(use_sharepoint=True)

    @patch("src.io_sharepoint.SHAREPOINT_SITE_URL", "")
    @patch("src.io_sharepoint.AZURE_CLIENT_SECRET", "secret")
    @patch("src.io_sharepoint.AZURE_CLIENT_ID", "test-id")
    @patch("src.io_sharepoint.AZURE_TENANT_ID", "test-tenant")
    def test_missing_site_url_raises(self):
        with pytest.raises(ValueError, match="SHAREPOINT_SITE_URL"):
            SharePointClient(use_sharepoint=True)

    @patch("src.io_sharepoint.SHAREPOINT_SITE_URL", "https://company.sharepoint.com/sites/Test")
    @patch("src.io_sharepoint.AZURE_CLIENT_SECRET", "secret")
    @patch("src.io_sharepoint.AZURE_CLIENT_ID", "test-id")
    @patch("src.io_sharepoint.AZURE_TENANT_ID", "test-tenant")
    def test_valid_credentials_accepted(self):
        client = SharePointClient(use_sharepoint=True)
        assert client.use_sharepoint is True


# =============================================================================
# TOKEN MANAGEMENT TESTS
# =============================================================================

class TestTokenManagement:
    """Test OAuth2 token acquisition and refresh."""

    @patch("src.io_sharepoint.SHAREPOINT_SITE_URL", "https://company.sharepoint.com/sites/Test")
    @patch("src.io_sharepoint.AZURE_CLIENT_SECRET", "secret")
    @patch("src.io_sharepoint.AZURE_CLIENT_ID", "test-id")
    @patch("src.io_sharepoint.AZURE_TENANT_ID", "test-tenant")
    def test_token_acquisition(self):
        client = SharePointClient(use_sharepoint=True)

        mock_app = MagicMock()
        mock_app.acquire_token_for_client.return_value = {
            "access_token": "test-token-abc",
            "expires_in": 3600,
        }

        with patch.dict("sys.modules", {"msal": MagicMock(ConfidentialClientApplication=lambda *a, **k: mock_app)}):
            token = client._get_access_token()

        assert token == "test-token-abc"
        assert client._access_token == "test-token-abc"
        assert client._token_expiry > 0

    @patch("src.io_sharepoint.SHAREPOINT_SITE_URL", "https://company.sharepoint.com/sites/Test")
    @patch("src.io_sharepoint.AZURE_CLIENT_SECRET", "secret")
    @patch("src.io_sharepoint.AZURE_CLIENT_ID", "test-id")
    @patch("src.io_sharepoint.AZURE_TENANT_ID", "test-tenant")
    def test_token_reuse_when_valid(self):
        import time
        client = SharePointClient(use_sharepoint=True)
        client._access_token = "cached-token"
        client._token_expiry = time.time() + 3600  # Still valid

        token = client._get_access_token()
        assert token == "cached-token"

    @patch("src.io_sharepoint.SHAREPOINT_SITE_URL", "https://company.sharepoint.com/sites/Test")
    @patch("src.io_sharepoint.AZURE_CLIENT_SECRET", "secret")
    @patch("src.io_sharepoint.AZURE_CLIENT_ID", "test-id")
    @patch("src.io_sharepoint.AZURE_TENANT_ID", "test-tenant")
    def test_token_refresh_when_expired(self):
        import time
        client = SharePointClient(use_sharepoint=True)
        client._access_token = "old-token"
        client._token_expiry = time.time() - 100  # Expired

        mock_app = MagicMock()
        mock_app.acquire_token_for_client.return_value = {
            "access_token": "new-token",
            "expires_in": 3600,
        }

        with patch.dict("sys.modules", {"msal": MagicMock(ConfidentialClientApplication=lambda *a, **k: mock_app)}):
            token = client._get_access_token()

        assert token == "new-token"

    @patch("src.io_sharepoint.SHAREPOINT_SITE_URL", "https://company.sharepoint.com/sites/Test")
    @patch("src.io_sharepoint.AZURE_CLIENT_SECRET", "secret")
    @patch("src.io_sharepoint.AZURE_CLIENT_ID", "test-id")
    @patch("src.io_sharepoint.AZURE_TENANT_ID", "test-tenant")
    def test_token_failure_raises_runtime_error(self):
        client = SharePointClient(use_sharepoint=True)

        mock_app = MagicMock()
        mock_app.acquire_token_for_client.return_value = {
            "error": "invalid_client",
            "error_description": "Bad credentials",
        }

        with patch.dict("sys.modules", {"msal": MagicMock(ConfidentialClientApplication=lambda *a, **k: mock_app)}):
            with pytest.raises(RuntimeError, match="Bad credentials"):
                client._get_access_token()


# =============================================================================
# SYNC HELPER TESTS
# =============================================================================

class TestSyncHelpers:
    """Test sync_from_sharepoint and sync_to_sharepoint convenience functions."""

    def test_sync_from_sharepoint_local(self, tmp_dirs):
        raw_dir, processed_dir = tmp_dirs

        with patch("src.io_sharepoint.RAW_DATA_DIR", raw_dir):
            client = SharePointClient(use_sharepoint=False)
            downloaded = sync_from_sharepoint(
                client=client,
                remote_folder="",
                local_folder=processed_dir,
                file_pattern="*.csv",
            )

        assert len(downloaded) == 1
        assert "data.csv" in downloaded[0]

    def test_sync_to_sharepoint_local(self, tmp_dirs):
        raw_dir, processed_dir = tmp_dirs

        with patch("src.io_sharepoint.PROCESSED_DATA_DIR", processed_dir):
            client = SharePointClient(use_sharepoint=False)
            uploaded = sync_to_sharepoint(
                client=client,
                local_folder=processed_dir,
                remote_folder="Outputs",
                file_pattern="*.csv",
            )

        assert len(uploaded) == 1
        assert "output.csv" in uploaded

    def test_sync_to_sharepoint_no_matching_files(self, tmp_dirs):
        raw_dir, processed_dir = tmp_dirs

        with patch("src.io_sharepoint.PROCESSED_DATA_DIR", processed_dir):
            client = SharePointClient(use_sharepoint=False)
            uploaded = sync_to_sharepoint(
                client=client,
                local_folder=processed_dir,
                remote_folder="Outputs",
                file_pattern="*.parquet",
            )

        assert uploaded == []
