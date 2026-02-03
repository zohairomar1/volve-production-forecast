"""
SharePoint Integration Module
=============================

Handles file operations with SharePoint via Microsoft Graph API.

This module provides a stubbed interface for SharePoint operations.
To enable full functionality, configure Azure AD credentials in .env file.

Usage:
    # Local mode (default)
    client = SharePointClient()  # Uses local filesystem

    # SharePoint mode (requires credentials)
    client = SharePointClient(use_sharepoint=True)
    files = client.list_files("Raw Data")
    client.download_file("Raw Data/volve_data.csv", "/local/path.csv")
    client.upload_file("/local/processed.csv", "Processed/processed.csv")
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import shutil

from .config import (
    SHAREPOINT_SITE_URL,
    SHAREPOINT_RAW_FOLDER,
    SHAREPOINT_PROCESSED_FOLDER,
    AZURE_TENANT_ID,
    AZURE_CLIENT_ID,
    AZURE_CLIENT_SECRET,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)


class SharePointClient:
    """
    Client for SharePoint file operations.

    Supports two modes:
    - Local mode: Uses local filesystem (data/raw, data/processed)
    - SharePoint mode: Uses Microsoft Graph API

    Parameters
    ----------
    use_sharepoint : bool
        Whether to use SharePoint API. Defaults to False (local mode).
    """

    # Refresh token 5 minutes before expiry (tokens typically last 3600s)
    TOKEN_REFRESH_BUFFER = 300

    def __init__(self, use_sharepoint: bool = False):
        self.use_sharepoint = use_sharepoint
        self._access_token = None
        self._token_expiry = 0.0

        if use_sharepoint:
            self._validate_credentials()

    def _validate_credentials(self) -> None:
        """Check if SharePoint credentials are configured."""
        if not all([AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET]):
            raise ValueError(
                "SharePoint credentials not configured. "
                "Set AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET "
                "in your .env file."
            )

        if not SHAREPOINT_SITE_URL:
            raise ValueError(
                "SHAREPOINT_SITE_URL not configured in .env file."
            )

    def _get_access_token(self) -> str:
        """
        Get OAuth2 access token from Azure AD.

        Caches the token and automatically refreshes before expiry.

        Returns
        -------
        str
            Access token for Microsoft Graph API.
        """
        if self._access_token and time.time() < self._token_expiry - self.TOKEN_REFRESH_BUFFER:
            return self._access_token

        try:
            from msal import ConfidentialClientApplication
        except ImportError:
            raise ImportError("msal package required for SharePoint integration")

        app = ConfidentialClientApplication(
            AZURE_CLIENT_ID,
            authority=f"https://login.microsoftonline.com/{AZURE_TENANT_ID}",
            client_credential=AZURE_CLIENT_SECRET,
        )

        result = app.acquire_token_for_client(
            scopes=["https://graph.microsoft.com/.default"]
        )

        if "access_token" in result:
            self._access_token = result["access_token"]
            # Azure AD tokens include expires_in (seconds); default to 3600 if missing
            self._token_expiry = time.time() + result.get("expires_in", 3600)
            return self._access_token
        else:
            raise RuntimeError(f"Failed to get access token: {result.get('error_description')}")

    def _graph_request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a request to Microsoft Graph API.

        Parameters
        ----------
        method : str
            HTTP method (GET, POST, PUT, DELETE).
        endpoint : str
            API endpoint path.
        **kwargs
            Additional arguments for requests.

        Returns
        -------
        Dict
            API response.
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required for SharePoint integration")

        token = self._get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        url = f"https://graph.microsoft.com/v1.0{endpoint}"
        response = requests.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()

        if response.content:
            return response.json()
        return {}

    def list_files(
        self,
        folder_path: str = "",
        file_extension: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List files in a SharePoint folder.

        Parameters
        ----------
        folder_path : str
            Path within the document library.
        file_extension : str, optional
            Filter by file extension (e.g., ".csv").

        Returns
        -------
        List[Dict]
            List of file metadata dictionaries.
        """
        if not self.use_sharepoint:
            # Local mode: list files in raw data directory
            local_path = RAW_DATA_DIR / folder_path if folder_path else RAW_DATA_DIR
            if not local_path.exists():
                return []

            files = []
            for f in local_path.iterdir():
                if f.is_file():
                    if file_extension and not f.suffix.lower() == file_extension.lower():
                        continue
                    files.append({
                        "name": f.name,
                        "path": str(f),
                        "size": f.stat().st_size,
                        "modified": f.stat().st_mtime,
                    })
            return files

        # SharePoint mode
        site_path = SHAREPOINT_SITE_URL.split("/sites/")[-1]
        drive_path = f"{SHAREPOINT_RAW_FOLDER}/{folder_path}".strip("/")

        endpoint = f"/sites/{site_path}/drive/root:/{drive_path}:/children"

        try:
            result = self._graph_request("GET", endpoint)
            files = []
            for item in result.get("value", []):
                if "file" in item:
                    if file_extension:
                        if not item["name"].lower().endswith(file_extension.lower()):
                            continue
                    files.append({
                        "name": item["name"],
                        "path": item.get("webUrl", ""),
                        "size": item.get("size", 0),
                        "modified": item.get("lastModifiedDateTime", ""),
                        "id": item.get("id", ""),
                    })
            return files
        except Exception as e:
            print(f"Error listing files: {e}")
            return []

    def download_file(
        self,
        remote_path: str,
        local_path: str,
    ) -> bool:
        """
        Download a file from SharePoint.

        Parameters
        ----------
        remote_path : str
            Path to file in SharePoint.
        local_path : str
            Local destination path.

        Returns
        -------
        bool
            True if successful.
        """
        if not self.use_sharepoint:
            # Local mode: copy from raw to destination
            source = RAW_DATA_DIR / remote_path
            if source.exists():
                shutil.copy(source, local_path)
                return True
            return False

        # SharePoint mode
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required")

        site_path = SHAREPOINT_SITE_URL.split("/sites/")[-1]
        drive_path = f"{SHAREPOINT_RAW_FOLDER}/{remote_path}".strip("/")

        endpoint = f"/sites/{site_path}/drive/root:/{drive_path}:/content"

        token = self._get_access_token()
        headers = {"Authorization": f"Bearer {token}"}
        url = f"https://graph.microsoft.com/v1.0{endpoint}"

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(response.content)

        return True

    def upload_file(
        self,
        local_path: str,
        remote_path: str,
    ) -> bool:
        """
        Upload a file to SharePoint.

        Parameters
        ----------
        local_path : str
            Path to local file.
        remote_path : str
            Destination path in SharePoint.

        Returns
        -------
        bool
            True if successful.
        """
        if not self.use_sharepoint:
            # Local mode: copy to processed directory
            dest = PROCESSED_DATA_DIR / remote_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(local_path, dest)
            return True

        # SharePoint mode
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required")

        site_path = SHAREPOINT_SITE_URL.split("/sites/")[-1]
        drive_path = f"{SHAREPOINT_PROCESSED_FOLDER}/{remote_path}".strip("/")

        endpoint = f"/sites/{site_path}/drive/root:/{drive_path}:/content"

        token = self._get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
        }
        url = f"https://graph.microsoft.com/v1.0{endpoint}"

        with open(local_path, "rb") as f:
            response = requests.put(url, headers=headers, data=f)

        response.raise_for_status()
        return True

    def get_latest_file(
        self,
        folder_path: str = "",
        file_extension: str = ".csv",
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recently modified file in a folder.

        Parameters
        ----------
        folder_path : str
            Folder to search in.
        file_extension : str
            File extension filter.

        Returns
        -------
        Dict or None
            File metadata or None if no files found.
        """
        files = self.list_files(folder_path, file_extension)
        if not files:
            return None

        # Sort by modified time (descending)
        if self.use_sharepoint:
            files.sort(key=lambda x: x.get("modified", ""), reverse=True)
        else:
            files.sort(key=lambda x: x.get("modified", 0), reverse=True)

        return files[0]


def sync_from_sharepoint(
    client: Optional[SharePointClient] = None,
    remote_folder: str = "",
    local_folder: Optional[Path] = None,
    file_pattern: str = "*.csv",
) -> List[str]:
    """
    Sync files from SharePoint to local folder.

    Convenience function for downloading files.

    Parameters
    ----------
    client : SharePointClient, optional
        SharePoint client instance.
    remote_folder : str
        Remote folder path.
    local_folder : Path, optional
        Local destination folder.
    file_pattern : str
        File pattern to match.

    Returns
    -------
    List[str]
        List of downloaded file paths.
    """
    client = client or SharePointClient()
    local_folder = local_folder or RAW_DATA_DIR

    ext = file_pattern.replace("*", "")
    files = client.list_files(remote_folder, ext)

    downloaded = []
    for file_info in files:
        local_path = local_folder / file_info["name"]
        if client.download_file(file_info["name"], str(local_path)):
            downloaded.append(str(local_path))

    return downloaded


def sync_to_sharepoint(
    client: Optional[SharePointClient] = None,
    local_folder: Optional[Path] = None,
    remote_folder: str = "",
    file_pattern: str = "*.csv",
) -> List[str]:
    """
    Sync files from local folder to SharePoint.

    Convenience function for uploading files.

    Parameters
    ----------
    client : SharePointClient, optional
        SharePoint client instance.
    local_folder : Path, optional
        Local source folder.
    remote_folder : str
        Remote destination folder.
    file_pattern : str
        File pattern to match.

    Returns
    -------
    List[str]
        List of uploaded file names.
    """
    client = client or SharePointClient()
    local_folder = local_folder or PROCESSED_DATA_DIR

    uploaded = []
    ext = file_pattern.replace("*", "")

    for f in local_folder.iterdir():
        if f.is_file() and f.suffix == ext:
            remote_path = f"{remote_folder}/{f.name}".strip("/")
            if client.upload_file(str(f), remote_path):
                uploaded.append(f.name)

    return uploaded
