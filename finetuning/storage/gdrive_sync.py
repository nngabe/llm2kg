"""
Google Drive synchronization for datasets and checkpoints.

Uses pydrive2 for authentication and file operations.
"""

import os
import logging
from typing import Optional, List
from pathlib import Path

try:
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
    PYDRIVE_AVAILABLE = True
except ImportError:
    PYDRIVE_AVAILABLE = False

from ..config import StorageConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GDriveSync:
    """
    Synchronizes files with Google Drive.

    Supports:
    - Uploading datasets after generation
    - Syncing training checkpoints periodically
    - Downloading datasets to resume training
    """

    def __init__(self, config: StorageConfig):
        if not PYDRIVE_AVAILABLE:
            raise ImportError(
                "pydrive2 is required for Google Drive sync. "
                "Install with: pip install pydrive2"
            )

        self.config = config
        self._drive = None
        self._authenticated = False

    def authenticate(self, credentials_path: Optional[str] = None):
        """
        Authenticate with Google Drive.

        Uses OAuth2 flow. On first run, will open browser for authentication.
        Credentials are saved for subsequent runs.

        Args:
            credentials_path: Path to client_secrets.json or saved credentials
        """
        if credentials_path is None:
            credentials_path = self.config.gdrive_credentials_path

        gauth = GoogleAuth()

        # Try to load saved credentials
        gauth.LoadCredentialsFile(credentials_path)

        if gauth.credentials is None:
            # Authenticate if no credentials
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            # Refresh if expired
            gauth.Refresh()
        else:
            # Initialize with valid credentials
            gauth.Authorize()

        # Save credentials for next time
        gauth.SaveCredentialsFile(credentials_path)

        self._drive = GoogleDrive(gauth)
        self._authenticated = True
        logger.info("Google Drive authentication successful")

    def _ensure_authenticated(self):
        """Ensure we're authenticated before operations."""
        if not self._authenticated:
            self.authenticate()

    def _get_or_create_folder(
        self,
        folder_name: str,
        parent_id: Optional[str] = None,
    ) -> str:
        """Get existing folder ID or create new one."""
        self._ensure_authenticated()

        # Search for existing folder
        query = f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent_id:
            query += f" and '{parent_id}' in parents"

        file_list = self._drive.ListFile({"q": query}).GetList()

        if file_list:
            return file_list[0]["id"]

        # Create new folder
        folder_metadata = {
            "title": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            folder_metadata["parents"] = [{"id": parent_id}]

        folder = self._drive.CreateFile(folder_metadata)
        folder.Upload()

        logger.info(f"Created Google Drive folder: {folder_name}")
        return folder["id"]

    def upload_file(
        self,
        local_path: str,
        folder_id: Optional[str] = None,
        overwrite: bool = True,
    ) -> str:
        """
        Upload a single file to Google Drive.

        Args:
            local_path: Path to local file
            folder_id: Google Drive folder ID (uses config default if not provided)
            overwrite: Whether to overwrite existing file with same name

        Returns:
            Google Drive file ID
        """
        self._ensure_authenticated()

        if folder_id is None:
            folder_id = self.config.gdrive_folder_id

        file_name = os.path.basename(local_path)

        # Check for existing file
        if overwrite and folder_id:
            query = f"title='{file_name}' and '{folder_id}' in parents and trashed=false"
            existing = self._drive.ListFile({"q": query}).GetList()
            if existing:
                # Delete existing file
                existing[0].Delete()

        # Upload new file
        file_metadata = {"title": file_name}
        if folder_id:
            file_metadata["parents"] = [{"id": folder_id}]

        gfile = self._drive.CreateFile(file_metadata)
        gfile.SetContentFile(local_path)
        gfile.Upload()

        logger.info(f"Uploaded {file_name} to Google Drive")
        return gfile["id"]

    def upload_directory(
        self,
        local_dir: str,
        folder_id: Optional[str] = None,
        folder_name: Optional[str] = None,
    ) -> str:
        """
        Upload a directory and its contents to Google Drive.

        Args:
            local_dir: Path to local directory
            folder_id: Parent folder ID in Google Drive
            folder_name: Name for the folder in Drive (defaults to local dir name)

        Returns:
            Google Drive folder ID of uploaded directory
        """
        self._ensure_authenticated()

        if folder_id is None:
            folder_id = self.config.gdrive_folder_id

        if folder_name is None:
            folder_name = os.path.basename(local_dir)

        # Create folder in Drive
        drive_folder_id = self._get_or_create_folder(folder_name, folder_id)

        # Upload all files
        for root, dirs, files in os.walk(local_dir):
            # Get relative path from local_dir
            rel_path = os.path.relpath(root, local_dir)

            # Create subfolders
            current_folder_id = drive_folder_id
            if rel_path != ".":
                for part in rel_path.split(os.sep):
                    current_folder_id = self._get_or_create_folder(part, current_folder_id)

            # Upload files
            for file_name in files:
                local_path = os.path.join(root, file_name)
                self.upload_file(local_path, current_folder_id)

        logger.info(f"Uploaded directory {local_dir} to Google Drive")
        return drive_folder_id

    def download_file(
        self,
        file_id: str,
        local_path: str,
    ):
        """Download a file from Google Drive."""
        self._ensure_authenticated()

        gfile = self._drive.CreateFile({"id": file_id})
        gfile.GetContentFile(local_path)

        logger.info(f"Downloaded file to {local_path}")

    def download_directory(
        self,
        folder_id: str,
        local_dir: str,
    ):
        """Download a folder and its contents from Google Drive."""
        self._ensure_authenticated()

        os.makedirs(local_dir, exist_ok=True)

        # List all files in folder
        query = f"'{folder_id}' in parents and trashed=false"
        file_list = self._drive.ListFile({"q": query}).GetList()

        for item in file_list:
            local_path = os.path.join(local_dir, item["title"])

            if item["mimeType"] == "application/vnd.google-apps.folder":
                # Recursively download subfolder
                self.download_directory(item["id"], local_path)
            else:
                # Download file
                item.GetContentFile(local_path)
                logger.info(f"Downloaded {item['title']}")

        logger.info(f"Downloaded folder to {local_dir}")

    def sync_checkpoints(
        self,
        checkpoint_dir: str,
        folder_id: Optional[str] = None,
    ):
        """
        Sync training checkpoints to Google Drive.

        Only uploads new checkpoints that haven't been synced yet.
        """
        self._ensure_authenticated()

        if folder_id is None:
            folder_id = self.config.gdrive_folder_id

        # Create checkpoints folder in Drive
        checkpoints_folder_id = self._get_or_create_folder("checkpoints", folder_id)

        # Get list of already uploaded checkpoints
        query = f"'{checkpoints_folder_id}' in parents and trashed=false"
        existing_files = self._drive.ListFile({"q": query}).GetList()
        existing_names = {f["title"] for f in existing_files}

        # Upload new checkpoints
        for item in os.listdir(checkpoint_dir):
            if item.startswith("checkpoint-") and item not in existing_names:
                item_path = os.path.join(checkpoint_dir, item)
                if os.path.isdir(item_path):
                    self.upload_directory(item_path, checkpoints_folder_id, item)
                else:
                    self.upload_file(item_path, checkpoints_folder_id)

        logger.info("Checkpoint sync complete")

    def upload_dataset(
        self,
        local_path: str,
        dataset_name: str = "dataset",
    ) -> str:
        """
        Upload a dataset directory to Google Drive.

        Args:
            local_path: Path to dataset directory
            dataset_name: Name for the dataset folder in Drive

        Returns:
            Google Drive folder ID
        """
        folder_id = self.config.gdrive_folder_id

        # Create datasets folder
        datasets_folder_id = self._get_or_create_folder("datasets", folder_id)

        # Upload dataset
        return self.upload_directory(local_path, datasets_folder_id, dataset_name)

    def list_files(self, folder_id: Optional[str] = None) -> List[dict]:
        """List files in a Google Drive folder."""
        self._ensure_authenticated()

        if folder_id is None:
            folder_id = self.config.gdrive_folder_id

        query = f"'{folder_id}' in parents and trashed=false"
        file_list = self._drive.ListFile({"q": query}).GetList()

        return [
            {"id": f["id"], "title": f["title"], "mimeType": f["mimeType"]}
            for f in file_list
        ]
