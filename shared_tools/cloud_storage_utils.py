# shared_tools/cloud_storage_utils.py

import logging
from google.cloud import storage
from google.oauth2 import service_account
from google.auth.exceptions import DefaultCredentialsError # Import this for specific error handling
from typing import Optional, Dict, Any
import json
import os
import io # For handling file-like objects
import base64 # For base64 encoding/decoding

# Import config_manager to get GCS bucket name and credentials path
from config.config_manager import config_manager
# Import analytics_tracker for logging events - it will use the already initialized Firebase
# Removed direct log_event call from here to simplify error handling within this utility
# and allow for graceful fallback in calling modules (like vector_utils).
# from utils.analytics_tracker import log_event # Removed from here
# from utils.user_manager import UserManager # Removed from here, if you need user context for logging, pass it or handle upstream

logger = logging.getLogger(__name__)

class CloudStorageUtilsWrapper:
    """
    A wrapper class for Google Cloud Storage operations.
    This class encapsulates the GCS client initialization and provides methods
    for uploading, downloading, deleting, and reading files from GCS.
    It integrates with config_manager for bucket name.
    """
    def __init__(self, config_manager_instance):
        self.config_manager = config_manager_instance
        self._gcs_client: Optional[storage.Client] = None
        self._gcs_bucket_name: Optional[str] = None
        self._gcs_bucket: Optional[storage.Bucket] = None
        self._initialize_gcs_client()

    def _initialize_gcs_client(self):
        """
        Initializes the Google Cloud Storage client and bucket.
        Gracefully handles missing credentials by setting clients to None
        and logging warnings/errors, allowing the application to run without GCS
        functionality if not configured.
        """
        # Do not re-initialize if already done
        if self._gcs_client is not None:
            return

        self._gcs_bucket_name = self.config_manager.get_secret("gcs_bucket_name")

        if not self._gcs_bucket_name:
            logger.warning("GCS bucket name not configured. Google Cloud Storage functionality will be unavailable.")
            self._gcs_client = None
            self._gcs_bucket = None
            return # Exit early if no bucket name

        try:
            credentials_path = self.config_manager.get_secret("gcs_credentials_path")
            credentials = None

            if credentials_path and os.path.exists(credentials_path):
                # Use service account credentials if path is provided and file exists
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                self._gcs_client = storage.Client(credentials=credentials)
                logger.info(f"GCS client initialized with service account file from '{credentials_path}'.")
            else:
                # Fallback to default credentials (e.g., GOOGLE_APPLICATION_CREDENTIALS, GCE metadata, ADC)
                # This is the path that likely led to the original error if GOOGLE_APPLICATION_CREDENTIALS
                # was pointing to a non-existent file.
                self._gcs_client = storage.Client()
                if credentials_path: # Path was provided but file didn't exist
                    logger.warning(
                        f"GCS credentials file not found at '{credentials_path}'. "
                        "Attempting default GCS client initialization via environment/ADC."
                    )
                else: # No path was provided
                    logger.warning(
                        "No GCS credentials path configured. "
                        "Attempting default GCS client initialization via environment/ADC."
                    )

            # Attempt to get the bucket only if client initialized successfully
            if self._gcs_client:
                self._gcs_bucket = self._gcs_client.get_bucket(self._gcs_bucket_name)
                logger.info(f"Connected to GCS bucket: {self._gcs_bucket_name}")
            else:
                # If client didn't initialize, bucket won't either
                logger.error("GCS client could not be initialized, so bucket connection skipped.")
                self._gcs_bucket = None


        except DefaultCredentialsError as e:
            logger.error(
                f"Error initializing GCS client due to missing or invalid credentials: {e}. "
                "Please ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set "
                "to a valid service account key file, or you are running on GCP with ADC enabled."
            )
            self._gcs_client = None
            self._gcs_bucket = None
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during GCS client or bucket initialization: {e}",
                exc_info=True
            )
            self._gcs_client = None
            self._gcs_bucket = None

    def _ensure_gcs_ready(self) -> bool:
        """Helper to ensure GCS client and bucket are initialized."""
        if self._gcs_client is None or self._gcs_bucket is None:
            logger.error("Google Cloud Storage is not initialized or configured. Operation aborted.")
            return False
        return True

    async def upload_file_to_storage(self, user_id: str, blob_name: str, file_content_base64: str) -> Dict[str, Any]:
        """Uploads a file (base64 encoded content) to the GCS bucket."""
        if not self._ensure_gcs_ready():
            return {"success": False, "message": "GCS not available for upload.", "url": None}

        try:
            file_content_bytes = base64.b64decode(file_content_base64)
            blob = self._gcs_bucket.blob(blob_name)
            blob.upload_from_file(io.BytesIO(file_content_bytes))
            file_url = f"gs://{self._gcs_bucket_name}/{blob_name}"
            logger.info(f"File {blob_name} uploaded to GCS for user {user_id}. URL: {file_url}")
            # Analytics logging can be handled by the caller (e.g., vector_utils) if it needs user_id
            return {"success": True, "message": f"File '{blob_name}' uploaded successfully.", "url": file_url}
        except Exception as e:
            logger.error(f"Error uploading file '{blob_name}' to GCS: {e}", exc_info=True)
            # Analytics logging can be handled by the caller (e.g., vector_utils)
            return {"success": False, "message": f"Failed to upload file '{blob_name}': {e}", "url": None}

    async def download_file_from_storage(self, user_id: str, blob_name: str) -> Dict[str, Any]:
        """Downloads a file from the GCS bucket and returns its base64 encoded content."""
        if not self._ensure_gcs_ready():
            return {"success": False, "message": "GCS not available for download.", "content_base64": None}

        try:
            blob = self._gcs_bucket.blob(blob_name)
            if not blob.exists():
                logger.warning(f"Blob '{blob_name}' not found in GCS bucket '{self._gcs_bucket_name}'.")
                return {"success": False, "message": f"File '{blob_name}' not found.", "content_base64": None}

            file_content_bytes = blob.download_as_bytes()
            file_content_base64 = base64.b64encode(file_content_bytes).decode('utf-8')
            logger.info(f"File {blob_name} downloaded from GCS for user {user_id}.")
            return {"success": True, "message": f"File '{blob_name}' downloaded successfully.", "content_base64": file_content_base64}
        except Exception as e:
            logger.error(f"Error downloading file '{blob_name}' from GCS: {e}", exc_info=True)
            return {"success": False, "message": f"Failed to download file '{blob_name}': {e}", "content_base64": None}

    async def delete_file_from_storage(self, user_id: str, blob_name: str) -> Dict[str, Any]:
        """Deletes a file from the GCS bucket."""
        if not self._ensure_gcs_ready():
            return {"success": False, "message": "GCS not available for deletion."}

        try:
            blob = self._gcs_bucket.blob(blob_name)
            if blob.exists():
                blob.delete()
                logger.info(f"File {blob_name} deleted from GCS for user {user_id}.")
                return {"success": True, "message": f"File '{blob_name}' deleted successfully."}
            else:
                logger.warning(f"Blob '{blob_name}' not found for deletion in GCS bucket '{self._gcs_bucket_name}'.")
                return {"success": True, "message": f"File '{blob_name}' was not found, no action needed."}
        except Exception as e:
            logger.error(f"Error deleting file '{blob_name}' from GCS: {e}", exc_info=True)
            return {"success": False, "message": f"Failed to delete file '{blob_name}': {e}"}

    async def read_file_content(self, user_id: str, blob_name: str) -> Dict[str, Any]:
        """Reads the content of a file directly from GCS without downloading to disk."""
        if not self._ensure_gcs_ready():
            return {"success": False, "message": "GCS not available for reading content.", "content": None}

        try:
            blob = self._gcs_bucket.blob(blob_name)
            if not blob.exists():
                logger.warning(f"Blob '{blob_name}' not found for reading content in GCS bucket '{self._gcs_bucket_name}'.")
                return {"success": False, "message": f"File '{blob_name}' not found.", "content": None}

            content = blob.download_as_text() # Reads as string
            logger.info(f"Content of {blob_name} read from GCS for user {user_id}.")
            return {"success": True, "message": f"Content of '{blob_name}' read successfully.", "content": content}
        except Exception as e:
            logger.error(f"Error reading content of file '{blob_name}' from GCS: {e}", exc_info=True)
            return {"success": False, "message": f"Failed to read content of file '{blob_name}': {e}", "content": None}
