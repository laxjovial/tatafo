# shared_tools/cloud_storage_utils.py

import logging
from google.cloud import storage
from google.oauth2 import service_account
from typing import Optional, Dict, Any
import json
import os
import io # For handling file-like objects
import base64 # For base64 encoding/decoding

# Import config_manager to get GCS bucket name and credentials path
from config.config_manager import config_manager
# Import analytics_tracker for logging events - it will use the already initialized Firebase
from utils.analytics_tracker import log_event

logger = logging.getLogger(__name__)

class CloudStorageUtilsWrapper:
    """
    A wrapper class for Google Cloud Storage operations.
    This class encapsulates the GCS client initialization and provides methods
    for uploading, downloading, deleting, and reading files from GCS.
    It integrates with config_manager for bucket name and analytics_tracker for logging.
    """
    def __init__(self, config_manager_instance):
        self.config_manager = config_manager_instance
        self._gcs_client = None
        self._gcs_bucket_name = None
        self._gcs_bucket = None
        self._initialize_gcs_client()

    def _initialize_gcs_client(self):
        """Initializes the Google Cloud Storage client and bucket."""
        if self._gcs_client is None:
            try:
                # GCS bucket name can be retrieved from secrets.toml (top-level key)
                self._gcs_bucket_name = self.config_manager.get_secret("gcs_bucket_name")
                
                if not self._gcs_bucket_name:
                    raise ValueError("GCS bucket_name not specified in secrets.toml.")

                # Check for GOOGLE_APPLICATION_CREDENTIALS environment variable
                if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                    self._gcs_client = storage.Client() # Auto-detects from GOOGLE_APPLICATION_CREDENTIALS env var
                    logger.info("GCS client initialized using GOOGLE_APPLICATION_CREDENTIALS env var.")
                else:
                    # Fallback for local testing or other environments where default credentials apply
                    # This will try to use credentials from the environment (e.g., gcloud login, GCE metadata)
                    self._gcs_client = storage.Client()
                    logger.warning("GOOGLE_APPLICATION_CREDENTIALS env var not found. Relying on default application credentials for GCS client.")
                
                self._gcs_bucket = self._gcs_client.bucket(self._gcs_bucket_name)
                logger.info(f"GCS bucket '{self._gcs_bucket_name}' selected.")

            except Exception as e:
                logger.error(f"Error initializing GCS client: {e}")
                self._gcs_client = None # Ensure client is None on failure
                raise # Re-raise to propagate the error

    async def upload_file_to_storage(
        self,
        user_id: str,
        file_name: str,
        file_content_base64: str # Accepts base64 content directly
    ) -> Dict[str, Any]:
        """
        Uploads a file (from base64 encoded content) to Google Cloud Storage.

        Args:
            user_id (str): The ID of the user uploading the file.
            file_name (str): The name of the file to upload.
            file_content_base64 (str): The base64 encoded content of the file.

        Returns:
            Dict[str, Any]: A dictionary indicating success and potentially the file URL.
        """
        try:
            if not self._gcs_bucket:
                self._initialize_gcs_client() # Attempt to re-initialize if not ready
            if not self._gcs_bucket:
                raise ValueError("GCS bucket not initialized.")

            # Decode the base64 content
            file_content_bytes = base64.b64decode(file_content_base64)

            # Define the path in the bucket (e.g., user_uploads/user123/document.pdf)
            destination_blob_name = f"uploads/{user_id}/{file_name}"
            
            blob = self._gcs_bucket.blob(destination_blob_name)
            
            # Upload from bytes
            blob.upload_from_string(file_content_bytes)
            
            gcs_uri = f"gs://{self._gcs_bucket_name}/{destination_blob_name}"
            logger.info(f"File {file_name} uploaded to {gcs_uri}")
            await log_event('cloud_storage_operation', {
                'operation': 'upload',
                'file_path': destination_blob_name,
                'status': 'success',
                'storage_provider': 'GCS'
            }, user_id=user_id, success=True)
            return {"success": True, "message": "File uploaded successfully.", "file_url": gcs_uri}
        except Exception as e:
            logger.error(f"Failed to upload file {file_name} to GCS blob {destination_blob_name}: {e}", exc_info=True)
            await log_event('cloud_storage_operation', {
                'operation': 'upload',
                'file_path': destination_blob_name,
                'status': 'failure',
                'storage_provider': 'GCS',
                'error_message': str(e)
            }, user_id=user_id, success=False, error_message=str(e))
            return {"success": False, "message": f"Failed to upload file: {e}"}

    async def download_file_from_storage(
        self,
        user_id: str,
        file_name: str
    ) -> Dict[str, Any]:
        """
        Downloads a file from Google Cloud Storage and returns its content as a string.

        Args:
            user_id (str): The ID of the user who owns the file.
            file_name (str): The name of the file to download.

        Returns:
            Dict[str, Any]: A dictionary indicating success and the file content (decoded string).
        """
        try:
            if not self._gcs_bucket:
                self._initialize_gcs_client()
            if not self._gcs_bucket:
                raise ValueError("GCS bucket not initialized.")

            source_blob_name = f"uploads/{user_id}/{file_name}"
            blob = self._gcs_bucket.blob(source_blob_name)
            
            # Download as bytes and then decode to string
            file_content_bytes = blob.download_as_bytes()
            file_content_str = file_content_bytes.decode('utf-8')
            
            logger.info(f"File {source_blob_name} downloaded and read.")
            await log_event('cloud_storage_operation', {
                'operation': 'download',
                'file_path': source_blob_name,
                'status': 'success',
                'storage_provider': 'GCS'
            }, user_id=user_id, success=True)
            return {"success": True, "message": "File downloaded successfully.", "content": file_content_str}
        except Exception as e:
            logger.error(f"Failed to download file {source_blob_name} from GCS: {e}", exc_info=True)
            await log_event('cloud_storage_operation', {
                'operation': 'download',
                'file_path': source_blob_name,
                'status': 'failure',
                'storage_provider': 'GCS',
                'error_message': str(e)
            }, user_id=user_id, success=False, error_message=str(e))
            return {"success": False, "message": f"Failed to download file: {e}"}

    async def delete_file_from_storage(
        self,
        user_id: str,
        file_name: str
    ) -> Dict[str, Any]:
        """
        Deletes a file from Google Cloud Storage.

        Args:
            user_id (str): The ID of the user who owns the file.
            file_name (str): The name of the file to delete.

        Returns:
            Dict[str, Any]: A dictionary indicating success.
        """
        try:
            if not self._gcs_bucket:
                self._initialize_gcs_client()
            if not self._gcs_bucket:
                raise ValueError("GCS bucket not initialized.")

            blob_name = f"uploads/{user_id}/{file_name}"
            blob = self._gcs_bucket.blob(blob_name)
            blob.delete()
            
            logger.info(f"File {blob_name} deleted from GCS.")
            await log_event('cloud_storage_operation', {
                'operation': 'delete',
                'file_path': blob_name,
                'status': 'success',
                'storage_provider': 'GCS'
            }, user_id=user_id, success=True)
            return {"success": True, "message": "File deleted successfully."}
        except Exception as e:
            logger.error(f"Failed to delete file {blob_name} from GCS: {e}", exc_info=True)
            await log_event('cloud_storage_operation', {
                'operation': 'delete',
                'file_path': blob_name,
                'status': 'failure',
                'storage_provider': 'GCS',
                'error_message': str(e)
            }, user_id=user_id, success=False, error_message=str(e))
            return {"success": False, "message": f"Failed to delete file: {e}"}

    async def read_file_from_storage_to_bytes(
        self,
        user_id: str,
        file_name: str
    ) -> Dict[str, Any]:
        """
        Reads a file from Google Cloud Storage into bytes.

        Args:
            user_id (str): The ID of the user who owns the file.
            file_name (str): The name of the file to read.

        Returns:
            Dict[str, Any]: A dictionary indicating success and the file content as bytes.
        """
        try:
            if not self._gcs_bucket:
                self._initialize_gcs_client()
            if not self._gcs_bucket:
                raise ValueError("GCS bucket not initialized.")

            blob_name = f"uploads/{user_id}/{file_name}"
            blob = self._gcs_bucket.blob(blob_name)
            contents = blob.download_as_bytes()
            
            logger.info(f"File {blob_name} read from GCS into bytes.")
            await log_event('cloud_storage_operation', {
                'operation': 'read_bytes',
                'file_path': blob_name,
                'status': 'success',
                'storage_provider': 'GCS'
            }, user_id=user_id, success=True)
            return {"success": True, "message": "File read successfully.", "content": contents}
        except Exception as e:
            logger.error(f"Failed to read file {blob_name} from GCS into bytes: {e}", exc_info=True)
            await log_event('cloud_storage_operation', {
                'operation': 'read_bytes',
                'file_path': blob_name,
                'status': 'failure',
                'storage_provider': 'GCS',
                'error_message': str(e)
            }, user_id=user_id, success=False, error_message=str(e))
            return {"success": False, "message": f"Failed to read file: {e}"}

# CLI Test (optional) - Ensure mocks are updated to reflect class structure
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch, AsyncMock
    import asyncio
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    # Mock config_manager for local testing
    class MockConfigManager:
        _instance = None
        _is_loaded = False
        def __init__(self):
            if MockConfigManager._instance is None:
                MockConfigManager._instance = self
            self._config_data = {
                'cloud_storage': {
                    'gcs': {
                        'credentials_path': 'mock-credentials.json' # This won't exist, will trigger warning
                    }
                },
                'app_id': 'test-app-id-cli',
                'analytics': {
                    'enabled': True,
                    'log_tool_usage': True,
                    'log_query_failures': True
                }
            }
            # Mock secrets data, including the GCS bucket name
            self._secrets_data = {
                'gcs_bucket_name': 'mock-gcs-bucket-from-secrets', # Simulate secret from .streamlit/secrets.toml
                'firebase_config': json.dumps({"projectId": "mock-project-id"}) # For app_id fallback in FirestoreManager
            }
            self._is_loaded = True
        
        def get(self, key, default=None):
            parts = key.split('.')
            val = self._config_data
            for part in parts:
                if isinstance(val, dict) and part in val:
                    val = val[part]
                else:
                    return default
            return val
        
        def get_secret(self, key, default=None):
            return self._secrets_data.get(key, default)

        def set_secret(self, key, value):
            pass # No-op for mock
        
        def get_api_provider_config(self, domain: str, provider_name: str) -> Optional[Dict[str, Any]]:
            return None # Not relevant for this module

        def get_domain_api_providers(self, domain: str) -> Dict[str, Any]:
            return {}

    # Patch the actual imports for testing
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager

    # Mock firebase_admin.firestore and auth for analytics initialization
    mock_db_for_analytics = MagicMock()
    mock_auth_for_analytics = MagicMock()
    mock_auth_for_analytics.currentUser = MagicMock(uid="test_cli_user")
    mock_db_for_analytics.collection.return_value.add = AsyncMock(return_value=MagicMock(id="mock_doc_id"))

    # Patch firebase_admin.firestore and auth for the local import within log_event
    with patch.dict(sys.modules, {
        'firebase_admin.firestore': MagicMock(firestore=MagicMock()),
        'firebase_admin.auth': MagicMock(auth=MagicMock())
    }):
        sys.modules['firebase_admin.firestore'].firestore.CollectionReference = MagicMock()
        sys.modules['firebase_admin.firestore'].firestore.DocumentReference = MagicMock()
            
        with patch('firebase_admin.get_app', return_value=MagicMock(project_id="mock-project-id-from-app")):
            if 'analytics_initialized_backend' not in globals() or not globals()['analytics_initialized_backend']:
                # Mock the initialize_analytics function itself if it's called
                with patch('utils.analytics_tracker.initialize_analytics') as mock_init_analytics:
                    mock_init_analytics(
                        mock_db_for_analytics,
                        mock_auth_for_analytics,
                        "test-app-id-cli",
                        "test_cli_user"
                    )
                    globals()['analytics_initialized_backend'] = True
                    logger.info("Analytics tracker initialized with mocks for CLI test.")

    # Mock google.cloud.storage for GCS operations
    mock_blob = MagicMock()
    mock_blob.upload_from_string = MagicMock() # Changed from upload_from_filename
    mock_blob.download_as_bytes = MagicMock(return_value=b"mock file content") # Changed from download_to_filename
    mock_blob.delete = MagicMock()

    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    mock_gcs_client_instance = MagicMock()
    mock_gcs_client_instance.bucket.return_value = mock_bucket

    # Patch the actual storage.Client constructor
    with patch('google.cloud.storage.Client', return_value=mock_gcs_client_instance) as mock_gcs_client_constructor:
        async def run_gcs_tests():
            print("\n--- Testing GCS Utility Functions (Class-based) ---")
            test_user_id = "test_user_123"
            test_file_name = "test_document.txt"
            test_content_base64 = base64.b64encode(b"This is a test file content.").decode('utf-8')
            test_blob_name = f"uploads/{test_user_id}/{test_file_name}"

            # Instantiate the wrapper
            gcs_wrapper = CloudStorageUtilsWrapper(MockConfigManager())

            # Test upload_file_to_storage
            print("\n--- Test 1: upload_file_to_storage (Success) ---")
            mock_db_for_analytics.collection.return_value.add.reset_mock()
            upload_result = await gcs_wrapper.upload_file_to_storage(test_user_id, test_file_name, test_content_base64)
            print(f"Upload Result: {upload_result}")
            assert upload_result["success"] is True
            assert f"gs://{MockConfigManager().get_secret('gcs_bucket_name')}/{test_blob_name}" == upload_result["file_url"]
            mock_blob.upload_from_string.assert_called_once_with(base64.b64decode(test_content_base64))
            mock_db_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_db_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "cloud_storage_operation"
            assert logged_data["details"]["operation"] == "upload"
            assert logged_data["success"] is True
            print("Test 1 Passed (and analytics logged success).")

            # Test download_file_from_storage
            print("\n--- Test 2: download_file_from_storage (Success) ---")
            mock_db_for_analytics.collection.return_value.add.reset_mock()
            download_result = await gcs_wrapper.download_file_from_storage(test_user_id, test_file_name)
            print(f"Download Result: {download_result}")
            assert download_result["success"] is True
            assert download_result["content"] == b"mock file content".decode('utf-8')
            mock_blob.download_as_bytes.assert_called_once()
            mock_db_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_db_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "cloud_storage_operation"
            assert logged_data["details"]["operation"] == "download"
            assert logged_data["success"] is True
            print("Test 2 Passed (and analytics logged success).")

            # Test read_file_from_storage_to_bytes
            print("\n--- Test 3: read_file_from_storage_to_bytes (Success) ---")
            mock_db_for_analytics.collection.return_value.add.reset_mock()
            read_result = await gcs_wrapper.read_file_from_storage_to_bytes(test_user_id, test_file_name)
            print(f"Read Bytes Result: {read_result}")
            assert read_result["success"] is True
            assert read_result["content"] == b"mock file content"
            mock_db_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_db_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "cloud_storage_operation"
            assert logged_data["details"]["operation"] == "read_bytes"
            assert logged_data["success"] is True
            print("Test 3 Passed (and analytics logged success).")

            # Test delete_file_from_storage
            print("\n--- Test 4: delete_file_from_storage (Success) ---")
            mock_db_for_analytics.collection.return_value.add.reset_mock()
            delete_result = await gcs_wrapper.delete_file_from_storage(test_user_id, test_file_name)
            print(f"Delete Result: {delete_result}")
            assert delete_result["success"] is True
            mock_blob.delete.assert_called_once()
            mock_db_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_db_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "cloud_storage_operation"
            assert logged_data["details"]["operation"] == "delete"
            assert logged_data["success"] is True
            print("Test 4 Passed (and analytics logged success).")

            # Test upload_file_to_storage (Failure - e.g., no bucket)
            print("\n--- Test 5: upload_file_to_storage (Failure - No Bucket) ---")
            # Temporarily set _gcs_bucket to None to simulate failure
            gcs_wrapper._gcs_bucket = None 
            # Mock _initialize_gcs_client to raise an error
            with patch.object(gcs_wrapper, '_initialize_gcs_client', side_effect=ValueError("Mock GCS Init Error")):
                mock_db_for_analytics.collection.return_value.add.reset_mock()
                upload_fail_result = await gcs_wrapper.upload_file_to_storage(test_user_id, "fail_blob.txt", test_content_base64)
                print(f"Upload Fail Result: {upload_fail_result}")
                assert upload_fail_result["success"] is False
                mock_db_for_analytics.collection.return_value.add.assert_called_once()
                args, kwargs = mock_db_for_analytics.collection.return_value.add.call_args
                logged_data = args[0]
                assert logged_data["event_type"] == "cloud_storage_operation"
                assert logged_data["details"]["operation"] == "upload"
                assert logged_data["success"] is False
                assert "Mock GCS Init Error" in logged_data["error_message"]
                print("Test 5 Passed (and analytics logged failure).")
            # Restore bucket for subsequent tests if any, though none follow in this block
            gcs_wrapper._initialize_gcs_client() 

            print("\nAll GCS utility tests completed.")

        asyncio.run(run_gcs_tests())

