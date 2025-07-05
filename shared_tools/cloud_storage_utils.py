# shared_tools/cloud_storage_utils.py

import logging
from google.cloud import storage
from google.oauth2 import service_account
from typing import Optional, Dict, Any
import json
import os
import io # For handling file-like objects
import asyncio # For async operations in mocks/tests

# Import config_manager to get GCS bucket name and credentials path
from config.config_manager import config_manager
# Import analytics_tracker for logging events - it will use the already initialized Firebase
from utils.analytics_tracker import log_event # Removed initialize_analytics as it's done in main.py

logger = logging.getLogger(__name__)

_gcs_client = None
_gcs_bucket_name = None
_gcs_bucket = None # GCS Bucket object

# --- REMOVED: Firebase Admin SDK Initialization block ---
# Firebase Admin SDK and analytics initialization should happen ONLY in backend/main.py
# This module will rely on Firebase being initialized there.


def _initialize_gcs_client():
    """Initializes the Google Cloud Storage client."""
    global _gcs_client, _gcs_bucket_name, _gcs_bucket
    if _gcs_client is None:
        try:
            # GCS bucket name can be retrieved from secrets.toml (top-level key)
            _gcs_bucket_name = config_manager.get_secret("gcs_bucket_name")
            
            # GCS credentials can be handled by GOOGLE_APPLICATION_CREDENTIALS env var
            # or implicitly by default credentials in cloud environments.
            # For Codespaces, setting GOOGLE_APPLICATION_CREDENTIALS env var is best practice
            # with your service account key.
            
            # Check for GOOGLE_APPLICATION_CREDENTIALS environment variable
            if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                _gcs_client = storage.Client() # Auto-detects from GOOGLE_APPLICATION_CREDENTIALS env var
                logger.info("GCS client initialized using GOOGLE_APPLICATION_CREDENTIALS env var.")
            else:
                # Fallback for local testing or other environments where default credentials apply
                # This will try to use credentials from the environment (e.g., gcloud login, GCE metadata)
                _gcs_client = storage.Client()
                logger.warning("GOOGLE_APPLICATION_CREDENTIALS env var not found. Relying on default application credentials for GCS client.")
            
            if not _gcs_bucket_name:
                raise ValueError("GCS bucket_name not specified in secrets.toml.")
            
            _gcs_bucket = _gcs_client.bucket(_gcs_bucket_name)
            logger.info(f"GCS bucket '{_gcs_bucket_name}' selected.")

        except Exception as e:
            logger.error(f"Error initializing GCS client: {e}")
            _gcs_client = None # Ensure client is None on failure
            raise # Re-raise to propagate the error


def get_gcs_client():
    """Returns the initialized GCS client."""
    if _gcs_client is None:
        _initialize_gcs_client()
    return _gcs_client

def get_gcs_bucket():
    """Returns the initialized GCS bucket object."""
    if _gcs_bucket is None:
        _initialize_gcs_client()
    return _gcs_bucket

async def upload_file_to_gcs(
    source_file_path: str,
    destination_blob_name: str,
    user_id: Optional[str] = "backend_system_user"
) -> Optional[str]:
    """
    Uploads a file to Google Cloud Storage.

    Args:
        source_file_path (str): The path to the file to upload.
        destination_blob_name (str): The desired path/name of the file in the GCS bucket.
                                      (e.g., 'user_uploads/user123/document.pdf')
        user_id (str, optional): The ID of the user performing the upload for analytics.

    Returns:
        Optional[str]: The GCS URI of the uploaded file (e.g., 'gs://your-bucket-name/path/to/file.pdf')
                       or None if the upload fails.
    """
    try:
        bucket = get_gcs_bucket()
        if not bucket:
            raise ValueError("GCS bucket not initialized.")

        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_path)
        
        gcs_uri = f"gs://{_gcs_bucket_name}/{destination_blob_name}"
        logger.info(f"File {source_file_path} uploaded to {gcs_uri}")
        await log_event('cloud_storage_operation', {
            'operation': 'upload',
            'file_path': destination_blob_name,
            'status': 'success',
            'storage_provider': 'GCS'
        }, user_id=user_id, success=True)
        return gcs_uri
    except Exception as e:
        logger.error(f"Failed to upload file {source_file_path} to GCS blob {destination_blob_name}: {e}", exc_info=True)
        await log_event('cloud_storage_operation', {
            'operation': 'upload',
            'file_path': destination_blob_name,
            'status': 'failure',
            'storage_provider': 'GCS',
            'error_message': str(e)
        }, user_id=user_id, success=False, error_message=str(e))
        return None

async def download_file_from_gcs(
    source_blob_name: str,
    destination_file_path: str,
    user_id: Optional[str] = "backend_system_user"
) -> bool:
    """
    Downloads a file from Google Cloud Storage to a local path.

    Args:
        source_blob_name (str): The name of the file (blob) in the GCS bucket.
        destination_file_path (str): The local path where the file should be saved.
        user_id (str, optional): The ID of the user performing the download for analytics.

    Returns:
        bool: True if the download is successful, False otherwise.
    """
    try:
        bucket = get_gcs_bucket()
        if not bucket:
            raise ValueError("GCS bucket not initialized.")

        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_path)
        
        logger.info(f"File {source_blob_name} downloaded to {destination_file_path}")
        await log_event('cloud_storage_operation', {
            'operation': 'download',
            'file_path': source_blob_name,
            'status': 'success',
            'storage_provider': 'GCS'
        }, user_id=user_id, success=True)
        return True
    except Exception as e:
        logger.error(f"Failed to download file {source_blob_name} from GCS: {e}", exc_info=True)
        await log_event('cloud_storage_operation', {
            'operation': 'download',
            'file_path': source_blob_name,
            'status': 'failure',
            'storage_provider': 'GCS',
            'error_message': str(e)
        }, user_id=user_id, success=False, error_message=str(e))
        return False

async def delete_file_from_gcs(
    blob_name: str,
    user_id: Optional[str] = "backend_system_user"
) -> bool:
    """
    Deletes a file from Google Cloud Storage.

    Args:
        blob_name (str): The name of the file (blob) in the GCS bucket to delete.
        user_id (str, optional): The ID of the user performing the deletion for analytics.

    Returns:
        bool: True if the deletion is successful, False otherwise.
    """
    try:
        bucket = get_gcs_bucket()
        if not bucket:
            raise ValueError("GCS bucket not initialized.")

        blob = bucket.blob(blob_name)
        blob.delete()
        
        logger.info(f"File {blob_name} deleted from GCS.")
        await log_event('cloud_storage_operation', {
            'operation': 'delete',
            'file_path': blob_name,
            'status': 'success',
            'storage_provider': 'GCS'
        }, user_id=user_id, success=True)
        return True
    except Exception as e:
        logger.error(f"Failed to delete file {blob_name} from GCS: {e}", exc_info=True)
        await log_event('cloud_storage_operation', {
            'operation': 'delete',
            'file_path': blob_name,
            'status': 'failure',
            'storage_provider': 'GCS',
            'error_message': str(e)
        }, user_id=user_id, success=False, error_message=str(e))
        return False

async def read_file_from_gcs_to_bytes(
    blob_name: str,
    user_id: Optional[str] = "backend_system_user"
) -> Optional[bytes]:
    """
    Reads a file from Google Cloud Storage into bytes.

    Args:
        blob_name (str): The name of the file (blob) in the GCS bucket.
        user_id (str, optional): The ID of the user performing the read for analytics.

    Returns:
        Optional[bytes]: The content of the file as bytes, or None if the read fails.
    """
    try:
        bucket = get_gcs_bucket()
        if not bucket:
            raise ValueError("GCS bucket not initialized.")

        blob = bucket.blob(blob_name)
        contents = blob.download_as_bytes()
        
        logger.info(f"File {blob_name} read from GCS into bytes.")
        await log_event('cloud_storage_operation', {
            'operation': 'read_bytes',
            'file_path': blob_name,
            'status': 'success',
            'storage_provider': 'GCS'
        }, user_id=user_id, success=True)
        return contents
    except Exception as e:
        logger.error(f"Failed to read file {blob_name} from GCS into bytes: {e}", exc_info=True)
        await log_event('cloud_storage_operation', {
            'operation': 'read_bytes',
            'file_path': blob_name,
            'status': 'failure',
            'storage_provider': 'GCS',
            'error_message': str(e)
        }, user_id=user_id, success=False, error_message=str(e))
        return None

# CLI Test (optional) - Ensure mocks are updated to reflect removal of Firebase init
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch, AsyncMock # Ensure AsyncMock is imported

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
                        # 'bucket_name' is now expected from secrets, not config.yml
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
    # These mocks are needed because log_event imports them.
    mock_db_for_analytics = MagicMock()
    mock_auth_for_analytics = MagicMock()
    mock_auth_for_analytics.currentUser = MagicMock(uid="test_cli_user")
    mock_db_for_analytics.collection.return_value.add = AsyncMock(return_value=MagicMock(id="mock_doc_id"))

    # Patch firebase_admin.firestore and auth for the local import within log_event
    with patch.dict(sys.modules, {
        'firebase_admin.firestore': MagicMock(firestore=MagicMock()),
        'firebase_admin.auth': MagicMock(auth=MagicMock())
    }):
        # Ensure the patched modules have the necessary attributes
        sys.modules['firebase_admin.firestore'].firestore.CollectionReference = MagicMock()
        sys.modules['firebase_admin.firestore'].firestore.DocumentReference = MagicMock()
        
        # We need to manually set the global db and auth_sdk for initialize_analytics if it's called
        # in a test context where main.py hasn't run.
        # For this specific module's CLI test, we'll manually call initialize_analytics with mocks
        # as it's a standalone test for this module's functions.
        # In the actual backend runtime, main.py ensures analytics is initialized.
        
        # Mock firebase_admin.get_app() if it's called by analytics_tracker directly
        with patch('firebase_admin.get_app', return_value=MagicMock(project_id="mock-project-id-from-app")):
            # Re-initialize analytics with mocks if it hasn't been already
            # This is specifically for the `if __name__ == "__main__"` block
            if 'analytics_initialized_backend' not in globals() or not globals()['analytics_initialized_backend']:
                initialize_analytics(
                    mock_db_for_analytics,
                    mock_auth_for_analytics,
                    "test-app-id-cli", # Use a test app_id for this mock context
                    "test_cli_user"
                )
                globals()['analytics_initialized_backend'] = True
                logger.info("Analytics tracker initialized with mocks for CLI test.")


    # Mock google.cloud.storage for GCS operations
    mock_blob = MagicMock()
    mock_blob.upload_from_filename = MagicMock()
    mock_blob.download_to_filename = MagicMock()
    mock_blob.download_as_bytes = MagicMock(return_value=b"mock file content")
    mock_blob.delete = MagicMock()

    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    mock_gcs_client_instance = MagicMock()
    mock_gcs_client_instance.bucket.return_value = mock_bucket

    # Patch the actual storage.Client constructor
    with patch('google.cloud.storage.Client', return_value=mock_gcs_client_instance) as mock_gcs_client_constructor:
        # Reset global client/bucket to force re-initialization with mock
        _gcs_client = None
        _gcs_bucket = None

        async def run_gcs_tests():
            print("\n--- Testing GCS Utility Functions ---")
            test_user_id = "test_user_123"
            test_file_path = "/tmp/test_upload.txt"
            test_blob_name = "user_uploads/test_user_123/test_document.txt"
            test_download_path = "/tmp/test_download.txt"

            # Create a dummy file for upload
            with open(test_file_path, "w") as f:
                f.write("This is a test file content.")

            # Test upload_file_to_gcs
            print("\n--- Test 1: upload_file_to_gcs (Success) ---")
            mock_db_for_analytics.collection.return_value.add.reset_mock() # Reset mock for analytics
            gcs_uri = await upload_file_to_gcs(test_file_path, test_blob_name, user_id=test_user_id)
            print(f"Uploaded GCS URI: {gcs_uri}")
            assert gcs_uri is not None
            assert f"gs://{MockConfigManager().get_secret('gcs_bucket_name')}/{test_blob_name}" == gcs_uri
            mock_blob.upload_from_filename.assert_called_once_with(test_file_path)
            mock_db_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_db_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "cloud_storage_operation"
            assert logged_data["details"]["operation"] == "upload"
            assert logged_data["success"] is True
            print("Test 1 Passed (and analytics logged success).")

            # Test download_file_from_gcs
            print("\n--- Test 2: download_file_from_gcs (Success) ---")
            mock_db_for_analytics.collection.return_value.add.reset_mock()
            download_success = await download_file_from_gcs(test_blob_name, test_download_path, user_id=test_user_id)
            print(f"Download Success: {download_success}")
            assert download_success is True
            mock_blob.download_to_filename.assert_called_once_with(test_download_path)
            mock_db_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_db_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "cloud_storage_operation"
            assert logged_data["details"]["operation"] == "download"
            assert logged_data["success"] is True
            print("Test 2 Passed (and analytics logged success).")

            # Test read_file_from_gcs_to_bytes
            print("\n--- Test 3: read_file_from_gcs_to_bytes (Success) ---")
            mock_db_for_analytics.collection.return_value.add.reset_mock()
            file_contents = await read_file_from_gcs_to_bytes(test_blob_name, user_id=test_user_id)
            print(f"File Contents (bytes): {file_contents}")
            assert file_contents == b"mock file content"
            mock_blob.download_as_bytes.assert_called_once()
            mock_db_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_db_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "cloud_storage_operation"
            assert logged_data["details"]["operation"] == "read_bytes"
            assert logged_data["success"] is True
            print("Test 3 Passed (and analytics logged success).")

            # Test delete_file_from_gcs
            print("\n--- Test 4: delete_file_from_gcs (Success) ---")
            mock_db_for_analytics.collection.return_value.add.reset_mock()
            delete_success = await delete_file_from_gcs(test_blob_name, user_id=test_user_id)
            print(f"Delete Success: {delete_success}")
            assert delete_success is True
            mock_blob.delete.assert_called_once()
            mock_db_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_db_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "cloud_storage_operation"
            assert logged_data["details"]["operation"] == "delete"
            assert logged_data["success"] is True
            print("Test 4 Passed (and analytics logged success).")

            # Test upload_file_to_gcs (Failure - e.g., no bucket)
            print("\n--- Test 5: upload_file_to_gcs (Failure - No Bucket) ---")
            # Temporarily set _gcs_bucket to None to simulate failure
            global _gcs_bucket
            original_gcs_bucket = _gcs_bucket
            _gcs_bucket = None
            # Mock _initialize_gcs_client to raise an error
            with patch('shared_tools.cloud_storage_utils._initialize_gcs_client', side_effect=ValueError("Mock GCS Init Error")):
                mock_db_for_analytics.collection.return_value.add.reset_mock()
                gcs_uri_fail = await upload_file_to_gcs(test_file_path, "fail_blob.txt", user_id=test_user_id)
                print(f"Upload Fail URI: {gcs_uri_fail}")
                assert gcs_uri_fail is None
                mock_db_for_analytics.collection.return_value.add.assert_called_once()
                args, kwargs = mock_db_for_analytics.collection.return_value.add.call_args
                logged_data = args[0]
                assert logged_data["event_type"] == "cloud_storage_operation"
                assert logged_data["details"]["operation"] == "upload"
                assert logged_data["success"] is False
                assert "Mock GCS Init Error" in logged_data["error_message"]
                print("Test 5 Passed (and analytics logged failure).")
            _gcs_bucket = original_gcs_bucket # Restore bucket

            # Clean up dummy file
            os.remove(test_file_path)
            if os.path.exists(test_download_path):
                os.remove(test_download_path)

            print("\nAll GCS utility tests completed.")

        asyncio.run(run_gcs_tests())
