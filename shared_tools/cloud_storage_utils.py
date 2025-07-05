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
# Import analytics_tracker for logging events
from utils.analytics_tracker import log_event, initialize_analytics

logger = logging.getLogger(__name__)

_gcs_client = None
_gcs_bucket_name = None
_gcs_bucket = None # GCS Bucket object

# --- Firebase Admin SDK Initialization (for analytics context) ---
# This block is duplicated for each backend file to ensure analytics_tracker is initialized
# even if a file is run standalone or imported in a specific order.
import firebase_admin
from firebase_admin import credentials, auth, firestore

if not firebase_admin._apps:
    try:
        firebase_config_str = config_manager.get_secret("firebase_config")
        if not firebase_config_str:
            raise ValueError("Firebase configuration not found in secrets.")
        
        firebase_config = json.loads(firebase_config_str)
        
        if os.environ.get("FIREBASE_ADMIN_CERT"):
            cred = credentials.Certificate(json.loads(os.environ.get("FIREBASE_ADMIN_CERT")))
        else:
            logger.warning("FIREBASE_ADMIN_CERT environment variable not found. Firebase Admin SDK functionality may be limited.")
            cred = credentials.Certificate({
                "type": "service_account",
                "project_id": firebase_config.get("projectId", "mock-project-id"),
                "private_key_id": "mock-key-id",
                "private_key": "-----BEGIN PRIVATE KEY-----\nMOCK_PRIVATE_KEY\n-----END PRIVATE KEY-----\n",
                "client_email": "mock-client@mock-project-id.iam.gserviceaccount.com",
                "client_id": "mock-client-id",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/mock-client%40{firebase_config.get('projectId', 'mock-project-id')}.iam.gserviceaccount.com",
                "universe_domain": "googleapis.com"
            })

        firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully in cloud_storage_utils.")
    except Exception as e:
        logger.error(f"Error initializing Firebase Admin SDK in cloud_storage_utils: {e}")

# Initialize analytics_tracker for backend context
if 'analytics_initialized_backend' not in globals(): # Use globals() for module-level check
    if firebase_admin._apps:
        try:
            db_instance = firestore.client()
            auth_instance = auth
            app_id_for_analytics = config_manager.get("app_id", firebase_config.get("projectId", "default-streamlit-app-id"))
            initialize_analytics(db_instance, auth_instance, app_id_for_analytics, "backend_system_user")
            globals()['analytics_initialized_backend'] = True
            logger.info("Analytics tracker initialized for cloud_storage_utils with live Firebase.")
        except Exception as e:
            logger.error(f"Failed to initialize analytics with live Firebase Admin SDK in cloud_storage_utils: {e}")
            mock_db = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: None})()})()
            mock_auth = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': None})()})()
            app_id_for_analytics = config_manager.get("app_id", "default-streamlit-app-id")
            initialize_analytics(mock_db, mock_auth, app_id_for_analytics, "backend_system_user")
            globals()['analytics_initialized_backend'] = True
            logger.warning("Analytics tracker initialized with mock Firebase for cloud_storage_utils.")
    else:
        mock_db = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: None})()})()
        mock_auth = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': None})()})()
        app_id_for_analytics = config_manager.get("app_id", "default-streamlit-app-id")
        initialize_analytics(mock_db, mock_auth, app_id_for_analytics, "backend_system_user")
        globals()['analytics_initialized_backend'] = True
        logger.warning("Analytics tracker initialized with mock Firebase for cloud_storage_utils (Admin SDK not available).")


def _initialize_gcs_client():
    """Initializes the Google Cloud Storage client."""
    global _gcs_client, _gcs_bucket_name, _gcs_bucket
    if _gcs_client is None:
        try:
            gcs_config = config_manager.get("cloud_storage.gcs")
            if not gcs_config:
                raise ValueError("Google Cloud Storage configuration not found in config.yml.")

            _gcs_bucket_name = gcs_config.get("bucket_name")
            credentials_path = gcs_config.get("credentials_path")

            if not _gcs_bucket_name:
                raise ValueError("GCS bucket_name not specified in config.yml.")

            if credentials_path and os.path.exists(credentials_path):
                credentials_obj = service_account.Credentials.from_service_account_file(credentials_path)
                _gcs_client = storage.Client(credentials=credentials_obj)
                logger.info(f"GCS client initialized using credentials from {credentials_path}")
            elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                _gcs_client = storage.Client() # Auto-detects from GOOGLE_APPLICATION_CREDENTIALS env var
                logger.info("GCS client initialized using GOOGLE_APPLICATION_CREDENTIALS env var.")
            else:
                _gcs_client = storage.Client() # Attempts to use default credentials (e.g., from GCE metadata)
                logger.warning("GCS client initialized without explicit credentials path or GOOGLE_APPLICATION_CREDENTIALS. Relying on default application credentials.")
            
            _gcs_bucket = _gcs_client.bucket(_gcs_bucket_name)
            logger.info(f"GCS bucket '{_gcs_bucket_name}' selected.")

        except Exception as e:
            logger.error(f"Error initializing GCS client: {e}")
            _gcs_client = None # Ensure client is None on failure
            raise

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

# CLI Test (optional)
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch

    logging.basicConfig(level=logging.INFO)

    # Mock config_manager and st.secrets for local testing
    class MockSecrets:
        def __init__(self):
            self.firebase_config = json.dumps({"projectId": "mock-project-id"})
            # Add other secrets if needed for full config_manager mock

        def get(self, key, default=None):
            return getattr(self, key, default)
    
    class MockConfigManager:
        _instance = None
        _is_loaded = False
        def __init__(self):
            if MockConfigManager._instance is None:
                MockConfigManager._instance = self
            self._config_data = {
                'cloud_storage': {
                    'gcs': {
                        'bucket_name': 'mock-gcs-bucket',
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
            self._secrets_mock = MockSecrets()
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
            return self._secrets_mock.get(key, default)

        def set_secret(self, key, value):
            pass # No-op for mock
        
        def get_api_provider_config(self, domain: str, provider_name: str) -> Optional[Dict[str, Any]]:
            return None # Not relevant for this module

        def get_domain_api_providers(self, domain: str) -> Dict[str, Any]:
            return {}

    # Patch the actual imports for testing
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager

    # Mock firebase_admin for analytics initialization
    mock_db_for_analytics = MagicMock()
    mock_auth_for_analytics = MagicMock()
    mock_auth_for_analytics.currentUser = MagicMock(uid="test_cli_user")
    mock_db_for_analytics.collection.return_value.add = AsyncMock(return_value=MagicMock(id="mock_doc_id"))

    # Patch firebase_admin.firestore for the local import within log_event
    with patch.dict(sys.modules, {'firebase_admin.firestore': MagicMock(firestore=MagicMock())}):
        sys.modules['firebase_admin.firestore'].firestore.CollectionReference = MagicMock()
        sys.modules['firebase_admin.firestore'].firestore.DocumentReference = MagicMock()
        # Re-initialize analytics with mocks
        initialize_analytics(
            mock_db_for_analytics,
            mock_auth_for_analytics,
            "test-app-id-cli",
            "test_cli_user"
        )
        globals()['analytics_initialized_backend'] = True # Set global flag for test


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
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            gcs_uri = await upload_file_to_gcs(test_file_path, test_blob_name, user_id=test_user_id)
            print(f"Uploaded GCS URI: {gcs_uri}")
            assert gcs_uri is not None
            assert f"gs://{MockConfigManager().get('cloud_storage.gcs.bucket_name')}/{test_blob_name}" == gcs_uri
            mock_blob.upload_from_filename.assert_called_once_with(test_file_path)
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "cloud_storage_operation"
            assert logged_data["details"]["operation"] == "upload"
            assert logged_data["success"] is True
            print("Test 1 Passed (and analytics logged success).")

            # Test download_file_from_gcs
            print("\n--- Test 2: download_file_from_gcs (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            download_success = await download_file_from_gcs(test_blob_name, test_download_path, user_id=test_user_id)
            print(f"Download Success: {download_success}")
            assert download_success is True
            mock_blob.download_to_filename.assert_called_once_with(test_download_path)
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "cloud_storage_operation"
            assert logged_data["details"]["operation"] == "download"
            assert logged_data["success"] is True
            print("Test 2 Passed (and analytics logged success).")

            # Test read_file_from_gcs_to_bytes
            print("\n--- Test 3: read_file_from_gcs_to_bytes (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            file_contents = await read_file_from_gcs_to_bytes(test_blob_name, user_id=test_user_id)
            print(f"File Contents (bytes): {file_contents}")
            assert file_contents == b"mock file content"
            mock_blob.download_as_bytes.assert_called_once()
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "cloud_storage_operation"
            assert logged_data["details"]["operation"] == "read_bytes"
            assert logged_data["success"] is True
            print("Test 3 Passed (and analytics logged success).")

            # Test delete_file_from_gcs
            print("\n--- Test 4: delete_file_from_gcs (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            delete_success = await delete_file_from_gcs(test_blob_name, user_id=test_user_id)
            print(f"Delete Success: {delete_success}")
            assert delete_success is True
            mock_blob.delete.assert_called_once()
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
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
                mock_analytics_tracker_db.collection.return_value.add.reset_mock()
                gcs_uri_fail = await upload_file_to_gcs(test_file_path, "fail_blob.txt", user_id=test_user_id)
                print(f"Upload Fail URI: {gcs_uri_fail}")
                assert gcs_uri_fail is None
                mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
                args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
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
