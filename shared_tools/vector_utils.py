# shared_tools/vector_utils.py

import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, GooglePalmEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI, GooglePalm
from langchain_community.chat_models import ChatOpenAI, ChatGooglePalm

# Import config_manager to get configurations
from config.config_manager import config_manager
# Import analytics_tracker for logging events
from utils.analytics_tracker import log_event, initialize_analytics
# Import GCS utilities
from shared_tools.cloud_storage_utils import upload_file_to_gcs, download_file_from_gcs, read_file_from_gcs_to_bytes, delete_file_from_gcs

# Import Firebase Admin SDK components for backend initialization (if needed for context)
import firebase_admin
from firebase_admin import credentials, auth, firestore
import json

logger = logging.getLogger(__name__)

# --- Firebase Admin SDK Initialization (for analytics context) ---
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
        logger.info("Firebase Admin SDK initialized successfully in vector_utils.")
    except Exception as e:
        logger.error(f"Error initializing Firebase Admin SDK in vector_utils: {e}")

# Initialize analytics_tracker for backend context
if 'analytics_initialized_backend' not in globals(): # Use globals() for module-level check
    if firebase_admin._apps:
        try:
            db_instance = firestore.client()
            auth_instance = auth
            app_id_for_analytics = config_manager.get("app_id", firebase_config.get("projectId", "default-streamlit-app-id"))
            initialize_analytics(db_instance, auth_instance, app_id_for_analytics, "backend_system_user")
            globals()['analytics_initialized_backend'] = True
            logger.info("Analytics tracker initialized for vector_utils with live Firebase.")
        except Exception as e:
            logger.error(f"Failed to initialize analytics with live Firebase Admin SDK in vector_utils: {e}")
            mock_db = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: None})()})()
            mock_auth = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': None})()})()
            app_id_for_analytics = config_manager.get("app_id", "default-streamlit-app-id")
            initialize_analytics(mock_db, mock_auth, app_id_for_analytics, "backend_system_user")
            globals()['analytics_initialized_backend'] = True
            logger.warning("Analytics tracker initialized with mock Firebase for vector_utils.")
    else:
        mock_db = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: None})()})()
        mock_auth = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': None})()})()
        app_id_for_analytics = config_manager.get("app_id", "default-streamlit-app-id")
        initialize_analytics(mock_db, mock_auth, app_id_for_analytics, "backend_system_user")
        globals()['analytics_initialized_backend'] = True
        logger.warning("Analytics tracker initialized with mock Firebase for vector_utils (Admin SDK not available).")


# Base directory for storing ChromaDB collections and temporary uploaded files
BASE_VECTOR_DIR = Path("vector_db_data")
# Temporary local directory for uploaded files before/after GCS operations
TEMP_UPLOAD_DIR = Path("temp_uploads") 

# Ensure temporary upload directory exists
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_embedding_model(provider: str = "openai", model_name: Optional[str] = None):
    """
    Returns an embedding model based on the specified provider.
    """
    if provider == "openai":
        openai_api_key = config_manager.get_secret("openai_api_key")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found in secrets.")
        return OpenAIEmbeddings(openai_api_key=openai_api_key, model=model_name or "text-embedding-ada-002")
    elif provider == "google":
        google_api_key = config_manager.get_secret("google_api_key")
        if not google_api_key:
            raise ValueError("Google API key not found in secrets.")
        return GooglePalmEmbeddings(google_api_key=google_api_key, model_name=model_name or "models/embedding-001")
    elif provider == "huggingface":
        # Using a common sentence-transformer model for demonstration
        return HuggingFaceEmbeddings(model_name=model_name or "all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

async def load_documents_from_gcs(
    gcs_blob_path: str,
    user_id: str,
    temp_local_path: Optional[Path] = None
) -> List[Any]: # Returns List[Document] from langchain
    """
    Downloads a document from GCS, loads it using Langchain loaders, and returns.
    The downloaded file is temporarily stored and then removed.

    Args:
        gcs_blob_path (str): The full path to the blob in GCS (e.g., 'user_uploads/user123/document.pdf').
        user_id (str): The ID of the user for analytics logging.
        temp_local_path (Path, optional): Specific local path to download to. Defaults to TEMP_UPLOAD_DIR.

    Returns:
        List[Document]: A list of Langchain Document objects.
    """
    if temp_local_path is None:
        # Create a unique temporary local path for the downloaded file
        local_file_name = Path(gcs_blob_path).name
        temp_local_path = TEMP_UPLOAD_DIR / user_id / local_file_name
    
    temp_local_path.parent.mkdir(parents=True, exist_ok=True) # Ensure user-specific temp dir exists

    logger.info(f"Attempting to download {gcs_blob_path} to {temp_local_path}")
    download_success = await download_file_from_gcs(gcs_blob_path, str(temp_local_path), user_id=user_id)

    if not download_success:
        logger.error(f"Failed to download document from GCS: {gcs_blob_path}")
        await log_event('document_processing', {
            'operation': 'load_from_gcs',
            'file_path': gcs_blob_path,
            'status': 'failure',
            'reason': 'download_failed'
        }, user_id=user_id, success=False, error_message=f"Failed to download {gcs_blob_path} from GCS.")
        return []

    try:
        file_extension = temp_local_path.suffix.lower()
        loader = None
        if file_extension == ".pdf":
            loader = PyPDFLoader(str(temp_local_path))
        elif file_extension in [".docx", ".doc"]:
            loader = Docx2txtLoader(str(temp_local_path))
        elif file_extension == ".txt":
            loader = TextLoader(str(temp_local_path))
        else:
            raise ValueError(f"Unsupported file type for RAG: {file_extension}")

        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} pages/chunks from {temp_local_path}")
        await log_event('document_processing', {
            'operation': 'load_from_gcs',
            'file_path': gcs_blob_path,
            'status': 'success',
            'num_documents': len(documents)
        }, user_id=user_id, success=True)
        return documents
    except Exception as e:
        logger.error(f"Error loading document {temp_local_path} for RAG: {e}", exc_info=True)
        await log_event('document_processing', {
            'operation': 'load_from_gcs',
            'file_path': gcs_blob_path,
            'status': 'failure',
            'reason': 'document_load_error',
            'error_message': str(e)
        }, user_id=user_id, success=False, error_message=str(e))
        return []
    finally:
        # Clean up the temporary local file
        if temp_local_path.exists():
            os.remove(temp_local_path)
            logger.debug(f"Cleaned up temporary file: {temp_local_path}")


def split_documents(documents: List[Any], chunk_size: int, chunk_overlap: int) -> List[Any]:
    """
    Splits a list of Langchain Documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks.")
    return chunks

async def create_and_store_embeddings(
    documents: List[Any],
    collection_name: str,
    user_id: str,
    embedding_provider: str = "openai",
    embedding_model_name: Optional[str] = None
) -> Chroma:
    """
    Creates embeddings for documents and stores them in a ChromaDB vector store.
    """
    try:
        embeddings = get_embedding_model(embedding_provider, embedding_model_name)
        
        # ChromaDB persists to disk. The persist_directory should be unique per user/collection.
        persist_directory = BASE_VECTOR_DIR / user_id / collection_name
        persist_directory.mkdir(parents=True, exist_ok=True)
        
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(persist_directory)
        )
        vector_store.persist()
        logger.info(f"Embeddings created and stored in ChromaDB at {persist_directory} for collection '{collection_name}'.")
        await log_event('vector_db_operation', {
            'operation': 'create_and_store_embeddings',
            'collection_name': collection_name,
            'num_documents': len(documents),
            'embedding_provider': embedding_provider,
            'status': 'success'
        }, user_id=user_id, success=True)
        return vector_store
    except Exception as e:
        logger.error(f"Error creating and storing embeddings for collection '{collection_name}': {e}", exc_info=True)
        await log_event('vector_db_operation', {
            'operation': 'create_and_store_embeddings',
            'collection_name': collection_name,
            'num_documents': len(documents),
            'embedding_provider': embedding_provider,
            'status': 'failure',
            'error_message': str(e)
        }, user_id=user_id, success=False, error_message=str(e))
        raise

async def get_vector_store(
    collection_name: str,
    user_id: str,
    embedding_provider: str = "openai",
    embedding_model_name: Optional[str] = None
) -> Optional[Chroma]:
    """
    Retrieves an existing ChromaDB vector store.
    """
    try:
        embeddings = get_embedding_model(embedding_provider, embedding_model_name)
        persist_directory = BASE_VECTOR_DIR / user_id / collection_name
        
        if not persist_directory.exists():
            logger.warning(f"Vector store for collection '{collection_name}' at {persist_directory} does not exist.")
            await log_event('vector_db_operation', {
                'operation': 'get_vector_store',
                'collection_name': collection_name,
                'status': 'failure',
                'reason': 'not_found'
            }, user_id=user_id, success=False, error_message="Vector store not found.")
            return None
            
        vector_store = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embeddings
        )
        logger.info(f"Retrieved ChromaDB vector store from {persist_directory} for collection '{collection_name}'.")
        await log_event('vector_db_operation', {
            'operation': 'get_vector_store',
            'collection_name': collection_name,
            'status': 'success'
        }, user_id=user_id, success=True)
        return vector_store
    except Exception as e:
        logger.error(f"Error retrieving vector store for collection '{collection_name}': {e}", exc_info=True)
        await log_event('vector_db_operation', {
            'operation': 'get_vector_store',
            'collection_name': collection_name,
            'status': 'failure',
            'error_message': str(e)
        }, user_id=user_id, success=False, error_message=str(e))
        raise

async def query_vector_store(
    vector_store: Chroma,
    query_text: str,
    user_id: str,
    k: int = 4
) -> List[Dict[str, Any]]:
    """
    Queries the vector store for relevant documents.
    """
    try:
        docs = vector_store.similarity_search(query_text, k=k)
        results = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        logger.info(f"Queried vector store for '{query_text}', found {len(results)} results.")
        await log_event('vector_db_operation', {
            'operation': 'query_vector_store',
            'query_text': query_text,
            'num_results': len(results),
            'status': 'success'
        }, user_id=user_id, success=True)
        return results
    except Exception as e:
        logger.error(f"Error querying vector store for '{query_text}': {e}", exc_info=True)
        await log_event('vector_db_operation', {
            'operation': 'query_vector_store',
            'query_text': query_text,
            'status': 'failure',
            'error_message': str(e)
        }, user_id=user_id, success=False, error_message=str(e))
        raise

async def delete_vector_store_collection(
    collection_name: str,
    user_id: str
) -> bool:
    """
    Deletes a specific ChromaDB collection for a user.
    """
    try:
        persist_directory = BASE_VECTOR_DIR / user_id / collection_name
        if persist_directory.exists():
            shutil.rmtree(persist_directory)
            logger.info(f"ChromaDB collection '{collection_name}' for user '{user_id}' deleted from {persist_directory}.")
            await log_event('vector_db_operation', {
                'operation': 'delete_collection',
                'collection_name': collection_name,
                'status': 'success'
            }, user_id=user_id, success=True)
            return True
        else:
            logger.warning(f"Attempted to delete non-existent ChromaDB collection '{collection_name}' for user '{user_id}'.")
            await log_event('vector_db_operation', {
                'operation': 'delete_collection',
                'collection_name': collection_name,
                'status': 'failure',
                'reason': 'not_found'
            }, user_id=user_id, success=False, error_message="Collection not found for deletion.")
            return False
    except Exception as e:
        logger.error(f"Error deleting ChromaDB collection '{collection_name}' for user '{user_id}': {e}", exc_info=True)
        await log_event('vector_db_operation', {
            'operation': 'delete_collection',
            'collection_name': collection_name,
            'status': 'failure',
            'error_message': str(e)
        }, user_id=user_id, success=False, error_message=str(e))
        return False


# CLI Test (optional)
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch, AsyncMock
    import tempfile

    logging.basicConfig(level=logging.INFO)

    # Mock config_manager and st.secrets for local testing
    class MockSecrets:
        def __init__(self):
            self.openai_api_key = "sk-mock-openai-key"
            self.google_api_key = "AIzaSy-mock-google-key"
            self.firebase_config = json.dumps({"projectId": "mock-project-id"})

        def get(self, key, default=None):
            return getattr(self, key, default)
    
    class MockConfigManager:
        _instance = None
        _is_loaded = False
        def __init__(self):
            if MockConfigManager._instance is None:
                MockConfigManager._instance = self
            self._config_data = {
                'rag': {
                    'chunk_size': 500,
                    'chunk_overlap': 50,
                    'embedding_provider': 'openai',
                    'embedding_model_name': 'text-embedding-ada-002',
                    'max_query_results_k': 4
                },
                'app_id': 'test-app-id-cli',
                'analytics': {
                    'enabled': True,
                    'log_tool_usage': True,
                    'log_query_failures': True
                },
                'cloud_storage': { # Mock GCS config for cloud_storage_utils
                    'gcs': {
                        'bucket_name': 'mock-gcs-bucket',
                        'credentials_path': 'mock-credentials.json'
                    }
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
            pass
        
        def get_api_provider_config(self, domain: str, provider_name: str) -> Optional[Dict[str, Any]]:
            return None

        def get_domain_api_providers(self, domain: str) -> Dict[str, Any]:
            return {}

    # Patch config_manager
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager

    # Mock firebase_admin for analytics initialization
    mock_db_for_analytics = MagicMock()
    mock_auth_for_analytics = MagicMock()
    mock_auth_for_analytics.currentUser = MagicMock(uid="test_cli_user")
    mock_db_for_analytics.collection.return_value.add = AsyncMock(return_value=MagicMock(id="mock_doc_id"))

    with patch.dict(sys.modules, {'firebase_admin.firestore': MagicMock(firestore=MagicMock())}):
        sys.modules['firebase_admin.firestore'].firestore.CollectionReference = MagicMock()
        sys.modules['firebase_admin.firestore'].firestore.DocumentReference = MagicMock()
        initialize_analytics(
            mock_db_for_analytics,
            mock_auth_for_analytics,
            "test-app-id-cli",
            "test_cli_user"
        )
        globals()['analytics_initialized_backend'] = True

    # Mock GCS utilities
    mock_upload_file_to_gcs = AsyncMock(return_value="gs://mock-bucket/mock-path/test.pdf")
    mock_download_file_from_gcs = AsyncMock(return_value=True)
    mock_read_file_from_gcs_to_bytes = AsyncMock(return_value=b"Mock PDF content")
    mock_delete_file_from_gcs = AsyncMock(return_value=True)

    # Patch cloud_storage_utils functions
    with patch('shared_tools.cloud_storage_utils.upload_file_to_gcs', new=mock_upload_file_to_gcs), \
         patch('shared_tools.cloud_storage_utils.download_file_from_gcs', new=mock_download_file_from_gcs), \
         patch('shared_tools.cloud_storage_utils.read_file_from_gcs_to_bytes', new=mock_read_file_from_gcs_to_bytes), \
         patch('shared_tools.cloud_storage_utils.delete_file_from_gcs', new=mock_delete_file_from_gcs):

        # Mock Langchain loaders to return dummy documents
        mock_pdf_loader = MagicMock()
        mock_pdf_loader.load.return_value = [
            MagicMock(page_content="Mock PDF content page 1.", metadata={"source": "mock.pdf", "page": 0}),
            MagicMock(page_content="Mock PDF content page 2.", metadata={"source": "mock.pdf", "page": 1})
        ]
        with patch('langchain_community.document_loaders.PyPDFLoader', return_value=mock_pdf_loader):
            # Mock ChromaDB for vector store operations
            mock_chroma_instance = MagicMock()
            mock_chroma_instance.persist = MagicMock()
            mock_chroma_instance.similarity_search.return_value = [
                MagicMock(page_content="Relevant chunk 1.", metadata={"source": "doc1.pdf"}),
                MagicMock(page_content="Relevant chunk 2.", metadata={"source": "doc2.txt"})
            ]

            with patch('langchain_community.vectorstores.Chroma.from_documents', return_value=mock_chroma_instance) as mock_chroma_from_documents, \
                 patch('langchain_community.vectorstores.Chroma', return_value=mock_chroma_instance) as mock_chroma_constructor, \
                 patch('langchain_community.embeddings.OpenAIEmbeddings') as MockOpenAIEmbeddings, \
                 patch('langchain_community.embeddings.GooglePalmEmbeddings') as MockGooglePalmEmbeddings, \
                 patch('langchain_community.embeddings.HuggingFaceEmbeddings') as MockHuggingFaceEmbeddings:

                MockOpenAIEmbeddings.return_value = MagicMock() # Mock the embedding model instance
                MockGooglePalmEmbeddings.return_value = MagicMock()
                MockHuggingFaceEmbeddings.return_value = MagicMock()

                async def run_vector_tests():
                    print("\n--- Testing Vector Utility Functions with GCS Integration ---")
                    test_user_id = "test_user_456"
                    test_collection_name = "test_medical_docs"
                    test_gcs_blob_path = "user_uploads/test_user_456/sample.pdf"
                    test_local_temp_path = TEMP_UPLOAD_DIR / test_user_id / "sample.pdf"
                    
                    # Ensure temporary directories are clean
                    if TEMP_UPLOAD_DIR.exists():
                        shutil.rmtree(TEMP_UPLOAD_DIR)
                    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                    
                    # Create dummy local file for testing the path construction
                    test_local_temp_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(test_local_temp_path, "w") as f:
                        f.write("dummy content")


                    # Test load_documents_from_gcs
                    print("\n--- Test 1: load_documents_from_gcs (Success) ---")
                    mock_analytics_tracker_db.collection.return_value.add.reset_mock()
                    documents = await load_documents_from_gcs(test_gcs_blob_path, test_user_id, temp_local_path=test_local_temp_path)
                    print(f"Loaded {len(documents)} documents.")
                    assert len(documents) == 2
                    mock_download_file_from_gcs.assert_called_once_with(test_gcs_blob_path, str(test_local_temp_path), user_id=test_user_id)
                    mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
                    args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
                    logged_data = args[0]
                    assert logged_data["event_type"] == "document_processing"
                    assert logged_data["details"]["operation"] == "load_from_gcs"
                    assert logged_data["success"] is True
                    print("Test 1 Passed (and analytics logged success).")
                    mock_download_file_from_gcs.reset_mock() # Reset for next test

                    # Test create_and_store_embeddings
                    print("\n--- Test 2: create_and_store_embeddings (Success) ---")
                    mock_analytics_tracker_db.collection.return_value.add.reset_mock()
                    vector_store = await create_and_store_embeddings(documents, test_collection_name, test_user_id)
                    assert vector_store is not None
                    mock_chroma_from_documents.assert_called_once()
                    mock_chroma_instance.persist.assert_called_once()
                    mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
                    args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
                    logged_data = args[0]
                    assert logged_data["event_type"] == "vector_db_operation"
                    assert logged_data["details"]["operation"] == "create_and_store_embeddings"
                    assert logged_data["success"] is True
                    print("Test 2 Passed (and analytics logged success).")

                    # Test get_vector_store
                    print("\n--- Test 3: get_vector_store (Success) ---")
                    mock_analytics_tracker_db.collection.return_value.add.reset_mock()
                    retrieved_vector_store = await get_vector_store(test_collection_name, test_user_id)
                    assert retrieved_vector_store is not None
                    mock_chroma_constructor.assert_called_with(
                        persist_directory=str(BASE_VECTOR_DIR / test_user_id / test_collection_name),
                        embedding_function=MockOpenAIEmbeddings.return_value
                    )
                    mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
                    args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
                    logged_data = args[0]
                    assert logged_data["event_type"] == "vector_db_operation"
                    assert logged_data["details"]["operation"] == "get_vector_store"
                    assert logged_data["success"] is True
                    print("Test 3 Passed (and analytics logged success).")

                    # Test query_vector_store
                    print("\n--- Test 4: query_vector_store (Success) ---")
                    mock_analytics_tracker_db.collection.return_value.add.reset_mock()
                    query_results = await query_vector_store(vector_store, "What is this document about?", test_user_id)
                    print(f"Query Results: {query_results}")
                    assert len(query_results) == 2
                    assert "Relevant chunk 1." in query_results[0]["page_content"]
                    mock_chroma_instance.similarity_search.assert_called_once_with("What is this document about?", k=4)
                    mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
                    args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
                    logged_data = args[0]
                    assert logged_data["event_type"] == "vector_db_operation"
                    assert logged_data["details"]["operation"] == "query_vector_store"
                    assert logged_data["success"] is True
                    print("Test 4 Passed (and analytics logged success).")

                    # Test delete_vector_store_collection
                    print("\n--- Test 5: delete_vector_store_collection (Success) ---")
                    mock_analytics_tracker_db.collection.return_value.add.reset_mock()
                    delete_success = await delete_vector_store_collection(test_collection_name, test_user_id)
                    print(f"Delete Collection Success: {delete_success}")
                    assert delete_success is True
                    # Verify that the directory was attempted to be removed (mocking shutil.rmtree)
                    # For actual test, you'd check if the directory is gone.
                    mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
                    args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
                    logged_data = args[0]
                    assert logged_data["event_type"] == "vector_db_operation"
                    assert logged_data["details"]["operation"] == "delete_collection"
                    assert logged_data["success"] is True
                    print("Test 5 Passed (and analytics logged success).")

                    # Test load_documents_from_gcs (Failure - download fails)
                    print("\n--- Test 6: load_documents_from_gcs (Failure - Download) ---")
                    mock_download_file_from_gcs.return_value = False # Simulate download failure
                    mock_analytics_tracker_db.collection.return_value.add.reset_mock()
                    documents_fail = await load_documents_from_gcs("nonexistent/file.pdf", test_user_id)
                    print(f"Loaded documents (failure): {len(documents_fail)}")
                    assert len(documents_fail) == 0
                    mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
                    args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
                    logged_data = args[0]
                    assert logged_data["event_type"] == "document_processing"
                    assert logged_data["details"]["operation"] == "load_from_gcs"
                    assert logged_data["success"] is False
                    assert "download_failed" in logged_data["details"]["reason"]
                    print("Test 6 Passed (and analytics logged failure).")
                    mock_download_file_from_gcs.return_value = True # Reset mock

                    # Clean up temporary files and directories
                    if TEMP_UPLOAD_DIR.exists():
                        shutil.rmtree(TEMP_UPLOAD_DIR)
                    if BASE_VECTOR_DIR.exists():
                        shutil.rmtree(BASE_VECTOR_DIR)
                    print("Cleaned up temporary directories.")

                asyncio.run(run_vector_tests())
