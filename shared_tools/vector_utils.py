# shared_tools/vector_utils.py

import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
import base64 # For decoding base64 file content

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,        # For .csv files
    UnstructuredExcelLoader, # For .xls, .xlsx files
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, GooglePalmEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI, GooglePalm
from langchain_community.chat_models import ChatOpenAI, ChatGooglePalm

# Import config_manager to get configurations
from config.config_manager import config_manager
# Import analytics_tracker for logging events - it will use the already initialized Firebase
from utils.analytics_tracker import log_event # Removed initialize_analytics as it's done in main.py
# Import GCS utilities - these are now functions, not a class
from shared_tools.cloud_storage_utils import upload_file_to_gcs, download_file_from_gcs, read_file_from_gcs_to_bytes, delete_file_from_gcs

# Removed Firebase Admin SDK imports as it's initialized in main.py
# import firebase_admin
# from firebase_admin import credentials, auth, firestore
# import json

logger = logging.getLogger(__name__)

# --- REMOVED: Firebase Admin SDK Initialization block ---
# Firebase Admin SDK and analytics initialization should happen ONLY in backend/main.py
# This module will rely on Firebase being initialized there.


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

async def process_uploaded_document(
    file_name: str,
    file_content_base64: str,
    content_type: str,
    user_id: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    embedding_provider: Optional[str] = None,
    embedding_model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Handles the entire document processing pipeline:
    1. Decodes base64 content and saves temporarily.
    2. Uploads to GCS.
    3. Loads document, splits into chunks.
    4. Creates/updates vector store.
    5. Deletes temporary local file.
    """
    local_file_path = TEMP_UPLOAD_DIR / user_id / file_name
    local_file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Decode base64 content and save to a temporary local file
        file_content_bytes = base64.b64decode(file_content_base64)
        with open(local_file_path, "wb") as f:
            f.write(file_content_bytes)
        logger.info(f"Decoded and saved temporary file: {local_file_path}")

        # Upload to GCS
        gcs_blob_name = f"user_documents/{user_id}/{file_name}"
        gcs_uri = await upload_file_to_gcs(str(local_file_path), gcs_blob_name, user_id=user_id)

        if not gcs_uri:
            raise Exception("Failed to upload document to GCS.")

        # Load document from local path (after upload)
        documents = []
        file_extension = Path(file_name).suffix.lower()
        loader = None
        if file_extension == ".pdf":
            loader = PyPDFLoader(str(local_file_path))
        elif file_extension in [".docx", ".doc"]:
            loader = Docx2txtLoader(str(local_file_path))
        elif file_extension == ".txt":
            loader = TextLoader(str(local_file_path))
        elif file_extension == ".csv":
            loader = CSVLoader(str(local_file_path))
        elif file_extension in [".xls", ".xlsx"]:
            loader = UnstructuredExcelLoader(str(local_file_path))
        else:
            raise ValueError(f"Unsupported file type for RAG: {file_extension}")

        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} pages/chunks from {file_name}")

        # Get RAG configuration from config_manager
        rag_config = config_manager.get("rag", {})
        
        # Use provided chunk_size/overlap or fallback to config
        effective_chunk_size = chunk_size or rag_config.get("chunk_size", 1000)
        effective_chunk_overlap = chunk_overlap or rag_config.get("chunk_overlap", 100)
        effective_embedding_provider = embedding_provider or rag_config.get("embedding_provider", "openai")
        effective_embedding_model_name = embedding_model_name or rag_config.get("embedding_model_name")

        # Split documents
        chunks = split_documents(documents, effective_chunk_size, effective_chunk_overlap)
        
        # Create/update vector store
        # Use a collection name based on user_id and file_name for uniqueness
        collection_name = f"{user_id}_{Path(file_name).stem.replace('.', '_')}"
        vector_store = await create_and_store_embeddings(
            chunks,
            collection_name,
            user_id,
            embedding_provider=effective_embedding_provider,
            embedding_model_name=effective_embedding_model_name
        )

        if vector_store:
            await log_event('document_processing', {
                'operation': 'full_pipeline',
                'file_name': file_name,
                'gcs_uri': gcs_uri,
                'status': 'success',
                'num_chunks': len(chunks),
                'collection_name': collection_name
            }, user_id=user_id, success=True)
            return {"success": True, "message": "Document processed and indexed successfully.", "collection_name": collection_name, "gcs_uri": gcs_uri}
        else:
            raise Exception("Failed to create/update vector store.")

    except Exception as e:
        logger.error(f"Error processing document {file_name} for user {user_id}: {e}", exc_info=True)
        await log_event('document_processing', {
            'operation': 'full_pipeline',
            'file_name': file_name,
            'status': 'failure',
            'error_message': str(e)
        }, user_id=user_id, success=False, error_message=str(e))
        return {"success": False, "message": f"Failed to process document: {e}"}
    finally:
        # Clean up the temporary local file
        if local_file_path.exists():
            os.remove(local_file_path)
            logger.debug(f"Cleaned up temporary file: {local_file_path}")


async def query_documents(
    query_text: str,
    user_id: str,
    collection_name: Optional[str] = None, # Optional: query a specific collection
    k: int = 4,
    embedding_provider: Optional[str] = None,
    embedding_model_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Queries documents from a user's vector store(s).
    If collection_name is None, it might query all available collections for the user (more complex).
    For simplicity, we'll assume a default or infer a collection if not provided.
    """
    try:
        # For a single document, the collection name might be derived from the document ID
        # For multiple documents, you might have a "default" collection or need to list them.
        # For now, let's assume `collection_name` is provided or a default is used.
        # If no collection_name is provided, we might need a way to list all user collections.
        # For this example, let's assume a default or a specific collection is targeted.
        
        # If collection_name is not provided, we need a strategy.
        # One strategy: query a "default" collection or the most recently added one.
        # Another: iterate through all collections for the user (more complex, requires Firestore query for collections).
        
        # For now, let's enforce collection_name for simplicity, or use a placeholder.
        # In a real app, you'd track which documents belong to which collections in Firestore.
        if not collection_name:
            # This is a placeholder. In a real app, you'd fetch the user's active/default collection.
            # For demonstration, let's assume a generic collection or raise an error.
            # For now, let's use a generic collection name if not provided, but it's less robust.
            # A better approach would be to store collection names associated with user in Firestore.
            logger.warning(f"No collection_name provided for query. Using a default/generic approach. Consider storing user collections in Firestore.")
            # This will likely fail if no such generic collection exists.
            # Raising an error is safer for now if specific collection is expected.
            raise ValueError("A specific 'collection_name' is required for querying documents.")
            
        vector_store = await get_vector_store(
            collection_name,
            user_id,
            embedding_provider=embedding_provider,
            embedding_model_name=embedding_model_name
        )

        if not vector_store:
            logger.warning(f"Vector store for collection '{collection_name}' not found for user {user_id}. Cannot query.")
            return []

        results = await query_vector_store(vector_store, query_text, user_id, k=k)
        return results

    except Exception as e:
        logger.error(f"Error querying documents for user {user_id} in collection '{collection_name}': {e}", exc_info=True)
        await log_event('document_query', {
            'query_text': query_text,
            'collection_name': collection_name,
            'status': 'failure',
            'error_message': str(e)
        }, user_id=user_id, success=False, error_message=str(e))
        raise # Re-raise the exception for FastAPI to handle


# Utility functions (split_documents, create_and_store_embeddings, get_vector_store, query_vector_store, delete_vector_store_collection)
# These are already defined as module-level functions in your provided code.
# No changes needed to their definitions here.

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


# CLI Test (optional) - Ensure mocks are updated to reflect removal of Firebase init
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch, AsyncMock # Ensure AsyncMock is imported
    import tempfile
    import firebase_admin # Re-import for the test block only
    from firebase_admin import firestore, auth, credentials # Re-import for the test block only

    logging.basicConfig(level=logging.INFO)

    # Mock config_manager for local testing
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
            # Mock secrets data, including the GCS bucket name
            self._secrets_data = {
                'gcs_bucket_name': 'mock-gcs-bucket-from-secrets', # Simulate secret from .streamlit/secrets.toml
                'firebase_config': json.dumps({"projectId": "mock-project-id"}), # For app_id fallback in FirestoreManager
                'openai_api_key': 'sk-mock-openai-key',
                'google_api_key': 'AIzaSy-mock-google-key'
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

    # Patch config_manager
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager

    # Mock firebase_admin for analytics initialization
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
        
        # Mock firebase_admin.get_app() if it's called by analytics_tracker directly
        with patch('firebase_admin.get_app', return_value=MagicMock(project_id="mock-project-id-from-app")):
            # Re-initialize analytics with mocks if it hasn't been already
            # This is specifically for the `if __name__ == "__main__"` block
            # We need to import initialize_analytics here specifically for the test block.
            from utils.analytics_tracker import initialize_analytics
            if 'analytics_initialized_backend' not in globals() or not globals()['analytics_initialized_backend']:
                initialize_analytics(
                    mock_db_for_analytics,
                    mock_auth_for_analytics,
                    "test-app-id-cli", # Use a test app_id for this mock context
                    "test_cli_user"
                )
                globals()['analytics_initialized_backend'] = True
                logger.info("Analytics tracker initialized with mocks for CLI test.")


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
        mock_docx_loader = MagicMock()
        mock_docx_loader.load.return_value = [
            MagicMock(page_content="Mock DOCX content page 1.", metadata={"source": "mock.docx", "page": 0})
        ]
        mock_txt_loader = MagicMock()
        mock_txt_loader.load.return_value = [
            MagicMock(page_content="Mock TXT content.", metadata={"source": "mock.txt"})
        ]
        mock_csv_loader = MagicMock()
        mock_csv_loader.load.return_value = [
            MagicMock(page_content="Mock CSV content: header1,header2\nvalue1,value2", metadata={"source": "mock.csv", "row": 0})
        ]
        mock_excel_loader = MagicMock()
        mock_excel_loader.load.return_value = [
            MagicMock(page_content="Mock Excel content: Sheet1\nColA\tColB\nValA\tValB", metadata={"source": "mock.xlsx", "sheet": "Sheet1"})
        ]

        with patch('langchain_community.document_loaders.PyPDFLoader', return_value=mock_pdf_loader), \
             patch('langchain_community.document_loaders.Docx2txtLoader', return_value=mock_docx_loader), \
             patch('langchain_community.document_loaders.TextLoader', return_value=mock_txt_loader), \
             patch('langchain_community.document_loaders.CSVLoader', return_value=mock_csv_loader), \
             patch('langchain_community.document_loaders.UnstructuredExcelLoader', return_value=mock_excel_loader):
            
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
                    print("\n--- Testing Vector Utility Functions with GCS Integration and new formats ---")
                    test_user_id = "test_user_456"
                    test_collection_name = "test_medical_docs"
                    
                    # Ensure temporary directories are clean
                    if TEMP_UPLOAD_DIR.exists():
                        shutil.rmtree(TEMP_UPLOAD_DIR)
                    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                    
                    # Create dummy local files for testing the path construction and loader selection
                    test_files = {
                        "sample.pdf": "user_uploads/test_user_456/general/sample.pdf",
                        "sample.docx": "user_uploads/test_user_456/general/sample.docx",
                        "sample.txt": "user_uploads/test_user_456/general/sample.txt",
                        "sample.csv": "user_uploads/test_user_456/general/sample.csv",
                        "sample.xlsx": "user_uploads/test_user_456/general/sample.xlsx",
                    }
                    for fname, blob_path in test_files.items():
                        local_path = TEMP_UPLOAD_DIR / test_user_id / fname
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(local_path, "w") as f:
                            f.write(f"dummy content for {fname}")

                    # Test process_uploaded_document for PDF
                    print("\n--- Test 1: process_uploaded_document (PDF) ---")
                    mock_db_for_analytics.collection.return_value.add.reset_mock()
                    # Simulate base64 content
                    dummy_pdf_content_b64 = base64.b64encode(b"This is dummy PDF content.").decode('utf-8')
                    result = await process_uploaded_document(
                        "sample_uploaded.pdf", dummy_pdf_content_b64, "application/pdf", test_user_id
                    )
                    print(f"Process Uploaded PDF Result: {result}")
                    assert result["success"] is True
                    assert "collection_name" in result
                    assert "gcs_uri" in result
                    mock_upload_file_to_gcs.assert_called_once()
                    # The loader.load() is mocked to return 2 documents
                    mock_chroma_from_documents.assert_called_once() # Called for creating embeddings
                    mock_chroma_instance.persist.assert_called_once()
                    print("Test 1 Passed.")
                    mock_upload_file_to_gcs.reset_mock()
                    mock_chroma_from_documents.reset_mock()
                    mock_chroma_instance.persist.reset_mock()

                    # Test query_documents
                    print("\n--- Test 2: query_documents (Success) ---")
                    mock_db_for_analytics.collection.return_value.add.reset_mock()
                    query_results = await query_documents(
                        "What is the capital of France?", 
                        test_user_id, 
                        collection_name=test_collection_name # Need a collection name
                    )
                    print(f"Query Results: {query_results}")
                    assert len(query_results) == 2 # Mock similarity_search returns 2
                    mock_chroma_constructor.assert_called_once() # Called for get_vector_store
                    mock_chroma_instance.similarity_search.assert_called_once_with("What is the capital of France?", k=4)
                    print("Test 2 Passed.")
                    mock_chroma_constructor.reset_mock()
                    mock_chroma_instance.similarity_search.reset_mock()

                    # Test delete_vector_store_collection
                    print("\n--- Test 3: delete_vector_store_collection (Success) ---")
                    mock_db_for_analytics.collection.return_value.add.reset_mock()
                    # Create a dummy persistence directory for deletion test
                    (BASE_VECTOR_DIR / test_user_id / test_collection_name).mkdir(parents=True, exist_ok=True)
                    delete_success = await delete_vector_store_collection(test_collection_name, test_user_id)
                    print(f"Delete Success: {delete_success}")
                    assert delete_success is True
                    assert not (BASE_VECTOR_DIR / test_user_id / test_collection_name).exists()
                    print("Test 3 Passed.")

                    # Clean up dummy files
                    for fname in test_files:
                        local_path = TEMP_UPLOAD_DIR / test_user_id / fname
                        if local_path.exists():
                            os.remove(local_path)
                    if TEMP_UPLOAD_DIR.exists():
                        shutil.rmtree(TEMP_UPLOAD_DIR)

                    print("\nAll Vector utility tests completed.")

                asyncio.run(run_vector_tests())
