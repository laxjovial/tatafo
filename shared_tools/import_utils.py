# shared_tools/import_utils.py

import logging
from pathlib import Path
import shutil
import os
from typing import List, Dict, Any, Optional, Union
import pandas as pd # For CSV/Excel
import io # For handling uploaded file bytes

# Document loaders (e.g., for PDF, DOCX)
# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# For text splitting and embeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OpenAIEmbeddings, GooglePalmEmbeddings, HuggingFaceEmbeddings

# For vector store
# from langchain_community.vectorstores import FAISS

# Import config_manager and user_manager for RBAC checks
from config.config_manager import config_manager
from utils.user_manager import get_user_tier_capability, get_current_user

# Import specific LLM and Embedding utilities (if not using Langchain's direct loaders)
# from shared_tools.llm_embedding_utils import get_embedding_model # Assuming this exists or will be created

logger = logging.getLogger(__name__)

# Base directories for uploads and vector stores
BASE_UPLOAD_DIR = Path("uploads")
BASE_VECTOR_DIR = Path("vector_stores")

# Supported document extensions
# Added .csv, .xls, .xlsx
SUPPORTED_DOC_EXTS = [".pdf", ".docx", ".txt", ".csv", ".xls", ".xlsx"]

# --- Document Processing and Indexing ---
def _load_document_content(file_path: Path) -> List[Dict[str, Any]]:
    """
    Loads content from a document based on its file extension.
    Returns a list of dictionaries, where each dict represents a 'page' or 'chunk'
    with 'page_content' and 'metadata'.
    """
    file_extension = file_path.suffix.lower()
    docs_content = []

    try:
        if file_extension == ".pdf":
            # loader = PyPDFLoader(str(file_path)) # Uncomment for real Langchain
            # docs = loader.load()
            # For mock:
            docs = [{"page_content": f"Mock content from PDF: {file_path.name}. This is a test document.", "metadata": {"source": str(file_path)}}]
            logger.info(f"Mock loaded PDF: {file_path.name}")
        elif file_extension == ".docx":
            # loader = Docx2txtLoader(str(file_path)) # Uncomment for real Langchain
            # docs = loader.load()
            # For mock:
            docs = [{"page_content": f"Mock content from DOCX: {file_path.name}. This is a test document.", "metadata": {"source": str(file_path)}}]
            logger.info(f"Mock loaded DOCX: {file_path.name}")
        elif file_extension == ".txt":
            # loader = TextLoader(str(file_path), encoding='utf-8') # Uncomment for real Langchain
            # docs = loader.load()
            # For mock:
            docs = [{"page_content": f"Mock content from TXT: {file_path.name}. This is a test document.", "metadata": {"source": str(file_path)}}]
            logger.info(f"Mock loaded TXT: {file_path.name}")
        elif file_extension in [".csv", ".xls", ".xlsx"]:
            # Handle structured data (CSV, Excel)
            df: pd.DataFrame
            if file_extension == ".csv":
                df = pd.read_csv(file_path)
            elif file_extension == ".xls" or file_extension == ".xlsx":
                df = pd.read_excel(file_path)
            
            # Convert DataFrame to a string representation for chunking/embedding
            # Option 1: To string (less structured)
            # text_content = df.to_string(index=False)
            # Option 2: To markdown (more structured, better for LLM context)
            text_content = df.to_markdown(index=False)
            
            docs_content.append({
                "page_content": text_content,
                "metadata": {"source": str(file_path), "file_type": file_extension, "num_rows": len(df), "num_columns": len(df.columns)}
            })
            logger.info(f"Loaded {file_extension} file and converted to text: {file_path.name}")
            return docs_content # Return directly as it's already in the desired format
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Convert Langchain Document objects to our dictionary format if using real loaders
        # For mock, 'docs' is already in the desired format
        return docs
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}", exc_info=True)
        raise ValueError(f"Failed to load document {file_path.name}: {e}")

def _split_documents(documents: List[Dict[str, Any]], user_token: str) -> List[Dict[str, Any]]:
    """
    Splits loaded documents into smaller chunks using RecursiveCharacterTextSplitter.
    Applies RBAC for chunk size and overlap.
    """
    # Get chunking parameters from RBAC capabilities or config defaults
    chunk_size = get_user_tier_capability(user_token, 'document_chunk_size', config_manager.get('rag.chunk_size', 1000))
    chunk_overlap = get_user_tier_capability(user_token, 'document_chunk_overlap', config_manager.get('rag.chunk_overlap', 100))

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=chunk_size,
    #     chunk_overlap=chunk_overlap,
    #     length_function=len,
    #     is_separator_regex=False,
    # )
    
    # Mock splitting for now
    split_docs = []
    for doc in documents:
        content = doc['page_content']
        metadata = doc['metadata']
        # Simple mock split: just create one chunk for now, or split by a fixed size
        if len(content) > chunk_size:
            # Simulate splitting into multiple chunks
            for i in range(0, len(content), chunk_size - chunk_overlap):
                chunk = content[i : i + chunk_size]
                split_docs.append({"page_content": chunk, "metadata": {**metadata, "chunk_idx": len(split_docs)}})
        else:
            split_docs.append({"page_content": content, "metadata": metadata})
    
    logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks with size {chunk_size}.")
    return split_docs

def _embed_and_index_documents(chunks: List[Dict[str, Any]], vector_store_path: Path) -> None:
    """
    Embeds document chunks and indexes them into a FAISS vector store.
    """
    # embedding_model = get_embedding_model() # Assuming this function exists in shared_tools.llm_embedding_utils

    # Convert chunks to Langchain Document objects if using FAISS.from_documents
    # langchain_docs = [Document(page_content=chunk['page_content'], metadata=chunk['metadata']) for chunk in chunks]

    # For mock:
    if not chunks:
        logger.warning("No chunks to embed and index.")
        return

    # Simulate creating/updating FAISS index
    if vector_store_path.exists():
        # vectorstore = FAISS.load_local(str(vector_store_path), embedding_model, allow_dangerous_deserialization=True)
        # vectorstore.add_documents(langchain_docs)
        logger.info(f"Mock updated existing vector store at {vector_store_path} with {len(chunks)} new chunks.")
    else:
        # vectorstore = FAISS.from_documents(langchain_docs, embedding_model)
        logger.info(f"Mock created new vector store at {vector_store_path} with {len(chunks)} chunks.")
    
    # vectorstore.save_local(str(vector_store_path))
    # Simulate saving a dummy file to indicate success
    (vector_store_path / "index.faiss").touch()
    (vector_store_path / "index.pkl").touch()
    logger.info(f"Mock saved vector store to {vector_store_path}.")


def process_upload(uploaded_file: Any, user_token: str, section: str) -> str:
    """
    Handles the entire document upload, processing, and indexing workflow.
    Applies RBAC checks for document upload capability.

    Args:
        uploaded_file (Any): The file object from Streamlit's st.file_uploader.
        user_token (str): The unique identifier for the user.
        section (str): The application section (e.g., "medical", "legal").

    Returns:
        str: A success or error message.
    """
    logger.info(f"Processing upload for user: {user_token}, section: {section}, file: {uploaded_file.name}")

    # RBAC Check for Document Upload Enabled
    if not get_user_tier_capability(user_token, 'document_upload_enabled', False):
        return "Error: Document upload is not enabled for your current tier."

    # Create user-specific upload and vector store directories
    user_upload_dir = BASE_UPLOAD_DIR / user_token / section
    user_vector_store_dir = BASE_VECTOR_DIR / user_token / section
    
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    user_vector_store_dir.mkdir(parents=True, exist_ok=True)

    file_path = user_upload_dir / uploaded_file.name
    file_extension = Path(uploaded_file.name).suffix.lower()

    if file_extension not in SUPPORTED_DOC_EXTS:
        return f"Error: Unsupported file type '{file_extension}'. Supported types are: {', '.join(SUPPORTED_DOC_EXTS)}"

    try:
        # Save the uploaded file locally
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"File saved to: {file_path}")

        # Load document content
        documents = _load_document_content(file_path)
        
        # Split documents into chunks
        chunks = _split_documents(documents, user_token) # Pass user_token for chunking RBAC

        # Embed and index chunks into vector store
        _embed_and_index_documents(chunks, user_vector_store_dir)

        return f"Document '{uploaded_file.name}' processed and indexed successfully for section '{section}'."

    except ValueError as ve:
        logger.error(f"Processing failed for {uploaded_file.name} due to data error: {ve}", exc_info=True)
        return f"Error processing document: {ve}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during processing {uploaded_file.name}: {e}", exc_info=True)
        return f"An unexpected error occurred during document processing: {e}"

def clear_indexed_data(user_token: str, section: str) -> str:
    """
    Removes all uploaded files and indexed vector data for a specific user and section.

    Args:
        user_token (str): The unique identifier for the user.
        section (str): The application section (e.g., "medical", "legal").

    Returns:
        str: A success or error message.
    """
    logger.info(f"Clearing indexed data for user: {user_token}, section: {section}")

    user_upload_dir = BASE_UPLOAD_DIR / user_token / section
    user_vector_store_dir = BASE_VECTOR_DIR / user_token / section

    messages = []

    try:
        if user_upload_dir.exists():
            shutil.rmtree(user_upload_dir)
            messages.append(f"Removed uploaded files from: {user_upload_dir}")
        else:
            messages.append(f"No uploaded files found at: {user_upload_dir}")

        if user_vector_store_dir.exists():
            shutil.rmtree(user_vector_store_dir)
            messages.append(f"Removed indexed vector data from: {user_vector_store_dir}")
        else:
            messages.append(f"No indexed data found at: {user_vector_store_dir}")
        
        if not messages:
            return "No data found to clear for this user and section."

        return "\n".join(messages)
    except Exception as e:
        logger.error(f"Error clearing data for user {user_token}, section {section}: {e}", exc_info=True)
        return f"An error occurred while clearing data: {e}"


# CLI Test (optional)
if __name__ == "__main__":
    import shutil
    from unittest.mock import MagicMock, patch
    import sys
    import os

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.openai = {"api_key": "sk-mock-openai-key-12345"}
            self.google = {"api_key": "AIzaSy-mock-google-key"}
            self.user_tokens = {
                "free_user_token": "mock_free_token",
                "pro_user_token": "mock_pro_token",
                "premium_user_token": "mock_premium_token",
                "admin_user_token": "mock_admin_token"
            }
            self.firebase_config = "{}" # Mock empty config for Firebase if not set

        def get(self, key, default=None):
            parts = key.split('.')
            val = self
            for part in parts:
                if hasattr(val, part):
                    val = getattr(val, part)
                elif isinstance(val, dict) and part in val:
                    val = val[part]
                else:
                    return default
            return val
    
    class MockConfigManager:
        _instance = None
        _is_loaded = False
        def __init__(self):
            if MockConfigManager._instance is not None:
                raise Exception("ConfigManager is a singleton. Use get_instance().")
            MockConfigManager._instance = self
            self._config_data = {
                'llm': {'max_summary_input_chars': 10000},
                'rag': {
                    'chunk_size': 500,
                    'chunk_overlap': 50,
                    'max_query_results_k': 10
                },
                'web_scraping': {
                    'user_agent': 'Mozilla/5.0 (Test; Python)',
                    'timeout_seconds': 5,
                    'max_search_results': 5
                },
                'tiers': {}, # This will be overridden by tiers.yaml
                'default_user_tier': 'free',
                'default_user_roles': ['user'],
                'api_configs': []
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
            return st.secrets.get(key, default)

        def set_secret(self, key, value):
            setattr(st.secrets, key, value)


    # Mock user_manager.get_current_user and get_user_tier_capability for testing RBAC
    class MockUserManager:
        _mock_users = {
            "mock_free_token": {"user_id": "mock_free_token", "username": "FreeUser", "email": "free@example.com", "tier": "free", "roles": ["user"]},
            "mock_pro_token": {"user_id": "mock_pro_token", "username": "ProUser", "email": "pro@example.com", "tier": "pro", "roles": ["user"]},
            "mock_premium_token": {"user_id": "mock_premium_token", "username": "PremiumUser", "email": "premium@example.com", "tier": "premium", "roles": ["user"]},
            "mock_admin_token": {"user_id": "mock_admin_token", "username": "AdminUser", "email": "admin@example.com", "tier": "admin", "roles": ["user", "admin"]},
        }
        _rbac_capabilities = {
            'capabilities': {
                'document_upload_enabled': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'document_chunk_size': {
                    'default': 500,
                    'roles': {'pro': 1000, 'premium': 2000, 'admin': 5000}
                },
                'document_chunk_overlap': {
                    'default': 50,
                    'roles': {'pro': 100, 'premium': 200, 'admin': 500}
                }
            }
        }
        _tier_hierarchy = {
            "free": 0, "user": 1, "basic": 2, "pro": 3, "premium": 4, "admin": 99
        }

        def get_current_user(self) -> Dict[str, Any]:
            return getattr(self, '_current_mock_user', {})

        def get_user_tier_capability(self, user_token: Optional[str], capability_key: str, default_value: Any = None) -> Any:
            user_info = self._mock_users.get(user_token, {})
            user_id = user_info.get('user_id')
            user_tier = user_info.get('tier', 'free')
            user_roles = user_info.get('roles', [])

            if "admin" in user_roles:
                if isinstance(default_value, bool): return True
                if isinstance(default_value, (int, float)): return float('inf')
                return default_value
            
            capability_config = self._rbac_capabilities.get('capabilities', {}).get(capability_key)
            if not capability_config:
                return default_value

            for role in user_roles:
                if role in capability_config.get('roles', {}):
                    return capability_config['roles'][role]
            
            return capability_config.get('default', default_value)

    # Patch the actual imports for testing
    import streamlit as st_mock
    if not hasattr(st_mock, 'secrets'):
        st_mock.secrets = MockSecrets()
    
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager
    sys.modules['utils.user_manager'] = MockUserManager()
    sys.modules['utils.user_manager']._RBAC_CAPABILITIES = MockUserManager()._rbac_capabilities
    sys.modules['utils.user_manager']._TIER_HIERARCHY = MockUserManager()._tier_hierarchy


    print("\n--- Testing import_utils functions ---")

    test_user_free = sys.modules['utils.user_manager']._mock_users["mock_free_token"]['user_id']
    test_user_pro = sys.modules['utils.user_manager']._mock_users["mock_pro_token"]['user_id']
    test_user_premium = sys.modules['utils.user_manager']._mock_users["mock_premium_token"]['user_id']
    test_user_admin = sys.modules['utils.user_manager']._mock_users["mock_admin_token"]['user_id']

    test_section = "test_section"

    # Clean up directories from previous runs
    if BASE_UPLOAD_DIR.exists():
        shutil.rmtree(BASE_UPLOAD_DIR)
    if BASE_VECTOR_DIR.exists():
        shutil.rmtree(BASE_VECTOR_DIR)
    
    BASE_UPLOAD_DIR.mkdir(exist_ok=True)
    BASE_VECTOR_DIR.mkdir(exist_ok=True)


    # Create dummy files for testing
    test_upload_dir = BASE_UPLOAD_DIR / "temp_test_uploads"
    test_upload_dir.mkdir(parents=True, exist_ok=True)

    # Mock uploaded file object
    class MockUploadedFile:
        def __init__(self, name, content):
            self.name = name
            self._content = content
        def getbuffer(self):
            return self._content.encode('utf-8') # Assume text content for simplicity

    # Create dummy CSV
    csv_content = "header1,header2\nvalue1,valueA\nvalue2,valueB"
    csv_file = MockUploadedFile("sample.csv", csv_content)
    
    # Create dummy XLSX (requires openpyxl to write, mock for simplicity)
    # For a real test, you'd need a pre-existing XLSX or a more complex generation.
    # Here, we'll mock pd.read_excel.
    mock_excel_df = pd.DataFrame({"ColA": [1, 2], "ColB": ["X", "Y"]})
    xlsx_file = MockUploadedFile("sample.xlsx", "dummy_xlsx_bytes") # Content won't be read directly by mock
    
    # Create dummy PDF (mock PyPDFLoader)
    pdf_file = MockUploadedFile("sample.pdf", "dummy_pdf_bytes")

    # Create dummy TXT
    txt_content = "This is a sample text document for testing."
    txt_file = MockUploadedFile("sample.txt", txt_content)

    # Patch pandas.read_excel for mock
    with patch('pandas.read_excel', return_value=mock_excel_df):
        # Test 1: Pro user, CSV upload
        print("\n--- Test 1: Pro user, CSV upload ---")
        sys.modules['utils.user_manager']._current_mock_user = test_user_pro
        result_csv = process_upload(csv_file, test_user_pro, test_section)
        print(f"Result for CSV upload (Pro user): {result_csv}")
        assert "processed and indexed successfully" in result_csv
        assert (BASE_UPLOAD_DIR / test_user_pro / test_section / "sample.csv").exists()
        assert (BASE_VECTOR_DIR / test_user_pro / test_section / "index.faiss").exists()
        print("Test 1 Passed.")

        # Test 2: Premium user, XLSX upload
        print("\n--- Test 2: Premium user, XLSX upload ---")
        sys.modules['utils.user_manager']._current_mock_user = test_user_premium
        result_xlsx = process_upload(xlsx_file, test_user_premium, test_section)
        print(f"Result for XLSX upload (Premium user): {result_xlsx}")
        assert "processed and indexed successfully" in result_xlsx
        assert (BASE_UPLOAD_DIR / test_user_premium / test_section / "sample.xlsx").exists()
        assert (BASE_VECTOR_DIR / test_user_premium / test_section / "index.faiss").exists()
        print("Test 2 Passed.")

        # Test 3: Free user, document upload disabled
        print("\n--- Test 3: Free user, upload disabled ---")
        sys.modules['utils.user_manager']._current_mock_user = test_user_free
        result_free = process_upload(pdf_file, test_user_free, test_section)
        print(f"Result for PDF upload (Free user): {result_free}")
        assert "Error: Document upload is not enabled for your current tier." in result_free
        print("Test 3 Passed.")

        # Test 4: Admin user, TXT upload (should work)
        print("\n--- Test 4: Admin user, TXT upload ---")
        sys.modules['utils.user_manager']._current_mock_user = test_user_admin
        result_txt = process_upload(txt_file, test_user_admin, test_section)
        print(f"Result for TXT upload (Admin user): {result_txt}")
        assert "processed and indexed successfully" in result_txt
        assert (BASE_UPLOAD_DIR / test_user_admin / test_section / "sample.txt").exists()
        assert (BASE_VECTOR_DIR / test_user_admin / test_section / "index.faiss").exists()
        print("Test 4 Passed.")

        # Test 5: Unsupported file type
        print("\n--- Test 5: Unsupported file type ---")
        sys.modules['utils.user_manager']._current_mock_user = test_user_pro
        unsupported_file = MockUploadedFile("image.jpg", "dummy_image_bytes")
        result_unsupported = process_upload(unsupported_file, test_user_pro, test_section)
        print(f"Result for unsupported file: {result_unsupported}")
        assert "Error: Unsupported file type '.jpg'" in result_unsupported
        print("Test 5 Passed.")

    # Test 6: Clear indexed data
    print("\n--- Test 6: Clear indexed data for Pro user ---")
    clear_result = clear_indexed_data(test_user_pro, test_section)
    print(f"Clear data result (Pro user): {clear_result}")
    assert "Removed uploaded files" in clear_result
    assert "Removed indexed vector data" in clear_result
    assert not (BASE_UPLOAD_DIR / test_user_pro / test_section).exists()
    assert not (BASE_VECTOR_DIR / test_user_pro / test_section).exists()
    print("Test 6 Passed.")

    # Clean up dummy test directories
    if BASE_UPLOAD_DIR.exists():
        shutil.rmtree(BASE_UPLOAD_DIR)
    if BASE_VECTOR_DIR.exists():
        shutil.rmtree(BASE_VECTOR_DIR)
    print("\nCleaned up test directories.")
