# shared_tools/vector_utils.py

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# For embeddings and vector store
# from langchain_community.embeddings import OpenAIEmbeddings, GooglePalmEmbeddings, HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document # For Langchain Document objects

# Import config_manager and user_manager for RBAC checks
from config.config_manager import config_manager
from utils.user_manager import get_user_tier_capability, get_current_user

logger = logging.getLogger(__name__)

# Base directory for all vector stores
BASE_VECTOR_DIR = Path("vector_stores")

# --- Embedding Model Initialization (Lazy Loading) ---
_embedding_model_instance = None

def get_embedding_model():
    """
    Initializes and returns the embedding model instance.
    Uses configuration from config_manager.
    """
    global _embedding_model_instance
    if _embedding_model_instance is None:
        embedding_provider = config_manager.get("llm.embedding_provider", "openai") # New config key for embedding provider
        embedding_model_name = config_manager.get("llm.embedding_model_name", "text-embedding-ada-002") # New config key for embedding model name
        api_key = None

        if embedding_provider == "openai":
            api_key = config_manager.get_secret("openai_api_key")
            if not api_key:
                logger.error("OpenAI API key not found in secrets for embeddings.")
                raise ValueError("OpenAI API key is required for OpenAI embeddings.")
            # from langchain_community.embeddings import OpenAIEmbeddings # Uncomment in real setup
            # _embedding_model_instance = OpenAIEmbeddings(model=embedding_model_name, openai_api_key=api_key)
            logger.warning("Using mock embedding model. Replace with actual Langchain embedding model.")
            class MockEmbeddings:
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return [[0.1] * 1536 for _ in texts] # Mock embedding vector size for ada-002
                def embed_query(self, text: str) -> List[float]:
                    return [0.1] * 1536
            _embedding_model_instance = MockEmbeddings()
        elif embedding_provider == "google":
            api_key = config_manager.get_secret("google_api_key")
            if not api_key:
                logger.error("Google API key not found in secrets for embeddings.")
                raise ValueError("Google API key is required for Google embeddings.")
            # from langchain_community.embeddings import GooglePalmEmbeddings # Uncomment in real setup
            # _embedding_model_instance = GooglePalmEmbeddings(google_api_key=api_key) # GooglePalmEmbeddings often doesn't take model_name directly
            logger.warning("Using mock embedding model. Replace with actual Langchain embedding model.")
            class MockEmbeddings:
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return [[0.2] * 768 for _ in texts] # Mock embedding vector size for Google
                def embed_query(self, text: str) -> List[float]:
                    return [0.2] * 768
            _embedding_model_instance = MockEmbeddings()
        elif embedding_provider == "huggingface":
            # For local HuggingFace models, no API key is typically needed
            # from langchain_community.embeddings import HuggingFaceEmbeddings # Uncomment in real setup
            # _embedding_model_instance = HuggingFaceEmbeddings(model_name=embedding_model_name)
            logger.warning("Using mock embedding model. Replace with actual Langchain embedding model.")
            class MockEmbeddings:
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return [[0.3] * 384 for _ in texts] # Mock embedding vector size for HF
                def embed_query(self, text: str) -> List[float]:
                    return [0.3] * 384
            _embedding_model_instance = MockEmbeddings()
        else:
            raise ValueError(f"Unsupported embedding provider: {embedding_provider}")
    return _embedding_model_instance

# --- Vector Store Operations ---
def build_vectorstore(
    documents: List[Dict[str, Any]],
    vector_store_path: Path,
    user_token: str = "default"
) -> None:
    """
    Builds or updates a FAISS vector store from a list of document chunks.
    Applies RBAC for max_query_results_k (though this is more for retrieval,
    it can influence how the vector store is built if it has a max capacity).

    Args:
        documents (List[Dict[str, Any]]): A list of document chunks, each with 'page_content' and 'metadata'.
        vector_store_path (Path): The path where the FAISS vector store should be saved/loaded from.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks.
    """
    logger.info(f"Building/updating vector store at: {vector_store_path} for user: {user_token}")

    if not documents:
        logger.warning("No documents provided to build vector store.")
        return

    embedding_model = get_embedding_model()

    # RBAC check for max_query_results_k (relevant for retrieval, but good to note here)
    # max_k = get_user_tier_capability(user_token, 'document_query_max_results_k', config_manager.get('rag.max_query_results_k', 5))
    # This capability is primarily used by the query tool, not directly by build_vectorstore.

    # Convert dictionary chunks to Langchain Document objects
    # langchain_docs = [Document(page_content=doc['page_content'], metadata=doc['metadata']) for doc in documents]

    try:
        if vector_store_path.exists() and (vector_store_path / "index.faiss").exists() and (vector_store_path / "index.pkl").exists():
            # vectorstore = FAISS.load_local(str(vector_store_path), embedding_model, allow_dangerous_deserialization=True)
            # vectorstore.add_documents(langchain_docs)
            logger.info(f"Mock loaded existing FAISS and added {len(documents)} new documents.")
        else:
            # vectorstore = FAISS.from_documents(langchain_docs, embedding_model)
            logger.info(f"Mock created new FAISS from {len(documents)} documents.")
        
        # vectorstore.save_local(str(vector_store_path))
        # Simulate saving dummy files to indicate success
        vector_store_path.mkdir(parents=True, exist_ok=True)
        (vector_store_path / "index.faiss").touch()
        (vector_store_path / "index.pkl").touch()
        logger.info(f"Mock saved FAISS index to {vector_store_path}.")

    except Exception as e:
        logger.error(f"Error building/updating vector store at {vector_store_path}: {e}", exc_info=True)
        raise ValueError(f"Failed to build/update vector store: {e}")

def load_vectorstore(vector_store_path: Path) -> Any: # Returns FAISS object
    """
    Loads an existing FAISS vector store from a given path.

    Args:
        vector_store_path (Path): The path to the FAISS vector store.

    Returns:
        Any: The loaded FAISS vector store object.

    Raises:
        ValueError: If the vector store does not exist or cannot be loaded.
    """
    logger.info(f"Loading vector store from: {vector_store_path}")

    if not vector_store_path.exists() or not (vector_store_path / "index.faiss").exists() or not (vector_store_path / "index.pkl").exists():
        raise ValueError(f"Vector store not found at {vector_store_path}. Please upload and index documents first.")

    embedding_model = get_embedding_model()

    try:
        # vectorstore = FAISS.load_local(str(vector_store_path), embedding_model, allow_dangerous_deserialization=True)
        logger.info(f"Mock loaded vector store from {vector_store_path}.")
        # Return a mock object that simulates the functionality needed by retrieval
        class MockVectorStore:
            def __init__(self, path):
                self.path = path
            def similarity_search(self, query: str, k: int = 4) -> List[Any]:
                logger.info(f"Mock similarity search for query: '{query}' with k={k}")
                # Simulate returning mock document chunks
                return [
                    type('obj', (object,), {'page_content': f"Mock document chunk 1 for '{query}' from {self.path}", 'metadata': {'source': 'mock_doc_1.pdf', 'chunk_idx': 0}}),
                    type('obj', (object,), {'page_content': f"Mock document chunk 2 for '{query}' from {self.path}", 'metadata': {'source': 'mock_doc_2.txt', 'chunk_idx': 1}}),
                ][:k] # Return up to k mock results
        return MockVectorStore(vector_store_path)
    except Exception as e:
        logger.error(f"Error loading vector store from {vector_store_path}: {e}", exc_info=True)
        raise ValueError(f"Failed to load vector store: {e}")

def load_docs_from_json_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Loads documents from a JSON file.
    Expected JSON format: a list of objects, each with 'page_content' and 'metadata'.
    This is typically used for pre-processed documents.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"JSON document file not found at {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file must contain a list of documents.")
            # Basic validation for expected keys
            for doc in data:
                if not isinstance(doc, dict) or 'page_content' not in doc or 'metadata' not in doc:
                    raise ValueError("Each document in JSON must be an object with 'page_content' and 'metadata' keys.")
            logger.info(f"Loaded {len(data)} documents from JSON file: {file_path}")
            return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {file_path}: {e}", exc_info=True)
        raise ValueError(f"Invalid JSON format in document file: {e}")
    except Exception as e:
        logger.error(f"Error loading documents from JSON file {file_path}: {e}", exc_info=True)
        raise ValueError(f"Failed to load documents from JSON file: {e}")


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
                'llm': {
                    'provider': 'openai',
                    'model_name': 'gpt-3.5-turbo',
                    'temperature': 0.5,
                    'max_tokens': 4096,
                    'max_summary_input_chars': 10000,
                    'embedding_provider': 'openai', # New config for embeddings
                    'embedding_model_name': 'text-embedding-ada-002' # New config for embeddings
                },
                'rag': {
                    'chunk_size': 500,
                    'chunk_overlap': 50,
                    'max_query_results_k': 10 # Default config value
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
            # Simulate returning mock API key
            if key == "openai_api_key": return "MOCK_OPENAI_KEY_123"
            if key == "google_api_key": return "MOCK_GOOGLE_KEY_456"
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
                'document_query_enabled': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'document_query_max_results_k': {
                    'default': 3,
                    'roles': {'pro': 5, 'premium': 10, 'admin': 20}
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

    # Reset _embedding_model_instance for each test run
    global _embedding_model_instance
    _embedding_model_instance = None

    test_user = "test_user_vector"
    test_section = "test_section_vector"
    vector_store_path = BASE_VECTOR_DIR / test_user / test_section

    # Clean up directories from previous runs
    if BASE_VECTOR_DIR.exists():
        shutil.rmtree(BASE_VECTOR_DIR)
    BASE_VECTOR_DIR.mkdir(exist_ok=True)

    sample_documents = [
        {"page_content": "This is the first document chunk.", "metadata": {"source": "doc_a.pdf", "page": 1}},
        {"page_content": "This is the second document chunk.", "metadata": {"source": "doc_b.txt", "page": 0}},
    ]

    print("\n--- Testing build_vectorstore function ---")

    # Test 1: Build for Pro user
    print("\n--- Test 1: Build for Pro user ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    build_vectorstore(sample_documents, vector_store_path, user_token=test_user_pro)
    assert (vector_store_path / "index.faiss").exists()
    assert (vector_store_path / "index.pkl").exists()
    print("Test 1 Passed: Vector store built for Pro user.")

    # Test 2: Update for Premium user (should append)
    print("\n--- Test 2: Update for Premium user ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_premium
    additional_documents = [
        {"page_content": "This is a new document chunk.", "metadata": {"source": "doc_c.docx", "page": 0}}
    ]
    build_vectorstore(additional_documents, vector_store_path, user_token=test_user_premium)
    # Since it's a mock, we just check if files exist, not actual content merge
    assert (vector_store_path / "index.faiss").exists()
    assert (vector_store_path / "index.pkl").exists()
    print("Test 2 Passed: Vector store updated for Premium user (mocked).")

    # Test 3: Build with no documents
    print("\n--- Test 3: Build with no documents ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result_no_docs = build_vectorstore([], vector_store_path, user_token=test_user_pro)
    assert result_no_docs is None # Should just log warning and return None
    print("Test 3 Passed: Handled no documents gracefully.")

    print("\n--- Testing load_vectorstore function ---")

    # Test 4: Load existing vector store
    print("\n--- Test 4: Load existing vector store ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    loaded_vectorstore = load_vectorstore(vector_store_path)
    assert loaded_vectorstore is not None
    print("Test 4 Passed: Vector store loaded.")

    # Test 5: Load non-existent vector store
    print("\n--- Test 5: Load non-existent vector store ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    non_existent_path = BASE_VECTOR_DIR / "non_existent_user" / "non_existent_section"
    try:
        load_vectorstore(non_existent_path)
        assert False, "Expected ValueError for non-existent vector store but got none."
    except ValueError as e:
        print(f"Caught expected error: {e}")
        assert "Vector store not found" in str(e)
    print("Test 5 Passed: Handled non-existent vector store.")

    print("\n--- Testing load_docs_from_json_file function ---")
    
    # Create a dummy JSON file for testing load_docs_from_json_file
    json_test_dir = Path("./json_test_docs")
    json_test_dir.mkdir(exist_ok=True)
    json_file_path = json_test_dir / "test_docs.json"
    invalid_json_file_path = json_test_dir / "invalid_test_docs.json"
    empty_json_file_path = json_test_dir / "empty_test_docs.json"

    valid_json_content = [
        {"page_content": "Content 1", "metadata": {"id": 1}},
        {"page_content": "Content 2", "metadata": {"id": 2, "tag": "test"}}
    ]
    with open(json_file_path, "w") as f:
        json.dump(valid_json_content, f)

    invalid_json_content = "This is not a JSON array of objects."
    with open(invalid_json_file_path, "w") as f:
        f.write(invalid_json_content)
    
    empty_json_content = []
    with open(empty_json_file_path, "w") as f:
        json.dump(empty_json_content, f)


    # Test 6: Load valid JSON
    print("\n--- Test 6: Load valid JSON ---")
    loaded_docs = load_docs_from_json_file(json_file_path)
    print(f"Loaded docs from JSON: {loaded_docs}")
    assert len(loaded_docs) == 2
    assert loaded_docs[0]["page_content"] == "Content 1"
    print("Test 6 Passed: Loaded valid JSON.")

    # Test 7: Load invalid JSON format
    print("\n--- Test 7: Load invalid JSON format ---")
    try:
        load_docs_from_json_file(invalid_json_file_path)
        assert False, "Expected ValueError for invalid JSON format."
    except ValueError as e:
        print(f"Caught expected error: {e}")
        assert "Invalid JSON format" in str(e)
    print("Test 7 Passed: Handled invalid JSON format.")

    # Test 8: Load non-existent JSON file
    print("\n--- Test 8: Load non-existent JSON file ---")
    try:
        load_docs_from_json_file(json_test_dir / "non_existent.json")
        assert False, "Expected FileNotFoundError for non-existent JSON file."
    except FileNotFoundError as e:
        print(f"Caught expected error: {e}")
        assert "not found" in str(e)
    print("Test 8 Passed: Handled non-existent JSON file.")

    # Test 9: Load empty JSON file (empty list)
    print("\n--- Test 9: Load empty JSON file ---")
    loaded_empty_docs = load_docs_from_json_file(empty_json_file_path)
    print(f"Loaded empty docs from JSON: {loaded_empty_docs}")
    assert len(loaded_empty_docs) == 0
    print("Test 9 Passed: Loaded empty JSON file gracefully.")


    print("\nAll vector_utils tests passed (mocked embeddings and FAISS).")

    # Clean up test directories
    if BASE_VECTOR_DIR.exists():
        shutil.rmtree(BASE_VECTOR_DIR)
    if json_test_dir.exists():
        shutil.rmtree(json_test_dir)
    print(f"\nCleaned up test directories: {BASE_VECTOR_DIR}, {json_test_dir}")
