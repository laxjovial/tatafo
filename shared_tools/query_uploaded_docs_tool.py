# shared_tools/query_uploaded_docs_tool.py

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import vector_utils for loading the vector store
from shared_tools.vector_utils import load_vectorstore, BASE_VECTOR_DIR

# Import export_utils for exporting results
from shared_tools.export_utils import export_vector_results

# Import config_manager and user_manager for RBAC checks
from config.config_manager import config_manager
from utils.user_manager import get_user_tier_capability, get_current_user

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@tool
def query_uploaded_docs(
    query: str,
    user_token: str = "default",
    section: str = "general",
    k: Optional[int] = None,
    export: bool = False
) -> str:
    """
    Queries the user's uploaded and indexed documents within a specific section
    (e.g., "medical", "legal", "finance") to retrieve relevant information.
    This tool is essential for Retrieval Augmented Generation (RAG) to provide
    answers based on private or specialized knowledge bases.

    Args:
        query (str): The natural language query to search for within the documents.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks and user-specific vector stores.
        section (str, optional): The application section where the documents are indexed
                                 (e.g., "medical", "legal", "finance"). Defaults to "general".
        k (int, optional): The number of top relevant document chunks to retrieve.
                           If not provided, it will be determined by user's tier capability.
        export (bool, optional): If True, the retrieved results will be exported to a Markdown file.
                                 This action is subject to the 'chart_export_enabled' RBAC capability.

    Returns:
        str: A formatted string containing the retrieved document chunks and their sources,
             or an error message if the vector store is not found or access is denied.
    """
    logger.info(f"Tool: query_uploaded_docs called for user: {user_token}, section: {section}, query: '{query[:50]}...'")

    # RBAC Check for Document Query Enabled
    if not get_user_tier_capability(user_token, 'document_query_enabled', False):
        return "Error: Document querying is not enabled for your current tier."
    
    # Get user's allowed k (number of results) from RBAC capabilities
    if k is None:
        k = get_user_tier_capability(user_token, 'document_query_max_results_k', config_manager.get('rag.max_query_results_k', 5))
    
    # Check if export is enabled for the user's tier (if export is requested)
    is_export_allowed = get_user_tier_capability(user_token, 'chart_export_enabled', False) # Reusing chart_export_enabled for document export for now

    vector_store_path = BASE_VECTOR_DIR / user_token / section

    try:
        # Load the user-specific vector store
        vectorstore = load_vectorstore(vector_store_path)
        
        # Perform similarity search
        # The MockVectorStore from vector_utils.py will simulate this
        retrieved_docs = vectorstore.similarity_search(query, k=k)
        
        if not retrieved_docs:
            return f"No relevant documents found for your query in the '{section}' section."

        formatted_results = []
        raw_results_for_export = [] # To store dicts for export_vector_results

        for i, doc in enumerate(retrieved_docs):
            # Ensure doc has page_content and metadata attributes/keys
            content = getattr(doc, 'page_content', doc.get('page_content', 'N/A'))
            metadata = getattr(doc, 'metadata', doc.get('metadata', {}))
            source = metadata.get('source', 'Unknown Source')
            chunk_idx = metadata.get('chunk_idx', 'N/A')

            formatted_results.append(
                f"--- Document Chunk {i+1} ---\n"
                f"Source: {source}\n"
                f"Chunk Index: {chunk_idx}\n"
                f"Content:\n{content}\n"
            )
            raw_results_for_export.append({
                "page_content": content,
                "metadata": metadata
            })
        
        final_response = "\n\n".join(formatted_results)

        if export and is_export_allowed:
            export_message = export_vector_results(raw_results_for_export, user_token, section, file_prefix=f"{section}_query_results")
            final_response += f"\n\n{export_message}"
        elif export and not is_export_allowed:
            final_response += "\n\nWarning: Export was requested but is not enabled for your current tier."

        return final_response

    except ValueError as ve:
        logger.error(f"Error querying documents for user {user_token}, section {section}: {ve}", exc_info=True)
        return f"Error: {ve}"
    except Exception as e:
        logger.critical(f"An unexpected error occurred during document query for user {user_token}, section {section}: {e}", exc_info=True)
        return f"An unexpected error occurred during document query: {e}"


# CLI Test (optional)
if __name__ == "__main__":
    import shutil
    from unittest.mock import MagicMock, patch
    import sys
    import os

    logging.basicConfig(level=logging.INFO)

    pass


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
                },
                'chart_export_enabled': { # Used for document export as well
                    'default': False,
                    'roles': {'premium': True, 'admin': True}
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

    # Mock vector_utils functions
    sys.modules['shared_tools.vector_utils']._embedding_model_instance = None # Reset mock embedding
    sys.modules['shared_tools.vector_utils'].get_embedding_model() # Initialize mock embedding

    class MockLoadedVectorStore:
        def __init__(self, path):
            self.path = path
        def similarity_search(self, query: str, k: int = 4) -> List[Any]:
            mock_docs = [
                type('obj', (object,), {'page_content': f"Mock chunk 1 for '{query}' from {self.path}. This chunk is very relevant.", 'metadata': {'source': 'doc_A.pdf', 'chunk_idx': 0}}),
                type('obj', (object,), {'page_content': f"Mock chunk 2 for '{query}' from {self.path}. This chunk provides more details.", 'metadata': {'source': 'doc_B.txt', 'chunk_idx': 1}}),
                type('obj', (object,), {'page_content': f"Mock chunk 3 for '{query}' from {self.path}. This chunk is also relevant.", 'metadata': {'source': 'doc_C.docx', 'chunk_idx': 2}}),
                type('obj', (object,), {'page_content': f"Mock chunk 4 for '{query}' from {self.path}. This chunk is less relevant.", 'metadata': {'source': 'doc_D.csv', 'chunk_idx': 3}}),
                type('obj', (object,), {'page_content': f"Mock chunk 5 for '{query}' from {self.path}. This chunk is the least relevant.", 'metadata': {'source': 'doc_E.xlsx', 'chunk_idx': 4}}),
            ]
            return mock_docs[:k]

    sys.modules['shared_tools.vector_utils'].load_vectorstore = MagicMock(return_value=MockLoadedVectorStore(Path("mock_vector_store_path")))
    sys.modules['shared_tools.vector_utils'].BASE_VECTOR_DIR = Path("./mock_vector_stores") # Use a mock base dir for tests

    # Mock export_utils functions
    sys.modules['shared_tools.export_utils'].BASE_EXPORT_DIR = Path("./mock_exports") # Use a mock base dir for tests
    # Mock firestore.SERVER_TIMESTAMP for export_utils
    class MockFirestore:
        SERVER_TIMESTAMP = "MOCK_TIMESTAMP"
    sys.modules['firebase_admin.firestore'] = MockFirestore()

    # Ensure mock directories exist for tests
    sys.modules['shared_tools.vector_utils'].BASE_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    sys.modules['shared_tools.export_utils'].BASE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    (sys.modules['shared_tools.vector_utils'].BASE_VECTOR_DIR / "test_user_pro" / "medical" / "index.faiss").touch()
    (sys.modules['shared_tools.vector_utils'].BASE_VECTOR_DIR / "test_user_pro" / "medical" / "index.pkl").touch()
    (sys.modules['shared_tools.vector_utils'].BASE_VECTOR_DIR / "test_user_premium" / "legal" / "index.faiss").touch()
    (sys.modules['shared_tools.vector_utils'].BASE_VECTOR_DIR / "test_user_premium" / "legal" / "index.pkl").touch()
    (sys.modules['shared_tools.vector_utils'].BASE_VECTOR_DIR / "test_user_admin" / "finance" / "index.faiss").touch()
    (sys.modules['shared_tools.vector_utils'].BASE_VECTOR_DIR / "test_user_admin" / "finance" / "index.pkl").touch()


    test_user_free = sys.modules['utils.user_manager']._mock_users["mock_free_token"]['user_id']
    test_user_pro = sys.modules['utils.user_manager']._mock_users["mock_pro_token"]['user_id']
    test_user_premium = sys.modules['utils.user_manager']._mock_users["mock_premium_token"]['user_id']
    test_user_admin = sys.modules['utils.user_manager']._mock_users["mock_admin_token"]['user_id']

    print("\n--- Testing query_uploaded_docs function ---")

    # Test 1: Pro user, medical section, default k (5)
    print("\n--- Test 1: Pro user, medical section, default k ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result1 = query_uploaded_docs("symptoms of common cold", user_token=test_user_pro, section="medical")
    print(f"Result 1 (Pro user, medical):\n{result1[:200]}...")
    assert "Mock chunk 1 for 'symptoms of common cold'" in result1
    assert "Mock chunk 5 for 'symptoms of common cold'" in result1 # Should get up to k=5
    assert "Error: Document querying is not enabled" not in result1
    print("Test 1 Passed.")

    # Test 2: Premium user, legal section, explicit k=2, with export
    print("\n--- Test 2: Premium user, legal section, k=2, with export ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_premium
    result2 = query_uploaded_docs("contract law basics", user_token=test_user_premium, section="legal", k=2, export=True)
    print(f"Result 2 (Premium user, legal, k=2, export):\n{result2[:200]}...")
    assert "Mock chunk 1 for 'contract law basics'" in result2
    assert "Mock chunk 2 for 'contract law basics'" in result2
    assert "Mock chunk 3 for 'contract law basics'" not in result2 # Should be limited to k=2
    assert "Vector search results exported to:" in result2 # Should export
    assert (sys.modules['shared_tools.export_utils'].BASE_EXPORT_DIR / test_user_premium / "legal").exists()
    print("Test 2 Passed.")

    # Test 3: Free user, document querying disabled
    print("\n--- Test 3: Free user, querying disabled ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result3 = query_uploaded_docs("any query", user_token=test_user_free, section="general")
    print(f"Result 3 (Free user): {result3}")
    assert "Error: Document querying is not enabled for your current tier." in result3
    print("Test 3 Passed.")

    # Test 4: Admin user, finance section, explicit k=15 (admin override), with export
    print("\n--- Test 4: Admin user, finance section, k=15, with export ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    result4 = query_uploaded_docs("stock market trends", user_token=test_user_admin, section="finance", k=15, export=True)
    print(f"Result 4 (Admin user, finance, k=15, export):\n{result4[:200]}...")
    assert "Mock chunk 1 for 'stock market trends'" in result4
    assert "Vector search results exported to:" in result4 # Should export
    assert (sys.modules['shared_tools.export_utils'].BASE_EXPORT_DIR / test_user_admin / "finance").exists()
    print("Test 4 Passed.")

    # Test 5: Pro user, export requested but not allowed (chart_export_enabled is False for Pro by default)
    print("\n--- Test 5: Pro user, export requested but not allowed ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result5 = query_uploaded_docs("pro export test", user_token=test_user_pro, section="medical", export=True)
    print(f"Result 5 (Pro user, export requested):\n{result5[:200]}...")
    assert "Warning: Export was requested but is not enabled for your current tier." in result5
    assert "Mock chunk 1" in result5 # Should still return results
    print("Test 5 Passed.")

    # Test 6: Vector store not found
    print("\n--- Test 6: Vector store not found ---")
    sys.modules['shared_tools.vector_utils'].load_vectorstore.side_effect = ValueError("Vector store not found at mock_path")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result6 = query_uploaded_docs("non-existent store", user_token=test_user_pro, section="non_existent")
    print(f"Result 6 (Non-existent store): {result6}")
    assert "Error: Vector store not found at mock_path" in result6
    sys.modules['shared_tools.vector_utils'].load_vectorstore.side_effect = None # Reset mock
    print("Test 6 Passed.")

    print("\nAll query_uploaded_docs tests passed (mocked vector store and RBAC).")

    # Clean up mock directories
    if sys.modules['shared_tools.vector_utils'].BASE_VECTOR_DIR.exists():
        shutil.rmtree(sys.modules['shared_tools.vector_utils'].BASE_VECTOR_DIR)
    if sys.modules['shared_tools.export_utils'].BASE_EXPORT_DIR.exists():
        shutil.rmtree(sys.modules['shared_tools.export_utils'].BASE_EXPORT_DIR)
    print(f"\nCleaned up mock directories: {sys.modules['shared_tools.vector_utils'].BASE_VECTOR_DIR}, {sys.modules['shared_tools.export_utils'].BASE_EXPORT_DIR}")
