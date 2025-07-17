# shared_tools/query_uploaded_docs_tool.py

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Corrected import for load_vectorstore - now importing the module-level function
from shared_tools.vector_utils import load_vectorstore, BASE_VECTOR_DIR

# Import export_utils for exporting results
from shared_tools.export_utils import export_vector_results # Assuming this is available and correct

# Import config_manager and user_manager for RBAC checks
from config.config_manager import config_manager # Ensure this is correctly importable
from utils.user_manager import get_user_tier_capability, get_current_user # Ensure these are correctly importable

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@tool
async def query_uploaded_docs( # Made the function async
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
                                 (e.g., "medical", "legal", "finance", "general").
                                 Defaults to "general".
        k (int, optional): The number of top relevant documents to retrieve. Defaults to 4.
        export (bool, optional): If True, the retrieved results will be exported to a file.
                                 Defaults to False.

    Returns:
        str: A summary of the relevant information found, or an error/status message.
    """
    user_id = user_token # Use user_token as user_id for consistency

    logger.info(f"Tool: query_uploaded_docs called for user '{user_id}' in section '{section}' with query: '{query}'")

    # RBAC Check: Document querying access
    if not get_user_tier_capability(user_id, 'document_query_enabled', False):
        return "Error: Document querying is not enabled for your current tier. Please upgrade your plan."
    
    # RBAC Check: Specific section access (if sections are tied to RBAC)
    # This is a placeholder; implement specific section access logic if needed in config_manager
    # For example: if section == "medical" and not get_user_tier_capability(user_id, 'medical_docs_access', False):
    #    return "Error: Access to 'medical' document section is not enabled for your tier."

    try:
        # Load the vector store for the specific user and section
        # Now calling the module-level async function
        vectorstore = await load_vectorstore(user_id=user_id, section=section)

        if vectorstore is None:
            return f"Error: Could not load vector store for user '{user_id}' in section '{section}'. It might not exist or be accessible."

        # Perform the similarity search
        docs = vectorstore.similarity_search(query, k=k if k is not None else 4) # Use k from args, default to 4

        if not docs:
            log_event(user_id, "query_uploaded_docs", "no_results", {"query": query, "section": section})
            return "No relevant information found in uploaded documents."

        # Concatenate document content for response
        # Ensure that `docs` elements have a `page_content` attribute as returned by the mock vector store.
        relevant_content = "\\n\\n".join([doc.page_content for doc in docs])
        
        summary = f"Information from uploaded documents for your query '{query}':\\n\\n{relevant_content}"

        # RBAC Check: Export capability
        if export:
            if get_user_tier_capability(user_id, 'document_export_enabled', False):
                export_path = export_vector_results(user_id, query, docs)
                summary += f"\\n\\nQuery results exported to: {export_path}"
                log_event(user_id, "query_uploaded_docs", "exported", {"query": query, "section": section, "export_path": str(export_path)})
            else:
                summary += "\\n\\nWarning: Export was requested but is not enabled for your current tier."
                log_event(user_id, "query_uploaded_docs", "export_denied", {"query": query, "section": section})

        log_event(user_id, "query_uploaded_docs", "success", {"query": query, "section": section})
        return summary

    except Exception as e:
        logger.error(f"Error querying uploaded documents for user {user_id}, section {section}: {e}", exc_info=True)
        log_event(user_id, "query_uploaded_docs", "error", {"query": query, "section": section, "error": str(e)})
        return f"Error: An unexpected error occurred while querying documents: {e}"

# CLI Test (optional)
if __name__ == "__main__":
    import sys
    import asyncio
    from unittest.mock import MagicMock, patch

    logging.basicConfig(level=logging.INFO)

    # Mock UserManager and config_manager for testing
    class MockUserManager:
        _rbac_capabilities = {
            'capabilities': {
                'document_query_enabled': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'document_export_enabled': {
                    'default': False,
                    'roles': {'premium': True, 'admin': True}
                }
            }
        }
        _tier_hierarchy = {
            "free": 0, "user": 1, "basic": 2, "pro": 3, "premium": 4, "admin": 99
        }
        def get_user_tier_capability(self, user_token: Optional[str], capability_key: str, default_value: Any = None, user_tier: Optional[str] = None, user_roles: Optional[List[str]] = None) -> Any:
            user_info = {"tier": user_tier if user_tier else "free", "roles": user_roles if user_roles else []}
            if user_token == "test_user_pro": user_info = {"tier": "pro", "roles": ["user"]}
            if user_token == "test_user_premium": user_info = {"tier": "premium", "roles": ["user"]}
            if user_token == "test_user_free": user_info = {"tier": "free", "roles": ["user"]}
            if user_token == "test_user_admin": user_info = {"tier": "admin", "roles": ["user", "admin"]}

            if "admin" in user_info["roles"]:
                return True # Admin bypasses all specific capability checks

            capability_config = self._rbac_capabilities.get('capabilities', {}).get(capability_key)
            if not capability_config:
                return default_value
            
            for role in user_info["roles"]:
                if role in capability_config.get('roles', {}):
                    return capability_config['roles'][role]
            
            if user_info["tier"] in capability_config.get('tiers', {}):
                return capability_config['tiers'][user_info["tier"]]

            return capability_config.get('default', default_value)

    # Patch relevant modules
    sys.modules['config.config_manager'] = MagicMock()
    sys.modules['utils.user_manager'] = MockUserManager()
    sys.modules['utils.analytics_tracker'] = MagicMock()
    sys.modules['utils.analytics_tracker'].log_event = MagicMock()
    
    # Mock the vector store and export functions
    class MockVectorStore:
        def similarity_search(self, query: str, k: int) -> List[Any]:
            if "report" in query.lower():
                # For testing export, return a document with content
                mock_doc_content = "This is a mock report with key points."
                # Langchain Document objects have page_content attribute
                mock_doc = MagicMock()
                mock_doc.page_content = mock_doc_content
                return [mock_doc]
            elif "no info" in query.lower():
                return []
            # Simulate general results
            doc1 = MagicMock()
            doc1.page_content = f"Mock chunk 1 for '{query}'"
            doc2 = MagicMock()
            doc2.page_content = f"Mock chunk 2 for '{query}'"
            return [doc1, doc2][:k] # Return up to k mock documents
    
    # Mock load_vectorstore to return our MockVectorStore
    # Patch the actual module-level load_vectorstore from shared_tools.vector_utils
    with patch('shared_tools.vector_utils.load_vectorstore', new=MagicMock(return_value=MockVectorStore())):
        with patch('shared_tools.export_utils.export_vector_results', new=MagicMock(return_value="/tmp/mock_export.txt")):
            async def run_tests():
                test_user_pro = "test_user_pro"
                test_user_premium = "test_user_premium"
                test_user_free = "test_user_free"
                test_user_admin = "test_user_admin"

                # Test 1: Pro user, general query (allowed)
                print("\n--- Test 1: Pro user, general query ---")
                result1 = await query_uploaded_docs("What is the main topic?", user_token=test_user_pro)
                print(f"Result 1 (Pro user): {result1}")
                assert "Mock chunk 1" in result1
                sys.modules['utils.analytics_tracker'].log_event.assert_called_once_with(test_user_pro, "query_uploaded_docs", "success", {"query": "What is the main topic?", "section": "general"})
                sys.modules['utils.analytics_tracker'].log_event.reset_mock()
                print("Test 1 Passed.")

                # Test 2: Free user, query denied by RBAC
                print("\n--- Test 2: Free user, query denied by RBAC ---")
                result2 = await query_uploaded_docs("some query", user_token=test_user_free)
                print(f"Result 2 (Free user): {result2}")
                assert "Error: Document querying is not enabled for your current tier." in result2
                sys.modules['utils.analytics_tracker'].log_event.assert_not_called() # No success event if denied
                print("Test 2 Passed.")

                # Test 3: No relevant info found
                print("\n--- Test 3: No relevant info found ---")
                result3 = await query_uploaded_docs("no info", user_token=test_user_pro)
                print(f"Result 3 (Pro user, no info): {result3}")
                assert "No relevant information found in uploaded documents." in result3
                sys.modules['utils.analytics_tracker'].log_event.assert_called_once_with(test_user_pro, "query_uploaded_docs", "no_results", {"query": "no info", "section": "general"})
                sys.modules['utils.analytics_tracker'].log_event.reset_mock()
                print("Test 3 Passed.")

                # Test 4: Premium user, query and export (allowed)
                print("\n--- Test 4: Premium user, query and export ---")
                result4 = await query_uploaded_docs("report details", user_token=test_user_premium, export=True)
                print(f"Result 4 (Premium user, export requested):\\n{result4[:200]}...")
                assert "Query results exported to: /tmp/mock_export.txt" in result4
                assert "mock report" in result4
                sys.modules['utils.analytics_tracker'].log_event.assert_called_once_with(test_user_premium, "query_uploaded_docs", "exported", {"query": "report details", "section": "general", "export_path": "/tmp/mock_export.txt"})
                sys.modules['utils.analytics_tracker'].log_event.reset_mock()
                print("Test 4 Passed.")

                # Test 5: Pro user, export requested but denied by RBAC
                print("\n--- Test 5: Pro user, export requested but denied by RBAC ---")
                result5 = await query_uploaded_docs("another query", user_token=test_user_pro, export=True)
                print(f"Result 5 (Pro user, export requested):\\n{result5[:200]}...")
                assert "Warning: Export was requested but is not enabled for your current tier." in result5
                assert "Mock chunk 1" in result5 # Should still return results
                sys.modules['utils.analytics_tracker'].log_event.assert_called_once_with(test_user_pro, "query_uploaded_docs", "export_denied", {"query": "another query", "section": "general"})
                sys.modules['utils.analytics_tracker'].log_event.reset_mock()
                print("Test 5 Passed.")

                # Test 6: Admin user, full access (allowed)
                print("\n--- Test 6: Admin user, full access ---")
                result6 = await query_uploaded_docs("admin query", user_token=test_user_admin, export=True, section="finance")
                print(f"Result 6 (Admin user): {result6}")
                assert "Query results exported to: /tmp/mock_export.txt" in result6
                assert "Mock chunk 1" in result6
                sys.modules['utils.analytics_tracker'].log_event.assert_called_once_with(test_user_admin, "query_uploaded_docs", "exported", {"query": "admin query", "section": "finance", "export_path": "/tmp/mock_export.txt"})
                sys.modules['utils.analytics_tracker'].log_event.reset_mock()
                print("Test 6 Passed.")

                # Test 7: Error during vector store loading
                print("\n--- Test 7: Error during vector store loading ---")
                # Temporarily mock load_vectorstore to raise an exception
                original_load_vectorstore = load_vectorstore
                load_vectorstore_mock = MagicMock(side_effect=Exception("Simulated vector store load error"))
                sys.modules['shared_tools.vector_utils'].load_vectorstore = load_vectorstore_mock
                
                result7 = await query_uploaded_docs("error query", user_token=test_user_pro)
                print(f"Result 7 (Error loading store): {result7}")
                assert "Error: An unexpected error occurred while querying documents" in result7
                assert "Simulated vector store load error" in result7
                sys.modules['utils.analytics_tracker'].log_event.assert_called_once()
                sys.modules['utils.analytics_tracker'].log_event.reset_mock()
                # Restore original load_vectorstore
                sys.modules['shared_tools.vector_utils'].load_vectorstore = original_load_vectorstore
                print("Test 7 Passed.")

            asyncio.run(run_tests())
            print("\nAll query_uploaded_docs tests passed (mocked vector store and RBAC).")

            # Clean up mock export file if it was created
            if Path("/tmp/mock_export.txt").exists():
                Path("/tmp/mock_export.txt").unlink()
