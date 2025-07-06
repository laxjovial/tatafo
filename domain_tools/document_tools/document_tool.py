# domain_tools/document_tools/document_tool.py

import logging
from typing import Optional, Dict, Any, List
import asyncio
import base64 # Needed for file content handling if process_uploaded_document is a tool
from pathlib import Path

from langchain_core.tools import tool

# Import the VectorUtilsWrapper class
from shared_tools.vector_utils import VectorUtilsWrapper

# Import config_manager to access configurations
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability
# Import analytics_tracker
from utils import analytics_tracker # Import the module

# Import generic tools that DocumentTools might wrap
from shared_tools.scraper_tool import scrape_web
from shared_tools.doc_summarizer import summarize_document


logger = logging.getLogger(__name__)

class DocumentTools:
    """
    A collection of tools for interacting with user-uploaded documents,
    including querying, uploading/processing, summarizing, and managing them.
    """
    def __init__(
        self,
        vector_utils_wrapper: VectorUtilsWrapper,
        config_manager: Any,
        firestore_manager: Any, # FirestoreManager instance for process_uploaded_document tool
        cloud_storage_utils: Any, # CloudStorageUtilsWrapper instance for process_uploaded_document tool
        log_event_func: Any # log_event function for process_uploaded_document tool
    ):
        # Store the instantiated VectorUtilsWrapper and other managers
        self.vector_utils_wrapper = vector_utils_wrapper
        self.config_manager = config_manager
        self.firestore_manager = firestore_manager
        self.cloud_storage_utils = cloud_storage_utils
        self.log_event_func = log_event_func
        logger.info("DocumentTools initialized with VectorUtilsWrapper and other managers.")

    @tool
    async def document_query_uploaded_docs(
        self,
        query: str,
        user_token: str = "default",
        export: Optional[bool] = False,
        k: int = 5
    ) -> str:
        """
        Queries previously uploaded and indexed documents for a user using vector similarity search.
        This tool allows searching through documents that the user has uploaded to the system.
        It leverages a vector database for efficient retrieval of relevant information.

        Args:
            query (str): The search query to find relevant information in uploaded documents.
            user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                        Used for RBAC capability checks and to identify user's documents.
            export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
            k (int): The number of top relevant documents or chunks to retrieve. Defaults to 5.

        Returns:
            str: A string containing the combined content of the relevant document chunks,
                 or a message indicating no data/results found, or the export path if exported.
        """
        logger.info(f"Tool: document_query_uploaded_docs called with query: '{query}' for user: '{user_token}'")

        # RBAC check for document query capability
        if not get_user_tier_capability(user_token, 'document_query_enabled', False):
            error_msg = "Error: Access to document query tools is not enabled for your current tier."
            await analytics_tracker.log_tool_usage(
                tool_name="document_query_uploaded_docs",
                tool_params={"query": query, "export": export, "k": k},
                user_token=user_token,
                success=False,
                error_message=error_msg
            )
            return error_msg

        # Get max results allowed by user's tier if 'k' is not explicitly set or exceeds limit
        max_results_allowed = get_user_tier_capability(user_token, 'document_query_max_results_k', 4)
        if k > max_results_allowed:
            logger.warning(f"User {user_token} requested {k} results, but tier limits to {max_results_allowed}. Adjusting k.")
            k = max_results_allowed

        try:
            # Call the method on the stored VectorUtilsWrapper instance
            result = await self.vector_utils_wrapper.query_uploaded_docs(
                query_text=query,
                user_token=user_token,
                export=export,
                k=k
            )
            
            # Log successful tool usage
            await analytics_tracker.log_tool_usage(
                tool_name="document_query_uploaded_docs",
                tool_params={"query": query, "export": export, "k": k},
                user_token=user_token,
                success=True
            )
            return result
        except Exception as e:
            error_msg = f"An error occurred while querying uploaded documents: {e}"
            logger.error(error_msg, exc_info=True)
            await analytics_tracker.log_tool_usage(
                tool_name="document_query_uploaded_docs",
                tool_params={"query": query, "export": export, "k": k},
                user_token=user_token,
                success=False,
                error_message=error_msg
            )
            return f"Error: {error_msg}. Please try again later."

    @tool
    async def document_process_uploaded_document(
        self,
        file_name: str,
        file_content_base64: str,
        user_token: str = "default"
    ) -> str:
        """
        Uploads a document to cloud storage and processes it for vector indexing,
        making it searchable via document_query_uploaded_docs.

        Args:
            file_name (str): The original name of the file (e.g., "my_report.pdf").
            file_content_base64 (str): The base64 encoded content of the file.
            user_token (str): The unique identifier for the user. Defaults to "default".

        Returns:
            str: A message indicating the success or failure of the document processing,
                 including the document ID if successful.
        """
        logger.info(f"Tool: document_process_uploaded_document called for file: '{file_name}' by user: '{user_token}'")

        # RBAC check for document upload capability
        if not get_user_tier_capability(user_token, 'document_upload_enabled', False):
            error_msg = "Error: Access to document upload tools is not enabled for your current tier."
            await analytics_tracker.log_tool_usage(
                tool_name="document_process_uploaded_document",
                tool_params={"file_name": file_name},
                user_token=user_token,
                success=False,
                error_message=error_msg
            )
            return error_msg

        try:
            # Call the method on the stored VectorUtilsWrapper instance, passing all required managers
            result = await self.vector_utils_wrapper.process_uploaded_document(
                user_id=user_token, # user_token maps to user_id in vector_utils
                file_name=file_name,
                file_content_base64=file_content_base64,
                firestore_manager=self.firestore_manager,
                cloud_storage_utils=self.cloud_storage_utils,
                config_manager=self.config_manager,
                log_event_func=self.log_event_func
            )

            if result["success"]:
                success_msg = f"Document '{file_name}' uploaded and indexed successfully with ID: {result.get('document_id')}."
                await analytics_tracker.log_tool_usage(
                    tool_name="document_process_uploaded_document",
                    tool_params={"file_name": file_name},
                    user_token=user_token,
                    success=True
                )
                return success_msg
            else:
                error_msg = f"Failed to process document '{file_name}': {result.get('message', 'Unknown error.')}"
                await analytics_tracker.log_tool_usage(
                    tool_name="document_process_uploaded_document",
                    tool_params={"file_name": file_name},
                    user_token=user_token,
                    success=False,
                    error_message=error_msg
                )
                return error_msg
        except Exception as e:
            error_msg = f"An unexpected error occurred while processing document '{file_name}': {e}"
            logger.error(error_msg, exc_info=True)
            await analytics_tracker.log_tool_usage(
                tool_name="document_process_uploaded_document",
                tool_params={"file_name": file_name},
                user_token=user_token,
                success=False,
                error_message=error_msg
            )
            return f"Error processing document: {e}"

    @tool
    async def document_summarize_document_by_path(self, file_path_str: str, user_token: str = "default") -> str:
        """
        Summarizes the content of a document located at the given file path.
        The file path should be accessible by the system (e.g., in the 'uploads' directory).
        This tool wraps the generic `summarize_document` tool.

        Args:
            file_path_str (str): The full path to the document file to be summarized.
                                 Example: "uploads/default/document/my_report.pdf"
            user_token (str): The unique identifier for the user. Defaults to "default".

        Returns:
            str: A concise summary of the document content.
        """
        logger.info(f"Tool: document_summarize_document_by_path called for file: '{file_path_str}' by user: '{user_token}'")

        # RBAC check for summarization capability
        if not get_user_tier_capability(user_token, 'summarization_enabled', False):
            error_msg = "Error: Access to document summarization tools is not enabled for your current tier."
            await analytics_tracker.log_tool_usage(
                tool_name="document_summarize_document_by_path",
                tool_params={"file_path_str": file_path_str},
                user_token=user_token,
                success=False,
                error_message=error_msg
            )
            return error_msg

        try:
            # The summarize_document tool expects a Path object
            file_path = Path(file_path_str)
            summary = await summarize_document(file_path)

            await analytics_tracker.log_tool_usage(
                tool_name="document_summarize_document_by_path",
                tool_params={"file_path_str": file_path_str},
                user_token=user_token,
                success=True
            )
            return summary
        except Exception as e:
            error_msg = f"An error occurred during document summarization: {e}"
            logger.error(error_msg, exc_info=True)
            await analytics_tracker.log_tool_usage(
                tool_name="document_summarize_document_by_path",
                tool_params={"file_path_str": file_path_str},
                user_token=user_token,
                success=False,
                error_message=error_msg
            )
            return f"Error summarizing document: {e}"

    @tool
    async def document_search_web(self, query: str, user_token: str = "default", max_chars: int = 2000) -> str:
        """
        Searches the web for general document-related information using a smart search fallback mechanism.
        This tool wraps the generic `scrape_web` tool, providing a document-specific interface.

        Args:
            query (str): The document-related search query (e.g., "best practices for data privacy documents", "history of legal documents").
            user_token (str): The unique identifier for the user. Defaults to "default".
            max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.

        Returns:
            str: A string containing relevant information from the web, or an error message.
        """
        logger.info(f"Tool: document_search_web called with query: '{query}' for user: '{user_token}'")

        # RBAC check for web search capability (generic web search, not specific to document domain)
        if not get_user_tier_capability(user_token, 'web_search_enabled', False):
            error_msg = "Error: Access to web search tools is not enabled for your current tier."
            await analytics_tracker.log_tool_usage(
                tool_name="document_search_web",
                tool_params={"query": query, "max_chars": max_chars},
                user_token=user_token,
                success=False,
                error_message=error_msg
            )
            return error_msg

        try:
            result = await scrape_web(query=query, user_token=user_token, max_chars=max_chars)
            await analytics_tracker.log_tool_usage(
                tool_name="document_search_web",
                tool_params={"query": query, "max_chars": max_chars},
                user_token=user_token,
                success=True
            )
            return result
        except Exception as e:
            error_msg = f"An error occurred during web search: {e}"
            logger.error(error_msg, exc_info=True)
            await analytics_tracker.log_tool_usage(
                tool_name="document_search_web",
                tool_params={"query": query, "max_chars": max_chars},
                user_token=user_token,
                success=False,
                error_message=error_msg
            )
            return f"Error performing web search: {e}"

# CLI Test (optional)
if __name__ == "__main__":
    from unittest.mock import MagicMock, AsyncMock, patch
    import sys
    import shutil
    from pathlib import Path
    import os
    from datetime import datetime, timezone

    # Mock config_manager and analytics_tracker for testing context
    class MockConfigManager:
        def get(self, key, default=None):
            if key == "analytics.log_tool_usage": return True
            if key == "web_scraping.timeout_seconds": return 5
            if key == "web_scraping.max_search_results": return 5
            if key == "api_configs": return [] # No external search APIs for this test
            if key == "cloud_storage_bucket_name": return "mock-test-bucket"
            return default
        def get_secret(self, key):
            if key == "gcs_bucket_name": return "mock-test-bucket"
            return "mock_api_key"

    class MockUserManager:
        _mock_users = {
            "test_user_pro": {"user_id": "test_user_pro", "username": "ProUser", "email": "pro@example.com", "tier": "pro", "roles": ["user"]},
            "test_user_free": {"user_id": "test_user_free", "username": "FreeUser", "email": "free@example.com", "tier": "free", "roles": ["user"]},
        }
        _rbac_capabilities = {
            'capabilities': {
                'document_query_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
                'document_query_max_results_k': {'default': 4, 'tiers': {'pro': 10, 'premium': 20}},
                'document_upload_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
                'summarization_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
                'web_search_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
                'web_search_max_chars': {'default': 500, 'tiers': {'pro': 1000, 'premium': 2000}},
                'web_search_max_results': {'default': 2, 'tiers': {'pro': 5, 'premium': 10}},
            }
        }

        def get_user_tier_capability(self, user_token: Optional[str], capability_key: str, default_value: Any = None) -> Any:
            user_info = self._mock_users.get(user_token, {})
            user_tier = user_info.get('tier', 'free')
            user_roles = user_info.get('roles', [])

            capability_config = self._rbac_capabilities['capabilities'].get(capability_key, {})

            # Check role-based access first
            for role in user_roles:
                if capability_config.get('roles', {}).get(role, False):
                    return True # Role grants access

            # Check tier-based access
            if user_tier in capability_config.get('tiers', {}):
                return capability_config['tiers'][user_tier]

            return capability_config.get('default', default_value)

    # Patch the actual imports for testing
    original_get_user_tier_capability = sys.modules['utils.user_manager'].get_user_tier_capability
    sys.modules['utils.user_manager'].get_user_tier_capability = MockUserManager().get_user_tier_capability

    original_config_manager_instance = config_manager
    sys.modules['config.config_manager'].config_manager = MockConfigManager()

    original_analytics_tracker_log_tool_usage = analytics_tracker.log_tool_usage
    mock_analytics_tracker_log_tool_usage = AsyncMock()
    analytics_tracker.log_tool_usage = mock_analytics_tracker_log_tool_usage

    # Mock external dependencies for tools
    class MockVectorUtilsWrapper:
        def __init__(self, firestore_manager, cloud_storage_utils, config_manager):
            self.firestore_manager = firestore_manager
            self.cloud_storage_utils = cloud_storage_utils
            self.config_manager = config_manager
            self.mock_indexed_docs = {} # Simulate indexed documents

        async def query_uploaded_docs(self, query_text: str, user_token: str, export: bool, k: int) -> str:
            if "test document" in query_text.lower() and user_token == "test_user_pro":
                return f"Mocked document query result for '{query_text}' (k={k}). This is a test document about AI and machine learning."
            if "no results" in query_text.lower():
                return "No relevant information found in uploaded documents."
            return f"Mocked document query for '{query_text}'. Found some general info."

        async def process_uploaded_document(self, user_id: str, file_name: str, file_content_base64: str, firestore_manager: Any, cloud_storage_utils: Any, config_manager: Any, log_event_func: Any) -> Dict[str, Any]:
            if "fail" in file_name:
                return {"success": False, "message": "Simulated processing failure."}
            
            doc_id = f"mock_doc_{len(self.mock_indexed_docs) + 1}"
            self.mock_indexed_docs[doc_id] = {
                "user_id": user_id,
                "file_name": file_name,
                "content_preview": file_content_base64[:50],
                "indexed_at": datetime.now(timezone.utc).isoformat()
            }
            return {"success": True, "message": "Mocked processing success.", "document_id": doc_id}


    class MockSummarizeDocument:
        async def __call__(self, file_path: Path) -> str:
            if "empty.txt" in file_path.name:
                return "Mocked summary: The document is empty."
            return f"Mocked summary of {file_path.name}: This document discusses various aspects of {file_path.stem.replace('_', ' ')}."

    class MockScrapeWeb:
        async def __call__(self, query: str, user_token: str, max_chars: int) -> str:
            if "web search success" in query.lower():
                return f"Search results for '{query}': This is a mocked web page content up to {max_chars} characters."
            return "No relevant information found on the web."

    # Mock FirestoreManager and CloudStorageUtils for the DocumentTools constructor
    class MockFirestoreManagerForDT:
        async def add_document_metadata(self, *args, **kwargs): return MagicMock(id="mock_doc_id")
        async def update_document_metadata(self, *args, **kwargs): return True
        async def get_document_metadata(self, *args, **kwargs): return {}
        async def get_analytics_events(self, *args, **kwargs): return []

    class MockCloudStorageUtilsForDT:
        async def upload_file_to_storage(self, *args, **kwargs): return {"success": True, "file_url": "mock_url"}
        async def delete_file_from_storage(self, *args, **kwargs): return {"success": True}
        async def download_file_from_storage(self, *args, **kwargs): return {"success": True, "content": "mock_content"}
        async def read_file_from_storage_to_bytes(self, *args, **kwargs): return {"success": True, "content": b"mock_bytes"}

    # Patch the generic tools that DocumentTools wraps
    original_summarize_document = sys.modules['shared_tools.doc_summarizer'].summarize_document
    original_scrape_web = sys.modules['shared_tools.scraper_tool'].scrape_web

    sys.modules['shared_tools.doc_summarizer'].summarize_document = MockSummarizeDocument()
    sys.modules['shared_tools.scraper_tool'].scrape_web = MockScrapeWeb()

    async def run_document_tools_tests():
        print("\n--- Testing DocumentTools ---")
        test_user_pro = "test_user_pro"
        test_user_free = "test_user_free"

        # Instantiate mocks for DocumentTools constructor
        mock_firestore_for_dt = MockFirestoreManagerForDT()
        mock_cloud_storage_for_dt = MockCloudStorageUtilsForDT()
        mock_config_for_dt = MockConfigManager()
        mock_log_event_for_dt = mock_analytics_tracker_log_tool_usage # Use the patched analytics logger

        # Instantiate MockVectorUtilsWrapper with its own internal mocks
        mock_vector_utils = MockVectorUtilsWrapper(
            firestore_manager=mock_firestore_for_dt,
            cloud_storage_utils=mock_cloud_storage_for_dt,
            config_manager=mock_config_for_dt
        )

        document_tools = DocumentTools(
            vector_utils_wrapper=mock_vector_utils,
            config_manager=mock_config_for_dt,
            firestore_manager=mock_firestore_for_dt,
            cloud_storage_utils=mock_cloud_storage_for_dt,
            log_event_func=mock_log_event_for_dt
        )

        # Test 1: document_query_uploaded_docs (Success for Pro User)
        print("\n--- Test 1: document_query_uploaded_docs (Success) ---")
        mock_analytics_tracker_log_tool_usage.reset_mock()
        result_query = await document_tools.document_query_uploaded_docs(
            query="test document",
            user_token=test_user_pro,
            k=5
        )
        print(f"Query Result: {result_query}")
        assert "Mocked document query result for 'test document' (k=5)." in result_query
        mock_analytics_tracker_log_tool_usage.assert_called_once_with(
            tool_name="document_query_uploaded_docs",
            tool_params={"query": "test document", "export": False, "k": 5},
            user_token=test_user_pro,
            success=True
        )
        mock_analytics_tracker_log_tool_usage.reset_mock()
        print("Test 1 Passed.")

        # Test 2: document_query_uploaded_docs (RBAC Denied for Free User)
        print("\n--- Test 2: document_query_uploaded_docs (RBAC Denied) ---")
        result_query_rbac_denied = await document_tools.document_query_uploaded_docs(
            query="sensitive data",
            user_token=test_user_free,
            k=5
        )
        print(f"Query Result (RBAC Denied): {result_query_rbac_denied}")
        assert "Error: Access to document query tools is not enabled for your current tier." in result_query_rbac_denied
        mock_analytics_tracker_log_tool_usage.assert_called_once_with(
            tool_name="document_query_uploaded_docs",
            tool_params={"query": "sensitive data", "export": False, "k": 5},
            user_token=test_user_free,
            success=False,
            error_message="Error: Access to document query tools is not enabled for your current tier."
        )
        mock_analytics_tracker_log_tool_usage.reset_mock()
        print("Test 2 Passed.")

        # Test 3: document_query_uploaded_docs (k limit by tier)
        print("\n--- Test 3: document_query_uploaded_docs (k limit) ---")
        result_query_k_limit = await document_tools.document_query_uploaded_docs(
            query="another document",
            user_token=test_user_pro,
            k=15 # Pro tier limit is 10
        )
        print(f"Query Result (k limit): {result_query_k_limit}")
        assert "Mocked document query for 'another document' (k=10)." in result_query_k_limit # Should be 10, not 15
        mock_analytics_tracker_log_tool_usage.assert_called_once_with(
            tool_name="document_query_uploaded_docs",
            tool_params={"query": "another document", "export": False, "k": 10}, # Logged k should be 10
            user_token=test_user_pro,
            success=True
        )
        mock_analytics_tracker_log_tool_usage.reset_mock()
        print("Test 3 Passed.")

        # Test 4: document_process_uploaded_document (Success)
        print("\n--- Test 4: document_process_uploaded_document (Success) ---")
        mock_analytics_tracker_log_tool_usage.reset_mock()
        test_file_name = "report.pdf"
        test_file_content_base64 = base64.b64encode(b"This is a dummy PDF content.").decode('utf-8')
        result_upload = await document_tools.document_process_uploaded_document(
            file_name=test_file_name,
            file_content_base64=test_file_content_base64,
            user_token=test_user_pro
        )
        print(f"Upload Result: {result_upload}")
        assert "Document 'report.pdf' uploaded and indexed successfully with ID: mock_doc_1." in result_upload
        mock_analytics_tracker_log_tool_usage.assert_called_once_with(
            tool_name="document_process_uploaded_document",
            tool_params={"file_name": test_file_name},
            user_token=test_user_pro,
            success=True
        )
        mock_analytics_tracker_log_tool_usage.reset_mock()
        print("Test 4 Passed.")

        # Test 5: document_process_uploaded_document (RBAC Denied)
        print("\n--- Test 5: document_process_uploaded_document (RBAC Denied) ---")
        mock_analytics_tracker_log_tool_usage.reset_mock()
        result_upload_rbac_denied = await document_tools.document_process_uploaded_document(
            file_name="secret.docx",
            file_content_base64=base64.b64encode(b"Secret content.").decode('utf-8'),
            user_token=test_user_free
        )
        print(f"Upload Result (RBAC Denied): {result_upload_rbac_denied}")
        assert "Error: Access to document upload tools is not enabled for your current tier." in result_upload_rbac_denied
        mock_analytics_tracker_log_tool_usage.assert_called_once_with(
            tool_name="document_process_uploaded_document",
            tool_params={"file_name": "secret.docx"},
            user_token=test_user_free,
            success=False,
            error_message="Error: Access to document upload tools is not enabled for your current tier."
        )
        mock_analytics_tracker_log_tool_usage.reset_mock()
        print("Test 5 Passed.")

        # Test 6: document_summarize_document_by_path (Success)
        print("\n--- Test 6: document_summarize_document_by_path (Success) ---")
        mock_analytics_tracker_log_tool_usage.reset_mock()
        # Create a dummy file for summarization test
        dummy_file_path = Path("uploads") / test_user_pro / "document" / "test_report.txt"
        dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_file_path.write_text("This is a dummy test report content for summarization.")

        result_summarize = await document_tools.document_summarize_document_by_path(
            file_path_str=str(dummy_file_path),
            user_token=test_user_pro
        )
        print(f"Summarize Result: {result_summarize}")
        assert "Mocked summary of test_report.txt" in result_summarize
        mock_analytics_tracker_log_tool_usage.assert_called_once_with(
            tool_name="document_summarize_document_by_path",
            tool_params={"file_path_str": str(dummy_file_path)},
            user_token=test_user_pro,
            success=True
        )
        mock_analytics_tracker_log_tool_usage.reset_mock()
        print("Test 6 Passed.")
        if dummy_file_path.exists():
            os.remove(dummy_file_path) # Clean up dummy file

        # Test 7: document_summarize_document_by_path (RBAC Denied)
        print("\n--- Test 7: document_summarize_document_by_path (RBAC Denied) ---")
        mock_analytics_tracker_log_tool_usage.reset_mock()
        result_summarize_rbac_denied = await document_tools.document_summarize_document_by_path(
            file_path_str="uploads/test_user_free/document/secret_doc.pdf",
            user_token=test_user_free
        )
        print(f"Summarize Result (RBAC Denied): {result_summarize_rbac_denied}")
        assert "Error: Access to document summarization tools is not enabled for your current tier." in result_summarize_rbac_denied
        mock_analytics_tracker_log_tool_usage.assert_called_once_with(
            tool_name="document_summarize_document_by_path",
            tool_params={"file_path_str": "uploads/test_user_free/document/secret_doc.pdf"},
            user_token=test_user_free,
            success=False,
            error_message="Error: Access to document summarization tools is not enabled for your current tier."
        )
        mock_analytics_tracker_log_tool_usage.reset_mock()
        print("Test 7 Passed.")

        # Test 8: document_search_web (Success)
        print("\n--- Test 8: document_search_web (Success) ---")
        mock_analytics_tracker_log_tool_usage.reset_mock()
        result_web_search = await document_tools.document_search_web(
            query="web search success",
            user_token=test_user_pro,
            max_chars=500
        )
        print(f"Web Search Result: {result_web_search}")
        assert "Search results for 'web search success': This is a mocked web page content up to 500 characters." in result_web_search
        mock_analytics_tracker_log_tool_usage.assert_called_once_with(
            tool_name="document_search_web",
            tool_params={"query": "web search success", "max_chars": 500},
            user_token=test_user_pro,
            success=True
        )
        mock_analytics_tracker_log_tool_usage.reset_mock()
        print("Test 8 Passed.")

        # Test 9: document_search_web (RBAC Denied)
        print("\n--- Test 9: document_search_web (RBAC Denied) ---")
        mock_analytics_tracker_log_tool_usage.reset_mock()
        result_web_search_rbac_denied = await document_tools.document_search_web(
            query="public records",
            user_token=test_user_free,
            max_chars=500
        )
        print(f"Web Search Result (RBAC Denied): {result_web_search_rbac_denied}")
        assert "Error: Access to web search tools is not enabled for your current tier." in result_web_search_rbac_denied
        mock_analytics_tracker_log_tool_usage.assert_called_once_with(
            tool_name="document_search_web",
            tool_params={"query": "public records", "max_chars": 500},
            user_token=test_user_free,
            success=False,
            error_message="Error: Access to web search tools is not enabled for your current tier."
        )
        mock_analytics_tracker_log_tool_usage.reset_mock()
        print("Test 9 Passed.")

        print("\nAll DocumentTools tests completed.")

    finally:
        # Restore original imports
        sys.modules['utils.user_manager'].get_user_tier_capability = original_get_user_tier_capability
        sys.modules['config.config_manager'].config_manager = original_config_manager_instance
        analytics_tracker.log_tool_usage = original_analytics_tracker_log_tool_usage
        sys.modules['shared_tools.doc_summarizer'].summarize_document = original_summarize_document
        sys.modules['shared_tools.scraper_tool'].scrape_web = original_scrape_web

        # Clean up dummy directories if they were created by tests
        test_user_pro_uploads_dir = Path("uploads") / "test_user_pro"
        if test_user_pro_uploads_dir.exists():
            shutil.rmtree(test_user_pro_uploads_dir, ignore_errors=True)
            logger.info(f"Cleaned up {test_user_pro_uploads_dir}")

    if __name__ == "__main__":
        asyncio.run(run_document_tools_tests())
