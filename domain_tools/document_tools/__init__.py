# domain_tools/document_tools/__init__.py

import logging
from typing import Optional, Dict, Any

# Import the internal tool functions from the document_tool module
from .document_tool import (
    query_uploaded_docs_internal,
    # Add other document-related internal functions here if they become tools
)

# Import the tool decorator from langchain_core
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

class DocumentTools:
    """
    A collection of document-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application,
    allowing dependency injection of core managers.
    """
    def __init__(self, vector_utils_wrapper: Any, firestore_manager: Any, cloud_storage_utils: Any, config_manager: Any, log_event: Any):
        """
        Initializes the DocumentTools with necessary dependencies.

        Args:
            vector_utils_wrapper (Any): The VectorUtilsWrapper instance from main.py.
                                        This provides access to the underlying vector_utils_module functions.
            firestore_manager (Any): The FirestoreManager instance.
            cloud_storage_utils (Any): The CloudStorageUtilsWrapper instance.
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
        """
        self.vector_utils_wrapper = vector_utils_wrapper
        self.firestore_manager = firestore_manager
        self.cloud_storage_utils = cloud_storage_utils
        self.config_manager = config_manager
        self.log_event = log_event
        logger.info("DocumentTools initialized.")

    @tool
    async def query_uploaded_docs(self, query_text: str, user_token: str = "default", collection_name: Optional[str] = None, k: int = 5) -> str:
        """
        Queries the user's uploaded documents for relevant information based on a natural language query.
        This tool is useful for finding specific details within documents that have been previously uploaded
        and processed for vector search.

        Args:
            query_text (str): The natural language query to search for in the documents.
            user_token (str, optional): The unique identifier for the user. Defaults to "default".
            collection_name (str, optional): The specific collection of documents to query. If not provided,
                                             the user's default collection will be used.
            k (int, optional): The number of top relevant document chunks to retrieve. Defaults to 5.

        Returns:
            str: A formatted string containing the most relevant document chunks and their sources,
                 or a message indicating that no relevant information was found.
        """
        logger.info(f"Tool: query_uploaded_docs called for user {user_token} with query: '{query_text}'")

        if not get_user_tier_capability(user_token, 'document_query_enabled', False):
            return "Error: Document querying is not enabled for your current tier."
        
        # Call the internal function, passing all required dependencies
        return await query_uploaded_docs_internal(
            query_text=query_text,
            user_id=user_token, # user_token is used as user_id for analytics and internal functions
            firestore_manager=self.firestore_manager,
            cloud_storage_utils=self.cloud_storage_utils,
            config_manager_instance=self.config_manager,
            log_event_func=self.log_event,
            collection_name=collection_name,
            k=k
        )

    # You can add more document-related tools here as methods of this class
    # For example, a tool to summarize a specific document, or list uploaded documents.

