# domain_tools/legal_tools/__init__.py

import logging
from typing import Any, Optional

from .legal_tool import (
    perform_legal_research,
    legal_search_web,
    legal_query_uploaded_docs, # This will be wrapped by DocumentTools
    legal_summarize_document_by_path # This will be wrapped by DocumentTools
)

logger = logging.getLogger(__name__)

class LegalTools:
    """
    A collection of legal-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: Any):
        """
        Initializes the LegalTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
            document_tools (Any): The DocumentTools instance for document querying and summarization.
        """
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools
        logger.info("LegalTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_token.

    async def perform_legal_research(self, query: str, jurisdiction: Optional[str] = None, case_type: Optional[str] = None, user_token: str = "default") -> str:
        """
        Performs legal research based on a query, optionally filtered by jurisdiction and case type.
        """
        return await perform_legal_research(query=query, jurisdiction=jurisdiction, case_type=case_type, user_token=user_token)

    async def legal_search_web(self, query: str, user_token: str = "default", max_chars: int = 2000) -> str:
        """
        Searches the web for general legal-related information.
        """
        return await legal_search_web(query=query, user_token=user_token, max_chars=max_chars)

    async def legal_query_uploaded_docs(self, query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed legal documents for a user using vector similarity search.
        """
        # This now calls the DocumentTools instance
        return await self.document_tools.document_query_uploaded_docs(
            query=query,
            user_token=user_token,
            section="legal", # Specific collection for legal documents
            export=export,
            k=k
        )

    async def legal_summarize_document_by_path(self, file_path_str: str) -> str:
        """
        Summarizes a document related to legal matters located at the given file path.
        """
        # This now calls the DocumentTools instance
        return await self.document_tools.document_summarize_document_by_path(file_path_str=file_path_str)

    def get_tools(self):
        """
        Returns a list of tool functions exposed by this class.
        """
        return [
            self.perform_legal_research,
            self.legal_search_web,
            self.legal_query_uploaded_docs,
            self.legal_summarize_document_by_path
        ]
