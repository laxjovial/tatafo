# domain_tools/legal_tools/__init__.py

import logging
from typing import Optional, Dict, Any, List

# Import individual tool functions from the legal_tool module
from .legal_tool import (
    get_legal_definition,
    search_legal_precedents,
    get_legal_aid_info,
    legal_search_web,
    legal_query_uploaded_docs,
    legal_summarize_document_by_path # Added
)

logger = logging.getLogger(__name__)

class LegalTools:
    """
    A collection of legal-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: Any): # Added document_tools
        """
        Initializes the LegalTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
            document_tools (Any): The DocumentTools instance for document querying. # Added
        """
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools # Stored
        logger.info("LegalTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_token.

    async def get_legal_definition(self, term: str, user_token: str = "default") -> str:
        """
        Retrieves the legal definition and key elements of a specific legal term.
        """
        return await get_legal_definition(term=term, user_token=user_token)

    async def search_legal_precedents(self, query: str, user_token: str = "default") -> str:
        """
        Searches for significant legal precedents or case law relevant to a query.
        """
        return await search_legal_precedents(query=query, user_token=user_token)

    async def get_legal_aid_info(self, location: str, user_token: str = "default") -> str:
        """
        Retrieves information about legal aid organizations or services available in a specific location.
        """
        return await get_legal_aid_info(location=location, user_token=user_token)

    async def legal_search_web(self, query: str, user_token: str = "default", max_chars: int = 2000) -> str:
        """
        Searches the web for general legal information using a smart search fallback mechanism.
        """
        return await legal_search_web(query=query, user_token=user_token, max_chars=max_chars)

    async def legal_query_uploaded_docs(self, query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed legal documents for a user.
        """
        # This now calls the DocumentTools instance
        return await self.document_tools.query_uploaded_docs(
            query_text=query,
            user_token=user_token,
            collection_name="legal", # Specific collection for legal documents
            export=export,
            k=k
        )

    async def legal_summarize_document_by_path(self, file_path_str: str) -> str:
        """
        Summarizes a document related to legal matters located at the given file path.
        """
        return await legal_summarize_document_by_path(file_path_str=file_path_str) # Call the function from legal_tool.py

