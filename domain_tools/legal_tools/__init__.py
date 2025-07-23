# domain_tools/legal_tools/__init__.py

import logging
from typing import Any, Optional, List # Ensure List is imported

# Import individual tool functions from the legal_tool module
from .legal_tool import (
    perform_legal_research,
    legal_search_web,
    legal_query_uploaded_docs,
    legal_summarize_document_by_path
)

# Import UserProfile for type hinting
from backend.models.user_models import UserProfile # Added UserProfile import

# Import DocumentTools for type hinting in the LegalTools class
from domain_tools.document_tools.document_tool import DocumentTools


logger = logging.getLogger(__name__)

class LegalTools:
    """
    A collection of legal-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: DocumentTools):
        """
        Initializes the LegalTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
            document_tools (DocumentTools): The DocumentTools instance for document querying.
        """
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools
        logger.info("LegalTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_context.

    async def perform_legal_research(self, query: str, jurisdiction: Optional[str] = None, case_type: Optional[str] = None, user_context: Optional[UserProfile] = None, provider: str = "lexisnexis", user_api_keys: List[str] = []) -> str:
        """
        Performs legal research on a given query.
        """
        return await perform_legal_research(
            query=query,
            jurisdiction=jurisdiction,
            case_type=case_type,
            user_context=user_context,
            provider=provider,
            user_api_keys=user_api_keys
        )

    async def legal_search_web(self, query: str, user_context: Optional[UserProfile] = None, max_chars: int = 2000) -> str:
        """
        Searches the web for general legal-related information.
        """
        return await legal_search_web(
            query=query,
            user_context=user_context,
            max_chars=max_chars
        )

    async def legal_query_uploaded_docs(self, query: str, user_context: Optional[UserProfile] = None, export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed legal documents for a user using vector similarity search.
        """
        # This now calls the standalone function, passing the DocumentTools instance
        return await legal_query_uploaded_docs(
            query=query,
            user_context=user_context,
            export=export,
            k=k,
            document_tools=self.document_tools # Pass the instance
        )

    async def legal_summarize_document_by_path(self, file_path_str: str, user_context: Optional[UserProfile] = None) -> str:
        """
        Summarizes a document related to legal matters located at the given file path.
        """
        # This now calls the standalone function, passing the DocumentTools instance
        return await legal_summarize_document_by_path(
            file_path_str=file_path_str,
            user_context=user_context,
            document_tools=self.document_tools # Pass the instance
        )

    def get_tools(self):
        """
        Returns a list of tool functions exposed by this class.
        This is typically used for registering tools with an LLM agent.
        """
        # When registering with an LLM, you want the actual tool functions,
        # which are the standalone ones in legal_tool.py.
        # This assumes the LLM integration can directly use these decorated functions.
        return [
            perform_legal_research,
            legal_search_web,
            legal_query_uploaded_docs,
            legal_summarize_document_by_path
        ]
