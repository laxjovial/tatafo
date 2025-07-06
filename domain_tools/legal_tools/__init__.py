# domain_tools/legal_tools/__init__.py

import logging
from typing import Optional, Dict, Any, List

# Import individual tool functions from the legal_tool module
from .legal_tool import (
    get_legal_definition,
    search_legal_precedents,
    get_legal_aid_info,
    legal_search_web,
    legal_query_uploaded_docs # Note: This function was cut off in the provided content, assuming it exists.
)

logger = logging.getLogger(__name__)

class LegalTools:
    """
    A collection of legal-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any):
        """
        Initializes the LegalTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
        """
        self.config_manager = config_manager
        self.log_event = log_event
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
        return await legal_query_uploaded_docs(query=query, user_token=user_token, export=export, k=k)

