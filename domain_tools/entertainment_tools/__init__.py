# domain_tools/entertainment_tools/__init__.py

import logging
from typing import Any, Optional # Import Optional

from .entertainment_tool import (
    search_movies,
    search_tv_shows,
    entertainment_search_web,
    entertainment_query_uploaded_docs,
    entertainment_summarize_document_by_path
)

logger = logging.getLogger(__name__)

class EntertainmentTools:
    """
    A collection of entertainment-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: Any):
        """
        Initializes the EntertainmentTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
            document_tools (Any): The DocumentTools instance for document querying and summarization.
        """
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools
        logger.info("EntertainmentTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_token.

    async def search_movies(self, query: str, user_token: str = "default") -> str:
        """
        Searches for movie information based on a query.
        """
        return await search_movies(query=query, user_token=user_token)

    async def search_tv_shows(self, query: str, user_token: str = "default") -> str:
        """
        Searches for TV show information based on a query.
        """
        return await search_tv_shows(query=query, user_token=user_token)

    async def entertainment_search_web(self, query: str, user_token: str = "default", max_chars: int = 2000) -> str:
        """
        Searches the web for general entertainment-related information.
        """
        return await entertainment_search_web(query=query, user_token=user_token, max_chars=max_chars)

    async def entertainment_query_uploaded_docs(self, query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed entertainment documents for a user using vector similarity search.
        """
        return await self.document_tools.document_query_uploaded_docs(
            query=query,
            user_token=user_token,
            section="entertainment",
            export=export,
            k=k
        )

    async def entertainment_summarize_document_by_path(self, file_path_str: str) -> str:
        """
        Summarizes a document related to entertainment located at the given file path.
        """
        return await self.document_tools.document_summarize_document_by_path(file_path_str=file_path_str)

    def get_tools(self):
        """
        Returns a list of tool functions exposed by this class.
        """
        return [
            self.search_movies,
            self.search_tv_shows,
            self.entertainment_search_web,
            self.entertainment_query_uploaded_docs,
            self.entertainment_summarize_document_by_path
        ]
