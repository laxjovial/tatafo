# domain_tools/entertainment_tools/__init__.py

import logging
from typing import Any, Optional, List # Ensure List is imported

# Import individual tool functions from the entertainment_tool module
from .entertainment_tool import (
    search_movies,
    search_tv_shows,
    entertainment_search_web,
    entertainment_query_uploaded_docs,
    entertainment_summarize_document_by_path
)

# Import UserProfile for type hinting
from backend.models.user_models import UserProfile # Added UserProfile import

# Import DocumentTools for type hinting in the EntertainmentTools class
from domain_tools.document_tools.document_tool import DocumentTools


logger = logging.getLogger(__name__)

class EntertainmentTools:
    """
    A collection of entertainment-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: DocumentTools):
        """
        Initializes the EntertainmentTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
            document_tools (DocumentTools): The DocumentTools instance for document querying.
        """
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools
        logger.info("EntertainmentTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_context.

    async def search_movies(self, query: str, user_context: Optional[UserProfile] = None, provider: str = "tmdb", user_api_keys: List[str] = []) -> str:
        """
        Searches for movies.
        """
        return await search_movies(
            query=query,
            user_context=user_context,
            provider=provider,
            user_api_keys=user_api_keys
        )

    async def search_tv_shows(self, query: str, user_context: Optional[UserProfile] = None, provider: str = "tmdb", user_api_keys: List[str] = []) -> str:
        """
        Searches for TV shows.
        """
        return await search_tv_shows(
            query=query,
            user_context=user_context,
            provider=provider,
            user_api_keys=user_api_keys
        )

    async def entertainment_search_web(self, query: str, user_context: Optional[UserProfile] = None, max_chars: int = 2000) -> str:
        """
        Searches the web for general entertainment-related information.
        """
        return await entertainment_search_web(
            query=query,
            user_context=user_context,
            max_chars=max_chars
        )

    async def entertainment_query_uploaded_docs(self, query: str, user_context: Optional[UserProfile] = None, export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed entertainment documents for a user using vector similarity search.
        """
        # This now calls the standalone function, passing the DocumentTools instance
        return await entertainment_query_uploaded_docs(
            query=query,
            user_context=user_context,
            export=export,
            k=k,
            document_tools=self.document_tools # Pass the instance
        )

    async def entertainment_summarize_document_by_path(self, file_path_str: str, user_context: Optional[UserProfile] = None) -> str:
        """
        Summarizes a document related to entertainment located at the given file path.
        """
        # This now calls the standalone function, passing the DocumentTools instance
        return await entertainment_summarize_document_by_path(
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
        # which are the standalone ones in entertainment_tool.py.
        # This assumes the LLM integration can directly use these decorated functions.
        return [
            search_movies,
            search_tv_shows,
            entertainment_search_web,
            entertainment_query_uploaded_docs,
            entertainment_summarize_document_by_path
        ]
