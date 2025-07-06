# domain_tools/entertainment_tools/__init__.py

import logging
from typing import Optional, Dict, Any, List

# Import individual tool functions from the entertainment_tool module
from .entertainment_tool import (
    get_movie_info,
    get_tv_show_info,
    search_upcoming_entertainment_events,
    entertainment_search_web, # Added
    entertainment_query_uploaded_docs, # Added
    entertainment_summarize_document_by_path # Added
)

logger = logging.getLogger(__name__)

class EntertainmentTools:
    """
    A collection of entertainment-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: Any): # Added document_tools
        """
        Initializes the EntertainmentTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
            document_tools (Any): The DocumentTools instance for document querying. # Added
        """
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools # Stored
        logger.info("EntertainmentTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_token.

    async def get_movie_info(self, title: str, year: Optional[int] = None, user_token: str = "default") -> str:
        """
        Retrieves information about a movie, including its director, cast, plot, and IMDb rating.
        """
        return await get_movie_info(title=title, year=year, user_token=user_token)

    async def get_tv_show_info(self, title: str, user_token: str = "default") -> str:
        """
        Retrieves information about a TV show, including its creator, plot, and IMDb rating.
        """
        return await get_tv_show_info(title=title, user_token=user_token)

    async def search_upcoming_entertainment_events(self, event_type: Optional[str] = None, location: Optional[str] = None, date: Optional[str] = None, user_token: str = "default") -> str:
        """
        Searches for upcoming entertainment events (e.g., music concerts, festivals, conventions)
        optionally filtered by type, location, or date.
        """
        return await search_upcoming_entertainment_events(event_type=event_type, location=location, date=date, user_token=user_token)

    async def entertainment_search_web(self, query: str, user_token: str = "default", max_chars: int = 2000) -> str:
        """
        Searches the web for general entertainment information using a smart search fallback mechanism.
        """
        return await entertainment_search_web(query=query, user_token=user_token, max_chars=max_chars)

    async def entertainment_query_uploaded_docs(self, query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed entertainment documents for a user.
        """
        # This now calls the DocumentTools instance
        return await self.document_tools.query_uploaded_docs(
            query_text=query,
            user_token=user_token,
            collection_name="entertainment", # Specific collection for entertainment documents
            export=export,
            k=k
        )

    async def entertainment_summarize_document_by_path(self, file_path_str: str) -> str:
        """
        Summarizes a document related to entertainment (e.g., movie reviews, script excerpts) located at the given file path.
        """
        return await entertainment_summarize_document_by_path(file_path_str=file_path_str) # Call the function from entertainment_tool.py

