# domain_tools/sports_tools/__init__.py

import logging
from typing import Any, Optional # Import Optional

from .sports_tool import (
    get_latest_scores,
    get_upcoming_events,
    sports_search_web,
    sports_query_uploaded_docs,
    sports_summarize_document_by_path
)

logger = logging.getLogger(__name__)

class SportsTools:
    """
    A collection of sports-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: Any):
        """
        Initializes the SportsTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
            document_tools (Any): The DocumentTools instance for document querying and summarization.
        """
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools
        logger.info("SportsTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_token.

    async def get_latest_scores(self, league: Optional[str] = None, team: Optional[str] = None, user_token: str = "default") -> str:
        """
        Retrieves the latest scores for a specified sports league or team.
        """
        return await get_latest_scores(league=league, team=team, user_token=user_token)

    async def get_upcoming_events(self, league: Optional[str] = None, team: Optional[str] = None, user_token: str = "default") -> str:
        """
        Retrieves upcoming sports events for a specified league or team.
        """
        return await get_upcoming_events(league=league, team=team, user_token=user_token)

    async def sports_search_web(self, query: str, user_token: str = "default", max_chars: int = 2000) -> str:
        """
        Searches the web for general sports-related information.
        """
        return await sports_search_web(query=query, user_token=user_token, max_chars=max_chars)

    async def sports_query_uploaded_docs(self, query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed sports documents for a user using vector similarity search.
        """
        return await self.document_tools.document_query_uploaded_docs(
            query=query,
            user_token=user_token,
            section="sports",
            export=export,
            k=k
        )

    async def sports_summarize_document_by_path(self, file_path_str: str) -> str:
        """
        Summarizes a document related to sports located at the given file path.
        """
        return await self.document_tools.document_summarize_document_by_path(file_path_str=file_path_str)

    def get_tools(self):
        """
        Returns a list of tool functions exposed by this class.
        """
        return [
            self.get_latest_scores,
            self.get_upcoming_events,
            self.sports_search_web,
            self.sports_query_uploaded_docs,
            self.sports_summarize_document_by_path
        ]


