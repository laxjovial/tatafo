# domain_tools/travel_tools/__init__.py

import logging
from typing import Any, Optional # Import Optional

from .travel_tool import (
    search_flights,
    search_hotels,
    get_destination_info,
    travel_search_web,
    travel_query_uploaded_docs,
    travel_summarize_document_by_path
)

logger = logging.getLogger(__name__)

class TravelTools:
    """
    A collection of travel-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: Any):
        """
        Initializes the TravelTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
            document_tools (Any): The DocumentTools instance for document querying and summarization.
        """
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools
        logger.info("TravelTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_token.

    async def search_flights(self, origin: str, destination: str, departure_date: str, return_date: Optional[str] = None, user_token: str = "default") -> str:
        """
        Searches for flight information between two destinations.
        """
        return await search_flights(origin=origin, destination=destination, departure_date=departure_date, return_date=return_date, user_token=user_token)

    async def search_hotels(self, location: str, check_in_date: str, check_out_date: str, user_token: str = "default") -> str:
        """
        Searches for hotel information in a specific location.
        """
        return await search_hotels(location=location, check_in_date=check_in_date, check_out_date=check_out_date, user_token=user_token)

    async def get_destination_info(self, destination: str, user_token: str = "default") -> str:
        """
        Retrieves general information about a travel destination.
        """
        return await get_destination_info(destination=destination, user_token=user_token)

    async def travel_search_web(self, query: str, user_token: str = "default", max_chars: int = 2000) -> str:
        """
        Searches the web for general travel-related information.
        """
        return await travel_search_web(query=query, user_token=user_token, max_chars=max_chars)

    async def travel_query_uploaded_docs(self, query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed travel documents for a user using vector similarity search.
        """
        return await self.document_tools.document_query_uploaded_docs(
            query=query,
            user_token=user_token,
            section="travel",
            export=export,
            k=k
        )

    async def travel_summarize_document_by_path(self, file_path_str: str) -> str:
        """
        Summarizes a document related to travel located at the given file path.
        """
        return await self.document_tools.document_summarize_document_by_path(file_path_str=file_path_str)

    def get_tools(self):
        """
        Returns a list of tool functions exposed by this class.
        """
        return [
            self.search_flights,
            self.search_hotels,
            self.get_destination_info,
            self.travel_search_web,
            self.travel_query_uploaded_docs,
            self.travel_summarize_document_by_path
        ]
