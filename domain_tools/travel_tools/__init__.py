# domain_tools/travel_tools/__init__.py

import logging
from typing import Optional, Dict, Any

# Import individual tool functions from the travel_tool module
from .travel_tool import (
    search_flights,
    search_hotels,
    get_destination_info,
    travel_search_web, # Added
    travel_query_uploaded_docs, # Added
    travel_summarize_document_by_path # Added
)

logger = logging.getLogger(__name__)

class TravelTools:
    """
    A collection of travel-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: Any): # Added document_tools
        """
        Initializes the TravelTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
            document_tools (Any): The DocumentTools instance for document querying. # Added
        """
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools # Stored
        logger.info("TravelTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_token.

    async def search_flights(self, origin: str, destination: str, departure_date: str, return_date: Optional[str] = None, user_token: str = "default") -> str:
        """
        Searches for flights between an origin and destination on specified dates.
        """
        return await search_flights(origin=origin, destination=destination, departure_date=departure_date, return_date=return_date, user_token=user_token)

    async def search_hotels(self, location: str, checkin_date: str, checkout_date: str, user_token: str = "default") -> str:
        """
        Searches for hotels in a specified location for given check-in and check-out dates.
        """
        return await search_hotels(location=location, checkin_date=checkin_date, checkout_date=checkout_date, user_token=user_token)

    async def get_destination_info(self, city: str, user_token: str = "default") -> str:
        """
        Retrieves information about a travel destination, including its description, main attractions, and best time to visit.
        """
        return await get_destination_info(city=city, user_token=user_token)

    async def travel_search_web(self, query: str, user_token: str = "default", max_chars: int = 2000) -> str:
        """
        Searches the web for general travel information using a smart search fallback mechanism.
        """
        return await travel_search_web(query=query, user_token=user_token, max_chars=max_chars)

    async def travel_query_uploaded_docs(self, query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed travel documents for a user.
        """
        # This now calls the DocumentTools instance
        return await self.document_tools.query_uploaded_docs(
            query_text=query,
            user_token=user_token,
            collection_name="travel", # Specific collection for travel documents
            export=export,
            k=k
        )

    async def travel_summarize_document_by_path(self, file_path_str: str) -> str:
        """
        Summarizes a document related to travel (e.g., travel guides, visa applications) located at the given file path.
        """
        return await travel_summarize_document_by_path(file_path_str=file_path_str) # Call the function from travel_tool.py
