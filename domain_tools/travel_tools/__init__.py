# domain_tools/travel_tools/__init__.py

import logging
from typing import Optional, Dict, Any

# Import individual tool functions from the travel_tool module
from .travel_tool import (
    search_flights,
    search_hotels,
    get_destination_info
)

logger = logging.getLogger(__name__)

class TravelTools:
    """
    A collection of travel-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any):
        """
        Initializes the TravelTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
        """
        self.config_manager = config_manager
        self.log_event = log_event
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

