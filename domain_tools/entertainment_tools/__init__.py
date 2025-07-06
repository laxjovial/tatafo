# domain_tools/entertainment_tools/__init__.py

import logging
from typing import Optional, Dict, Any, List

# Import individual tool functions from the entertainment_tool module
from .entertainment_tool import (
    get_movie_info,
    get_tv_show_info,
    search_upcoming_entertainment_events
)

logger = logging.getLogger(__name__)

class EntertainmentTools:
    """
    A collection of entertainment-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any):
        """
        Initializes the EntertainmentTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
        """
        self.config_manager = config_manager
        self.log_event = log_event
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

