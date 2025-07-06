# domain_tools/sports_tools/__init__.py

import logging
from typing import Optional, Dict, Any

# Import individual tool functions from the sports_tool module
from .sports_tool import (
    get_sports_scores,
    get_team_info,
    search_player_stats, # Note: This function was cut off in the provided content, assuming it exists.
    sports_search_web # Assuming this is also part of the sports_tool.py
)

logger = logging.getLogger(__name__)

class SportsTools:
    """
    A collection of sports-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any):
        """
        Initializes the SportsTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
        """
        self.config_manager = config_manager
        self.log_event = log_event
        logger.info("SportsTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_token.

    async def get_sports_scores(self, sport: Optional[str] = None, team: Optional[str] = None, date: Optional[str] = None, user_token: str = "default") -> str:
        """
        Retrieves sports scores for various matches, optionally filtered by sport, team, or date.
        """
        return await get_sports_scores(sport=sport, team=team, date=date, user_token=user_token)

    async def get_team_info(self, team_name: str, sport: Optional[str] = None, user_token: str = "default") -> str:
        """
        Retrieves information about a specific sports team, including its league, coach, and key players.
        """
        return await get_team_info(team_name=team_name, sport=sport, user_token=user_token)

    async def search_player_stats(self, player_name: str, sport: Optional[str] = None, user_token: str = "default") -> str:
        """
        Searches for statistics and achievements of a specific sports player.
        """
        return await search_player_stats(player_name=player_name, sport=sport, user_token=user_token)

    async def sports_search_web(self, query: str, user_token: str = "default", max_chars: int = 2000) -> str:
        """
        Searches the web for general sports information using a smart search fallback mechanism.
        """
        return await sports_search_web(query=query, user_token=user_token, max_chars=max_chars)

