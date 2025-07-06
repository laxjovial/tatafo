# domain_tools/weather_tools/__init__.py

import logging
from typing import Any, Optional, List # Import Optional and List

from .weather_tool import (
    get_current_weather,
    get_weather_forecast,
    get_air_quality,
    weather_search_web,
    weather_query_uploaded_docs,
    weather_summarize_document_by_path
)

logger = logging.getLogger(__name__)

class WeatherTools:
    """
    A collection of weather-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: Any):
        """
        Initializes the WeatherTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
            document_tools (Any): The DocumentTools instance for document querying and summarization.
        """
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools
        logger.info("WeatherTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_token.

    async def get_current_weather(self, location: str, user_token: str = "default") -> str:
        """
        Retrieves the current weather conditions for a specified location.
        """
        return await get_current_weather(location=location, user_token=user_token)

    async def get_weather_forecast(self, location: str, days: Optional[int] = None, user_token: str = "default") -> str:
        """
        Retrieves the weather forecast for a specified location for a given number of days.
        """
        return await get_weather_forecast(location=location, days=days, user_token=user_token)

    async def get_air_quality(self, location: str, user_token: str = "default") -> str:
        """
        Retrieves air quality information for a specified location.
        """
        return await get_air_quality(location=location, user_token=user_token)

    async def weather_search_web(self, query: str, user_token: str = "default", max_chars: int = 2000) -> str:
        """
        Searches the web for general weather-related information.
        """
        return await weather_search_web(query=query, user_token=user_token, max_chars=max_chars)

    async def weather_query_uploaded_docs(self, query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed weather documents for a user using vector similarity search.
        """
        return await self.document_tools.document_query_uploaded_docs(
            query=query,
            user_token=user_token,
            section="weather",
            export=export,
            k=k
        )

    async def weather_summarize_document_by_path(self, file_path_str: str) -> str:
        """
        Summarizes a document related to weather or climate located at the given file path.
        """
        return await self.document_tools.document_summarize_document_by_path(file_path_str=file_path_str)

    def get_tools(self):
        """
        Returns a list of tool functions exposed by this class.
        """
        return [
            self.get_current_weather,
            self.get_weather_forecast,
            self.get_air_quality,
            self.weather_search_web,
            self.weather_query_uploaded_docs,
            self.weather_summarize_document_by_path
        ]


