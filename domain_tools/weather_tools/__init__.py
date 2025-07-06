# domain_tools/weather_tools/__init__.py

import logging
from typing import Optional, Dict, Any

# Import individual tool functions from the weather_tool module
from .weather_tool import (
    get_current_weather,
    get_weather_forecast,
    get_air_quality,
    weather_search_web, # Added
    weather_query_uploaded_docs, # Added
    weather_summarize_document_by_path # Added
)

logger = logging.getLogger(__name__)

class WeatherTools:
    """
    A collection of weather-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: Any): # Added document_tools
        """
        Initializes the WeatherTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
            document_tools (Any): The DocumentTools instance for document querying. # Added
        """
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools # Stored
        logger.info("WeatherTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_token.

    async def get_current_weather(self, location: str, user_token: str = "default", unit: str = "celsius") -> str:
        """
        Retrieves the current weather conditions for a specified location.
        """
        return await get_current_weather(location=location, user_token=user_token, unit=unit)

    async def get_weather_forecast(self, location: str, days: int = 3, user_token: str = "default", unit: str = "celsius") -> str:
        """
        Retrieves the weather forecast for a specified location for a number of upcoming days.
        """
        return await get_weather_forecast(location=location, days=days, user_token=user_token, unit=unit)

    async def get_air_quality(self, location: str, user_token: str = "default") -> str:
        """
        Retrieves the current air quality index (AQI) and main pollutants for a specified location.
        """
        return await get_air_quality(location=location, user_token=user_token)

    async def weather_search_web(self, query: str, user_token: str = "default", max_chars: int = 2000) -> str:
        """
        Searches the web for general weather information using a smart search fallback mechanism.
        """
        return await weather_search_web(query=query, user_token=user_token, max_chars=max_chars)

    async def weather_query_uploaded_docs(self, query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed weather documents for a user.
        """
        # This now calls the DocumentTools instance
        return await self.document_tools.query_uploaded_docs(
            query_text=query,
            user_token=user_token,
            collection_name="weather", # Specific collection for weather documents
            export=export,
            k=k
        )

    async def weather_summarize_document_by_path(self, file_path_str: str) -> str:
        """
        Summarizes a document related to weather or climate located at the given file path.
        """
        return await weather_summarize_document_by_path(file_path_str=file_path_str) # Call the function from weather_tool.py

