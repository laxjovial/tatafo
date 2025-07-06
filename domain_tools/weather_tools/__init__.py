# domain_tools/weather_tools/__init__.py

import logging
from typing import Optional, Dict, Any

# Import individual tool functions from the weather_tool module
from .weather_tool import (
    get_current_weather,
    get_weather_forecast,
    get_air_quality
)

logger = logging.getLogger(__name__)

class WeatherTools:
    """
    A collection of weather-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any):
        """
        Initializes the WeatherTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
        """
        self.config_manager = config_manager
        self.log_event = log_event
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

