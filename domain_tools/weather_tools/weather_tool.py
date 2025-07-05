# domain_tools/weather_tools/weather_tool.py

import logging
import requests
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta

# Import generic tools
from langchain_core.tools import tool
from shared_tools.query_uploaded_docs_tool import QueryUploadedDocs
from shared_tools.scraper_tool import scrape_web
from shared_tools.doc_summarizer import summarize_document

# Import config_manager to access API configurations and secrets
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability
# Import date_parser for date format flexibility
from utils.date_parser import parse_date_to_yyyymmdd

logger = logging.getLogger(__name__)

# --- Generic API Request Helper (copied for standalone tool file, ideally in shared utils) ---

def _get_nested_value(data: Dict[str, Any], path: List[str]):
    """Helper to get a value from a nested dictionary using a list of keys."""
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and key.isdigit(): # Handle list indices
            try:
                current = current[int(key)]
            except (IndexError, ValueError):
                return None
        else:
            return None
    return current

def _make_dynamic_api_request(
    domain: str,
    function_name: str,
    params: Dict[str, Any],
    user_token: str
) -> Optional[Dict[str, Any]]:
    """
    Makes an API request to the dynamically configured provider for a given domain and function.
    Handles API key retrieval, request construction, and basic error handling.
    Returns parsed JSON data or None on failure (triggering mock fallback).
    """
    # Get the default active API provider for the domain from config.yml
    active_provider_name = config_manager.get(f"api_defaults.{domain}")
    if not active_provider_name:
        logger.error(f"No default API provider configured for domain '{domain}'.")
        return None

    # Get the full configuration for the active provider from api_providers.yml
    provider_config = config_manager.get_api_provider_config(domain, active_provider_name)
    if not provider_config:
        logger.error(f"Configuration for API provider '{active_provider_name}' in domain '{domain}' not found in api_providers.yml.")
        return None

    base_url = provider_config.get("base_url")
    api_key_name = provider_config.get("api_key_name")
    api_key = config_manager.get_secret(api_key_name) if api_key_name else None

    # Special handling for Amadeus which uses client_id and client_secret for token
    if active_provider_name == "amadeus":
        api_secret_name = provider_config.get("api_secret_name")
        api_secret = config_manager.get_secret(api_secret_name) if api_secret_name else None
        token_endpoint = provider_config.get("token_endpoint")

        if not api_key or not api_secret or not token_endpoint:
            logger.warning(f"Amadeus API credentials (client_id/secret) or token_endpoint missing. Cannot make live Amadeus call.")
            return None
        
        # Get Amadeus access token (simplified for demonstration)
        try:
            token_response = requests.post(
                token_endpoint,
                data={'grant_type': 'client_credentials', 'client_id': api_key, 'client_secret': api_secret},
                timeout=5
            )
            token_response.raise_for_status()
            access_token = token_response.json().get('access_token')
            if not access_token:
                logger.error("Failed to get Amadeus access token.")
                return None
            headers = {"Authorization": f"Bearer {access_token}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting Amadeus access token: {e}")
            return None
    else:
        headers = {} # No special headers by default

    if not base_url:
        logger.error(f"Base URL not configured for API provider '{active_provider_name}' in domain '{domain}'.")
        return None

    function_details = provider_config.get("functions", {}).get(function_name)
    if not function_details:
        logger.error(f"Function '{function_name}' not configured for API provider '{active_provider_name}' in domain '{domain}'.")
        return None

    endpoint = function_details.get("endpoint")
    function_param = function_details.get("function_param") # For Alpha Vantage style 'function' param
    path_params = function_details.get("path_params", []) # For ExchangeRate-API style path params

    if not endpoint and not function_param:
        logger.error(f"Neither 'endpoint' nor 'function_param' defined for function '{function_name}'.")
        return None

    # Construct URL
    full_url = f"{base_url}{endpoint}" if endpoint else base_url

    # Add path parameters to URL if specified
    for p_param in path_params:
        if p_param in params:
            value = str(params.pop(p_param))
            full_url = full_url.replace(f"{{{p_param}}}", value)
        else:
            logger.warning(f"Missing path parameter '{p_param}' for function '{function_name}'.")
            return None # Cannot construct URL without required path params

    # Construct query parameters
    query_params = {}
    if function_param:
        query_params["function"] = function_param # Alpha Vantage specific

    # Add API key if it's a query param (not in path or header)
    if api_key_name and active_provider_name not in ["amadeus", "exchangerate_api"]: # Amadeus handled by headers, ExchangeRate by path
        param_name_in_url = provider_config.get("api_key_param_name", api_key_name.replace("_api_key", ""))
        if api_key: # Only add if key exists
            query_params[param_name_in_url] = api_key 
    elif active_provider_name == "exchangerate_api" and api_key:
        pass # Key is a path parameter, already handled above

    for param_key in function_details.get("required_params", []) + function_details.get("optional_params", []):
        if param_key in params:
            query_params[param_key] = params[param_key]
        elif param_key in function_details.get("required_params", []):
            logger.warning(f"Missing required parameter '{param_key}' for function '{function_name}'.")
            return None # Missing required param, cannot proceed

    try:
        logger.info(f"Making API call to: {full_url} with params: {query_params}")
        response = requests.get(full_url, params=query_params, headers=headers, timeout=config_manager.get("web_scraping.timeout_seconds", 15))
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        raw_data = response.json()
        
        # Check for API-specific error messages in the response body
        if "Error Message" in raw_data: # Alpha Vantage specific
            logger.error(f"API Error from {active_provider_name}: {raw_data['Error Message']}")
            return None
        if "Note" in raw_data and "Thank you for using Alpha Vantage!" in raw_data["Note"]: # Alpha Vantage rate limit
            logger.warning(f"API rate limit hit for {active_provider_name}: {raw_data['Note']}")
            return None
        if raw_data.get("status") == "error": # NewsAPI specific
            logger.error(f"API Error from {active_provider_name}: {raw_data.get('message', 'Unknown error')}")
            return None
        if raw_data.get("Error"): # OMDBAPI specific
            logger.error(f"API Error from {active_provider_name}: {raw_data.get('Error')}")
            return None
        if raw_data.get("status") and raw_data["status"].get("error_code"): # CoinGecko error
            logger.error(f"API Error from {active_provider_name}: {raw_data['status'].get('error_message', 'Unknown CoinGecko error')}")
            return None
        if raw_data.get("result") == "error": # ExchangeRate-API error
            logger.error(f"API Error from {active_provider_name}: {raw_data.get('error-type', 'Unknown ExchangeRate-API error')}")
            return None


        # Extract data based on response_path
        data_to_map = raw_data
        response_path = function_details.get("response_path")
        if response_path:
            data_to_map = _get_nested_value(raw_data, response_path)
            if data_to_map is None:
                logger.warning(f"Response path '{'.'.join(response_path)}' not found in API response from {active_provider_name}. Raw data: {raw_data}")
                return None

        # Apply data mapping
        mapped_data = {}
        data_map = function_details.get("data_map", {})
        if isinstance(data_to_map, list): # For lists of items (e.g., news articles, historical data)
            mapped_data_list = []
            for item in data_to_map:
                mapped_item = {}
                for mapped_key, original_key_path in data_map.items():
                    if isinstance(original_key_path, list): # Handle nested paths in data_map
                        mapped_item[mapped_key] = _get_nested_value(item, original_key_path)
                    elif '.' in str(original_key_path): # Handle dot-separated paths in data_map
                        mapped_item[mapped_key] = _get_nested_value(item, original_key_path.split('.'))
                    else: # Direct key or list index
                        if isinstance(original_key_path, int) and isinstance(item, list):
                            try: mapped_item[mapped_key] = item[original_key_path]
                            except IndexError: mapped_item[mapped_key] = None
                        else:
                            mapped_item[mapped_key] = item.get(original_key_path)
                mapped_data_list.append(mapped_item)
            return {"data": mapped_data_list} # Wrap list in a dict for consistent return
        elif isinstance(data_to_map, dict) and function_name == "get_historical_stock_prices" and active_provider_name == "alphavantage":
            # Special handling for Alpha Vantage TIME_SERIES_DAILY where keys are dates
            processed_data = {}
            for date_key, values in data_to_map.items():
                mapped_values = {}
                for mapped_key, original_key_path in data_map.items():
                    if isinstance(original_key_path, list):
                        mapped_values[mapped_key] = _get_nested_value(values, original_key_path)
                    elif '.' in str(original_key_path):
                        mapped_values[mapped_key] = _get_nested_value(values, original_key_path.split('.'))
                    else:
                        mapped_values[mapped_key] = values.get(original_key_path)
                processed_data[date_key] = mapped_values
            return {"data": processed_data}
        else: # For single object responses
            # Special handling for CoinGecko simple price, where response is { "bitcoin": { "usd": 20000 } }
            if function_name == "get_crypto_price" and active_provider_name == "coingecko":
                # params will contain 'ids' and 'vs_currencies'
                crypto_id = params.get("ids", "").lower()
                currency = params.get("vs_currencies", "").lower()
                if crypto_id in raw_data and currency in raw_data[crypto_id]:
                    mapped_data["price"] = raw_data[crypto_id][currency]
                    if f"{currency}_market_cap" in raw_data[crypto_id]:
                        mapped_data["market_cap"] = raw_data[crypto_id][f"{currency}_market_cap"]
                    if f"{currency}_24hr_vol" in raw_data[crypto_id]:
                        mapped_data["vol_24hr"] = raw_data[crypto_id][f"{currency}_24hr_vol"]
                    if f"{currency}_24hr_change" in raw_data[crypto_id]:
                        mapped_data["change_24hr"] = raw_data[crypto_id][f"{currency}_24hr_change"]
                    if "last_updated_at" in raw_data[crypto_id]:
                        mapped_data["last_updated"] = raw_data[crypto_id]["last_updated_at"]
                    return mapped_data
                else:
                    logger.warning(f"CoinGecko simple price response unexpected for {crypto_id}/{currency}: {raw_data}")
                    return None
            
            for mapped_key, original_key_path in data_map.items():
                if isinstance(original_key_path, list):
                    mapped_data[mapped_key] = _get_nested_value(data_to_map, original_key_path)
                elif '.' in str(original_key_path):
                    mapped_data[mapped_key] = _get_nested_value(data_to_map, original_key_path.split('.'))
                else:
                    mapped_data[mapped_key] = data_to_map.get(original_key_path)
            return mapped_data

    except requests.exceptions.Timeout:
        logger.error(f"API request to {active_provider_name} timed out for function '{function_name}'.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making API request to {active_provider_name} for function '{function_name}': {e}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from {active_provider_name} for function '{function_name}'.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during API call to {active_provider_name} for '{function_name}': {e}", exc_info=True)
        return None


# --- Mock Data for Fallback ---
_mock_weather_data = {
    "current_weather": {
        "london": {
            "location": "London, UK",
            "temperature_celsius": 18,
            "temperature_fahrenheit": 64,
            "condition": "Partly Cloudy",
            "humidity": 70,
            "wind_speed_kph": 15,
            "last_updated": datetime.now().isoformat()
        },
        "new_york": {
            "location": "New York, USA",
            "temperature_celsius": 25,
            "temperature_fahrenheit": 77,
            "condition": "Sunny",
            "humidity": 60,
            "wind_speed_kph": 10,
            "last_updated": datetime.now().isoformat()
        }
    },
    "weather_forecast": {
        "london": [
            {
                "date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
                "max_temp_celsius": 20,
                "min_temp_celsius": 12,
                "condition": "Light Rain",
                "precipitation_mm": 5
            },
            {
                "date": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
                "max_temp_celsius": 22,
                "min_temp_celsius": 14,
                "condition": "Partly Cloudy",
                "precipitation_mm": 0
            }
        ],
        "new_york": [
            {
                "date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
                "max_temp_celsius": 28,
                "min_temp_celsius": 20,
                "condition": "Sunny",
                "precipitation_mm": 0
            },
            {
                "date": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
                "max_temp_celsius": 26,
                "min_temp_celsius": 18,
                "condition": "Thunderstorms",
                "precipitation_mm": 15
            }
        ]
    }
}

@tool
def get_current_weather(location: str, user_token: str = "default") -> str:
    """
    Retrieves the current weather conditions for a specified location (city, country).
    Falls back to mock data if API key is missing or API call fails.

    Args:
        location (str): The city and optionally country (e.g., "London, UK", "New York").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of current weather information, or an error/fallback message.
    """
    logger.info(f"Tool: get_current_weather called for location='{location}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'weather_tool_access', False):
        return "Error: Access to weather tools is not enabled for your current tier."
    
    params = {"location": location}

    api_data = _make_dynamic_api_request(
        "weather", "get_current_weather",
        params,
        user_token
    )

    if api_data:
        try:
            loc = api_data.get("location")
            temp_c = api_data.get("temperature_celsius")
            temp_f = api_data.get("temperature_fahrenheit")
            condition = api_data.get("condition")
            humidity = api_data.get("humidity")
            wind_speed = api_data.get("wind_speed_kph")
            last_updated = api_data.get("last_updated")

            if loc and temp_c is not None and condition:
                response_str = (
                    f"Current Weather in {loc}:\n"
                    f"  Temperature: {temp_c}°C ({temp_f}°F)\n"
                    f"  Condition: {condition}\n"
                )
                if humidity is not None:
                    response_str += f"  Humidity: {humidity}%\n"
                if wind_speed is not None:
                    response_str += f"  Wind Speed: {wind_speed} kph\n"
                if last_updated:
                    # Attempt to parse and format if it's an ISO string
                    try:
                        last_updated_dt = datetime.fromisoformat(last_updated)
                        response_str += f"  Last Updated: {last_updated_dt.strftime('%Y-%m-%d %H:%M')}\n"
                    except ValueError:
                        response_str += f"  Last Updated: {last_updated}\n" # Use as is if not ISO
                return response_str
            else:
                logger.warning(f"Live API data for current weather in '{location}' is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live current weather for '{location}'. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live current weather data for '{location}': {e}")
            return f"Error parsing live data for '{location}'. Falling back to mock data."

    # Fallback to mock data
    mock_data_key = location.lower().replace(" ", "_").replace(",", "")
    mock_data = _mock_weather_data.get("current_weather", {}).get(mock_data_key)
    if mock_data:
        response_str = (
            f"Current Weather in {mock_data['location']} (Mock Data Fallback):\n"
            f"  Temperature: {mock_data['temperature_celsius']}°C ({mock_data['temperature_fahrenheit']}°F)\n"
            f"  Condition: {mock_data['condition']}\n"
            f"  Humidity: {mock_data['humidity']}%\n"
            f"  Wind Speed: {mock_data['wind_speed_kph']} kph\n"
        )
        try:
            last_updated_dt = datetime.fromisoformat(mock_data['last_updated'])
            response_str += f"  Last Updated: {last_updated_dt.strftime('%Y-%m-%d %H:%M')}\n"
        except ValueError:
            response_str += f"  Last Updated: {mock_data['last_updated']}\n"
        return response_str
    else:
        return f"Current weather information not found for '{location}'. (API/Mock Fallback Failed)"


@tool
def get_weather_forecast(location: str, days: int = 3, user_token: str = "default") -> str:
    """
    Retrieves the weather forecast for a specified location (city, country) for a number of upcoming days.
    The maximum number of forecast days depends on the API provider's capabilities.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        location (str): The city and optionally country (e.g., "London, UK", "New York").
        days (int, optional): The number of days for the forecast (e.g., 1, 3, 5). Defaults to 3.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of weather forecast information, or an error/fallback message.
    """
    logger.info(f"Tool: get_weather_forecast called for location='{location}', days='{days}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'weather_tool_access', False):
        return "Error: Access to weather tools is not enabled for your current tier."
    
    params = {"location": location, "days": days}

    api_data = _make_dynamic_api_request(
        "weather", "get_weather_forecast",
        params,
        user_token
    )

    if api_data and api_data.get("data"):
        forecast_days = api_data["data"]
        if forecast_days:
            response_str = f"Weather Forecast for {location}:\n"
            for i, day_data in enumerate(forecast_days[:days]): # Limit to requested number of days
                date_str = day_data.get('date', 'N/A')
                # Format date if it's a valid YYYY-MM-DD string
                try:
                    date_str = datetime.strptime(date_str, "%Y-%m-%d").strftime("%A, %B %d, %Y")
                except ValueError:
                    pass # Keep as is if not YYYY-MM-DD
                
                response_str += (
                    f"\nDay {i+1} ({date_str}):\n"
                    f"  Max Temp: {day_data.get('max_temp_celsius', 'N/A')}°C\n"
                    f"  Min Temp: {day_data.get('min_temp_celsius', 'N/A')}°C\n"
                    f"  Condition: {day_data.get('condition', 'N/A')}\n"
                )
                if day_data.get('precipitation_mm') is not None:
                    response_str += f"  Precipitation: {day_data.get('precipitation_mm', 'N/A')} mm\n"
            return response_str
        else:
            return f"No live weather forecast found for '{location}' for {days} days. Falling back to mock data."

    # Fallback to mock data
    mock_data_key = location.lower().replace(" ", "_").replace(",", "")
    mock_forecast = _mock_weather_data.get("weather_forecast", {}).get(mock_data_key, [])
    
    if mock_forecast:
        response_str = f"Weather Forecast for {location} (Mock Data Fallback):\n"
        for i, day_data in enumerate(mock_forecast[:days]): # Limit mock to requested days
            date_str = day_data.get('date', 'N/A')
            try:
                date_str = datetime.strptime(date_str, "%Y-%m-%d").strftime("%A, %B %d, %Y")
            except ValueError:
                pass
            response_str += (
                f"\nDay {i+1} ({date_str}):\n"
                f"  Max Temp: {day_data.get('max_temp_celsius', 'N/A')}°C\n"
                f"  Min Temp: {day_data.get('min_temp_celsius', 'N/A')}°C\n"
                f"  Condition: {day_data.get('condition', 'N/A')}\n"
            )
            if day_data.get('precipitation_mm') is not None:
                response_str += f"  Precipitation: {day_data.get('precipitation_mm', 'N/A')} mm\n"
        return response_str
    else:
        return f"Weather forecast information not found for '{location}'. (API/Mock Fallback Failed)"


# --- Existing Generic Tools (not directly using external APIs, but can be used in weather context) ---

@tool
def weather_search_web(query: str, user_token: str = "default", max_chars: int = 2000) -> str:
    """
    Searches the web for weather-related information using a smart search fallback mechanism.
    This tool wraps the generic `scrape_web` tool, providing a weather-specific interface.
    
    Args:
        query (str): The weather-related search query (e.g., "historical weather data for London", "impact of climate change on hurricanes").
        user_token (str): The unique identifier for the user. Defaults to "default".
        max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
    
    Returns:
        str: A string containing relevant information from the web.
    """
    logger.info(f"Tool: weather_search_web called with query: '{query}' for user: '{user_token}'")
    return scrape_web(query=query, user_token=user_token, max_chars=max_chars)

@tool
def weather_query_uploaded_docs(query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
    """
    Queries previously uploaded and indexed weather documents for a user using vector similarity search.
    This tool wraps the generic `QueryUploadedDocs` tool, fixing the section to "weather".
    
    Args:
        query (str): The search query to find relevant weather documents (e.g., "local climate report", "hurricane tracking data").
        user_token (str): The unique identifier for the user. Defaults to "default".
        export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
        k (int): The number of top relevant documents to retrieve. Defaults to 5.
    
    Returns:
        str: A string containing the combined content of the relevant document chunks,
             or a message indicating no data/results found, or the export path if exported.
    """
    logger.info(f"Tool: weather_query_uploaded_docs called with query: '{query}' for user: '{user_token}'")
    return QueryUploadedDocs(query=query, user_token=user_token, section="weather", export=export, k=k)

@tool
def weather_summarize_document_by_path(file_path_str: str) -> str:
    """
    Summarizes a document related to weather or climate information located at the given file path.
    The file path should be accessible by the system (e.g., in the 'uploads' directory).
    This tool wraps the generic `summarize_document` tool.
    
    Args:
        file_path_str (str): The full path to the document file to be summarized.
                              Example: "uploads/default/weather/climate_report.pdf"
    
    Returns:
        str: A concise summary of the document content.
    """
    logger.info(f"Tool: weather_summarize_document_by_path called for file: '{file_path_str}'")
    file_path = Path(file_path_str)
    if not file_path.exists():
        logger.error(f"Document not found at '{file_path_str}' for summarization.")
        return f"Error: Document not found at '{file_path_str}'."
    
    try:
        summary = summarize_document(file_path)
        return f"Summary of '{file_path.name}':\n{summary}"
    except ValueError as e:
        logger.error(f"Error summarizing document '{file_path_str}': {e}")
        return f"Error summarizing document: {e}"
    except Exception as e:
        logger.critical(f"An unexpected error occurred during summarization of '{file_path_str}': {e}", exc_info=True)
        return f"An unexpected error occurred during summarization: {e}"


# CLI Test (optional)
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch
    import shutil
    import os
    from shared_tools.vector_utils import BASE_VECTOR_DIR # For cleanup
    from shared_tools.python_interpreter_tool import python_interpreter_with_rbac # For testing REPL

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.weather_api_key = "MOCK_WEATHER_API_KEY"
            self.openai_api_key = "sk-mock-openai-key-12345"
            self.google_api_key = "AIzaSy-mock-google-key"
            self.firebase_config = "{}"
            self.serpapi_api_key = "MOCK_SERPAPI_KEY" # For scrape_web

        def get(self, key, default=None):
            return getattr(self, key, default)
    
    class MockConfigManager:
        _instance = None
        _is_loaded = False
        def __init__(self):
            if MockConfigManager._instance is not None:
                raise Exception("ConfigManager is a singleton. Use get_instance().")
            MockConfigManager._instance = self
            self._config_data = {
                'llm': {'max_summary_input_chars': 10000},
                'rag': {'chunk_size': 500, 'chunk_overlap': 50, 'max_query_results_k': 10},
                'web_scraping': {
                    'user_agent': 'Mozilla/5.0 (Test; Python)',
                    'timeout_seconds': 1 # Short timeout for mocks
                },
                'tiers': {},
                'default_user_tier': 'free',
                'default_user_roles': ['user'],
                'api_defaults': { # Mock api_defaults
                    'weather': 'weather_api'
                }
            }
            self._api_providers_data = { # Mock api_providers_data for weather
                "weather": {
                    "weather_api": {
                        "base_url": "https://api.example.com/weather",
                        "api_key_name": "weather_api_key",
                        "api_key_param_name": "key",
                        "functions": {
                            "get_current_weather": {
                                "endpoint": "/current.json",
                                "required_params": ["location"],
                                "response_path": ["current"],
                                "data_map": {
                                    "location": "location.name",
                                    "temperature_celsius": "temp_c",
                                    "temperature_fahrenheit": "temp_f",
                                    "condition": "condition.text",
                                    "humidity": "humidity",
                                    "wind_speed_kph": "wind_kph",
                                    "last_updated": "last_updated"
                                }
                            },
                            "get_weather_forecast": {
                                "endpoint": "/forecast.json",
                                "required_params": ["location", "days"],
                                "response_path": ["forecast", "forecastday"],
                                "data_map": {
                                    "date": "date",
                                    "max_temp_celsius": "day.maxtemp_c",
                                    "min_temp_celsius": "day.mintemp_c",
                                    "condition": "day.condition.text",
                                    "precipitation_mm": "day.totalprecip_mm"
                                }
                            }
                        }
                    }
                }
            }
            self._is_loaded = True
        
        def get(self, key, default=None):
            parts = key.split('.')
            val = self._config_data
            for part in parts:
                if isinstance(val, dict) and part in val:
                    val = val[part]
                else:
                    return default
            return val
        
        def get_secret(self, key, default=None):
            mock_secrets_instance = MockSecrets()
            return mock_secrets_instance.get(key, default)

        def set_secret(self, key, value):
            pass
        
        def get_api_provider_config(self, domain: str, provider_name: str) -> Optional[Dict[str, Any]]:
            return self._api_providers_data.get(domain, {}).get(provider_name)

        def get_domain_api_providers(self, domain: str) -> Dict[str, Any]:
            return self._api_providers_data.get(domain, {})


    # Mock user_manager.get_current_user and get_user_tier_capability for testing RBAC
    class MockUserManager:
        _mock_users = {
            "mock_free_token": {"user_id": "mock_free_token", "username": "FreeUser", "email": "free@example.com", "tier": "free", "roles": ["user"]},
            "mock_pro_token": {"user_id": "mock_pro_token", "username": "ProUser", "email": "pro@example.com", "tier": "pro", "roles": ["user"]},
            "mock_premium_token": {"user_id": "mock_premium_token", "username": "PremiumUser", "email": "premium@example.com", "tier": "premium", "roles": ["user"]},
            "mock_admin_token": {"user_id": "mock_admin_token", "username": "AdminUser", "email": "admin@example.com", "tier": "admin", "roles": ["user", "admin"]},
        }
        _rbac_capabilities = {
            'capabilities': {
                'weather_tool_access': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'data_analysis_enabled': { # For python interpreter
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'web_search_max_results': {
                    'default': 2,
                    'tiers': {'pro': 7, 'premium': 15}
                },
                'web_search_limit_chars': {
                    'default': 500,
                    'tiers': {'pro': 3000, 'premium': 10000}
                }
            }
        }
        _tier_hierarchy = {
            "free": 0, "user": 1, "basic": 2, "pro": 3, "premium": 4, "admin": 99
        }

        def get_current_user(self) -> Dict[str, Any]:
            return getattr(self, '_current_mock_user', {})

        def get_user_tier_capability(self, user_token: Optional[str], capability_key: str, default_value: Any = None) -> Any:
            user_info = self._mock_users.get(user_token, {})
            user_id = user_info.get('user_id')
            user_tier = user_info.get('tier', 'free')
            user_roles = user_info.get('roles', [])

            if "admin" in user_roles:
                if isinstance(default_value, bool): return True
                if isinstance(default_value, (int, float)): return float('inf')
                return default_value
            
            capability_config = self._rbac_capabilities.get('capabilities', {}).get(capability_key)
            if not capability_config:
                return default_value

            # Check roles first
            for role in user_roles:
                if role in capability_config.get('roles', {}):
                    return capability_config['roles'][role]
            
            # Then check tiers
            if user_tier in capability_config.get('tiers', {}):
                return capability_config['tiers'][user_tier]

            return capability_config.get('default', default_value)

    # Patch the actual imports for testing
    import streamlit as st_mock
    if not hasattr(st_mock, 'secrets'):
        st_mock.secrets = MockSecrets()
    
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager
    sys.modules['utils.user_manager'] = MockUserManager()
    sys.modules['utils.user_manager'].get_user_tier_capability = MockUserManager().get_user_tier_capability # Patch the function directly

    # Mock requests.get for external API calls
    original_requests_get = requests.get

    def mock_requests_get_dynamic(url, params, headers, timeout):
        # Simulate hypothetical Weather API responses
        if "api.example.com/weather" in url:
            if "/current.json" in url:
                location = params.get("location", "").lower()
                if "london" in location:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "location": {"name": "London", "country": "UK"},
                        "current": {
                            "temp_c": 18.5, "temp_f": 65.3, "condition": {"text": "Partly Cloudy"},
                            "humidity": 70, "wind_kph": 15.0, "last_updated": datetime.now().isoformat()
                        }
                    }
                    return mock_response
                elif "new york" in location:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "location": {"name": "New York", "country": "USA"},
                        "current": {
                            "temp_c": 25.0, "temp_f": 77.0, "condition": {"text": "Sunny"},
                            "humidity": 60, "wind_kph": 10.0, "last_updated": datetime.now().isoformat()
                        }
                    }
                    return mock_response
                else:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"current": {}} # No data for location
                    return mock_response
            
            elif "/forecast.json" in url:
                location = params.get("location", "").lower()
                days = params.get("days", 3)
                
                forecast_data = []
                if "london" in location:
                    for i in range(min(days, 2)): # Mock up to 2 days
                        forecast_data.append({
                            "date": (datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d"),
                            "day": {
                                "maxtemp_c": 20 + i, "mintemp_c": 12 + i,
                                "condition": {"text": "Light Rain" if i == 0 else "Partly Cloudy"},
                                "totalprecip_mm": 5 if i == 0 else 0
                            }
                        })
                elif "new york" in location:
                     for i in range(min(days, 2)): # Mock up to 2 days
                        forecast_data.append({
                            "date": (datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d"),
                            "day": {
                                "maxtemp_c": 28 - i, "mintemp_c": 20 - i,
                                "condition": {"text": "Sunny" if i == 0 else "Thunderstorms"},
                                "totalprecip_mm": 0 if i == 0 else 15
                            }
                        })

                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"forecast": {"forecastday": forecast_data}}
                return mock_response
        
        # Simulate scrape_web's internal requests.get if needed
        if "google.com/search" in url or "example.com" in url: # Mock for scrape_web
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = f"<html><body><h1>Search results for {params.get('q', 'weather')}</h1><p>Some weather related content from web search.</p></body></html>"
            return mock_response

        return original_requests_get(url, params=params, headers=headers, timeout=timeout)

    requests.get = mock_requests_get_dynamic

    test_user_pro = "mock_pro_token"
    test_user_free = "mock_free_token"

    print("\n--- Testing weather_tool functions ---")

    # Test get_current_weather
    print("\n--- Testing get_current_weather ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result_current_weather = get_current_weather("London, UK", user_token=test_user_pro)
    print(f"Current Weather (Pro User, API):\n{result_current_weather[:500]}...")
    assert "Current Weather in London, UK:" in result_current_weather
    assert "Temperature: 18.5°C (65.3°F)" in result_current_weather
    print("Test 1 Passed.")

    # Test get_current_weather (fallback)
    print("\n--- Testing get_current_weather (Fallback) ---")
    with patch('domain_tools.weather_tools.weather_tool._make_dynamic_api_request', return_value=None):
        result_current_weather_fallback = get_current_weather("Paris, France", user_token=test_user_pro)
        print(f"Current Weather (Pro User, Fallback):\n{result_current_weather_fallback[:500]}...")
        assert "Current Weather in London, UK (Mock Data Fallback):" in result_current_weather_fallback # Falls back to default mock
    print("Test 2 Passed.")

    # Test get_weather_forecast
    print("\n--- Testing get_weather_forecast ---")
    result_forecast = get_weather_forecast("New York, USA", days=2, user_token=test_user_pro)
    print(f"Weather Forecast (Pro User, API):\n{result_forecast[:500]}...")
    assert "Weather Forecast for New York, USA:" in result_forecast
    assert "Day 1 (" in result_forecast # Check for formatted date
    assert "Max Temp: 28.0°C" in result_forecast
    print("Test 3 Passed.")

    # Test get_weather_forecast (fallback)
    print("\n--- Testing get_weather_forecast (Fallback) ---")
    with patch('domain_tools.weather_tools.weather_tool._make_dynamic_api_request', return_value=None):
        result_forecast_fallback = get_weather_forecast("Tokyo, Japan", days=1, user_token=test_user_pro)
        print(f"Weather Forecast (Pro User, Fallback):\n{result_forecast_fallback[:500]}...")
        assert "Weather Forecast for Tokyo, Japan (Mock Data Fallback):" in result_forecast_fallback
    print("Test 4 Passed.")

    # Test RBAC for weather_tool_access (e.g., get_current_weather for free user)
    print("\n--- Testing RBAC for weather_tool_access (Free User) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result_rbac_denied = get_current_weather("Berlin, Germany", user_token=test_user_free)
    print(f"Current Weather (Free User, RBAC Denied): {result_rbac_denied}")
    assert "Error: Access to weather tools is not enabled for your current tier." in result_rbac_denied
    print("Test 5 Passed.")

    # Test weather_search_web
    print("\n--- Testing weather_search_web ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    search_web_query = "historical weather for July in London"
    search_web_result = weather_search_web(search_web_query, user_token=test_user_pro)
    print(f"Web Search Result for '{search_web_query}':\n{search_web_result[:500]}...")
    assert "Search results for historical weather for July in London" in search_web_result
    print("Test 6 Passed.")

    # Test weather_summarize_document_by_path (requires a dummy file)
    print("\n--- Testing weather_summarize_document_by_path ---")
    dummy_upload_dir = Path("uploads") / test_user_pro / "weather"
    dummy_upload_dir.mkdir(parents=True, exist_ok=True)
    dummy_file_path = dummy_upload_dir / "climate_study.txt"
    with open(dummy_file_path, "w") as f:
        f.write("This is a climate study report. It analyzes temperature trends and precipitation patterns in a specific region.")
    
    result_summary = weather_summarize_document_by_path(str(dummy_file_path))
    print(f"Climate Study Summary (Pro User): {result_summary}")
    assert "Mock summary of the provided text." in result_summary
    assert "temperature trends" in result_summary
    print("Test 7 Passed.")

    print("\nAll weather_tool tests completed.")

    # Restore original requests.get
    requests.get = original_requests_get

    # Clean up dummy files and directories
    test_user_dirs = [Path("uploads") / test_user_pro, BASE_VECTOR_DIR / test_user_pro]
    for d in test_user_dirs:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            print(f"Cleaned up {d}")
