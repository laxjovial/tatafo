# domain_tools/weather_tools/weather_tool.py

import logging
import requests
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta

# Import generic tools
from langchain_core.tools import tool
# REMOVED: from shared_tools.query_uploaded_docs_tool import QueryUploadedDocs
from shared_tools.scraper_tool import scrape_web
from shared_tools.doc_summarizer import summarize_document

# Import config_manager to access API configurations and secrets
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability
# Import date_parser for date format flexibility
from utils.date_parser import parse_date_to_yyyymmdd
# Import analytics_tracker
from utils import analytics_tracker # Import the module

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

async def _make_dynamic_api_request( # Made async to await analytics_tracker.log_tool_usage
    domain: str,
    function_name: str,
    params: Dict[str, Any],
    user_token: str
) -> Optional[Dict[str, Any]]:
    """
    Makes an API request to the dynamically configured provider for a given domain and function.
    Handles API key retrieval, request construction, and basic error handling.
    Returns parsed JSON data or None on failure (triggering mock fallback).
    Logs tool usage analytics.
    """
    # Check if analytics is enabled for logging tool usage
    log_tool_usage_enabled = config_manager.get("analytics.log_tool_usage", False)

    # Get the default active API provider for the domain from data/config.yml
    active_provider_name = config_manager.get(f"api_defaults.{domain}")
    if not active_provider_name:
        logger.error(f"No default API provider configured for domain '{domain}'.")
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=f"No default API provider configured for domain '{domain}'."
            )
        return None

    # Get the full configuration for the active provider from api_providers.yml
    provider_config = config_manager.get_api_provider_config(domain, active_provider_name)
    if not provider_config:
        logger.error(f"Configuration for API provider '{active_provider_name}' in domain '{domain}' not found in api_providers.yml.")
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=f"API provider config '{active_provider_name}' not found for domain '{domain}'."
            )
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
            if log_tool_usage_enabled:
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_token=user_token,
                    success=False,
                    error_message="Amadeus API credentials or token endpoint missing."
                )
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
                if log_tool_usage_enabled:
                    await analytics_tracker.log_tool_usage(
                        tool_name=f"{domain}_{function_name}",
                        tool_params=params,
                        user_token=user_token,
                        success=False,
                        error_message="Failed to get Amadeus access token."
                    )
                return None
            headers = {"Authorization": f"Bearer {access_token}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting Amadeus access token: {e}")
            if log_tool_usage_enabled:
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_token=user_token,
                    success=False,
                    error_message=f"Error getting Amadeus access token: {e}"
                )
            return None
    else:
        headers = {} # No special headers by default

    if not base_url:
        logger.error(f"Base URL not configured for API provider '{active_provider_name}' in domain '{domain}'.")
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=f"Base URL not configured for '{active_provider_name}'."
            )
        return None

    function_details = provider_config.get("functions", {}).get(function_name)
    if not function_details:
        logger.error(f"Function '{function_name}' not configured for API provider '{active_provider_name}' in domain '{domain}'.")
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=f"Function '{function_name}' not configured for '{active_provider_name}'."
            )
        return None

    endpoint = function_details.get("endpoint")
    function_param = function_details.get("function_param") # For Alpha Vantage style 'function' param
    path_params = function_details.get("path_params", []) # For ExchangeRate-API style path params

    if not endpoint and not function_param:
        logger.error(f"Neither 'endpoint' nor 'function_param' defined for function '{function_name}'.")
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=f"Endpoint or function_param missing for '{function_name}'."
            )
        return None

    # Construct URL
    full_url = f"{base_url}{endpoint}" if endpoint else base_url

    # Add path parameters to URL if specified
    for p_param in path_params:
        if p_param in params:
            value = str(params.pop(p_param))
            full_url = full_url.replace(f"{{{p_param}}}", value)
        else:
            error_msg = f"Missing path parameter '{p_param}' for function '{function_name}'."
            logger.warning(error_msg)
            if log_tool_usage_enabled:
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_token=user_token,
                    success=False,
                    error_message=error_msg
                )
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
            error_msg = f"Missing required parameter '{param_key}' for function '{function_name}'."
            logger.warning(error_msg)
            if log_tool_usage_enabled:
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_token=user_token,
                    success=False,
                    error_message=error_msg
                )
            return None # Missing required param, cannot proceed

    try:
        logger.info(f"Making API call to: {full_url} with params: {query_params}")
        response = requests.get(full_url, params=query_params, headers=headers, timeout=config_manager.get("web_scraping.timeout_seconds", 15))
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        raw_data = response.json()
        
        # Check for API-specific error messages in the response body
        api_error_message = None
        if "Error Message" in raw_data: # Alpha Vantage specific
            api_error_message = f"API Error from {active_provider_name}: {raw_data['Error Message']}"
        elif "Note" in raw_data and "Thank you for using Alpha Vantage!" in raw_data["Note"]: # Alpha Vantage rate limit
            api_error_message = f"API rate limit hit for {active_provider_name}: {raw_data['Note']}"
        elif raw_data.get("status") == "error": # NewsAPI specific
            api_error_message = f"API Error from {active_provider_name}: {raw_data.get('message', 'Unknown error')}"
        elif raw_data.get("Error"): # OMDBAPI specific
            api_error_message = f"API Error from {active_provider_name}: {raw_data.get('Error')}"
        elif raw_data.get("status") and raw_data["status"].get("error_code"): # CoinGecko error
            api_error_message = f"API Error from {active_provider_name}: {raw_data['status'].get('error_message', 'Unknown CoinGecko error')}"
        elif raw_data.get("result") == "error": # ExchangeRate-API error
            api_error_message = f"API Error from {active_provider_name}: {raw_data.get('error-type', 'Unknown ExchangeRate-API error')}"

        if api_error_message:
            logger.error(api_error_message)
            if log_tool_usage_enabled:
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_token=user_token,
                    success=False,
                    error_message=api_error_message
                )
            return None


        # Extract data based on response_path
        data_to_map = raw_data
        response_path = function_details.get("response_path")
        if response_path:
            data_to_map = _get_nested_value(raw_data, response_path)
            if data_to_map is None:
                error_msg = f"Response path '{'.'.join(response_path)}' not found in API response from {active_provider_name}. Raw data: {raw_data}"
                logger.warning(error_msg)
                if log_tool_usage_enabled:
                    await analytics_tracker.log_tool_usage(
                        tool_name=f"{domain}_{function_name}",
                        tool_params=params,
                        user_token=user_token,
                        success=False,
                        error_message=error_msg
                    )
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
            final_result = {"data": mapped_data_list} # Wrap list in a dict for consistent return
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
            final_result = {"data": processed_data}
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
                    final_result = mapped_data
                else:
                    error_msg = f"CoinGecko simple price response unexpected for {crypto_id}/{currency}: {raw_data}"
                    logger.warning(error_msg)
                    if log_tool_usage_enabled:
                        await analytics_tracker.log_tool_usage(
                            tool_name=f"{domain}_{function_name}",
                            tool_params=params,
                            user_token=user_token,
                            success=False,
                            error_message=error_msg
                        )
                    return None
            
            for mapped_key, original_key_path in data_map.items():
                if isinstance(original_key_path, list):
                    mapped_data[mapped_key] = _get_nested_value(data_to_map, original_key_path)
                elif '.' in str(original_key_path):
                    mapped_data[mapped_key] = _get_nested_value(data_to_map, original_key_path.split('.'))
                else:
                    mapped_data[mapped_key] = data_to_map.get(original_key_path)
            final_result = mapped_data

        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=True
            )
        return final_result

    except requests.exceptions.Timeout:
        error_msg = f"API request to {active_provider_name} timed out for function '{function_name}'."
        logger.error(error_msg)
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=error_msg
            )
        return None
    except requests.exceptions.RequestException as e:
        error_msg = f"Error making API request to {active_provider_name} for function '{function_name}': {e}"
        logger.error(error_msg)
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=error_msg
            )
        return None
    except json.JSONDecodeError:
        error_msg = f"Failed to decode JSON response from {active_provider_name} for function '{function_name}'."
        logger.error(error_msg)
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=error_msg
            )
        return None
    except Exception as e:
        error_msg = f"An unexpected error occurred during API call to {active_provider_name} for '{function_name}': {e}"
        logger.error(error_msg, exc_info=True)
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=error_msg
            )
        return None


# --- Mock Data for Fallback ---
_mock_weather_data = {
    "current_weather": {
        "london": {
            "location": "London, UK",
            "temperature_celsius": 18,
            "temperature_fahrenheit": 64.4,
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
    "forecast": {
        "london": [
            {
                "date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
                "max_temp_celsius": 20,
                "min_temp_celsius": 12,
                "condition": "Light Rain",
                "pop": 60 # Probability of precipitation
            },
            {
                "date": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
                "max_temp_celsius": 22,
                "min_temp_celsius": 14,
                "condition": "Cloudy",
                "pop": 20
            }
        ],
        "new_york": [
            {
                "date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
                "max_temp_celsius": 28,
                "min_temp_celsius": 19,
                "condition": "Sunny",
                "pop": 10
            },
            {
                "date": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
                "max_temp_celsius": 26,
                "min_temp_celsius": 18,
                "condition": "Partly Cloudy",
                "pop": 30
            }
        ]
    },
    "air_quality": {
        "london": {
            "location": "London, UK",
            "aqi": 35, # Good
            "pollutants": {"pm2_5": 8, "o3": 40},
            "last_updated": datetime.now().isoformat()
        },
        "new_york": {
            "location": "New York, USA",
            "aqi": 60, # Moderate
            "pollutants": {"pm2_5": 15, "co": 3},
            "last_updated": datetime.now().isoformat()
        }
    }
}

@tool
def get_current_weather(location: str, user_token: str = "default", unit: str = "celsius") -> str:
    """
    Retrieves the current weather conditions for a specified location.
    Can return temperature in 'celsius' or 'fahrenheit'.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        location (str): The city or location (e.g., "London", "New York").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
        unit (str, optional): The temperature unit ('celsius' or 'fahrenheit'). Defaults to 'celsius'.

    Returns:
        str: A formatted string of current weather information, or an error/fallback message.
    """
    logger.info(f"Tool: get_current_weather called for location: '{location}', unit: '{unit}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'weather_tool_access', False):
        return "Error: Access to weather tools is not enabled for your current tier."
    
    params = {"location": location, "unit": unit}
    api_data = asyncio.run(_make_dynamic_api_request("weather", "get_current_weather", params, user_token))

    if api_data:
        try:
            loc = api_data.get("location")
            temp_c = api_data.get("temperature_celsius")
            temp_f = api_data.get("temperature_fahrenheit")
            condition = api_data.get("condition")
            humidity = api_data.get("humidity")
            wind_speed = api_data.get("wind_speed_kph")
            last_updated = api_data.get("last_updated")

            temp_display = f"{temp_c}°C" if unit.lower() == "celsius" else f"{temp_f}°F"

            if loc and condition:
                response_str = (
                    f"Current Weather in {loc}:\n"
                    f"  Temperature: {temp_display}\n"
                    f"  Condition: {condition}\n"
                    f"  Humidity: {humidity}%\n"
                    f"  Wind Speed: {wind_speed} kph\n"
                )
                if last_updated:
                    try:
                        last_updated_dt = datetime.fromisoformat(last_updated)
                        response_str += f"  Last Updated: {last_updated_dt.strftime('%Y-%m-%d %H:%M')}\n"
                    except ValueError:
                        response_str += f"  Last Updated: {last_updated}\n"
                return response_str
            else:
                logger.warning(f"Live API data for current weather in '{location}' is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live current weather for '{location}'. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live current weather data for '{location}': {e}")
            return f"Error parsing live data for '{location}'. Falling back to mock data."

    # Fallback to mock data
    mock_data_key = location.lower().replace(" ", "_")
    mock_data = _mock_weather_data.get("current_weather", {}).get(mock_data_key)
    if mock_data:
        temp_display = f"{mock_data['temperature_celsius']}°C" if unit.lower() == "celsius" else f"{mock_data['temperature_fahrenheit']}°F"
        response_str = (
            f"Current Weather in {mock_data['location']} (Mock Data Fallback):\n"
            f"  Temperature: {temp_display}\n"
            f"  Condition: {mock_data['condition']}\n"
            f"  Humidity: {mock_data['humidity']}%\n"
            f"  Wind Speed: {mock_data['wind_speed_kph']} kph\n"
        )
        if mock_data.get('last_updated'):
            try:
                last_updated_dt = datetime.fromisoformat(mock_data['last_updated'])
                response_str += f"  Last Updated: {last_updated_dt.strftime('%Y-%m-%d %H:%M')}\n"
            except ValueError:
                response_str += f"  Last Updated: {mock_data['last_updated']}\n"
        return response_str
    else:
        return f"Current weather information not found for '{location}'. (API/Mock Fallback Failed)"


@tool
def get_weather_forecast(location: str, days: int = 3, user_token: str = "default", unit: str = "celsius") -> str:
    """
    Retrieves the weather forecast for a specified location for a number of upcoming days (default 3).
    Can return temperature in 'celsius' or 'fahrenheit'.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        location (str): The city or location (e.g., "London", "New York").
        days (int, optional): The number of days for the forecast (max 10). Defaults to 3.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
        unit (str, optional): The temperature unit ('celsius' or 'fahrenheit'). Defaults to 'celsius'.

    Returns:
        str: A formatted string of weather forecast information, or an error/fallback message.
    """
    logger.info(f"Tool: get_weather_forecast called for location: '{location}', days: {days}, unit: '{unit}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'weather_tool_access', False):
        return "Error: Access to weather tools is not enabled for your current tier."
    
    params = {"location": location, "days": days, "unit": unit}
    api_data = asyncio.run(_make_dynamic_api_request("weather", "get_weather_forecast", params, user_token))

    if api_data and api_data.get("data"):
        forecast_days = api_data["data"]
        if forecast_days:
            response_str = f"Weather Forecast for {location} ({days} days):\n"
            for i, day_data in enumerate(forecast_days[:days]): # Limit to requested days
                max_temp_c = day_data.get("max_temp_celsius")
                min_temp_c = day_data.get("min_temp_celsius")
                max_temp_f = day_data.get("max_temp_fahrenheit")
                min_temp_f = day_data.get("min_temp_fahrenheit")
                condition = day_data.get("condition")
                pop = day_data.get("pop") # Probability of precipitation

                max_temp_display = f"{max_temp_c}°C" if unit.lower() == "celsius" else f"{max_temp_f}°F"
                min_temp_display = f"{min_temp_c}°C" if unit.lower() == "celsius" else f"{min_temp_f}°F"

                response_str += (
                    f"  Day {i+1} ({day_data.get('date', 'N/A')}):\n"
                    f"    Max Temp: {max_temp_display}, Min Temp: {min_temp_display}\n"
                    f"    Condition: {condition}\n"
                    f"    Chance of Rain: {pop}%\n"
                )
            return response_str
        else:
            return f"No live weather forecast found for '{location}' for {days} days. Falling back to mock data."

    # Fallback to mock data
    mock_data_key = location.lower().replace(" ", "_")
    mock_forecast = _mock_weather_data.get("forecast", {}).get(mock_data_key, [])
    if mock_forecast:
        response_str = f"Weather Forecast for {location} (Mock Data Fallback):\n"
        for i, day_data in enumerate(mock_forecast[:days]):
            max_temp_c = day_data.get("max_temp_celsius")
            min_temp_c = day_data.get("min_temp_celsius")
            # Mock data might not have Fahrenheit, convert if necessary for display
            max_temp_f = round(max_temp_c * 9/5 + 32, 1) if max_temp_c is not None else 'N/A'
            min_temp_f = round(min_temp_c * 9/5 + 32, 1) if min_temp_c is not None else 'N/A'

            max_temp_display = f"{max_temp_c}°C" if unit.lower() == "celsius" else f"{max_temp_f}°F"
            min_temp_display = f"{min_temp_c}°C" if unit.lower() == "celsius" else f"{min_temp_f}°F"

            response_str += (
                f"  Day {i+1} ({day_data.get('date', 'N/A')}):\n"
                f"    Max Temp: {max_temp_display}, Min Temp: {min_temp_display}\n"
                f"    Condition: {day_data.get('condition')}\n"
                f"    Chance of Rain: {day_data.get('pop') or 'N/A'}%\n"
            )
        return response_str
    else:
        return f"Weather forecast not found for '{location}'. (API/Mock Fallback Failed)"


@tool
def get_air_quality(location: str, user_token: str = "default") -> str:
    """
    Retrieves the current air quality index (AQI) and main pollutants for a specified location.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        location (str): The city or location (e.g., "London", "New York").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of air quality information, or an error/fallback message.
    """
    logger.info(f"Tool: get_air_quality called for location: '{location}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'weather_tool_access', False):
        return "Error: Access to weather tools is not enabled for your current tier."
    
    params = {"location": location}
    api_data = asyncio.run(_make_dynamic_api_request("weather", "get_air_quality", params, user_token))

    if api_data:
        try:
            loc = api_data.get("location")
            aqi = api_data.get("aqi")
            pollutants = api_data.get("pollutants")
            last_updated = api_data.get("last_updated")

            pollutant_str = ", ".join([f"{p}: {v}" for p, v in pollutants.items()]) if pollutants else "N/A"

            if loc and aqi is not None:
                response_str = (
                    f"Air Quality in {loc}:\n"
                    f"  AQI: {aqi} ({'Good' if aqi <= 50 else 'Moderate' if aqi <= 100 else 'Unhealthy'})\n"
                    f"  Main Pollutants: {pollutant_str}\n"
                )
                if last_updated:
                    try:
                        last_updated_dt = datetime.fromisoformat(last_updated)
                        response_str += f"  Last Updated: {last_updated_dt.strftime('%Y-%m-%d %H:%M')}\n"
                    except ValueError:
                        response_str += f"  Last Updated: {last_updated}\n"
                return response_str
            else:
                logger.warning(f"Live API data for air quality in '{location}' is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live air quality for '{location}'. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live air quality data for '{location}': {e}")
            return f"Error parsing live data for '{location}'. Falling back to mock data."

    # Fallback to mock data
    mock_data_key = location.lower().replace(" ", "_")
    mock_data = _mock_weather_data.get("air_quality", {}).get(mock_data_key)
    if mock_data:
        pollutant_str = ", ".join([f"{p}: {v}" for p, v in mock_data.get('pollutants', {}).items()])
        response_str = (
            f"Air Quality in {mock_data['location']} (Mock Data Fallback):\n"
            f"  AQI: {mock_data['aqi']} ({'Good' if mock_data['aqi'] <= 50 else 'Moderate' if mock_data['aqi'] <= 100 else 'Unhealthy'})\n"
            f"  Main Pollutants: {pollutant_str}\n"
        )
        if mock_data.get('last_updated'):
            try:
                last_updated_dt = datetime.fromisoformat(mock_data['last_updated'])
                response_str += f"  Last Updated: {last_updated_dt.strftime('%Y-%m-%d %H:%M')}\n"
            except ValueError:
                response_str += f"  Last Updated: {mock_data['last_updated']}\n"
        return response_str
    else:
        return f"Air quality information not found for '{location}'. (API/Mock Fallback Failed)"


# --- Existing Generic Tools (not directly using external APIs, but can be used in weather context) ---

@tool
def weather_search_web(query: str, user_token: str = "default", max_chars: int = 2000) -> str:
    """
    Searches the web for weather-related information using a smart search fallback mechanism.
    This tool wraps the generic `scrape_web` tool, providing a weather-specific interface.
    
    Args:
        query (str): The weather-related search query (e.g., "impact of climate change on hurricanes", "best time to visit Paris weather-wise").
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
        query (str): The search query to find relevant weather documents (e.g., "my local weather station data", "climate patterns in my region").
        user_token (str): The unique identifier for the user. Defaults to "default".
        export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
        k (int): The number of top relevant documents to retrieve. Defaults to 5.
    
    Returns:
        str: A string containing the combined content of the relevant document chunks,
             or a message indicating no data/results found, or the export path if exported.
    """
    logger.info(f"Tool: weather_query_uploaded_docs called with query: '{query}' for user: '{user_token}'")
    # This will be replaced by a call to self.document_tools.query_uploaded_docs
    # For now, keeping the original call for review purposes.
    return QueryUploadedDocs(query=query, user_token=user_token, section="weather", export=export, k=k)

@tool
def weather_summarize_document_by_path(file_path_str: str) -> str:
    """
    Summarizes a document related to weather or climate located at the given file path.
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
    import asyncio
    from unittest.mock import MagicMock, AsyncMock, patch
    import shutil
    import os
    import sys # Import sys for patching modules
    from shared_tools.vector_utils import BASE_VECTOR_DIR # For cleanup
    # from shared_tools.python_interpreter_tool import python_interpreter_with_rbac # For testing REPL

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
                },
                'analytics': { # Mock analytics settings
                    'enabled': True,
                    'log_tool_usage': True,
                    'log_query_failures': True
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
                                "endpoint": "/current",
                                "required_params": ["location"],
                                "optional_params": ["unit"],
                                "response_path": ["data"],
                                "data_map": {
                                    "location": "location.name",
                                    "temperature_celsius": "current.temp_c",
                                    "temperature_fahrenheit": "current.temp_f",
                                    "condition": "current.condition.text",
                                    "humidity": "current.humidity",
                                    "wind_speed_kph": "current.wind_kph",
                                    "last_updated": "current.last_updated"
                                }
                            },
                            "get_weather_forecast": {
                                "endpoint": "/forecast",
                                "required_params": ["location", "days"],
                                "optional_params": ["unit"],
                                "response_path": ["forecast", "forecastday"],
                                "data_map": {
                                    "date": "date",
                                    "max_temp_celsius": "day.maxtemp_c",
                                    "min_temp_celsius": "day.mintemp_c",
                                    "max_temp_fahrenheit": "day.maxtemp_f",
                                    "min_temp_fahrenheit": "day.mintemp_f",
                                    "condition": "day.condition.text",
                                    "pop": "day.daily_chance_of_rain"
                                }
                            },
                            "get_air_quality": {
                                "endpoint": "/aqi",
                                "required_params": ["location"],
                                "response_path": ["data"],
                                "data_map": {
                                    "location": "location.name",
                                    "aqi": "current.air_quality.us-epa-index",
                                    "pollutants": "current.air_quality" # Special handling for pollutants
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
                'document_query_enabled': { # Added for document tool
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

    # Mock analytics_tracker
    mock_analytics_tracker_db = MagicMock()
    mock_analytics_tracker_auth = MagicMock()
    mock_analytics_tracker_auth.currentUser = MagicMock(uid="mock_user_123")
    mock_analytics_tracker_db.collection.return_value.add = AsyncMock(return_value=MagicMock(id="mock_doc_id"))

    # Patch firebase_admin.firestore for the local import within log_event
    with patch.dict(sys.modules, {'firebase_admin.firestore': MagicMock(firestore=MagicMock())}):
        sys.modules['firebase_admin.firestore'].firestore.CollectionReference = MagicMock()
        sys.modules['firebase_admin.firestore'].firestore.DocumentReference = MagicMock()
        
        # Initialize the actual analytics_tracker with mocks
        analytics_tracker.initialize_analytics(
            mock_analytics_tracker_db,
            mock_analytics_tracker_auth,
            "test_app_id_for_analytics",
            "mock_user_123"
        )

        # Mock requests.get for external API calls
        original_requests_get = requests.get

        def mock_requests_get_dynamic(url, params, headers, timeout):
            # Simulate hypothetical Weather API responses
            if "api.example.com/weather" in url:
                if "/current" in url:
                    location = params.get("location", "").lower()
                    if "london" in location:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "data": {
                                "location": {"name": "London"},
                                "current": {
                                    "temp_c": 18, "temp_f": 64.4,
                                    "condition": {"text": "Partly Cloudy"},
                                    "humidity": 70, "wind_kph": 15,
                                    "last_updated": datetime.now().isoformat()
                                }
                            }
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"data": {}}
                        return mock_response
                elif "/forecast" in url:
                    location = params.get("location", "").lower()
                    days = params.get("days", 3)
                    if "new york" in location:
                        forecast_data = []
                        for i in range(min(days, 2)): # Mock up to 2 days
                            forecast_data.append({
                                "date": (datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d"),
                                "day": {
                                    "maxtemp_c": 28 - i, "mintemp_c": 19 - i,
                                    "maxtemp_f": round((28 - i) * 9/5 + 32, 1), "mintemp_f": round((19 - i) * 9/5 + 32, 1),
                                    "condition": {"text": "Sunny" if i == 0 else "Partly Cloudy"},
                                    "daily_chance_of_rain": 10 + i*10
                                }
                            })
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"forecast": {"forecastday": forecast_data}}
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"forecast": {"forecastday": []}}
                        return mock_response
                elif "/aqi" in url:
                    location = params.get("location", "").lower()
                    if "london" in location:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "data": {
                                "location": {"name": "London"},
                                "current": {
                                    "air_quality": {
                                        "co": 1.5, "no2": 10, "o3": 40, "so2": 2, "pm2_5": 8, "pm10": 15,
                                        "us-epa-index": 35 # Good
                                    }
                                },
                                "last_updated": datetime.now().isoformat()
                            }
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"data": {}}
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

        # Mock for QueryUploadedDocs
        class MockQueryUploadedDocs:
            def __init__(self, query, user_token, section, export, k):
                self.query = query
                self.user_token = user_token
                self.section = section
                self.export = export
                self.k = k
            def __call__(self):
                return f"Mocked document query results for '{self.query}' in section '{self.section}'."

        # Mock for summarize_document
        class MockSummarizeDocument:
            def __call__(self, file_path):
                return f"Mocked summary of {file_path.name}"

        # Patch QueryUploadedDocs and summarize_document in the weather_tool module
        original_QueryUploadedDocs = sys.modules['domain_tools.weather_tools.weather_tool'].QueryUploadedDocs
        original_summarize_document = sys.modules['domain_tools.weather_tools.weather_tool'].summarize_document
        sys.modules['domain_tools.weather_tools.weather_tool'].QueryUploadedDocs = MockQueryUploadedDocs
        sys.modules['domain_tools.weather_tools.weather_tool'].summarize_document = MockSummarizeDocument()

        async def run_weather_tests():
            print("\n--- Testing weather_tool functions with Analytics ---")

            # Test get_current_weather (success)
            print("\n--- Test 1: get_current_weather (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock() # Reset mock call count
            result_current_weather = await get_current_weather("London", user_token=test_user_pro)
            print(f"Current Weather: {result_current_weather}")
            assert "Current Weather in London, UK:" in result_current_weather
            assert "Temperature: 18°C" in result_current_weather
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "weather_get_current_weather"
            assert logged_data["success"] is True
            print("Test 1 Passed (and analytics logged success).")

            # Test get_weather_forecast (API failure - no data found)
            print("\n--- Test 2: get_weather_forecast (API Failure) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_forecast = await get_weather_forecast("NonExistentCity", days=5, user_token=test_user_pro)
            print(f"Weather Forecast (API Error): {result_forecast}")
            assert "No live weather forecast found for 'NonExistentCity' for 5 days." in result_forecast
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "weather_get_weather_forecast"
            assert logged_data["success"] is False
            assert "Response path 'forecast.forecastday' not found" in logged_data["error_message"] or "incomplete" in logged_data["error_message"]
            print("Test 2 Passed (and analytics logged failure).")

            # Test get_air_quality (RBAC denied)
            print("\n--- Test 3: get_air_quality (RBAC Denied) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_air_quality_rbac_denied = await get_air_quality("Paris", user_token=test_user_free)
            print(f"Air Quality (Free User, RBAC Denied): {result_air_quality_rbac_denied}")
            assert "Error: Access to weather tools is not enabled for your current tier." in result_air_quality_rbac_denied
            # No analytics log expected here because RBAC check happens before _make_dynamic_api_request
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 3 Passed (RBAC correctly prevented call and no analytics logged).")

            # Test weather_search_web (generic tool, not using _make_dynamic_api_request)
            print("\n--- Test 4: weather_search_web (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_web_search = await weather_search_web("impact of El Nino on global weather", user_token=test_user_pro)
            print(f"Web Search Result: {result_web_search[:100]}...")
            assert "Search results for impact of El Nino on global weather" in result_web_search
            # Analytics for generic tools like scrape_web or summarize_document
            # would need to be integrated within those shared_tools themselves,
            # or wrapped by a higher-level agent logging.
            # For now, we are focusing on _make_dynamic_api_request.
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 4 Passed (no analytics expected for generic tool directly).")

            # Test 5: weather_query_uploaded_docs (generic tool)
            print("\n--- Test 5: weather_query_uploaded_docs (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            # Mock QueryUploadedDocs to simulate a response
            class MockQueryUploadedDocs:
                def __init__(self, query, user_token, section, export, k):
                    self.query = query
                    self.user_token = user_token
                    self.section = section
                    self.export = export
                    self.k = k
                def __call__(self):
                    return f"Mocked document query results for '{self.query}' in section '{self.section}'."
            
            # Temporarily replace QueryUploadedDocs with our mock
            import sys # Import sys for patching modules
            original_QueryUploadedDocs_in_test = sys.modules['domain_tools.weather_tools.weather_tool'].QueryUploadedDocs
            sys.modules['domain_tools.weather_tools.weather_tool'].QueryUploadedDocs = MockQueryUploadedDocs

            result_doc_query = await weather_query_uploaded_docs("my local weather station data", user_token=test_user_pro)
            print(f"Document Query Result: {result_doc_query}")
            assert "Mocked document query results for 'my local weather station data' in section 'weather'." in result_doc_query
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 5 Passed (no analytics expected for generic tool directly, will be logged by DocumentTools).")
            sys.modules['domain_tools.weather_tools.weather_tool'].QueryUploadedDocs = original_QueryUploadedDocs_in_test # Restore original

            # Test 6: weather_summarize_document_by_path (generic tool)
            print("\n--- Test 6: weather_summarize_document_by_path (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            # Create a dummy file for summarization test
            dummy_file_path = Path("uploads") / test_user_pro / "weather" / "climate_report.pdf"
            dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_file_path.write_text("This is a dummy climate report content for testing summarization.")

            result_summarize = await weather_summarize_document_by_path(str(dummy_file_path))
            print(f"Summarize Result: {result_summarize}")
            assert "Mocked summary of climate_report.pdf" in result_summarize
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 6 Passed (no analytics expected for generic tool directly).")

            print("\nAll weather_tool tests with analytics considerations completed.")

        await run_weather_tests()

        # Restore original requests.get
        requests.get = original_requests_get

        # Clean up dummy files and directories
        test_user_dirs = [Path("uploads") / test_user_pro, BASE_VECTOR_DIR / test_user_pro]
        for d in test_user_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print(f"Cleaned up {d}")
