# domain_tools/travel_tools/travel_tool.py

import logging
import requests
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta
import asyncio # Import asyncio

# Import generic tools
from langchain_core.tools import tool
# REMOVED: from shared_tools.query_uploaded_docs_tool import QueryUploadedDocs
from shared_tools.scrapper_tool import scrape_web
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
                error_msg = f"Response path '{'.'.join(response_path)}' not found in API response from {active_provider_name}. Raw: {raw_data}"
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
                error_message=e
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
_mock_travel_data = {
    "flight_search": {
        "origin": "JFK",
        "destination": "LAX",
        "departure_date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
        "return_date": (datetime.now() + timedelta(days=37)).strftime("%Y-%m-%d"),
        "adults": 1,
        "currency": "USD",
        "flights": [
            {
                "id": "FL001",
                "airline": "MockAir",
                "flight_number": "MA101",
                "departure_airport": "JFK",
                "arrival_airport": "LAX",
                "departure_time": "08:00",
                "arrival_time": "11:00",
                "price": 250.00,
                "currency": "USD"
            },
            {
                "id": "FL002",
                "airline": "GlobalWings",
                "flight_number": "GW202",
                "departure_airport": "JFK",
                "arrival_airport": "LAX",
                "departure_time": "10:00",
                "arrival_time": "13:30",
                "price": 300.00,
                "currency": "USD"
            }
        ]
    },
    "hotel_search": {
        "city_code": "PAR",
        "check_in_date": (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d"),
        "check_out_date": (datetime.now() + timedelta(days=65)).strftime("%Y-%m-%d"),
        "adults": 2,
        "hotels": [
            {
                "id": "HO001",
                "name": "Mock Hotel Paris",
                "address": "123 Rue de la Mock, Paris",
                "rating": 4.5,
                "price_per_night": 150.00,
                "currency": "EUR"
            },
            {
                "id": "HO002",
                "name": "Grand Mock Resort",
                "address": "456 Av. des Champs-Mock, Paris",
                "rating": 5.0,
                "price_per_night": 250.00,
                "currency": "EUR"
            }
        ]
    },
    "destination_info": {
        "paris": {
            "name": "Paris",
            "description": "Paris, France's capital, is a major European city and a global center for art, fashion, gastronomy and culture. Its 19th-century cityscape is crisscrossed by wide boulevards and the River Seine. Beyond such landmarks as the Eiffel Tower and the 12th-century Gothic Notre-Dame cathedral, the city is known for its cafe culture and designer boutiques along the Rue du Faubourg Saint-Honoré.",
            "best_time_to_visit": "Spring (April to June) and Autumn (September to November)",
            "currency": "Euro (EUR)",
            "language": "French"
        },
        "tokyo": {
            "name": "Tokyo",
            "description": "Tokyo, Japan’s bustling capital, mixes the ultramodern and the traditional, from neon-lit skyscrapers and anime shops to cherry-blossom-lined temples and gardens. The opulent Meiji Shinto Shrine is known for its towering gate and surrounding woods. The Imperial Palace sits amid large public gardens. The city's many museums offer exhibits ranging from classical art to a reconstructed Edo-period village.",
            "best_time_to_visit": "Spring (March to May) and Autumn (September to November)",
            "currency": "Japanese Yen (JPY)",
            "language": "Japanese"
        }
    }
}

@tool
async def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: Optional[str] = None,
    adults: int = 1,
    currency: str = "USD",
    user_token: str = "default"
) -> str:
    """
    Searches for flights between specified origin and destination airports for given dates.
    Uses IATA airport codes (e.g., "JFK", "LAX", "LHR", "CDG").
    Departure and return dates should be in YYYY-MM-DD format.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        origin (str): The IATA code of the departure airport.
        destination (str): The IATA code of the arrival airport.
        departure_date (str): The departure date in YYYY-MM-DD format.
        return_date (str, optional): The return date in YYYY-MM-DD format for round trips.
                                     Defaults to None for one-way.
        adults (int, optional): The number of adult passengers. Defaults to 1.
        currency (str, optional): The desired currency for prices (e.g., "USD", "EUR", "GBP").
                                  Defaults to "USD".
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of flight search results, or an error/fallback message.
    """
    logger.info(f"Tool: search_flights called for {origin} to {destination} on {departure_date} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'travel_tool_access', False):
        return "Error: Access to travel tools is not enabled for your current tier."

    # Validate date formats
    try:
        parsed_departure_date = parse_date_to_yyyymmdd(departure_date)
        if not parsed_departure_date:
            return "Error: Invalid departure date format. Please use YYYY-MM-DD or a recognizable date string."
        
        parsed_return_date = None
        if return_date:
            parsed_return_date = parse_date_to_yyyymmdd(return_date)
            if not parsed_return_date:
                return "Error: Invalid return date format. Please use YYYY-MM-DD or a recognizable date string."
            if parsed_return_date < parsed_departure_date:
                return "Error: Return date cannot be before departure date."
    except Exception as e:
        return f"Error parsing dates: {e}. Please ensure dates are valid."

    params = {
        "origin": origin,
        "destination": destination,
        "departure_date": parsed_departure_date,
        "return_date": parsed_return_date,
        "adults": adults,
        "currency": currency
    }
    api_data = await _make_dynamic_api_request("travel", "search_flights", params, user_token) # Await the async call

    if api_data and api_data.get("data"):
        flights = api_data["data"]
        if flights:
            response_str = f"Found {len(flights)} flights from {origin} to {destination} for {departure_date}:\n"
            for flight in flights:
                response_str += (
                    f"  Flight {flight.get('flight_number', 'N/A')} ({flight.get('airline', 'N/A')}):\n"
                    f"    Departure: {flight.get('departure_airport', 'N/A')} at {flight.get('departure_time', 'N/A')}\n"
                    f"    Arrival: {flight.get('arrival_airport', 'N/A')} at {flight.get('arrival_time', 'N/A')}\n"
                    f"    Price: {flight.get('price', 'N/A')} {flight.get('currency', 'N/A')}\n"
                )
            return response_str
        else:
            return f"No live flights found for {origin} to {destination} on {departure_date}. Falling back to mock data."

    # Fallback to mock data
    mock_data = _mock_travel_data.get("flight_search")
    if mock_data and mock_data["origin"].lower() == origin.lower() and mock_data["destination"].lower() == destination.lower():
        flights = mock_data["flights"]
        response_str = f"Found {len(flights)} flights from {origin} to {destination} for {departure_date} (Mock Data Fallback):\n"
        for flight in flights:
            response_str += (
                f"  Flight {flight['flight_number']} ({flight['airline']}):\n"
                f"    Departure: {flight['departure_airport']} at {flight['departure_time']}\n"
                f"    Arrival: {flight['arrival_airport']} at {flight['arrival_time']}\n"
                f"    Price: {flight['price']} {flight['currency']}\n"
            )
        return response_str
    else:
        return f"Flight information not found for {origin} to {destination}. (API/Mock Fallback Failed)"


@tool
async def search_hotels(
    city_code: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 1,
    user_token: str = "default"
) -> str:
    """
    Searches for hotels in a specified city for given check-in and check-out dates.
    Uses IATA city codes (e.g., "PAR" for Paris, "NYC" for New York).
    Dates should be in YYYY-MM-DD format.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        city_code (str): The IATA code of the city (e.g., "PAR", "NYC").
        check_in_date (str): The check-in date in YYYY-MM-DD format.
        check_out_date (str): The check-out date in YYYY-MM-DD format.
        adults (int, optional): The number of adult guests. Defaults to 1.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of hotel search results, or an error/fallback message.
    """
    logger.info(f"Tool: search_hotels called for {city_code} from {check_in_date} to {check_out_date} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'travel_tool_access', False):
        return "Error: Access to travel tools is not enabled for your current tier."

    # Validate date formats
    try:
        parsed_check_in_date = parse_date_to_yyyymmdd(check_in_date)
        if not parsed_check_in_date:
            return "Error: Invalid check-in date format. Please use YYYY-MM-DD or a recognizable date string."
        
        parsed_check_out_date = parse_date_to_yyyymmdd(check_out_date)
        if not parsed_check_out_date:
            return "Error: Invalid check-out date format. Please use YYYY-MM-DD or a recognizable date string."
        
        if parsed_check_out_date <= parsed_check_in_date:
            return "Error: Check-out date must be after check-in date."
    except Exception as e:
        return f"Error parsing dates: {e}. Please ensure dates are valid."

    params = {
        "city_code": city_code,
        "check_in_date": parsed_check_in_date,
        "check_out_date": parsed_check_out_date,
        "adults": adults
    }
    api_data = await _make_dynamic_api_request("travel", "search_hotels", params, user_token) # Await the async call

    if api_data and api_data.get("data"):
        hotels = api_data["data"]
        if hotels:
            response_str = f"Found {len(hotels)} hotels in {city_code} for dates {check_in_date} to {check_out_date}:\n"
            for hotel in hotels:
                response_str += (
                    f"  Hotel: {hotel.get('name', 'N/A')}\n"
                    f"    Address: {hotel.get('address', 'N/A')}\n"
                    f"    Rating: {hotel.get('rating', 'N/A')} stars\n"
                    f"    Price per night: {hotel.get('price_per_night', 'N/A')} {hotel.get('currency', 'N/A')}\n"
                )
            return response_str
        else:
            return f"No live hotels found in {city_code} for the specified dates. Falling back to mock data."

    # Fallback to mock data
    mock_data = _mock_travel_data.get("hotel_search")
    if mock_data and mock_data["city_code"].lower() == city_code.lower():
        hotels = mock_data["hotels"]
        response_str = f"Found {len(hotels)} hotels in {city_code} for dates {check_in_date} to {check_out_date} (Mock Data Fallback):\n"
        for hotel in hotels:
            response_str += (
                f"  Hotel: {hotel['name']}\n"
                f"    Address: {hotel['address']}\n"
                f"    Rating: {hotel['rating']} stars\n"
                f"    Price per night: {hotel['price_per_night']} {hotel['currency']}\n"
            )
        return response_str
    else:
        return f"Hotel information not found for {city_code}. (API/Mock Fallback Failed)"


@tool
async def get_destination_info(destination_name: str, user_token: str = "default") -> str:
    """
    Retrieves general information about a travel destination, including description,
    best time to visit, currency, and language.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        destination_name (str): The name of the destination (e.g., "Paris", "Tokyo", "London").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of destination information, or an error/fallback message.
    """
    logger.info(f"Tool: get_destination_info called for destination: '{destination_name}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'travel_tool_access', False):
        return "Error: Access to travel tools is not enabled for your current tier."
    
    params = {"destination_name": destination_name}
    api_data = await _make_dynamic_api_request("travel", "get_destination_info", params, user_token) # Await the async call

    if api_data:
        try:
            name = api_data.get("name")
            description = api_data.get("description")
            best_time = api_data.get("best_time_to_visit")
            currency = api_data.get("currency")
            language = api_data.get("language")

            if name and description:
                response_str = (
                    f"Information about {name}:\n"
                    f"  Description: {description}\n"
                )
                if best_time:
                    response_str += f"  Best time to visit: {best_time}\n"
                if currency:
                    response_str += f"  Local Currency: {currency}\n"
                if language:
                    response_str += f"  Official Language: {language}\n"
                return response_str
            else:
                logger.warning(f"Live API data for destination info for '{destination_name}' is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live destination information for '{destination_name}'. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live destination info data for '{destination_name}': {e}")
            return f"Error parsing live data for '{destination_name}'. Falling back to mock data."

    # Fallback to mock data
    mock_data_key = destination_name.lower().replace(" ", "_")
    mock_data = _mock_travel_data.get("destination_info", {}).get(mock_data_key)
    if mock_data:
        response_str = (
            f"Information about {mock_data['name']} (Mock Data Fallback):\n"
            f"  Description: {mock_data['description']}\n"
            f"  Best time to visit: {mock_data['best_time_to_visit']}\n"
            f"  Local Currency: {mock_data['currency']}\n"
            f"  Official Language: {mock_data['language']}\n"
        )
        return response_str
    else:
        return f"Destination information not found for '{destination_name}'. (API/Mock Fallback Failed)"


# --- Existing Generic Tools (not directly using external APIs, but can be used in travel context) ---

@tool
def travel_search_web(query: str, user_token: str = "default", max_chars: int = 2000) -> str:
    """
    Searches the web for travel-related information using a smart search fallback mechanism.
    This tool wraps the generic `scrape_web` tool, providing a travel-specific interface.
    
    Args:
        query (str): The travel-related search query (e.g., "visa requirements for Japan", "best beaches in Thailand").
        user_token (str): The unique identifier for the user. Defaults to "default".
        max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
    
    Returns:
        str: A string containing relevant information from the web.
    """
    logger.info(f"Tool: travel_search_web called with query: '{query}' for user: '{user_token}'")
    return scrape_web(query=query, user_token=user_token, max_chars=max_chars)

@tool
async def travel_query_uploaded_docs(query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
    """
    Queries previously uploaded and indexed travel documents for a user using vector similarity search.
    This tool wraps the generic `QueryUploadedDocs` tool, fixing the section to "travel".
    
    Args:
        query (str): The search query to find relevant travel documents (e.g., "my flight booking details", "packing list for Europe").
        user_token (str): The unique identifier for the user. Defaults to "default".
        export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
        k (int): The number of top relevant documents to retrieve. Defaults to 5.
    
    Returns:
        str: A string containing the combined content of the relevant document chunks,
             or a message indicating no data/results found, or the export path if exported.
    """
    logger.info(f"Tool: travel_query_uploaded_docs called with query: '{query}' for user: '{user_token}'")
    # This will be replaced by a call to self.document_tools.query_uploaded_docs
    # For now, keeping the original call for review purposes.
    # Assuming QueryUploadedDocs is an async tool or can be awaited
    # If QueryUploadedDocs is not async, remove 'await' and make this function non-async
    return f"Mocked document query results for '{query}' in section 'travel'." # Return mock string for now


@tool
async def travel_summarize_document_by_path(file_path_str: str) -> str:
    """
    Summarizes a document related to travel located at the given file path.
    The file path should be accessible by the system (e.g., in the 'uploads' directory).
    This tool wraps the generic `summarize_document` tool.
    
    Args:
        file_path_str (str): The full path to the document file to be summarized.
                              Example: "uploads/default/travel/visa_requirements.pdf"
    
    Returns:
        str: A concise summary of the document content.
    """
    logger.info(f"Tool: travel_summarize_document_by_path called for file: '{file_path_str}'")
    file_path = Path(file_path_str)
    if not file_path.exists():
        logger.error(f"Document not found at '{file_path_str}' for summarization.")
        return f"Error: Document not found at '{file_path_str}'."
    
    try:
        # Assuming summarize_document is an async tool or can be awaited
        summary = await summarize_document(file_path.read_text(), user_token="default") # Await and pass text content
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

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.amadeus_api_key = "MOCK_AMADEUS_CLIENT_ID"
            self.amadeus_api_secret = "MOCK_AMADEUS_CLIENT_SECRET"
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
                    'travel': 'amadeus'
                },
                'analytics': { # Mock analytics settings
                    'enabled': True,
                    'log_tool_usage': True,
                    'log_query_failures': True
                }
            }
            self._api_providers_data = { # Mock api_providers_data for travel
                "travel": {
                    "amadeus": {
                        "base_url": "https://api.example.com/amadeus",
                        "token_endpoint": "https://api.example.com/amadeus/oauth2/token",
                        "api_key_name": "amadeus_api_key",
                        "api_secret_name": "amadeus_api_secret",
                        "functions": {
                            "search_flights": {
                                "endpoint": "/v1/shopping/flight-offers",
                                "required_params": ["originLocationCode", "destinationLocationCode", "departureDate"],
                                "optional_params": ["returnDate", "adults", "currencyCode"],
                                "data_map": {
                                    "id": "id",
                                    "airline": ["validatingAirlineCodes", 0], # Mocking first airline
                                    "flight_number": ["itineraries", 0, "segments", 0, "number"],
                                    "departure_airport": ["itineraries", 0, "segments", 0, "departure", "iataCode"],
                                    "arrival_airport": ["itineraries", 0, "segments", 0, "arrival", "iataCode"],
                                    "departure_time": ["itineraries", 0, "segments", 0, "departure", "at"],
                                    "arrival_time": ["itineraries", 0, "segments", 0, "arrival", "at"],
                                    "price": ["price", "total"],
                                    "currency": ["price", "currency"]
                                }
                            },
                            "search_hotels": {
                                "endpoint": "/v1/reference-data/locations/hotels/by-city",
                                "required_params": ["cityCode"],
                                "optional_params": ["checkInDate", "checkOutDate", "adults"],
                                "data_map": {
                                    "id": "hotelId",
                                    "name": "name",
                                    "address": ["address", "lines", 0],
                                    "rating": "rating",
                                    "price_per_night": "price.amount", # Mocking a price field
                                    "currency": "price.currency"
                                }
                            },
                            "get_destination_info": { # Hypothetical endpoint for destination info
                                "endpoint": "/v1/travel/destinations",
                                "required_params": ["destinationName"],
                                "data_map": {
                                    "name": "name",
                                    "description": "description",
                                    "best_time_to_visit": "bestSeason",
                                    "currency": "currency",
                                    "language": "language"
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
                'travel_tool_access': {
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
                },
                'summarization_enabled': { # For summarize_document
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'llm_default_provider': { # For summarize_document
                    'default': 'gemini',
                    'tiers': {'pro': 'gemini', 'premium': 'openai', 'admin': 'gemini'}
                },
                'llm_default_model_name': { # For summarize_document
                    'default': 'gemini-1.5-flash',
                    'tiers': {'pro': 'gemini-1.5-flash', 'premium': 'gpt-4o', 'admin': 'gemini-1.5-flash'}
                },
                'llm_default_temperature': { # For summarize_document
                    'default': 0.7,
                    'tiers': {'pro': 0.5, 'premium': 0.3, 'admin': 0.7}
                },
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

        # Mock requests.get and requests.post for external API calls
        original_requests_get = requests.get
        original_requests_post = requests.post

        def mock_requests_post_dynamic(url, data, timeout):
            if "api.example.com/amadeus/oauth2/token" in url:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"access_token": "mock_amadeus_token_123"}
                return mock_response
            return original_requests_post(url, data=data, timeout=timeout)

        def mock_requests_get_dynamic(url, params, headers, timeout):
            # Simulate hypothetical Amadeus API responses
            if "api.example.com/amadeus" in url:
                if "/v1/shopping/flight-offers" in url:
                    origin = params.get("originLocationCode", "").lower()
                    destination = params.get("destinationLocationCode", "").lower()
                    if "jfk" in origin and "lax" in destination:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "data": [
                                {
                                    "id": "FL001",
                                    "validatingAirlineCodes": ["MA"],
                                    "itineraries": [{
                                        "segments": [{
                                            "number": "101",
                                            "departure": {"iataCode": "JFK", "at": "2025-08-01T08:00:00"},
                                            "arrival": {"iataCode": "LAX", "at": "2025-08-01T11:00:00"}
                                        }]
                                    }],
                                    "price": {"total": "250.00", "currency": "USD"}
                                }
                            ]
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"data": []}
                        return mock_response
                elif "/v1/reference-data/locations/hotels/by-city" in url:
                    city_code = params.get("cityCode", "").lower()
                    if "par" in city_code:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "data": [
                                {
                                    "hotelId": "HO001",
                                    "name": "Live Hotel Paris",
                                    "address": {"lines": ["1 Rue Live"]},
                                    "rating": 4.0,
                                    "price": {"amount": "180.00", "currency": "EUR"}
                                }
                            ]
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"data": []}
                        return mock_response
                elif "/v1/travel/destinations" in url:
                    destination_name = params.get("destinationName", "").lower()
                    if "paris" in destination_name:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "name": "Paris",
                            "description": "The city of lights, live version.",
                            "bestSeason": "Spring",
                            "currency": "EUR",
                            "language": "French"
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {}
                        return mock_response

            # Simulate scrape_web's internal requests.get if needed
            if "google.com/search" in url or "example.com" in url: # Mock for scrape_web
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.text = f"<html><body><h1>Search results for {params.get('q', 'travel')}</h1><p>Some travel related content from web search.</p></body></html>"
                return mock_response

            return original_requests_get(url, params=params, headers=headers, timeout=timeout)

        requests.get = mock_requests_get_dynamic
        requests.post = mock_requests_post_dynamic

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
            async def __call__(self): # Made async
                return f"Mocked document query results for '{self.query}' in section '{self.section}'."

        # Mock for summarize_document
        class MockSummarizeDocument:
            async def __call__(self, text_content, user_token): # Made async
                return f"Mocked summary of text for user {user_token}: {text_content[:50]}..."

        # Patch QueryUploadedDocs and summarize_document in the travel_tool module
        # original_QueryUploadedDocs = sys.modules['domain_tools.travel_tools.travel_tool'].QueryUploadedDocs # Not needed anymore
        original_summarize_document = sys.modules['domain_tools.travel_tools.travel_tool'].summarize_document
        # sys.modules['domain_tools.travel_tools.travel_tool'].QueryUploadedDocs = MockQueryUploadedDocs # Not needed anymore
        sys.modules['domain_tools.travel_tools.travel_tool'].summarize_document = MockSummarizeDocument()


        async def run_travel_tests():
            print("\n--- Testing travel_tool functions with Analytics ---")

            # Test search_flights (success)
            print("\n--- Test 1: search_flights (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_flights = await search_flights("JFK", "LAX", "2025-08-01", user_token=test_user_pro)
            print(f"Flight Search Result: {result_flights}")
            assert "Found 1 flights from JFK to LAX for 2025-08-01:" in result_flights
            assert "Price: 250.00 USD" in result_flights
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "travel_search_flights"
            assert logged_data["success"] is True
            print("Test 1 Passed (and analytics logged success).")

            # Test search_hotels (API failure - no data found)
            print("\n--- Test 2: search_hotels (API Failure) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_hotels = await search_hotels("XYZ", "2025-09-01", "2025-09-05", user_token=test_user_pro)
            print(f"Hotel Search Result (API Error): {result_hotels}")
            assert "No live hotels found in XYZ for the specified dates." in result_hotels
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "travel_search_hotels"
            assert logged_data["success"] is False
            assert "Response path 'data' not found" in logged_data["error_message"] or "incomplete" in logged_data["error_message"]
            print("Test 2 Passed (and analytics logged failure).")

            # Test get_destination_info (RBAC denied)
            print("\n--- Test 3: get_destination_info (RBAC Denied) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_destination_info_rbac_denied = await get_destination_info("Rome", user_token=test_user_free)
            print(f"Destination Info (Free User, RBAC Denied): {result_destination_info_rbac_denied}")
            assert "Error: Access to travel tools is not enabled for your current tier." in result_destination_info_rbac_denied
            # No analytics log expected here because RBAC check happens before _make_dynamic_api_request
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 3 Passed (RBAC correctly prevented call and no analytics logged).")

            # Test travel_search_web (generic tool, not using _make_dynamic_api_request)
            print("\n--- Test 4: travel_search_web (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_web_search = await travel_search_web("best travel insurance for Europe", user_token=test_user_pro)
            print(f"Web Search Result: {result_web_search[:100]}...")
            assert "Search results for best travel insurance for Europe" in result_web_search
            # Analytics for generic tools like scrape_web or summarize_document
            # would need to be integrated within those shared_tools themselves,
            # or wrapped by a higher-level agent logging.
            # For now, we are focusing on _make_dynamic_api_request.
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 4 Passed (no analytics expected for generic tool directly).\n")

            # Test 5: travel_query_uploaded_docs (generic tool)
            print("\n--- Test 5: travel_query_uploaded_docs (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_doc_query = await travel_query_uploaded_docs("my flight booking details", user_token=test_user_pro)
            print(f"Document Query Result: {result_doc_query}")
            assert "Mocked document query results for 'my flight booking details' in section 'travel'." in result_doc_query
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 5 Passed (no analytics expected for generic tool directly, will be logged by DocumentTools).")

            # Test 6: travel_summarize_document_by_path (generic tool)
            print("\n--- Test 6: travel_summarize_document_by_path (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            # Create a dummy file for summarization test
            dummy_file_path = Path("uploads") / test_user_pro / "travel" / "europe_itinerary.pdf"
            dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_file_path.write_text("This is a dummy travel itinerary for testing summarization.")

            result_summarize = await travel_summarize_document_by_path(str(dummy_file_path))
            print(f"Summarize Result: {result_summarize}")
            assert "Mocked summary of text for user default" in result_summarize
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 6 Passed (no analytics expected for generic tool directly).\n")


            print("\nAll travel_tool tests with analytics considerations completed.")

        # Ensure tests are only run when the script is executed directly
        if __name__ == "__main__":
            # Use asyncio.run to execute the async test function
            asyncio.run(run_travel_tests())

        # Restore original requests.get and requests.post
        requests.get = original_requests_get
        requests.post = original_requests_post

        # Restore original summarize_document
        sys.modules['domain_tools.travel_tools.travel_tool'].summarize_document = original_summarize_document

        # Clean up dummy files and directories
        test_user_dirs = [Path("uploads") / test_user_pro, BASE_VECTOR_DIR / test_user_pro]
        for d in test_user_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print(f"Cleaned up {d}")


