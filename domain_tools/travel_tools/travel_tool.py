# domain_tools/travel_tools/travel_tool.py

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
_mock_travel_data = {
    "flights": [
        {
            "flight_id": "FLT-001",
            "origin": "LAG",
            "destination": "NYC",
            "departure_date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
            "return_date": (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
            "airline": "MockAir",
            "price": "500 USD",
            "duration": "10h 30m",
            "stops": 1
        },
        {
            "flight_id": "FLT-002",
            "origin": "NYC",
            "destination": "LON",
            "departure_date": (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d"),
            "return_date": (datetime.now() + timedelta(days=17)).strftime("%Y-%m-%d"),
            "airline": "GlobalWings",
            "price": "650 USD",
            "duration": "7h 0m",
            "stops": 0
        }
    ],
    "hotels": [
        {
            "hotel_id": "HTL-001",
            "name": "Grand Central Hotel",
            "location": "New York, USA",
            "check_in": (datetime.now() + timedelta(days=20)).strftime("%Y-%m-%d"),
            "check_out": (datetime.now() + timedelta(days=23)).strftime("%Y-%m-%d"),
            "price_per_night": "150 USD",
            "rating": "4.5 stars",
            "amenities": ["WiFi", "Pool", "Gym"]
        },
        {
            "hotel_id": "HTL-002",
            "name": "Riverside Inn",
            "location": "London, UK",
            "check_in": (datetime.now() + timedelta(days=25)).strftime("%Y-%m-%d"),
            "check_out": (datetime.now() + timedelta(days=28)).strftime("%Y-%m-%d"),
            "price_per_night": "120 GBP",
            "rating": "4.0 stars",
            "amenities": ["WiFi", "Breakfast"]
        }
    ],
    "destinations": {
        "paris": {
            "name": "Paris, France",
            "description": "The 'City of Love', famous for its Eiffel Tower, Louvre Museum, and exquisite cuisine.",
            "attractions": ["Eiffel Tower", "Louvre Museum", "Notre Dame Cathedral"],
            "best_time_to_visit": "Spring (April-June) or Autumn (September-November)"
        },
        "tokyo": {
            "name": "Tokyo, Japan",
            "description": "A bustling metropolis blending traditional culture with futuristic technology.",
            "attractions": ["Tokyo Skytree", "Shibuya Crossing", "Senso-ji Temple"],
            "best_time_to_visit": "Spring (March-May) for cherry blossoms or Autumn (September-November) for foliage"
        }
    }
}

@tool
def search_flights(origin: str, destination: str, departure_date: str, return_date: Optional[str] = None, user_token: str = "default") -> str:
    """
    Searches for flight information between an origin and destination, for specified departure and optional return dates.
    Dates can be in various formats (e.g., 'YYYY-MM-DD', 'MM/DD/YYYY', 'August 15, 2024').
    Falls back to mock data if API key is missing or API call fails.

    Args:
        origin (str): The origin airport code (e.g., 'LAG' for Lagos) or city name.
        destination (str): The destination airport code (e.g., 'NYC' for New York) or city name.
        departure_date (str): The desired departure date.
        return_date (str, optional): The desired return date for a round trip.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of flight information, or an error/fallback message.
    """
    logger.info(f"Tool: search_flights called for origin='{origin}', destination='{destination}', departure_date='{departure_date}', return_date='{return_date}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'travel_tool_access', False):
        return "Error: Access to travel tools is not enabled for your current tier."
    
    parsed_departure_date = parse_date_to_yyyymmdd(departure_date)
    if not parsed_departure_date:
        return "Error: Could not parse the provided departure date. Please ensure the date is valid."
    
    parsed_return_date = None
    if return_date:
        parsed_return_date = parse_date_to_yyyymmdd(return_date)
        if not parsed_return_date:
            return "Error: Could not parse the provided return date. Please ensure the date is valid."

    params = {
        "origin": origin,
        "destination": destination,
        "departure_date": parsed_departure_date
    }
    if parsed_return_date:
        params["return_date"] = parsed_return_date

    api_data = _make_dynamic_api_request(
        "travel", "search_flights",
        params,
        user_token
    )

    if api_data and api_data.get("data"):
        flights = api_data["data"]
        if flights:
            response_str = "Found Flights:\n"
            for i, flight in enumerate(flights[:5]): # Limit to top 5 flights
                dep_date = flight.get('departure_date', 'N/A')
                ret_date = flight.get('return_date', 'N/A')
                try:
                    dep_date = datetime.strptime(dep_date, "%Y-%m-%d").strftime("%B %d, %Y")
                except ValueError: pass
                try:
                    ret_date = datetime.strptime(ret_date, "%Y-%m-%d").strftime("%B %d, %Y")
                except ValueError: pass

                response_str += (
                    f"{i+1}. From: {flight.get('origin', 'N/A')} to {flight.get('destination', 'N/A')}\n"
                    f"   Departure: {dep_date}\n"
                )
                if ret_date != 'N/A':
                    response_str += f"   Return: {ret_date}\n"
                response_str += (
                    f"   Airline: {flight.get('airline', 'N/A')}\n"
                    f"   Price: {flight.get('price', 'N/A')}\n"
                    f"   Duration: {flight.get('duration', 'N/A')}\n"
                    f"   Stops: {flight.get('stops', 'N/A')}\n\n"
                )
            return response_str
        else:
            return f"No live flights found for your criteria (origin='{origin}', destination='{destination}', departure_date='{departure_date}'). Falling back to mock data."

    # Fallback to mock data
    mock_flights = _mock_travel_data.get("flights", [])
    filtered_mock_flights = []
    for flight in mock_flights:
        match = True
        if origin.lower() not in flight.get("origin", "").lower() and origin.lower() not in flight.get("origin_city", "").lower():
            match = False
        if destination.lower() not in flight.get("destination", "").lower() and destination.lower() not in flight.get("destination_city", "").lower():
            match = False
        if parsed_departure_date and flight.get("departure_date") != parsed_departure_date:
            match = False
        if parsed_return_date and flight.get("return_date") != parsed_return_date:
            match = False
        if match:
            filtered_mock_flights.append(flight)

    if filtered_mock_flights:
        response_str = "Found Flights (Mock Data Fallback):\n"
        for i, flight in enumerate(filtered_mock_flights[:2]): # Limit mock to top 2
            dep_date = flight.get('departure_date', 'N/A')
            ret_date = flight.get('return_date', 'N/A')
            try:
                dep_date = datetime.strptime(dep_date, "%Y-%m-%d").strftime("%B %d, %Y")
            except ValueError: pass
            try:
                ret_date = datetime.strptime(ret_date, "%Y-%m-%d").strftime("%B %d, %Y")
            except ValueError: pass

            response_str += (
                f"{i+1}. From: {flight.get('origin', 'N/A')} to {flight.get('destination', 'N/A')}\n"
                f"   Departure: {dep_date}\n"
            )
            if ret_date != 'N/A':
                response_str += f"   Return: {ret_date}\n"
            response_str += (
                f"   Airline: {flight.get('airline', 'N/A')}\n"
                f"   Price: {flight.get('price', 'N/A')}\n"
                f"   Duration: {flight.get('duration', 'N/A')}\n"
                f"   Stops: {flight.get('stops', 'N/A')}\n\n"
            )
        return response_str
    else:
        return f"Flight information not found for your criteria. (API/Mock Fallback Failed)"


@tool
def search_hotels(location: str, check_in_date: str, check_out_date: str, user_token: str = "default") -> str:
    """
    Searches for hotel availability in a specified location for given check-in and check-out dates.
    Dates can be in various formats (e.g., 'YYYY-MM-DD', 'MM/DD/YYYY', 'September 1, 2024').
    Falls back to mock data if API key is missing or API call fails.

    Args:
        location (str): The city or specific area for the hotel search (e.g., "New York", "London City Centre").
        check_in_date (str): The desired check-in date.
        check_out_date (str): The desired check-out date.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of hotel information, or an error/fallback message.
    """
    logger.info(f"Tool: search_hotels called for location='{location}', check_in='{check_in_date}', check_out='{check_out_date}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'travel_tool_access', False):
        return "Error: Access to travel tools is not enabled for your current tier."
    
    parsed_check_in_date = parse_date_to_yyyymmdd(check_in_date)
    if not parsed_check_in_date:
        return "Error: Could not parse the provided check-in date. Please ensure the date is valid."
    
    parsed_check_out_date = parse_date_to_yyyymmdd(check_out_date)
    if not parsed_check_out_date:
        return "Error: Could not parse the provided check-out date. Please ensure the date is valid."

    params = {
        "location": location,
        "check_in": parsed_check_in_date,
        "check_out": parsed_check_out_date
    }

    api_data = _make_dynamic_api_request(
        "travel", "search_hotels",
        params,
        user_token
    )

    if api_data and api_data.get("data"):
        hotels = api_data["data"]
        if hotels:
            response_str = "Found Hotels:\n"
            for i, hotel in enumerate(hotels[:5]): # Limit to top 5 hotels
                check_in_str = hotel.get('check_in', 'N/A')
                check_out_str = hotel.get('check_out', 'N/A')
                try:
                    check_in_str = datetime.strptime(check_in_str, "%Y-%m-%d").strftime("%B %d, %Y")
                except ValueError: pass
                try:
                    check_out_str = datetime.strptime(check_out_str, "%Y-%m-%d").strftime("%B %d, %Y")
                except ValueError: pass

                response_str += (
                    f"{i+1}. Name: {hotel.get('name', 'N/A')}\n"
                    f"   Location: {hotel.get('location', 'N/A')}\n"
                    f"   Check-in: {check_in_str}\n"
                    f"   Check-out: {check_out_str}\n"
                    f"   Price per Night: {hotel.get('price_per_night', 'N/A')}\n"
                    f"   Rating: {hotel.get('rating', 'N/A')}\n"
                    f"   Amenities: {', '.join(hotel.get('amenities', []))}\n\n"
                )
            return response_str
        else:
            return f"No live hotels found for your criteria (location='{location}', check_in='{check_in_date}', check_out='{check_out_date}'). Falling back to mock data."

    # Fallback to mock data
    mock_hotels = _mock_travel_data.get("hotels", [])
    filtered_mock_hotels = []
    for hotel in mock_hotels:
        match = True
        if location.lower() not in hotel.get("location", "").lower() and location.lower() not in hotel.get("name", "").lower():
            match = False
        if parsed_check_in_date and hotel.get("check_in") != parsed_check_in_date:
            match = False
        if parsed_check_out_date and hotel.get("check_out") != parsed_check_out_date:
            match = False
        if match:
            filtered_mock_hotels.append(hotel)

    if filtered_mock_hotels:
        response_str = "Found Hotels (Mock Data Fallback):\n"
        for i, hotel in enumerate(filtered_mock_hotels[:2]): # Limit mock to top 2
            check_in_str = hotel.get('check_in', 'N/A')
            check_out_str = hotel.get('check_out', 'N/A')
            try:
                check_in_str = datetime.strptime(check_in_str, "%Y-%m-%d").strftime("%B %d, %Y")
            except ValueError: pass
            try:
                check_out_str = datetime.strptime(check_out_str, "%Y-%m-%d").strftime("%B %d, %Y")
            except ValueError: pass
            response_str += (
                f"{i+1}. Name: {hotel.get('name', 'N/A')}\n"
                f"   Location: {hotel.get('location', 'N/A')}\n"
                f"   Check-in: {check_in_str}\n"
                f"   Check-out: {check_out_str}\n"
                f"   Price per Night: {hotel.get('price_per_night', 'N/A')}\n"
                f"   Rating: {hotel.get('rating', 'N/A')}\n"
                f"   Amenities: {', '.join(hotel.get('amenities', []))}\n\n"
            )
        return response_str
    else:
        return f"Hotel information not found for your criteria. (API/Mock Fallback Failed)"


@tool
def get_destination_info(destination_name: str, user_token: str = "default") -> str:
    """
    Retrieves information about a travel destination, including its description, attractions, and best time to visit.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        destination_name (str): The name of the destination (e.g., "Paris", "Tokyo").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of destination information, or an error/fallback message.
    """
    logger.info(f"Tool: get_destination_info called for destination: {destination_name} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'travel_tool_access', False):
        return "Error: Access to travel tools is not enabled for your current tier."
    
    params = {"name": destination_name}

    api_data = _make_dynamic_api_request(
        "travel", "get_destination_info",
        params,
        user_token
    )

    if api_data:
        try:
            name = api_data.get("name")
            description = api_data.get("description")
            attractions = api_data.get("attractions")
            best_time_to_visit = api_data.get("best_time_to_visit")

            if name and description:
                response_str = (
                    f"Information for Destination: {name}\n"
                    f"  Description: {description}\n"
                )
                if attractions:
                    response_str += f"  Main Attractions: {', '.join(attractions)}\n"
                if best_time_to_visit:
                    response_str += f"  Best Time to Visit: {best_time_to_visit}\n"
                return response_str
            else:
                logger.warning(f"Live API data for destination '{destination_name}' is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live destination information for '{destination_name}'. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live destination data for '{destination_name}': {e}")
            return f"Error parsing live data for '{destination_name}'. Falling back to mock data."

    # Fallback to mock data
    mock_data_key = destination_name.lower().replace(" ", "_").replace(",", "")
    mock_data = _mock_travel_data.get("destinations", {}).get(mock_data_key)
    if mock_data:
        response_str = (
            f"Information for Destination: {mock_data['name']} (Mock Data Fallback)\n"
            f"  Description: {mock_data['description']}\n"
        )
        if mock_data.get('attractions'):
            response_str += f"  Main Attractions: {', '.join(mock_data['attractions'])}\n"
        if mock_data.get('best_time_to_visit'):
            response_str += f"  Best Time to Visit: {mock_data['best_time_to_visit']}\n"
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
        query (str): The travel-related search query (e.g., "best places to visit in Europe", "visa requirements for Nigeria").
        user_token (str): The unique identifier for the user. Defaults to "default".
        max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
    
    Returns:
        str: A string containing relevant information from the web.
    """
    logger.info(f"Tool: travel_search_web called with query: '{query}' for user: '{user_token}'")
    return scrape_web(query=query, user_token=user_token, max_chars=max_chars)

@tool
def travel_query_uploaded_docs(query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
    """
    Queries previously uploaded and indexed travel documents for a user using vector similarity search.
    This tool wraps the generic `QueryUploadedDocs` tool, fixing the section to "travel".
    
    Args:
        query (str): The search query to find relevant travel documents (e.g., "my travel itinerary for Paris", "packing list for a beach vacation").
        user_token (str): The unique identifier for the user. Defaults to "default".
        export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
        k (int): The number of top relevant documents to retrieve. Defaults to 5.
    
    Returns:
        str: A string containing the combined content of the relevant document chunks,
             or a message indicating no data/results found, or the export path if exported.
    """
    logger.info(f"Tool: travel_query_uploaded_docs called with query: '{query}' for user: '{user_token}'")
    return QueryUploadedDocs(query=query, user_token=user_token, section="travel", export=export, k=k)

@tool
def travel_summarize_document_by_path(file_path_str: str) -> str:
    """
    Summarizes a document related to travel (e.g., itineraries, travel guides) located at the given file path.
    The file path should be accessible by the system (e.g., in the 'uploads' directory).
    This tool wraps the generic `summarize_document` tool.
    
    Args:
        file_path_str (str): The full path to the document file to be summarized.
                              Example: "uploads/default/travel/europe_itinerary.pdf"
    
    Returns:
        str: A concise summary of the document content.
    """
    logger.info(f"Tool: travel_summarize_document_by_path called for file: '{file_path_str}'")
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
            self.travel_api_key = "MOCK_TRAVEL_API_KEY"
            self.amadeus_client_id = "MOCK_AMADEUS_CLIENT_ID"
            self.amadeus_client_secret = "MOCK_AMADEUS_CLIENT_SECRET"
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
                    'travel': 'travel_api' # Default to hypothetical API
                }
            }
            self._api_providers_data = { # Mock api_providers_data for travel
                "travel": {
                    "travel_api": { # Hypothetical generic travel API
                        "base_url": "https://api.example.com/travel",
                        "api_key_name": "travel_api_key",
                        "api_key_param_name": "api_key",
                        "functions": {
                            "search_flights": {
                                "endpoint": "/flights",
                                "required_params": ["origin", "destination", "departure_date"],
                                "optional_params": ["return_date"],
                                "response_path": ["data"],
                                "data_map": {
                                    "flight_id": "id",
                                    "origin": "origin",
                                    "destination": "destination",
                                    "departure_date": "departure_date",
                                    "return_date": "return_date",
                                    "airline": "airline",
                                    "price": "price",
                                    "duration": "duration",
                                    "stops": "stops"
                                }
                            },
                            "search_hotels": {
                                "endpoint": "/hotels",
                                "required_params": ["location", "check_in", "check_out"],
                                "response_path": ["data"],
                                "data_map": {
                                    "hotel_id": "id",
                                    "name": "name",
                                    "location": "location",
                                    "check_in": "check_in",
                                    "check_out": "check_out",
                                    "price_per_night": "price_per_night",
                                    "rating": "rating",
                                    "amenities": "amenities"
                                }
                            },
                            "get_destination_info": {
                                "endpoint": "/destinations",
                                "required_params": ["name"],
                                "response_path": ["data", 0], # Assuming first result is most relevant
                                "data_map": {
                                    "name": "name",
                                    "description": "description",
                                    "attractions": "attractions",
                                    "best_time_to_visit": "best_time"
                                }
                            }
                        }
                    },
                    "amadeus": { # Amadeus configuration for flight search (example from api_providers.yml)
                        "base_url": "https://test.api.amadeus.com/v2",
                        "api_key_name": "amadeus_client_id",
                        "api_secret_name": "amadeus_client_secret",
                        "token_endpoint": "https://test.api.amadeus.com/v1/security/oauth2/token",
                        "functions": {
                            "search_flights": { # This would be the actual Amadeus flight search
                                "endpoint": "/shopping/flight-offers",
                                "required_params": ["originLocationCode", "destinationLocationCode", "departureDate"],
                                "optional_params": ["returnDate", "adults", "travelClass", "max"],
                                "response_path": ["data"],
                                "data_map": {
                                    "flight_id": "id",
                                    "origin": "itineraries.segments.departure.iataCode", # Example nested path
                                    "destination": "itineraries.segments.arrival.iataCode",
                                    "departure_date": "itineraries.segments.departure.at",
                                    "return_date": "itineraries.segments.arrival.at",
                                    "airline": "validatingAirlineCodes", # This might need more complex mapping
                                    "price": "price.grandTotal",
                                    "duration": "itineraries.duration",
                                    "stops": "itineraries.segments.numberOfStops"
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
        # Simulate hypothetical Travel API responses
        if "api.example.com/travel" in url:
            if "/flights" in url:
                origin = params.get("origin", "").upper()
                destination = params.get("destination", "").upper()
                departure_date = params.get("departure_date")
                return_date = params.get("return_date")
                
                mock_flights = [
                    {
                        "id": "FLT-001",
                        "origin": "LAG", "destination": "NYC",
                        "departure_date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                        "return_date": (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
                        "airline": "MockAir", "price": "500 USD", "duration": "10h 30m", "stops": 1
                    },
                    {
                        "id": "FLT-002",
                        "origin": "NYC", "destination": "LON",
                        "departure_date": (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d"),
                        "return_date": (datetime.now() + timedelta(days=17)).strftime("%Y-%m-%d"),
                        "airline": "GlobalWings", "price": "650 USD", "duration": "7h 0m", "stops": 0
                    }
                ]
                
                filtered_flights = []
                for flight in mock_flights:
                    match = True
                    if origin and flight["origin"] != origin:
                        match = False
                    if destination and flight["destination"] != destination:
                        match = False
                    if departure_date and flight["departure_date"] != departure_date:
                        match = False
                    if return_date and flight.get("return_date") != return_date:
                        match = False
                    if match:
                        filtered_flights.append(flight)

                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"data": filtered_flights}
                return mock_response

            elif "/hotels" in url:
                location = params.get("location", "").lower()
                check_in = params.get("check_in")
                check_out = params.get("check_out")

                mock_hotels = [
                    {
                        "id": "HTL-001",
                        "name": "Grand Central Hotel",
                        "location": "New York, USA",
                        "check_in": (datetime.now() + timedelta(days=20)).strftime("%Y-%m-%d"),
                        "check_out": (datetime.now() + timedelta(days=23)).strftime("%Y-%m-%d"),
                        "price_per_night": "150 USD",
                        "rating": "4.5 stars",
                        "amenities": ["WiFi", "Pool", "Gym"]
                    },
                    {
                        "id": "HTL-002",
                        "name": "Riverside Inn",
                        "location": "London, UK",
                        "check_in": (datetime.now() + timedelta(days=25)).strftime("%Y-%m-%d"),
                        "check_out": (datetime.now() + timedelta(days=28)).strftime("%Y-%m-%d"),
                        "price_per_night": "120 GBP",
                        "rating": "4.0 stars",
                        "amenities": ["WiFi", "Breakfast"]
                    }
                ]
                
                filtered_hotels = []
                for hotel in mock_hotels:
                    match = True
                    if location and location not in hotel["location"].lower():
                        match = False
                    if check_in and hotel["check_in"] != check_in:
                        match = False
                    if check_out and hotel["check_out"] != check_out:
                        match = False
                    if match:
                        filtered_hotels.append(hotel)

                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"data": filtered_hotels}
                return mock_response
            
            elif "/destinations" in url:
                name = params.get("name", "").lower()
                if "paris" in name:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "data": [{
                            "name": "Paris, France",
                            "description": "City of Love.",
                            "attractions": ["Eiffel Tower", "Louvre"],
                            "best_time": "Spring"
                        }]
                    }
                    return mock_response
                else:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"data": []}
                    return mock_response
        
        # Simulate scrape_web's internal requests.get if needed
        if "google.com/search" in url or "example.com" in url: # Mock for scrape_web
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = f"<html><body><h1>Search results for {params.get('q', 'travel')}</h1><p>Some travel related content from web search.</p></body></html>"
            return mock_response

        return original_requests_get(url, params=params, headers=headers, timeout=timeout)

    requests.get = mock_requests_get_dynamic

    test_user_pro = "mock_pro_token"
    test_user_free = "mock_free_token"

    print("\n--- Testing travel_tool functions ---")

    # Test search_flights
    print("\n--- Testing search_flights ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result_flights = search_flights("LAG", "NYC", "2025-07-12", "2025-07-19", user_token=test_user_pro)
    print(f"Flights (Pro User, API):\n{result_flights[:500]}...")
    assert "Found Flights:" in result_flights
    assert "From: LAG to NYC" in result_flights
    assert "Departure: July 12, 2025" in result_flights
    print("Test 1 Passed.")

    # Test search_flights (flexible date format)
    result_flights_flex_date = search_flights("NYC", "LON", "July 15, 2025", "July 22, 2025", user_token=test_user_pro)
    print(f"Flights (Pro User, API - Flexible Date):\n{result_flights_flex_date[:500]}...")
    assert "Found Flights:" in result_flights_flex_date
    assert "From: NYC to LON" in result_flights_flex_date
    assert "Departure: July 15, 2025" in result_flights_flex_date
    print("Test 2 Passed.")

    # Test search_flights (fallback)
    print("\n--- Testing search_flights (Fallback) ---")
    with patch('domain_tools.travel_tools.travel_tool._make_dynamic_api_request', return_value=None):
        result_flights_fallback = search_flights("ABV", "DXB", "2025-08-01", user_token=test_user_pro)
        print(f"Flights (Pro User, Fallback):\n{result_flights_fallback[:500]}...")
        assert "Found Flights (Mock Data Fallback):" in result_flights_fallback
    print("Test 3 Passed.")

    # Test search_hotels
    print("\n--- Testing search_hotels ---")
    result_hotels = search_hotels("New York", "2025-07-20", "2025-07-23", user_token=test_user_pro)
    print(f"Hotels (Pro User, API):\n{result_hotels[:500]}...")
    assert "Found Hotels:" in result_hotels
    assert "Grand Central Hotel" in result_hotels
    assert "Check-in: July 20, 2025" in result_hotels
    print("Test 4 Passed.")

    # Test search_hotels (flexible date format)
    result_hotels_flex_date = search_hotels("London", "Sept 1, 2025", "Sept 5, 2025", user_token=test_user_pro)
    print(f"Hotels (Pro User, API - Flexible Date):\n{result_hotels_flex_date[:500]}...")
    assert "Found Hotels:" in result_hotels_flex_date
    assert "Riverside Inn" in result_hotels_flex_date
    assert "Check-in: September 01, 2025" in result_hotels_flex_date
    print("Test 5 Passed.")

    # Test get_destination_info
    print("\n--- Testing get_destination_info ---")
    result_destination = get_destination_info("Paris", user_token=test_user_pro)
    print(f"Destination Info (Pro User, API):\n{result_destination[:200]}...")
    assert "Information for Destination: Paris, France" in result_destination
    assert "City of Love." in result_destination
    print("Test 6 Passed.")

    # Test RBAC for travel_tool_access (e.g., search_flights for free user)
    print("\n--- Testing RBAC for travel_tool_access (Free User) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result_rbac_denied = search_flights("LAX", "MIA", "2025-09-01", user_token=test_user_free)
    print(f"Flights (Free User, RBAC Denied): {result_rbac_denied}")
    assert "Error: Access to travel tools is not enabled for your current tier." in result_rbac_denied
    print("Test 7 Passed.")

    # Test travel_search_web
    print("\n--- Testing travel_search_web ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    search_web_query = "travel guide to Bali"
    search_web_result = travel_search_web(search_web_query, user_token=test_user_pro)
    print(f"Web Search Result for '{search_web_query}':\n{search_web_result[:500]}...")
    assert "Search results for travel guide to Bali" in search_web_result
    print("Test 8 Passed.")

    # Test travel_summarize_document_by_path (requires a dummy file)
    print("\n--- Testing travel_summarize_document_by_path ---")
    dummy_upload_dir = Path("uploads") / test_user_pro / "travel"
    dummy_upload_dir.mkdir(parents=True, exist_ok=True)
    dummy_file_path = dummy_upload_dir / "itinerary.txt"
    with open(dummy_file_path, "w") as f:
        f.write("This is a travel itinerary for a trip to Rome. It includes visits to the Colosseum and Vatican City.")
    
    result_summary = travel_summarize_document_by_path(str(dummy_file_path))
    print(f"Itinerary Summary (Pro User): {result_summary}")
    assert "Mock summary of the provided text." in result_summary
    assert "trip to Rome" in result_summary
    print("Test 9 Passed.")

    print("\nAll travel_tool tests completed.")

    # Restore original requests.get
    requests.get = original_requests_get

    # Clean up dummy files and directories
    test_user_dirs = [Path("uploads") / test_user_pro, BASE_VECTOR_DIR / test_user_pro]
    for d in test_user_dirs:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            print(f"Cleaned up {d}")
