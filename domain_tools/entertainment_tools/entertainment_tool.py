# domain_tools/entertainment_tools/entertainment_tool.py

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
_mock_entertainment_data = {
    "movie_info": {
        "inception": {
            "title": "Inception",
            "director": "Christopher Nolan",
            "year": 2010,
            "genre": "Sci-Fi, Action",
            "plot": "A thief who steals corporate secrets through use of dream-sharing technology...",
            "imdb_rating": 8.8,
            "cast": ["Leonardo DiCaprio", "Joseph Gordon-Levitt"]
        },
        "the_matrix": {
            "title": "The Matrix",
            "director": "Lana Wachowski, Lilly Wachowski",
            "year": 1999,
            "genre": "Sci-Fi, Action",
            "plot": "A computer hacker learns from mysterious rebels about the true nature of his reality...",
            "imdb_rating": 8.7,
            "cast": ["Keanu Reeves", "Laurence Fishburne"]
        }
    },
    "tv_show_info": {
        "breaking_bad": {
            "title": "Breaking Bad",
            "creator": "Vince Gilligan",
            "year_start": 2008,
            "year_end": 2013,
            "genre": "Crime, Drama, Thriller",
            "plot": "A chemistry teacher turns to a life of crime after being diagnosed with lung cancer.",
            "imdb_rating": 9.5,
            "seasons": 5
        },
        "the_office_us": {
            "title": "The Office (US)",
            "creator": "Ricky Gervais, Stephen Merchant, Greg Daniels",
            "year_start": 2005,
            "year_end": 2013,
            "genre": "Comedy",
            "plot": "A mockumentary about the everyday lives of office employees.",
            "imdb_rating": 9.0,
            "seasons": 9
        }
    },
    "upcoming_events": [
        {
            "event_name": "Summer Music Festival",
            "type": "Music Concert",
            "date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "location": "Central Park, New York",
            "artists": ["Artist A", "Band B"],
            "ticket_info": "Tickets available on Ticketmaster."
        },
        {
            "event_name": "Sci-Fi Movie Convention",
            "type": "Convention",
            "date": (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d"),
            "location": "Los Angeles Convention Center",
            "guests": ["Actor X", "Director Y"],
            "ticket_info": "Early bird tickets on Eventbrite."
        }
    ]
}

@tool
def get_movie_info(title: str, year: Optional[int] = None, user_token: str = "default") -> str:
    """
    Retrieves information about a movie, including its director, cast, plot, and IMDb rating.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        title (str): The title of the movie (e.g., "Inception", "The Matrix").
        year (int, optional): The release year of the movie to refine the search.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of movie information, or an error/fallback message.
    """
    logger.info(f"Tool: get_movie_info called for title: '{title}', year: '{year}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'entertainment_tool_access', False):
        return "Error: Access to entertainment tools is not enabled for your current tier."
    
    params = {"title": title}
    if year: params["year"] = year

    api_data = asyncio.run(_make_dynamic_api_request("entertainment", "get_movie_info", params, user_token))

    if api_data:
        try:
            movie_title = api_data.get("title")
            director = api_data.get("director")
            release_year = api_data.get("year")
            genre = api_data.get("genre")
            plot = api_data.get("plot")
            imdb_rating = api_data.get("imdb_rating")
            cast = api_data.get("cast")

            if movie_title and plot:
                response_str = (
                    f"Information for Movie: {movie_title} ({release_year})\n"
                    f"  Director: {director}\n"
                    f"  Genre: {genre}\n"
                    f"  IMDb Rating: {imdb_rating}\n"
                    f"  Plot: {plot}\n"
                )
                if cast:
                    response_str += f"  Cast: {', '.join(cast)}\n"
                return response_str
            else:
                logger.warning(f"Live API data for movie '{title}' is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live movie information for '{title}'. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live movie info data for '{title}': {e}")
            return f"Error parsing live data for '{title}'. Falling back to mock data."

    # Fallback to mock data
    mock_data_key = title.lower().replace(" ", "_")
    mock_data = _mock_entertainment_data.get("movie_info", {}).get(mock_data_key)
    if mock_data and (not year or mock_data.get("year") == year):
        response_str = (
            f"Information for Movie: {mock_data['title']} ({mock_data['year']}) (Mock Data Fallback)\n"
            f"  Director: {mock_data['director']}\n"
            f"  Genre: {mock_data['genre']}\n"
            f"  IMDb Rating: {mock_data['imdb_rating']}\n"
            f"  Plot: {mock_data['plot']}\n"
        )
        if mock_data.get('cast'):
            response_str += f"  Cast: {', '.join(mock_data['cast'])}\n"
        return response_str
    else:
        return f"Movie information not found for '{title}'. (API/Mock Fallback Failed)"


@tool
def get_tv_show_info(title: str, user_token: str = "default") -> str:
    """
    Retrieves information about a TV show, including its creator, plot, and IMDb rating.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        title (str): The title of the TV show (e.g., "Breaking Bad", "The Office").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of TV show information, or an error/fallback message.
    """
    logger.info(f"Tool: get_tv_show_info called for title: '{title}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'entertainment_tool_access', False):
        return "Error: Access to entertainment tools is not enabled for your current tier."
    
    params = {"title": title}
    api_data = asyncio.run(_make_dynamic_api_request("entertainment", "get_tv_show_info", params, user_token))

    if api_data:
        try:
            show_title = api_data.get("title")
            creator = api_data.get("creator")
            year_start = api_data.get("year_start")
            year_end = api_data.get("year_end")
            genre = api_data.get("genre")
            plot = api_data.get("plot")
            imdb_rating = api_data.get("imdb_rating")
            seasons = api_data.get("seasons")

            if show_title and plot:
                response_str = (
                    f"Information for TV Show: {show_title} ({year_start}-{year_end if year_end else 'Present'})\n"
                    f"  Creator: {creator}\n"
                    f"  Genre: {genre}\n"
                    f"  IMDb Rating: {imdb_rating}\n"
                    f"  Seasons: {seasons}\n"
                    f"  Plot: {plot}\n"
                )
                return response_str
            else:
                logger.warning(f"Live API data for TV show '{title}' is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live TV show information for '{title}'. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live TV show info data for '{title}': {e}")
            return f"Error parsing live data for '{title}'. Falling back to mock data."

    # Fallback to mock data
    mock_data_key = title.lower().replace(" ", "_")
    # Handle "The Office" specifically for US vs UK versions if needed in mock
    if "the office" in mock_data_key and "us" in mock_data_key:
        mock_data_key = "the_office_us"
    mock_data = _mock_entertainment_data.get("tv_show_info", {}).get(mock_data_key)
    if mock_data:
        response_str = (
            f"Information for TV Show: {mock_data['title']} ({mock_data['year_start']}-{mock_data['year_end'] if mock_data['year_end'] else 'Present'}) (Mock Data Fallback)\n"
            f"  Creator: {mock_data['creator']}\n"
            f"  Genre: {mock_data['genre']}\n"
            f"  IMDb Rating: {mock_data['imdb_rating']}\n"
            f"  Seasons: {mock_data['seasons']}\n"
            f"  Plot: {mock_data['plot']}\n"
        )
        return response_str
    else:
        return f"TV show information not found for '{title}'. (API/Mock Fallback Failed)"


@tool
def search_upcoming_entertainment_events(event_type: Optional[str] = None, location: Optional[str] = None, date: Optional[str] = None, user_token: str = "default") -> str:
    """
    Searches for upcoming entertainment events (e.g., music concerts, festivals, conventions)
    optionally filtered by type, location, or date.
    Dates can be in various formats (e.g., 'YYYY-MM-DD', 'MM/DD/YYYY', 'July 5, 2025').
    Falls back to mock data if API key is missing or API call fails.

    Args:
        event_type (str, optional): The type of event (e.g., 'Music Concert', 'Convention', 'Play').
        location (str, optional): The city or venue of the event.
        date (str, optional): The specific date of the event.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of upcoming event information, or an error/fallback message.
    """
    logger.info(f"Tool: search_upcoming_entertainment_events called for type: '{event_type}', location: '{location}', date: '{date}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'entertainment_tool_access', False):
        return "Error: Access to entertainment tools is not enabled for your current tier."
    
    params = {}
    if event_type: params["type"] = event_type
    if location: params["location"] = location

    parsed_date = None
    if date:
        parsed_date = parse_date_to_yyyymmdd(date)
        if not parsed_date:
            return "Error: Could not parse the provided date. Please ensure the date is valid."
        params["date"] = parsed_date

    api_data = asyncio.run(_make_dynamic_api_request("entertainment", "search_upcoming_entertainment_events", params, user_token))

    if api_data and api_data.get("data"):
        events = api_data["data"]
        if events:
            response_str = "Upcoming Entertainment Events:\n"
            for i, event in enumerate(events[:5]): # Limit to top 5 events
                event_date_str = event.get('date', 'N/A')
                try:
                    event_date_str = datetime.strptime(event_date_str, "%Y-%m-%d").strftime("%B %d, %Y")
                except ValueError: pass

                response_str += (
                    f"{i+1}. Event: {event.get('event_name', 'N/A')} ({event.get('type', 'N/A')})\n"
                    f"   Date: {event_date_str}\n"
                    f"   Location: {event.get('location', 'N/A')}\n"
                    f"   Details: {event.get('artists', event.get('guests', ''))}\n" # Combine artists/guests
                    f"   Tickets: {event.get('ticket_info', 'N/A')}\n\n"
                )
            return response_str
        else:
            return f"No live upcoming entertainment events found for your criteria. Falling back to mock data."

    # Fallback to mock data
    mock_events = _mock_entertainment_data.get("upcoming_events", [])
    filtered_mock_events = []
    for event in mock_events:
        match = True
        if event_type and event.get("type", "").lower() != event_type.lower():
            match = False
        if location and location.lower() not in event.get("location", "").lower():
            match = False
        if parsed_date and event.get("date") != parsed_date:
            match = False
        if match:
            filtered_mock_events.append(event)

    if filtered_mock_events:
        response_str = "Upcoming Entertainment Events (Mock Data Fallback):\n"
        for i, event in enumerate(filtered_mock_events[:2]): # Limit mock to top 2
            event_date_str = event.get('date', 'N/A')
            try:
                event_date_str = datetime.strptime(event_date_str, "%Y-%m-%d").strftime("%B %d, %Y")
            except ValueError: pass
            response_str += (
                f"{i+1}. Event: {event.get('event_name', 'N/A')} ({event.get('type', 'N/A')})\n"
                f"   Date: {event_date_str}\n"
                f"   Location: {event.get('location', 'N/A')}\n"
                f"   Details: {event.get('artists', event.get('guests', ''))}\n"
                f"   Tickets: {event.get('ticket_info', 'N/A')}\n\n"
            )
        return response_str
    else:
        return f"Upcoming entertainment events not found for your criteria. (API/Mock Fallback Failed)"


# --- Existing Generic Tools (not directly using external APIs, but can be used in entertainment context) ---

@tool
def entertainment_search_web(query: str, user_token: str = "default", max_chars: int = 2000) -> str:
    """
    Searches the web for entertainment-related information using a smart search fallback mechanism.
    This tool wraps the generic `scrape_web` tool, providing an entertainment-specific interface.
    
    Args:
        query (str): The entertainment-related search query (e.g., "new movie releases 2024", "best TV series to binge-watch").
        user_token (str): The unique identifier for the user. Defaults to "default".
        max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
    
    Returns:
        str: A string containing relevant information from the web.
    """
    logger.info(f"Tool: entertainment_search_web called with query: '{query}' for user: '{user_token}'")
    return scrape_web(query=query, user_token=user_token, max_chars=max_chars)

@tool
def entertainment_query_uploaded_docs(query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
    """
    Queries previously uploaded and indexed entertainment documents for a user using vector similarity search.
    This tool wraps the generic `QueryUploadedDocs` tool, fixing the section to "entertainment".
    
    Args:
        query (str): The search query to find relevant entertainment documents (e.g., "script for my play", "fan theories about show X").
        user_token (str): The unique identifier for the user. Defaults to "default".
        export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
        k (int): The number of top relevant documents to retrieve. Defaults to 5.
    
    Returns:
        str: A string containing the combined content of the relevant document chunks,
             or a message indicating no data/results found, or the export path if exported.
    """
    logger.info(f"Tool: entertainment_query_uploaded_docs called with query: '{query}' for user: '{user_token}'")
    return QueryUploadedDocs(query=query, user_token=user_token, section="entertainment", export=export, k=k)

@tool
def entertainment_summarize_document_by_path(file_path_str: str) -> str:
    """
    Summarizes a document related to entertainment (e.g., movie reviews, script excerpts) located at the given file path.
    The file path should be accessible by the system (e.g., in the 'uploads' directory).
    This tool wraps the generic `summarize_document` tool.
    
    Args:
        file_path_str (str): The full path to the document file to be summarized.
                              Example: "uploads/default/entertainment/movie_review.pdf"
    
    Returns:
        str: A concise summary of the document content.
    """
    logger.info(f"Tool: entertainment_summarize_document_by_path called for file: '{file_path_str}'")
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
    from shared_tools.vector_utils import BASE_VECTOR_DIR # For cleanup
    from shared_tools.python_interpreter_tool import python_interpreter_with_rbac # For testing REPL

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.entertainment_api_key = "MOCK_ENTERTAINMENT_API_KEY"
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
                    'entertainment': 'entertainment_api'
                },
                'analytics': { # Mock analytics settings
                    'enabled': True,
                    'log_tool_usage': True,
                    'log_query_failures': True
                }
            }
            self._api_providers_data = { # Mock api_providers_data for entertainment
                "entertainment": {
                    "entertainment_api": {
                        "base_url": "https://api.example.com/entertainment",
                        "api_key_name": "entertainment_api_key",
                        "api_key_param_name": "api_key",
                        "functions": {
                            "get_movie_info": {
                                "endpoint": "/movies",
                                "required_params": ["title"],
                                "optional_params": ["year"],
                                "response_path": ["data", 0],
                                "data_map": {
                                    "title": "title",
                                    "director": "director",
                                    "year": "year",
                                    "genre": "genre",
                                    "plot": "plot",
                                    "imdb_rating": "rating",
                                    "cast": "cast"
                                }
                            },
                            "get_tv_show_info": {
                                "endpoint": "/tvshows",
                                "required_params": ["title"],
                                "response_path": ["data", 0],
                                "data_map": {
                                    "title": "title",
                                    "creator": "creator",
                                    "year_start": "start_year",
                                    "year_end": "end_year",
                                    "genre": "genre",
                                    "plot": "plot",
                                    "imdb_rating": "rating",
                                    "seasons": "num_seasons"
                                }
                            },
                            "search_upcoming_entertainment_events": {
                                "endpoint": "/events/upcoming",
                                "required_params": [],
                                "optional_params": ["type", "location", "date"],
                                "response_path": ["data"],
                                "data_map": {
                                    "event_name": "name",
                                    "type": "type",
                                    "date": "date",
                                    "location": "location",
                                    "artists": "artists",
                                    "guests": "guests",
                                    "ticket_info": "tickets"
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
                'entertainment_tool_access': {
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
            # Simulate hypothetical Entertainment API responses
            if "api.example.com/entertainment" in url:
                if "/movies" in url:
                    title = params.get("title", "").lower()
                    year = params.get("year")
                    
                    mock_movies = [
                        {
                            "title": "Inception",
                            "director": "Christopher Nolan",
                            "year": 2010,
                            "genre": "Sci-Fi, Action",
                            "plot": "A thief who steals corporate secrets...",
                            "rating": 8.8,
                            "cast": ["Leonardo DiCaprio"]
                        },
                        {
                            "title": "The Matrix",
                            "director": "Lana Wachowski",
                            "year": 1999,
                            "genre": "Sci-Fi, Action",
                            "plot": "A computer hacker learns...",
                            "rating": 8.7,
                            "cast": ["Keanu Reeves"]
                        }
                    ]
                    
                    filtered_movies = []
                    for movie in mock_movies:
                        match = True
                        if title and title not in movie["title"].lower():
                            match = False
                        if year and movie["year"] != year:
                            match = False
                        if match:
                            filtered_movies.append(movie)

                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"data": filtered_movies}
                    return mock_response
                elif "/tvshows" in url:
                    title = params.get("title", "").lower()
                    
                    mock_tv_shows = [
                        {
                            "title": "Breaking Bad",
                            "creator": "Vince Gilligan",
                            "start_year": 2008,
                            "end_year": 2013,
                            "genre": "Crime, Drama",
                            "plot": "A chemistry teacher turns to crime.",
                            "rating": 9.5,
                            "num_seasons": 5
                        },
                        {
                            "title": "The Office (US)",
                            "creator": "Greg Daniels",
                            "start_year": 2005,
                            "end_year": 2013,
                            "genre": "Comedy",
                            "plot": "A mockumentary about office employees.",
                            "rating": 9.0,
                            "num_seasons": 9
                        }
                    ]
                    
                    filtered_shows = []
                    for show in mock_tv_shows:
                        match = True
                        if title and title not in show["title"].lower():
                            match = False
                        if match:
                            filtered_shows.append(show)

                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"data": filtered_shows}
                    return mock_response
                elif "/events/upcoming" in url:
                    event_type = params.get("type", "").lower()
                    location = params.get("location", "").lower()
                    date = params.get("date")

                    mock_events = [
                        {
                            "name": "Summer Music Festival",
                            "type": "Music Concert",
                            "date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                            "location": "Central Park, New York",
                            "artists": ["Artist A"],
                            "tickets": "Ticketmaster."
                        },
                        {
                            "name": "Sci-Fi Movie Convention",
                            "type": "Convention",
                            "date": (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d"),
                            "location": "Los Angeles Convention Center",
                            "guests": ["Actor X"],
                            "tickets": "Eventbrite."
                        }
                    ]

                    filtered_events = []
                    for event in mock_events:
                        match = True
                        if event_type and event["type"].lower() != event_type:
                            match = False
                        if location and location not in event["location"].lower():
                            match = False
                        if date and event["date"] != date:
                            match = False
                        if match:
                            filtered_events.append(event)

                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"data": filtered_events}
                    return mock_response
                else:
                    mock_response = MagicMock()
                    mock_response.status_code = 400
                    mock_response.json.return_value = {"error": "Invalid endpoint"}
                    return mock_response
            
            # Simulate scrape_web's internal requests.get if needed
            if "google.com/search" in url or "example.com" in url: # Mock for scrape_web
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.text = f"<html><body><h1>Search results for {params.get('q', 'entertainment')}</h1><p>Some entertainment related content from web search.</p></body></html>"
                return mock_response

            return original_requests_get(url, params=params, headers=headers, timeout=timeout)

        requests.get = mock_requests_get_dynamic

        test_user_pro = "mock_pro_token"
        test_user_free = "mock_free_token"

        async def run_entertainment_tests():
            print("\n--- Testing entertainment_tool functions with Analytics ---")

            # Test get_movie_info (success)
            print("\n--- Test 1: get_movie_info (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock() # Reset mock call count
            result_movie_info = await get_movie_info("Inception", user_token=test_user_pro)
            print(f"Movie Info: {result_movie_info}")
            assert "Information for Movie: Inception (2010)" in result_movie_info
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "entertainment_get_movie_info"
            assert logged_data["success"] is True
            print("Test 1 Passed (and analytics logged success).")

            # Test get_tv_show_info (API failure - no data found)
            print("\n--- Test 2: get_tv_show_info (API Failure) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_tv_show_info = await get_tv_show_info("NonExistent Show", user_token=test_user_pro)
            print(f"TV Show Info (API Error): {result_tv_show_info}")
            assert "Could not retrieve complete live TV show information for 'NonExistent Show'." in result_tv_show_info
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "entertainment_get_tv_show_info"
            assert logged_data["success"] is False
            assert "Response path 'data.0' not found" in logged_data["error_message"] or "incomplete" in logged_data["error_message"]
            print("Test 2 Passed (and analytics logged failure).")

            # Test search_upcoming_entertainment_events (RBAC denied)
            print("\n--- Test 3: search_upcoming_entertainment_events (RBAC Denied) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_events_rbac_denied = await search_upcoming_entertainment_events(event_type="Concert", user_token=test_user_free)
            print(f"Upcoming Events (Free User, RBAC Denied): {result_events_rbac_denied}")
            assert "Error: Access to entertainment tools is not enabled for your current tier." in result_events_rbac_denied
            # No analytics log expected here because RBAC check happens before _make_dynamic_api_request
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 3 Passed (RBAC correctly prevented call and no analytics logged).")

            # Test entertainment_search_web (generic tool, not using _make_dynamic_api_request)
            print("\n--- Test 4: entertainment_search_web (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_web_search = await entertainment_search_web("best sci-fi movies of all time", user_token=test_user_pro)
            print(f"Web Search Result: {result_web_search[:100]}...")
            assert "Search results for best sci-fi movies of all time" in result_web_search
            # Analytics for generic tools like scrape_web or summarize_document
            # would need to be integrated within those shared_tools themselves,
            # or wrapped by a higher-level agent logging.
            # For now, we are focusing on _make_dynamic_api_request.
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 4 Passed (no analytics expected for generic tool directly).")

            print("\nAll entertainment_tool tests with analytics considerations completed.")

        await run_entertainment_tests()

        # Restore original requests.get
        requests.get = original_requests_get

        # Clean up dummy files and directories
        test_user_dirs = [Path("uploads") / test_user_pro, BASE_VECTOR_DIR / test_user_pro]
        for d in test_user_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print(f"Cleaned up {d}")
