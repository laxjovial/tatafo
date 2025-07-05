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
_mock_entertainment_data = {
    "movie_search": [
        {
            "title": "Inception",
            "year": "2010",
            "genre": "Action, Sci-Fi, Thriller",
            "director": "Christopher Nolan",
            "plot": "A thief who steals information by entering people's dreams is given the inverse task of planting an idea into the mind of a C.E.O.",
            "imdb_rating": "8.8",
            "poster_url": "https://placehold.co/100x150/000/FFF?text=Inception"
        },
        {
            "title": "The Matrix",
            "year": "1999",
            "genre": "Action, Sci-Fi",
            "director": "Lana Wachowski, Lilly Wachowski",
            "plot": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
            "imdb_rating": "8.7",
            "poster_url": "https://placehold.co/100x150/000/FFF?text=Matrix"
        }
    ],
    "event_search": [
        {
            "event_id": "EVT-2024-001",
            "name": "Summer Music Festival",
            "date": "2024-08-15",
            "location": "Central Park, New York",
            "genre": "Music",
            "description": "An annual festival featuring various musical artists.",
            "tickets_url": "http://example.com/tickets/musicfest"
        },
        {
            "event_id": "EVT-2024-002",
            "name": "Comedy Night Live",
            "date": "2024-07-20",
            "location": "The Laugh Factory, Los Angeles",
            "genre": "Comedy",
            "description": "A stand-up comedy show with popular comedians.",
            "tickets_url": "http://example.com/tickets/comedynight"
        }
    ],
    "artist_info": {
        "taylor_swift": {
            "name": "Taylor Swift",
            "genre": "Pop, Country",
            "active_since": "2004",
            "notable_albums": ["Fearless", "1989", "Midnights"],
            "website": "https://www.taylorswift.com"
        },
        "beyonce": {
            "name": "Beyoncé",
            "genre": "R&B, Pop, Soul",
            "active_since": "1997 (Destiny's Child), 2003 (Solo)",
            "notable_albums": ["Dangerously in Love", "Lemonade", "Renaissance"],
            "website": "https://www.beyonce.com"
        }
    }
}

@tool
def search_movies(title: str, year: Optional[int] = None, user_token: str = "default") -> str:
    """
    Searches for movie information by title and optional release year.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        title (str): The title of the movie (e.g., "Inception", "The Matrix").
        year (int, optional): The release year of the movie.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of movie information, or an error/fallback message.
    """
    logger.info(f"Tool: search_movies called for title='{title}', year='{year}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'entertainment_tool_access', False):
        return "Error: Access to entertainment tools is not enabled for your current tier."
    
    params = {"title": title}
    if year: params["year"] = year

    api_data = _make_dynamic_api_request(
        "entertainment", "search_movies",
        params,
        user_token
    )

    if api_data:
        try:
            movie_title = api_data.get("title")
            movie_year = api_data.get("year")
            genre = api_data.get("genre")
            director = api_data.get("director")
            plot = api_data.get("plot")
            imdb_rating = api_data.get("imdb_rating")
            poster_url = api_data.get("poster_url")

            if movie_title and plot:
                response_str = (
                    f"Information for Movie: {movie_title} ({movie_year})\n"
                    f"  Genre: {genre}\n"
                    f"  Director: {director}\n"
                    f"  Plot: {plot}\n"
                    f"  IMDb Rating: {imdb_rating}\n"
                )
                if poster_url:
                    response_str += f"  Poster: {poster_url}\n"
                return response_str
            else:
                logger.warning(f"Live API data for movie '{title}' is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live movie information for '{title}'. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live movie data for '{title}': {e}")
            return f"Error parsing live data for '{title}'. Falling back to mock data."

    # Fallback to mock data
    mock_movies = _mock_entertainment_data.get("movie_search", [])
    filtered_mock_movies = []
    for movie in mock_movies:
        match = True
        if title.lower() not in movie.get("title", "").lower():
            match = False
        if year and int(movie.get("year", 0)) != year:
            match = False
        if match:
            filtered_mock_movies.append(movie)

    if filtered_mock_movies:
        movie = filtered_mock_movies[0] # Take the first matching mock movie
        response_str = (
            f"Information for Movie: {movie.get('title', 'N/A')} ({movie.get('year', 'N/A')}) (Mock Data Fallback)\n"
            f"  Genre: {movie.get('genre', 'N/A')}\n"
            f"  Director: {movie.get('director', 'N/A')}\n"
            f"  Plot: {movie.get('plot', 'N/A')}\n"
            f"  IMDb Rating: {movie.get('imdb_rating', 'N/A')}\n"
        )
        if movie.get('poster_url'):
            response_str += f"  Poster: {movie.get('poster_url', 'N/A')}\n"
        return response_str
    else:
        return f"Movie information not found for '{title}'. (API/Mock Fallback Failed)"


@tool
def search_events(query: str, date: Optional[str] = None, location: Optional[str] = None, user_token: str = "default") -> str:
    """
    Searches for entertainment events based on a query, optional date, and optional location.
    Dates can be in various formats (e.g., 'YYYY-MM-DD', 'MM/DD/YYYY', 'August 15, 2024').
    Falls back to mock data if API key is missing or API call fails.

    Args:
        query (str): The search query for events (e.g., "music festival", "comedy show").
        date (str, optional): The date of the event to filter by.
        location (str, optional): The location of the event.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of event information, or an error/fallback message.
    """
    logger.info(f"Tool: search_events called with query='{query}', date='{date}', location='{location}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'entertainment_tool_access', False):
        return "Error: Access to entertainment tools is not enabled for your current tier."
    
    params = {"query": query}
    if location: params["location"] = location
    
    parsed_date = None
    if date:
        parsed_date = parse_date_to_yyyymmdd(date)
        if not parsed_date:
            return "Error: Could not parse the provided date. Please ensure the date is valid."
        params["date"] = parsed_date

    api_data = _make_dynamic_api_request(
        "entertainment", "search_events",
        params,
        user_token
    )

    if api_data and api_data.get("data"):
        events = api_data["data"]
        if events:
            response_str = "Found Entertainment Events:\n"
            for i, event in enumerate(events[:5]): # Limit to top 5 events
                event_date = event.get('date', 'N/A')
                # Format date if it's a valid YYYY-MM-DD string
                try:
                    event_date = datetime.strptime(event_date, "%Y-%m-%d").strftime("%B %d, %Y")
                except ValueError:
                    pass # Keep as is if not YYYY-MM-DD
                
                response_str += (
                    f"{i+1}. Name: {event.get('name', 'N/A')}\n"
                    f"   Date: {event_date}\n"
                    f"   Location: {event.get('location', 'N/A')}\n"
                    f"   Genre: {event.get('genre', 'N/A')}\n"
                    f"   Description: {event.get('description', 'N/A')}\n"
                    f"   Tickets: {event.get('tickets_url', 'N/A')}\n\n"
                )
            return response_str
        else:
            return f"No live entertainment events found for your criteria (query='{query}', date='{date}', location='{location}'). Falling back to mock data."

    # Fallback to mock data
    mock_events = _mock_entertainment_data.get("event_search", [])
    filtered_mock_events = []
    for event in mock_events:
        match = True
        if query and query.lower() not in event.get("name", "").lower() and query.lower() not in event.get("description", "").lower():
            match = False
        if location and event.get("location", "").lower() not in location.lower():
            match = False
        if parsed_date and event.get("date") != parsed_date:
            match = False
        if match:
            filtered_mock_events.append(event)

    if filtered_mock_events:
        response_str = "Found Entertainment Events (Mock Data Fallback):\n"
        for i, event in enumerate(filtered_mock_events[:2]): # Limit mock to top 2
            event_date = event.get('date', 'N/A')
            try:
                event_date = datetime.strptime(event_date, "%Y-%m-%d").strftime("%B %d, %Y")
            except ValueError:
                pass
            response_str += (
                f"{i+1}. Name: {event.get('name', 'N/A')}\n"
                f"   Date: {event_date}\n"
                f"   Location: {event.get('location', 'N/A')}\n"
                f"   Genre: {event.get('genre', 'N/A')}\n"
                f"   Description: {event.get('description', 'N/A')}\n"
                f"   Tickets: {event.get('tickets_url', 'N/A')}\n\n"
            )
        return response_str
    else:
        return f"Entertainment event information not found for your criteria. (API/Mock Fallback Failed)"


@tool
def get_artist_info(artist_name: str, user_token: str = "default") -> str:
    """
    Retrieves information about a musical artist or band.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        artist_name (str): The name of the artist or band (e.g., "Taylor Swift", "Beyoncé").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of artist information, or an error/fallback message.
    """
    logger.info(f"Tool: get_artist_info called for artist: {artist_name} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'entertainment_tool_access', False):
        return "Error: Access to entertainment tools is not enabled for your current tier."
    
    params = {"name": artist_name}

    api_data = _make_dynamic_api_request(
        "entertainment", "get_artist_info",
        params,
        user_token
    )

    if api_data:
        try:
            name = api_data.get("name")
            genre = api_data.get("genre")
            active_since = api_data.get("active_since")
            notable_albums = api_data.get("notable_albums")
            website = api_data.get("website")

            if name and genre:
                response_str = (
                    f"Information for Artist: {name}\n"
                    f"  Genre: {genre}\n"
                )
                if active_since:
                    response_str += f"  Active Since: {active_since}\n"
                if notable_albums:
                    response_str += f"  Notable Albums: {', '.join(notable_albums)}\n"
                if website:
                    response_str += f"  Website: {website}\n"
                return response_str
            else:
                logger.warning(f"Live API data for artist '{artist_name}' is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live artist information for '{artist_name}'. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live artist data for '{artist_name}': {e}")
            return f"Error parsing live data for '{artist_name}'. Falling back to mock data."

    # Fallback to mock data
    mock_data_key = artist_name.lower().replace(" ", "_")
    mock_data = _mock_entertainment_data.get("artist_info", {}).get(mock_data_key)
    if mock_data:
        response_str = (
            f"Information for Artist: {mock_data['name']} (Mock Data Fallback)\n"
            f"  Genre: {mock_data['genre']}\n"
        )
        if mock_data.get('active_since'):
            response_str += f"  Active Since: {mock_data['active_since']}\n"
        if mock_data.get('notable_albums'):
            response_str += f"  Notable Albums: {', '.join(mock_data['notable_albums'])}\n"
        if mock_data.get('website'):
            response_str += f"  Website: {mock_data['website']}\n"
        return response_str
    else:
        return f"Artist information not found for '{artist_name}'. (API/Mock Fallback Failed)"


# --- Existing Generic Tools (not directly using external APIs, but can be used in entertainment context) ---

@tool
def entertainment_search_web(query: str, user_token: str = "default", max_chars: int = 2000) -> str:
    """
    Searches the web for entertainment-related information using a smart search fallback mechanism.
    This tool wraps the generic `scrape_web` tool, providing an entertainment-specific interface.
    
    Args:
        query (str): The entertainment-related search query (e.g., "new movie releases", "concerts near me").
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
        query (str): The search query to find relevant entertainment documents (e.g., "review of album X", "script for movie Y").
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
    Summarizes a document related to entertainment (e.g., movie scripts, concert reviews) located at the given file path.
    The file path should be accessible by the system (e.g., in the 'uploads' directory).
    This tool wraps the generic `summarize_document` tool.
    
    Args:
        file_path_str (str): The full path to the document file to be summarized.
                              Example: "uploads/default/entertainment/movie_script.pdf"
    
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
                }
            }
            self._api_providers_data = { # Mock api_providers_data for entertainment
                "entertainment": {
                    "entertainment_api": {
                        "base_url": "https://api.example.com/entertainment",
                        "api_key_name": "entertainment_api_key",
                        "api_key_param_name": "api_key",
                        "functions": {
                            "search_movies": {
                                "endpoint": "/movies/search",
                                "required_params": ["title"],
                                "optional_params": ["year"],
                                "response_path": ["data", 0], # Assuming first result is most relevant
                                "data_map": {
                                    "title": "title",
                                    "year": "year",
                                    "genre": "genre",
                                    "director": "director",
                                    "plot": "plot",
                                    "imdb_rating": "imdb_rating",
                                    "poster_url": "poster_url"
                                }
                            },
                            "search_events": {
                                "endpoint": "/events/search",
                                "required_params": ["query"],
                                "optional_params": ["date", "location"],
                                "response_path": ["data"],
                                "data_map": {
                                    "event_id": "id",
                                    "name": "name",
                                    "date": "date",
                                    "location": "location",
                                    "genre": "genre",
                                    "description": "description",
                                    "tickets_url": "tickets_url"
                                }
                            },
                            "get_artist_info": {
                                "endpoint": "/artists",
                                "required_params": ["name"],
                                "response_path": ["data", 0],
                                "data_map": {
                                    "name": "name",
                                    "genre": "genre",
                                    "active_since": "active_since",
                                    "notable_albums": "albums",
                                    "website": "website"
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

    # Mock requests.get for external API calls
    original_requests_get = requests.get

    def mock_requests_get_dynamic(url, params, headers, timeout):
        # Simulate hypothetical Entertainment API responses
        if "api.example.com/entertainment" in url:
            if "/movies/search" in url:
                title = params.get("title", "").lower()
                year = params.get("year")
                
                mock_movies = [
                    {
                        "id": "MOV-001",
                        "title": "Inception",
                        "year": "2010",
                        "genre": "Sci-Fi",
                        "director": "Christopher Nolan",
                        "plot": "Dreams within dreams.",
                        "imdb_rating": "8.8",
                        "poster_url": "https://placehold.co/100x150/000/FFF?text=Inception"
                    },
                    {
                        "id": "MOV-002",
                        "title": "Interstellar",
                        "year": "2014",
                        "genre": "Sci-Fi",
                        "director": "Christopher Nolan",
                        "plot": "Space travel to save humanity.",
                        "imdb_rating": "8.6",
                        "poster_url": "https://placehold.co/100x150/000/FFF?text=Interstellar"
                    }
                ]
                
                filtered_movies = []
                for movie in mock_movies:
                    match = True
                    if title and title not in movie["title"].lower():
                        match = False
                    if year and int(movie["year"]) != year:
                        match = False
                    if match:
                        filtered_movies.append(movie)

                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"data": filtered_movies}
                return mock_response

            elif "/events/search" in url:
                query = params.get("query", "").lower()
                date = params.get("date")
                location = params.get("location", "").lower()

                mock_events = [
                    {
                        "id": "EVT-2024-001",
                        "name": "Summer Music Festival",
                        "date": "2024-08-15",
                        "location": "Central Park, New York",
                        "genre": "Music",
                        "description": "Annual music festival.",
                        "tickets_url": "http://example.com/musicfest"
                    },
                    {
                        "id": "EVT-2024-002",
                        "name": "Comedy Night Live",
                        "date": "2024-07-20",
                        "location": "The Laugh Factory, Los Angeles",
                        "genre": "Comedy",
                        "description": "Stand-up comedy show.",
                        "tickets_url": "http://example.com/comedynight"
                    }
                ]

                filtered_events = []
                for event in mock_events:
                    match = True
                    if query and not (query in event["name"].lower() or query in event["description"].lower()):
                        match = False
                    if date and event["date"] != date:
                        match = False
                    if location and location not in event["location"].lower():
                        match = False
                    if match:
                        filtered_events.append(event)

                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"data": filtered_events}
                return mock_response
            
            elif "/artists" in url:
                name = params.get("name", "").lower()
                if "taylor swift" in name:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "data": [{
                            "name": "Taylor Swift",
                            "genre": "Pop, Country",
                            "active_since": "2004",
                            "albums": ["Fearless", "1989"],
                            "website": "https://www.taylorswift.com"
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
            mock_response.text = f"<html><body><h1>Search results for {params.get('q', 'entertainment')}</h1><p>Some entertainment news snippet from web search.</p></body></html>"
            return mock_response

        return original_requests_get(url, params=params, headers=headers, timeout=timeout)

    requests.get = mock_requests_get_dynamic

    test_user_pro = "mock_pro_token"
    test_user_free = "mock_free_token"

    print("\n--- Testing entertainment_tool functions ---")

    # Test search_movies
    print("\n--- Testing search_movies ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result_movie = search_movies("Inception", year=2010, user_token=test_user_pro)
    print(f"Movie Info (Pro User, API):\n{result_movie[:500]}...")
    assert "Information for Movie: Inception (2010)" in result_movie
    assert "Dreams within dreams." in result_movie
    print("Test 1 Passed.")

    # Test search_movies (fallback)
    print("\n--- Testing search_movies (Fallback) ---")
    with patch('domain_tools.entertainment_tools.entertainment_tool._make_dynamic_api_request', return_value=None):
        result_movie_fallback = search_movies("Avatar", user_token=test_user_pro)
        print(f"Movie Info (Pro User, Fallback):\n{result_movie_fallback[:500]}...")
        assert "Information for Movie: Inception (2010) (Mock Data Fallback)" in result_movie_fallback # Falls back to a default mock if none match
    print("Test 2 Passed.")

    # Test search_events
    print("\n--- Testing search_events ---")
    result_event = search_events("music festival", date="2024-08-15", location="New York", user_token=test_user_pro)
    print(f"Events (Pro User, API):\n{result_event[:500]}...")
    assert "Found Entertainment Events:" in result_event
    assert "Summer Music Festival" in result_event
    assert "August 15, 2024" in result_event # Check formatted date
    print("Test 3 Passed.")

    # Test search_events (flexible date format)
    result_event_flex_date = search_events("comedy show", date="July 20, 2024", location="Los Angeles", user_token=test_user_pro)
    print(f"Events (Pro User, API - Flexible Date):\n{result_event_flex_date[:500]}...")
    assert "Found Entertainment Events:" in result_event_flex_date
    assert "Comedy Night Live" in result_event_flex_date
    assert "July 20, 2024" in result_event_flex_date
    print("Test 4 Passed.")

    # Test get_artist_info
    print("\n--- Testing get_artist_info ---")
    result_artist = get_artist_info("Taylor Swift", user_token=test_user_pro)
    print(f"Artist Info (Pro User, API):\n{result_artist[:200]}...")
    assert "Information for Artist: Taylor Swift" in result_artist
    assert "Pop, Country" in result_artist
    print("Test 5 Passed.")

    # Test RBAC for entertainment_tool_access (e.g., search_movies for free user)
    print("\n--- Testing RBAC for entertainment_tool_access (Free User) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result_rbac_denied = search_movies("Dune", user_token=test_user_free)
    print(f"Movies (Free User, RBAC Denied): {result_rbac_denied}")
    assert "Error: Access to entertainment tools is not enabled for your current tier." in result_rbac_denied
    print("Test 6 Passed.")

    # Test entertainment_search_web
    print("\n--- Testing entertainment_search_web ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    search_web_query = "upcoming concerts in Lagos"
    search_web_result = entertainment_search_web(search_web_query, user_token=test_user_pro)
    print(f"Web Search Result for '{search_web_query}':\n{search_web_result[:500]}...")
    assert "Search results for upcoming concerts in Lagos" in search_web_result
    print("Test 7 Passed.")

    # Test entertainment_summarize_document_by_path (requires a dummy file)
    print("\n--- Testing entertainment_summarize_document_by_path ---")
    dummy_upload_dir = Path("uploads") / test_user_pro / "entertainment"
    dummy_upload_dir.mkdir(parents=True, exist_ok=True)
    dummy_file_path = dummy_upload_dir / "movie_review.txt"
    with open(dummy_file_path, "w") as f:
        f.write("This is a review of the new action movie. The special effects were stunning, but the plot was weak.")
    
    result_summary = entertainment_summarize_document_by_path(str(dummy_file_path))
    print(f"Movie Review Summary (Pro User): {result_summary}")
    assert "Mock summary of the provided text." in result_summary
    assert "special effects were stunning" in result_summary
    print("Test 8 Passed.")

    print("\nAll entertainment_tool tests completed.")

    # Restore original requests.get
    requests.get = original_requests_get

    # Clean up dummy files and directories
    test_user_dirs = [Path("uploads") / test_user_pro, BASE_VECTOR_DIR / test_user_pro]
    for d in test_user_dirs:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            print(f"Cleaned up {d}")
