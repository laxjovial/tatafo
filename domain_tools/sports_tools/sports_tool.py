# domain_tools/sports_tools/sports_tool.py

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
# Import date_parser for date format flexibility (not directly used by current tools, but available)
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
_mock_sports_data = {
    "team_info": {
        "los_angeles_lakers": {
            "name": "Los Angeles Lakers",
            "sport": "Basketball",
            "league": "NBA",
            "city": "Los Angeles",
            "coach": "Darvin Ham",
            "championships": 17,
            "current_record": "47-35 (W-L)",
            "key_players": ["LeBron James", "Anthony Davis"]
        },
        "manchester_united": {
            "name": "Manchester United",
            "sport": "Soccer",
            "league": "Premier League",
            "city": "Manchester",
            "coach": "Erik ten Hag",
            "championships": 20,
            "current_record": "5th in Premier League",
            "key_players": ["Bruno Fernandes", "Marcus Rashford"]
        }
    },
    "player_stats": {
        "lebron_james": {
            "name": "LeBron James",
            "team": "Los Angeles Lakers",
            "sport": "Basketball",
            "position": "Small Forward",
            "points_per_game": 25.7,
            "rebounds_per_game": 7.3,
            "assists_per_game": 8.3,
            "career_championships": 4
        },
        "lionel_messi": {
            "name": "Lionel Messi",
            "team": "Inter Miami CF",
            "sport": "Soccer",
            "position": "Forward",
            "goals_this_season": 12,
            "assists_this_season": 9,
            "career_ballon_d_or": 8
        }
    },
    "game_results": [
        {
            "game_id": "NBA-20231024-LAL-DEN",
            "sport": "Basketball",
            "league": "NBA",
            "date": "2023-10-24",
            "home_team": "Los Angeles Lakers",
            "away_team": "Denver Nuggets",
            "home_score": 102,
            "away_score": 119,
            "winner": "Denver Nuggets",
            "summary": "Nuggets defeated Lakers in season opener."
        },
        {
            "game_id": "PL-20231105-MUN-MCI",
            "sport": "Soccer",
            "league": "Premier League",
            "date": "2023-11-05",
            "home_team": "Manchester United",
            "away_team": "Manchester City",
            "home_score": 0,
            "away_score": 3,
            "winner": "Manchester City",
            "summary": "Manchester City dominated the derby."
        }
    ],
    "upcoming_games": [
        {
            "game_id": "NBA-20240705-GSW-BOS",
            "sport": "Basketball",
            "league": "NBA",
            "date": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
            "time": "19:00 EST",
            "home_team": "Golden State Warriors",
            "away_team": "Boston Celtics",
            "venue": "Chase Center",
            "event_type": "Regular Season Game"
        },
        {
            "game_id": "PL-20240710-LIV-ARS",
            "sport": "Soccer",
            "league": "Premier League",
            "date": (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"),
            "time": "20:00 BST",
            "home_team": "Liverpool",
            "away_team": "Arsenal",
            "venue": "Anfield",
            "event_type": "League Match"
        }
    ]
}

@tool
def get_team_info(team_name: str, sport: Optional[str] = None, user_token: str = "default") -> str:
    """
    Retrieves information about a specific sports team, including its league, coach, championships, and key players.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        team_name (str): The full or partial name of the team (e.g., "Los Angeles Lakers", "Manchester United").
        sport (str, optional): The sport the team belongs to (e.g., "Basketball", "Soccer").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of team information, or an error/fallback message.
    """
    logger.info(f"Tool: get_team_info called for team='{team_name}', sport='{sport}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'sports_tool_access', False):
        return "Error: Access to sports tools is not enabled for your current tier."
    
    params = {"name": team_name}
    if sport: params["sport"] = sport

    api_data = _make_dynamic_api_request(
        "sports", "get_team_info",
        params,
        user_token
    )

    if api_data:
        try:
            name = api_data.get("name")
            sport_name = api_data.get("sport")
            league = api_data.get("league")
            city = api_data.get("city")
            coach = api_data.get("coach")
            championships = api_data.get("championships")
            current_record = api_data.get("current_record")
            key_players = api_data.get("key_players")

            if name and sport_name and league:
                response_str = (
                    f"Information for Team: {name} ({sport_name}, {league})\n"
                    f"  City: {city}\n"
                    f"  Coach: {coach}\n"
                )
                if championships is not None:
                    response_str += f"  Championships: {championships}\n"
                if current_record:
                    response_str += f"  Current Record: {current_record}\n"
                if key_players:
                    response_str += f"  Key Players: {', '.join(key_players)}\n"
                return response_str
            else:
                logger.warning(f"Live API data for team '{team_name}' is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live team information for '{team_name}'. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live team info data for '{team_name}': {e}")
            return f"Error parsing live data for '{team_name}'. Falling back to mock data."

    # Fallback to mock data
    mock_data_key = team_name.lower().replace(" ", "_")
    mock_data = _mock_sports_data.get("team_info", {}).get(mock_data_key)
    if mock_data:
        response_str = (
            f"Information for Team: {mock_data['name']} ({mock_data['sport']}, {mock_data['league']}) (Mock Data Fallback)\n"
            f"  City: {mock_data['city']}\n"
            f"  Coach: {mock_data['coach']}\n"
        )
        if mock_data.get('championships') is not None:
            response_str += f"  Championships: {mock_data['championships']}\n"
        if mock_data.get('current_record'):
            response_str += f"  Current Record: {mock_data['current_record']}\n"
        if mock_data.get('key_players'):
            response_str += f"  Key Players: {', '.join(mock_data['key_players'])}\n"
        return response_str
    else:
        return f"Team information not found for '{team_name}'. (API/Mock Fallback Failed)"


@tool
def get_player_stats(player_name: str, sport: Optional[str] = None, user_token: str = "default") -> str:
    """
    Retrieves statistics for a specific sports player.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        player_name (str): The full name of the player (e.g., "LeBron James", "Lionel Messi").
        sport (str, optional): The sport the player plays (e.g., "Basketball", "Soccer").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of player statistics, or an error/fallback message.
    """
    logger.info(f"Tool: get_player_stats called for player='{player_name}', sport='{sport}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'sports_tool_access', False):
        return "Error: Access to sports tools is not enabled for your current tier."
    
    params = {"name": player_name}
    if sport: params["sport"] = sport

    api_data = _make_dynamic_api_request(
        "sports", "get_player_stats",
        params,
        user_token
    )

    if api_data:
        try:
            name = api_data.get("name")
            team = api_data.get("team")
            sport_name = api_data.get("sport")
            position = api_data.get("position")
            
            response_str = (
                f"Statistics for Player: {name} ({sport_name})\n"
                f"  Team: {team}\n"
                f"  Position: {position}\n"
            )
            # Dynamically add other stats based on what's available in api_data
            for key, value in api_data.items():
                if key not in ["name", "team", "sport", "position"] and value is not None:
                    response_str += f"  {key.replace('_', ' ').title()}: {value}\n"
            
            return response_str
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live player stats data for '{player_name}': {e}")
            return f"Error parsing live data for '{player_name}'. Falling back to mock data."

    # Fallback to mock data
    mock_data_key = player_name.lower().replace(" ", "_")
    mock_data = _mock_sports_data.get("player_stats", {}).get(mock_data_key)
    if mock_data:
        response_str = (
            f"Statistics for Player: {mock_data['name']} ({mock_data['sport']}) (Mock Data Fallback)\n"
            f"  Team: {mock_data['team']}\n"
            f"  Position: {mock_data['position']}\n"
        )
        for key, value in mock_data.items():
            if key not in ["name", "team", "sport", "position"] and value is not None:
                response_str += f"  {key.replace('_', ' ').title()}: {value}\n"
        return response_str
    else:
        return f"Player statistics not found for '{player_name}'. (API/Mock Fallback Failed)"


@tool
def search_game_results(team_name: Optional[str] = None, sport: Optional[str] = None, league: Optional[str] = None, date: Optional[str] = None, user_token: str = "default") -> str:
    """
    Searches for past game results based on team name, sport, league, or a specific date.
    Dates can be in various formats (e.g., 'YYYY-MM-DD', 'MM/DD/YYYY', 'October 24, 2023').
    Falls back to mock data if API key is missing or API call fails.

    Args:
        team_name (str, optional): The name of a team involved in the game.
        sport (str, optional): The sport of the game (e.g., "Basketball", "Soccer").
        league (str, optional): The league of the game (e.g., "NBA", "Premier League").
        date (str, optional): The date of the game to search for.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of game results, or an error/fallback message.
    """
    logger.info(f"Tool: search_game_results called with team='{team_name}', sport='{sport}', league='{league}', date='{date}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'sports_tool_access', False):
        return "Error: Access to sports tools is not enabled for your current tier."
    
    params = {}
    if team_name: params["team_name"] = team_name
    if sport: params["sport"] = sport
    if league: params["league"] = league
    
    parsed_date = None
    if date:
        parsed_date = parse_date_to_yyyymmdd(date)
        if not parsed_date:
            return "Error: Could not parse the provided date. Please ensure the date is valid."
        params["date"] = parsed_date

    api_data = _make_dynamic_api_request(
        "sports", "search_game_results",
        params,
        user_token
    )

    if api_data and api_data.get("data"):
        games = api_data["data"]
        if games:
            response_str = "Found Game Results:\n"
            for i, game in enumerate(games[:5]): # Limit to top 5 games
                game_date_str = game.get('date', 'N/A')
                try:
                    game_date_str = datetime.strptime(game_date_str, "%Y-%m-%d").strftime("%B %d, %Y")
                except ValueError: pass

                response_str += (
                    f"{i+1}. {game.get('sport', 'N/A')} - {game.get('league', 'N/A')}\n"
                    f"   Date: {game_date_str}\n"
                    f"   Teams: {game.get('away_team', 'N/A')} vs {game.get('home_team', 'N/A')}\n"
                    f"   Score: {game.get('away_score', 'N/A')} - {game.get('home_score', 'N/A')}\n"
                    f"   Winner: {game.get('winner', 'N/A')}\n"
                    f"   Summary: {game.get('summary', 'N/A')}\n\n"
                )
            return response_str
        else:
            return f"No live game results found for your criteria. Falling back to mock data."

    # Fallback to mock data
    mock_games = _mock_sports_data.get("game_results", [])
    filtered_mock_games = []
    for game in mock_games:
        match = True
        if team_name and team_name.lower() not in game.get("home_team", "").lower() and team_name.lower() not in game.get("away_team", "").lower():
            match = False
        if sport and game.get("sport", "").lower() != sport.lower():
            match = False
        if league and game.get("league", "").lower() != league.lower():
            match = False
        if parsed_date and game.get("date") != parsed_date:
            match = False
        if match:
            filtered_mock_games.append(game)

    if filtered_mock_games:
        response_str = "Found Game Results (Mock Data Fallback):\n"
        for i, game in enumerate(filtered_mock_games[:2]): # Limit mock to top 2
            game_date_str = game.get('date', 'N/A')
            try:
                game_date_str = datetime.strptime(game_date_str, "%Y-%m-%d").strftime("%B %d, %Y")
            except ValueError: pass
            response_str += (
                f"{i+1}. {game.get('sport', 'N/A')} - {game.get('league', 'N/A')}\n"
                f"   Date: {game_date_str}\n"
                f"   Teams: {game.get('away_team', 'N/A')} vs {game.get('home_team', 'N/A')}\n"
                f"   Score: {game.get('away_score', 'N/A')} - {game.get('home_score', 'N/A')}\n"
                f"   Winner: {game.get('winner', 'N/A')}\n"
                f"   Summary: {game.get('summary', 'N/A')}\n\n"
            )
        return response_str
    else:
        return f"Game results information not found for your criteria. (API/Mock Fallback Failed)"


@tool
def get_upcoming_games(team_name: Optional[str] = None, sport: Optional[str] = None, league: Optional[str] = None, date: Optional[str] = None, user_token: str = "default") -> str:
    """
    Retrieves information about upcoming sports games based on team name, sport, league, or a specific date.
    Dates can be in various formats (e.g., 'YYYY-MM-DD', 'MM/DD/YYYY', 'July 5, 2025').
    Falls back to mock data if API key is missing or API call fails.

    Args:
        team_name (str, optional): The name of a team involved in the game.
        sport (str, optional): The sport of the game (e.g., "Basketball", "Soccer").
        league (str, optional): The league of the game (e.g., "NBA", "Premier League").
        date (str, optional): The date of the game to search for.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of upcoming game information, or an error/fallback message.
    """
    logger.info(f"Tool: get_upcoming_games called with team='{team_name}', sport='{sport}', league='{league}', date='{date}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'sports_tool_access', False):
        return "Error: Access to sports tools is not enabled for your current tier."
    
    params = {}
    if team_name: params["team_name"] = team_name
    if sport: params["sport"] = sport
    if league: params["league"] = league
    
    parsed_date = None
    if date:
        parsed_date = parse_date_to_yyyymmdd(date)
        if not parsed_date:
            return "Error: Could not parse the provided date. Please ensure the date is valid."
        params["date"] = parsed_date

    api_data = _make_dynamic_api_request(
        "sports", "get_upcoming_games",
        params,
        user_token
    )

    if api_data and api_data.get("data"):
        games = api_data["data"]
        if games:
            response_str = "Found Upcoming Games:\n"
            for i, game in enumerate(games[:5]): # Limit to top 5 games
                game_date_str = game.get('date', 'N/A')
                try:
                    game_date_str = datetime.strptime(game_date_str, "%Y-%m-%d").strftime("%B %d, %Y")
                except ValueError: pass

                response_str += (
                    f"{i+1}. {game.get('sport', 'N/A')} - {game.get('league', 'N/A')}\n"
                    f"   Date: {game_date_str} at {game.get('time', 'N/A')}\n"
                    f"   Teams: {game.get('away_team', 'N/A')} vs {game.get('home_team', 'N/A')}\n"
                    f"   Venue: {game.get('venue', 'N/A')}\n"
                    f"   Event Type: {game.get('event_type', 'N/A')}\n\n"
                )
            return response_str
        else:
            return f"No live upcoming games found for your criteria. Falling back to mock data."

    # Fallback to mock data
    mock_games = _mock_sports_data.get("upcoming_games", [])
    filtered_mock_games = []
    for game in mock_games:
        match = True
        if team_name and team_name.lower() not in game.get("home_team", "").lower() and team_name.lower() not in game.get("away_team", "").lower():
            match = False
        if sport and game.get("sport", "").lower() != sport.lower():
            match = False
        if league and game.get("league", "").lower() != league.lower():
            match = False
        if parsed_date and game.get("date") != parsed_date:
            match = False
        if match:
            filtered_mock_games.append(game)

    if filtered_mock_games:
        response_str = "Found Upcoming Games (Mock Data Fallback):\n"
        for i, game in enumerate(filtered_mock_games[:2]): # Limit mock to top 2
            game_date_str = game.get('date', 'N/A')
            try:
                game_date_str = datetime.strptime(game_date_str, "%Y-%m-%d").strftime("%B %d, %Y")
            except ValueError: pass
            response_str += (
                f"{i+1}. {game.get('sport', 'N/A')} - {game.get('league', 'N/A')}\n"
                f"   Date: {game_date_str} at {game.get('time', 'N/A')}\n"
                f"   Teams: {game.get('away_team', 'N/A')} vs {game.get('home_team', 'N/A')}\n"
                f"   Venue: {game.get('venue', 'N/A')}\n"
                f"   Event Type: {game.get('event_type', 'N/A')}\n\n"
            )
        return response_str
    else:
        return f"Upcoming games information not found for your criteria. (API/Mock Fallback Failed)"


# --- Existing Generic Tools (not directly using external APIs, but can be used in sports context) ---

@tool
def sports_search_web(query: str, user_token: str = "default", max_chars: int = 2000) -> str:
    """
    Searches the web for sports-related information using a smart search fallback mechanism.
    This tool wraps the generic `scrape_web` tool, providing a sports-specific interface.
    
    Args:
        query (str): The sports-related search query (e.g., "NBA playoff schedule", "latest transfer news for Real Madrid").
        user_token (str): The unique identifier for the user. Defaults to "default".
        max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
    
    Returns:
        str: A string containing relevant information from the web.
    """
    logger.info(f"Tool: sports_search_web called with query: '{query}' for user: '{user_token}'")
    return scrape_web(query=query, user_token=user_token, max_chars=max_chars)

@tool
def sports_query_uploaded_docs(query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
    """
    Queries previously uploaded and indexed sports documents for a user using vector similarity search.
    This tool wraps the generic `QueryUploadedDocs` tool, fixing the section to "sports".
    
    Args:
        query (str): The search query to find relevant sports documents (e.g., "team roster for 2023 season", "rules of cricket").
        user_token (str): The unique identifier for the user. Defaults to "default".
        export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
        k (int): The number of top relevant documents to retrieve. Defaults to 5.
    
    Returns:
        str: A string containing the combined content of the relevant document chunks,
             or a message indicating no data/results found, or the export path if exported.
    """
    logger.info(f"Tool: sports_query_uploaded_docs called with query: '{query}' for user: '{user_token}'")
    return QueryUploadedDocs(query=query, user_token=user_token, section="sports", export=export, k=k)

@tool
def sports_summarize_document_by_path(file_path_str: str) -> str:
    """
    Summarizes a document related to sports (e.g., game analyses, player profiles) located at the given file path.
    The file path should be accessible by the system (e.g., in the 'uploads' directory).
    This tool wraps the generic `summarize_document` tool.
    
    Args:
        file_path_str (str): The full path to the document file to be summarized.
                              Example: "uploads/default/sports/match_report.pdf"
    
    Returns:
        str: A concise summary of the document content.
    """
    logger.info(f"Tool: sports_summarize_document_by_path called for file: '{file_path_str}'")
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
            self.sports_api_key = "MOCK_SPORTS_API_KEY"
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
                    'sports': 'sports_api'
                }
            }
            self._api_providers_data = { # Mock api_providers_data for sports
                "sports": {
                    "sports_api": {
                        "base_url": "https://api.example.com/sports",
                        "api_key_name": "sports_api_key",
                        "api_key_param_name": "api_key",
                        "functions": {
                            "get_team_info": {
                                "endpoint": "/teams",
                                "required_params": ["name"],
                                "optional_params": ["sport"],
                                "response_path": ["data", 0], # Assuming first result is most relevant
                                "data_map": {
                                    "name": "name",
                                    "sport": "sport",
                                    "league": "league",
                                    "city": "city",
                                    "coach": "coach",
                                    "championships": "championships",
                                    "current_record": "record",
                                    "key_players": "players"
                                }
                            },
                            "get_player_stats": {
                                "endpoint": "/players",
                                "required_params": ["name"],
                                "optional_params": ["sport"],
                                "response_path": ["data", 0],
                                "data_map": {
                                    "name": "name",
                                    "team": "team",
                                    "sport": "sport",
                                    "position": "position",
                                    "points_per_game": "ppg",
                                    "rebounds_per_game": "rpg",
                                    "assists_per_game": "apg",
                                    "goals_this_season": "goals",
                                    "assists_this_season": "assists",
                                    "career_championships": "championships",
                                    "career_ballon_d_or": "ballon_dor"
                                }
                            },
                            "search_game_results": {
                                "endpoint": "/games/results",
                                "required_params": [],
                                "optional_params": ["team_name", "sport", "league", "date"],
                                "response_path": ["data"],
                                "data_map": {
                                    "game_id": "id",
                                    "sport": "sport",
                                    "league": "league",
                                    "date": "date",
                                    "home_team": "home_team",
                                    "away_team": "away_team",
                                    "home_score": "home_score",
                                    "away_score": "away_score",
                                    "winner": "winner",
                                    "summary": "summary"
                                }
                            },
                            "get_upcoming_games": {
                                "endpoint": "/games/upcoming",
                                "required_params": [],
                                "optional_params": ["team_name", "sport", "league", "date"],
                                "response_path": ["data"],
                                "data_map": {
                                    "game_id": "id",
                                    "sport": "sport",
                                    "league": "league",
                                    "date": "date",
                                    "time": "time",
                                    "home_team": "home_team",
                                    "away_team": "away_team",
                                    "venue": "venue",
                                    "event_type": "event_type"
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
                'sports_tool_access': {
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
        # Simulate hypothetical Sports API responses
        if "api.example.com/sports" in url:
            if "/teams" in url:
                name = params.get("name", "").lower()
                sport = params.get("sport", "").lower()
                
                mock_teams = [
                    {
                        "name": "Los Angeles Lakers", "sport": "Basketball", "league": "NBA",
                        "city": "Los Angeles", "coach": "Darvin Ham", "championships": 17,
                        "record": "47-35 (W-L)", "players": ["LeBron James", "Anthony Davis"]
                    },
                    {
                        "name": "Manchester United", "sport": "Soccer", "league": "Premier League",
                        "city": "Manchester", "coach": "Erik ten Hag", "championships": 20,
                        "record": "5th in Premier League", "players": ["Bruno Fernandes", "Marcus Rashford"]
                    }
                ]
                
                filtered_teams = []
                for team in mock_teams:
                    match = True
                    if name and name not in team["name"].lower():
                        match = False
                    if sport and team["sport"].lower() != sport:
                        match = False
                    if match:
                        filtered_teams.append(team)

                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"data": filtered_teams}
                return mock_response

            elif "/players" in url:
                name = params.get("name", "").lower()
                sport = params.get("sport", "").lower()

                mock_players = [
                    {
                        "name": "LeBron James", "team": "Los Angeles Lakers", "sport": "Basketball",
                        "position": "Small Forward", "ppg": 25.7, "rpg": 7.3, "apg": 8.3, "championships": 4
                    },
                    {
                        "name": "Lionel Messi", "team": "Inter Miami CF", "sport": "Soccer",
                        "position": "Forward", "goals": 12, "assists": 9, "ballon_dor": 8
                    }
                ]
                
                filtered_players = []
                for player in mock_players:
                    match = True
                    if name and name not in player["name"].lower():
                        match = False
                    if sport and player["sport"].lower() != sport:
                        match = False
                    if match:
                        filtered_players.append(player)

                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"data": filtered_players}
                return mock_response
            
            elif "/games/results" in url:
                team_name = params.get("team_name", "").lower()
                sport = params.get("sport", "").lower()
                league = params.get("league", "").lower()
                date = params.get("date")

                mock_game_results = [
                    {
                        "id": "NBA-20231024-LAL-DEN", "sport": "Basketball", "league": "NBA",
                        "date": "2023-10-24", "home_team": "Los Angeles Lakers", "away_team": "Denver Nuggets",
                        "home_score": 102, "away_score": 119, "winner": "Denver Nuggets",
                        "summary": "Nuggets defeated Lakers in season opener."
                    },
                    {
                        "id": "PL-20231105-MUN-MCI", "sport": "Soccer", "league": "Premier League",
                        "date": "2023-11-05", "home_team": "Manchester United", "away_team": "Manchester City",
                        "home_score": 0, "away_score": 3, "winner": "Manchester City",
                        "summary": "Manchester City dominated the derby."
                    }
                ]

                filtered_results = []
                for game in mock_game_results:
                    match = True
                    if team_name and not (team_name in game["home_team"].lower() or team_name in game["away_team"].lower()):
                        match = False
                    if sport and game["sport"].lower() != sport:
                        match = False
                    if league and game["league"].lower() != league:
                        match = False
                    if date and game["date"] != date:
                        match = False
                    if match:
                        filtered_results.append(game)

                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"data": filtered_results}
                return mock_response

            elif "/games/upcoming" in url:
                team_name = params.get("team_name", "").lower()
                sport = params.get("sport", "").lower()
                league = params.get("league", "").lower()
                date = params.get("date")

                mock_upcoming_games = [
                    {
                        "id": "NBA-20240705-GSW-BOS", "sport": "Basketball", "league": "NBA",
                        "date": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
                        "time": "19:00 EST", "home_team": "Golden State Warriors", "away_team": "Boston Celtics",
                        "venue": "Chase Center", "event_type": "Regular Season Game"
                    },
                    {
                        "id": "PL-20240710-LIV-ARS", "sport": "Soccer", "league": "Premier League",
                        "date": (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"),
                        "time": "20:00 BST", "home_team": "Liverpool", "away_team": "Arsenal",
                        "venue": "Anfield", "event_type": "League Match"
                    }
                ]

                filtered_upcoming = []
                for game in mock_upcoming_games:
                    match = True
                    if team_name and not (team_name in game["home_team"].lower() or team_name in game["away_team"].lower()):
                        match = False
                    if sport and game["sport"].lower() != sport:
                        match = False
                    if league and game["league"].lower() != league:
                        match = False
                    if date and game["date"] != date:
                        match = False
                    if match:
                        filtered_upcoming.append(game)

                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"data": filtered_upcoming}
                return mock_response
        
        # Simulate scrape_web's internal requests.get if needed
        if "google.com/search" in url or "example.com" in url: # Mock for scrape_web
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = f"<html><body><h1>Search results for {params.get('q', 'sports')}</h1><p>Some sports related content from web search.</p></body></html>"
            return mock_response

        return original_requests_get(url, params=params, headers=headers, timeout=timeout)

    requests.get = mock_requests_get_dynamic

    test_user_pro = "mock_pro_token"
    test_user_free = "mock_free_token"

    print("\n--- Testing sports_tool functions ---")

    # Test get_team_info
    print("\n--- Testing get_team_info ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result_team_info = get_team_info("Los Angeles Lakers", sport="Basketball", user_token=test_user_pro)
    print(f"Team Info (Pro User, API):\n{result_team_info[:500]}...")
    assert "Information for Team: Los Angeles Lakers (Basketball, NBA)" in result_team_info
    assert "Championships: 17" in result_team_info
    print("Test 1 Passed.")

    # Test get_team_info (fallback)
    print("\n--- Testing get_team_info (Fallback) ---")
    with patch('domain_tools.sports_tools.sports_tool._make_dynamic_api_request', return_value=None):
        result_team_info_fallback = get_team_info("Golden State Warriors", user_token=test_user_pro)
        print(f"Team Info (Pro User, Fallback):\n{result_team_info_fallback[:500]}...")
        assert "Information for Team: Los Angeles Lakers (Basketball, NBA) (Mock Data Fallback)" in result_team_info_fallback # Falls back to default mock
    print("Test 2 Passed.")

    # Test get_player_stats
    print("\n--- Testing get_player_stats ---")
    result_player_stats = get_player_stats("LeBron James", sport="Basketball", user_token=test_user_pro)
    print(f"Player Stats (Pro User, API):\n{result_player_stats[:500]}...")
    assert "Statistics for Player: LeBron James (Basketball)" in result_player_stats
    assert "Points Per Game: 25.7" in result_player_stats
    print("Test 3 Passed.")

    # Test search_game_results
    print("\n--- Testing search_game_results ---")
    result_game_results = search_game_results(team_name="Lakers", date="2023-10-24", user_token=test_user_pro)
    print(f"Game Results (Pro User, API):\n{result_game_results[:500]}...")
    assert "Found Game Results:" in result_game_results
    assert "Los Angeles Lakers vs Denver Nuggets" in result_game_results
    assert "October 24, 2023" in result_game_results # Check formatted date
    print("Test 4 Passed.")

    # Test search_game_results (flexible date format)
    result_game_results_flex_date = search_game_results(team_name="Manchester United", date="Nov 5, 2023", user_token=test_user_pro)
    print(f"Game Results (Pro User, API - Flexible Date):\n{result_game_results_flex_date[:500]}...")
    assert "Found Game Results:" in result_game_results_flex_date
    assert "Manchester United vs Manchester City" in result_game_results_flex_date
    assert "November 05, 2023" in result_game_results_flex_date
    print("Test 5 Passed.")

    # Test get_upcoming_games
    print("\n--- Testing get_upcoming_games ---")
    result_upcoming_games = get_upcoming_games(team_name="Golden State Warriors", user_token=test_user_pro)
    print(f"Upcoming Games (Pro User, API):\n{result_upcoming_games[:500]}...")
    assert "Found Upcoming Games:" in result_upcoming_games
    assert "Golden State Warriors vs Boston Celtics" in result_upcoming_games
    print("Test 6 Passed.")

    # Test RBAC for sports_tool_access (e.g., get_team_info for free user)
    print("\n--- Testing RBAC for sports_tool_access (Free User) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result_rbac_denied = get_team_info("Real Madrid", user_token=test_user_free)
    print(f"Team Info (Free User, RBAC Denied): {result_rbac_denied}")
    assert "Error: Access to sports tools is not enabled for your current tier." in result_rbac_denied
    print("Test 7 Passed.")

    # Test sports_search_web
    print("\n--- Testing sports_search_web ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    search_web_query = "latest F1 results"
    search_web_result = sports_search_web(search_web_query, user_token=test_user_pro)
    print(f"Web Search Result for '{search_web_query}':\n{search_web_result[:500]}...")
    assert "Search results for latest F1 results" in search_web_result
    print("Test 8 Passed.")

    # Test sports_summarize_document_by_path (requires a dummy file)
    print("\n--- Testing sports_summarize_document_by_path ---")
    dummy_upload_dir = Path("uploads") / test_user_pro / "sports"
    dummy_upload_dir.mkdir(parents=True, exist_ok=True)
    dummy_file_path = dummy_upload_dir / "match_analysis.txt"
    with open(dummy_file_path, "w") as f:
        f.write("This is a match analysis report for the recent basketball game. It highlights key offensive plays and defensive strategies.")
    
    result_summary = sports_summarize_document_by_path(str(dummy_file_path))
    print(f"Match Analysis Summary (Pro User): {result_summary}")
    assert "Mock summary of the provided text." in result_summary
    assert "offensive plays" in result_summary
    print("Test 9 Passed.")

    print("\nAll sports_tool tests completed.")

    # Restore original requests.get
    requests.get = original_requests_get

    # Clean up dummy files and directories
    test_user_dirs = [Path("uploads") / test_user_pro, BASE_VECTOR_DIR / test_user_pro]
    for d in test_user_dirs:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            print(f"Cleaned up {d}")

    requests.get = original_requests_get
