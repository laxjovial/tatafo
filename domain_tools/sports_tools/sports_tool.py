# domain_tools/sports_tools/sports_tool.py

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
_mock_sports_data = {
    "sports_scores": {
        "football_match_1": {
            "sport": "Football",
            "team_home": "Manchester United",
            "team_away": "Liverpool",
            "score_home": 2,
            "score_away": 1,
            "status": "Finished",
            "date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        },
        "basketball_game_1": {
            "sport": "Basketball",
            "team_home": "L.A. Lakers",
            "team_away": "Boston Celtics",
            "score_home": 110,
            "score_away": 108,
            "status": "Finished",
            "date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        }
    },
    "team_info": {
        "golden_state_warriors": {
            "team_name": "Golden State Warriors",
            "sport": "Basketball",
            "league": "NBA",
            "city": "San Francisco",
            "coach": "Steve Kerr",
            "key_players": ["Stephen Curry", "Klay Thompson"],
            "championships": 7
        },
        "real_madrid_cf": {
            "team_name": "Real Madrid CF",
            "sport": "Football",
            "league": "La Liga",
            "city": "Madrid",
            "coach": "Carlo Ancelotti",
            "key_players": ["Vinicius Jr.", "Jude Bellingham"],
            "championships": 36
        }
    },
    "player_stats": {
        "stephen_curry": {
            "player_name": "Stephen Curry",
            "team": "Golden State Warriors",
            "sport": "Basketball",
            "position": "Point Guard",
            "stats": {
                "points_per_game": 27.3,
                "assists_per_game": 6.8,
                "rebounds_per_game": 4.9,
                "three_point_percentage": 0.428
            },
            "achievements": ["4x NBA Champion", "2x MVP"]
        },
        "lionel_messi": {
            "player_name": "Lionel Messi",
            "team": "Inter Miami CF",
            "sport": "Football",
            "position": "Forward",
            "stats": {
                "goals_scored": 834,
                "assists": 371
            },
            "achievements": ["8x Ballon d'Or", "World Cup Winner"]
        }
    }
}

@tool
def get_sports_scores(sport: Optional[str] = None, team: Optional[str] = None, date: Optional[str] = None, user_token: str = "default") -> str:
    """
    Retrieves sports scores for various matches, optionally filtered by sport, team, or date.
    Dates can be in various formats (e.g., 'YYYY-MM-DD', 'MM/DD/YYYY', 'July 5, 2025').
    Falls back to mock data if API key is missing or API call fails.

    Args:
        sport (str, optional): The sport (e.g., "Football", "Basketball", "Tennis").
        team (str, optional): The name of a team involved in the match.
        date (str, optional): The date of the matches.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of sports scores, or an error/fallback message.
    """
    logger.info(f"Tool: get_sports_scores called for sport='{sport}', team='{team}', date='{date}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'sports_tool_access', False):
        return "Error: Access to sports tools is not enabled for your current tier."
    
    params = {}
    if sport: params["sport"] = sport
    if team: params["team"] = team
    
    parsed_date = None
    if date:
        parsed_date = parse_date_to_yyyymmdd(date)
        if not parsed_date:
            return "Error: Could not parse the provided date. Please ensure the date is valid."
        params["date"] = parsed_date

    api_data = asyncio.run(_make_dynamic_api_request("sports", "get_sports_scores", params, user_token))

    if api_data and api_data.get("data"):
        matches = api_data["data"]
        if matches:
            response_str = "Sports Scores:\n"
            for i, match in enumerate(matches[:5]): # Limit to top 5 matches
                response_str += (
                    f"{i+1}. Sport: {match.get('sport', 'N/A')}\n"
                    f"   Match: {match.get('team_home', 'N/A')} vs {match.get('team_away', 'N/A')}\n"
                    f"   Score: {match.get('score_home', 'N/A')} - {match.get('score_away', 'N/A')}\n"
                    f"   Status: {match.get('status', 'N/A')}\n"
                    f"   Date: {match.get('date', 'N/A')}\n\n"
                )
            return response_str
        else:
            return f"No live sports scores found for your criteria. Falling back to mock data."

    # Fallback to mock data
    mock_scores = _mock_sports_data.get("sports_scores", {})
    filtered_mock_scores = []
    for key, match in mock_scores.items():
        match_found = True
        if sport and match.get("sport", "").lower() != sport.lower():
            match_found = False
        if team and team.lower() not in match.get("team_home", "").lower() and \
           team.lower() not in match.get("team_away", "").lower():
            match_found = False
        if parsed_date and match.get("date") != parsed_date:
            match_found = False
        if match_found:
            filtered_mock_scores.append(match)

    if filtered_mock_scores:
        response_str = "Sports Scores (Mock Data Fallback):\n"
        for i, match in enumerate(filtered_mock_scores[:2]): # Limit mock to top 2
            response_str += (
                f"{i+1}. Sport: {match.get('sport', 'N/A')}\n"
                f"   Match: {match.get('team_home', 'N/A')} vs {match.get('team_away', 'N/A')}\n"
                f"   Score: {match.get('score_home', 'N/A')} - {match.get('score_away', 'N/A')}\n"
                f"   Status: {match.get('status', 'N/A')}\n"
                f"   Date: {match.get('date', 'N/A')}\n\n"
            )
        return response_str
    else:
        return f"Sports scores not found for your criteria. (API/Mock Fallback Failed)"


@tool
def get_team_info(team_name: str, sport: Optional[str] = None, user_token: str = "default") -> str:
    """
    Retrieves information about a specific sports team, including its league, coach, and key players.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        team_name (str): The full or partial name of the sports team (e.g., "Lakers", "Real Madrid").
        sport (str, optional): The sport the team plays (e.g., "Basketball", "Football").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of team information, or an error/fallback message.
    """
    logger.info(f"Tool: get_team_info called for team: '{team_name}', sport: '{sport}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'sports_tool_access', False):
        return "Error: Access to sports tools is not enabled for your current tier."
    
    params = {"team_name": team_name}
    if sport: params["sport"] = sport

    api_data = asyncio.run(_make_dynamic_api_request("sports", "get_team_info", params, user_token))

    if api_data:
        try:
            name = api_data.get("team_name")
            spt = api_data.get("sport")
            league = api_data.get("league")
            city = api_data.get("city")
            coach = api_data.get("coach")
            key_players = api_data.get("key_players")
            championships = api_data.get("championships")

            if name and spt:
                response_str = (
                    f"Information for Team: {name} ({spt})\n"
                    f"  League: {league}\n"
                    f"  City: {city}\n"
                )
                if coach:
                    response_str += f"  Coach: {coach}\n"
                if key_players:
                    response_str += f"  Key Players: {', '.join(key_players)}\n"
                if championships is not None:
                    response_str += f"  Championships: {championships}\n"
                return response_str
            else:
                logger.warning(f"Live API data for team '{team_name}' is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live team information for '{team_name}'. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live team info data for '{team_name}': {e}")
            return f"Error parsing live data for '{team_name}'. Falling back to mock data."

    # Fallback to mock data
    mock_data_key_prefix = team_name.lower().replace(" ", "_")
    mock_data = None
    for key, entry in _mock_sports_data.get("team_info", {}).items():
        if mock_data_key_prefix in key and (not sport or sport.lower() in entry.get("sport", "").lower()):
            mock_data = entry
            break

    if mock_data:
        response_str = (
            f"Information for Team: {mock_data['team_name']} ({mock_data['sport']}) (Mock Data Fallback)\n"
            f"  League: {mock_data['league']}\n"
            f"  City: {mock_data['city']}\n"
        )
        if mock_data.get('coach'):
            response_str += f"  Coach: {mock_data['coach']}\n"
        if mock_data.get('key_players'):
            response_str += f"  Key Players: {', '.join(mock_data['key_players'])}\n"
        if mock_data.get('championships') is not None:
            response_str += f"  Championships: {mock_data['championships']}\n"
        return response_str
    else:
        return f"Team information not found for '{team_name}'. (API/Mock Fallback Failed)"


@tool
def search_player_stats(player_name: str, sport: Optional[str] = None, user_token: str = "default") -> str:
    """
    Searches for statistics and achievements of a specific sports player.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        player_name (str): The full or partial name of the player (e.g., "LeBron James", "Lionel Messi").
        sport (str, optional): The sport the player plays.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of player statistics and achievements, or an error/fallback message.
    """
    logger.info(f"Tool: search_player_stats called for player: '{player_name}', sport: '{sport}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'sports_tool_access', False):
        return "Error: Access to sports tools is not enabled for your current tier."
    
    params = {"player_name": player_name}
    if sport: params["sport"] = sport

    api_data = asyncio.run(_make_dynamic_api_request("sports", "search_player_stats", params, user_token))

    if api_data:
        try:
            name = api_data.get("player_name")
            team = api_data.get("team")
            spt = api_data.get("sport")
            position = api_data.get("position")
            stats = api_data.get("stats")
            achievements = api_data.get("achievements")

            if name and spt:
                response_str = (
                    f"Statistics for Player: {name} ({spt})\n"
                    f"  Team: {team}\n"
                    f"  Position: {position}\n"
                )
                if stats:
                    response_str += "  Stats:\n"
                    for stat_name, stat_value in stats.items():
                        response_str += f"    - {stat_name.replace('_', ' ').title()}: {stat_value}\n"
                if achievements:
                    response_str += f"  Achievements: {', '.join(achievements)}\n"
                return response_str
            else:
                logger.warning(f"Live API data for player '{player_name}' is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live player statistics for '{player_name}'. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live player stats data for '{player_name}': {e}")
            return f"Error parsing live data for '{player_name}'. Falling back to mock data."

    # Fallback to mock data
    mock_data_key = player_name.lower().replace(" ", "_")
    mock_data = _mock_sports_data.get("player_stats", {}).get(mock_data_key)
    if mock_data and (not sport or sport.lower() in mock_data.get("sport", "").lower()):
        response_str = (
            f"Statistics for Player: {mock_data['player_name']} ({mock_data['sport']}) (Mock Data Fallback)\n"
            f"  Team: {mock_data['team']}\n"
            f"  Position: {mock_data['position']}\n"
        )
        if mock_data.get('stats'):
            response_str += "  Stats:\n"
            for stat_name, stat_value in mock_data['stats'].items():
                response_str += f"    - {stat_name.replace('_', ' ').title()}: {stat_value}\n"
        if mock_data.get('achievements'):
            response_str += f"  Achievements: {', '.join(mock_data['achievements'])}\n"
        return response_str
    else:
        return f"Player statistics for '{player_name}' not found. (API/Mock Fallback Failed)"


# --- Existing Generic Tools (not directly using external APIs, but can be used in sports context) ---

@tool
def sports_search_web(query: str, user_token: str = "default", max_chars: int = 2000) -> str:
    """
    Searches the web for sports-related information using a smart search fallback mechanism.
    This tool wraps the generic `scrape_web` tool, providing a sports-specific interface.
    
    Args:
        query (str): The sports-related search query (e.g., "latest NBA news", "history of the Olympics").
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
        query (str): The search query to find relevant sports documents (e.g., "my fantasy football league rules", "training regimen for marathon").
        user_token (str): The unique identifier for the user. Defaults to "default".
        export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
        k (int): The number of top relevant documents to retrieve. Defaults to 5.
    
    Returns:
        str: A string containing the combined content of the relevant document chunks,
             or a message indicating no data/results found, or the export path if exported.
    """
    logger.info(f"Tool: sports_query_uploaded_docs called with query: '{query}' for user: '{user_token}'")
    # This will be replaced by a call to self.document_tools.query_uploaded_docs
    # For now, keeping the original call for review purposes.
    return QueryUploadedDocs(query=query, user_token=user_token, section="sports", export=export, k=k)

@tool
def sports_summarize_document_by_path(file_path_str: str) -> str:
    """
    Summarizes a document related to sports (e.g., game analysis, athlete biographies) located at the given file path.
    The file path should be accessible by the system (e.g., in the 'uploads' directory).
    This tool wraps the generic `summarize_document` tool.
    
    Args:
        file_path_str (str): The full path to the document file to be summarized.
                              Example: "uploads/default/sports/team_strategy.pdf"
    
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
                },
                'analytics': { # Mock analytics settings
                    'enabled': True,
                    'log_tool_usage': True,
                    'log_query_failures': True
                }
            }
            self._api_providers_data = { # Mock api_providers_data for sports
                "sports": {
                    "sports_api": {
                        "base_url": "https://api.example.com/sports",
                        "api_key_name": "sports_api_key",
                        "api_key_param_name": "api_key",
                        "functions": {
                            "get_sports_scores": {
                                "endpoint": "/scores",
                                "required_params": [],
                                "optional_params": ["sport", "team", "date"],
                                "response_path": ["data"],
                                "data_map": {
                                    "sport": "sport_name",
                                    "team_home": "home_team",
                                    "team_away": "away_team",
                                    "score_home": "home_score",
                                    "score_away": "away_score",
                                    "status": "match_status",
                                    "date": "match_date"
                                }
                            },
                            "get_team_info": {
                                "endpoint": "/teams",
                                "required_params": ["team_name"],
                                "optional_params": ["sport"],
                                "response_path": ["data", 0],
                                "data_map": {
                                    "team_name": "name",
                                    "sport": "sport",
                                    "league": "league",
                                    "city": "city",
                                    "coach": "coach",
                                    "key_players": "players",
                                    "championships": "titles"
                                }
                            },
                            "search_player_stats": {
                                "endpoint": "/players",
                                "required_params": ["player_name"],
                                "optional_params": ["sport"],
                                "response_path": ["data", 0],
                                "data_map": {
                                    "player_name": "name",
                                    "team": "current_team",
                                    "sport": "sport",
                                    "position": "position",
                                    "stats": "stats",
                                    "achievements": "achievements"
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
            # Simulate hypothetical Sports API responses
            if "api.example.com/sports" in url:
                if "/scores" in url:
                    sport = params.get("sport", "").lower()
                    team = params.get("team", "").lower()
                    date = params.get("date")
                    
                    mock_scores = [
                        {
                            "sport_name": "Football",
                            "home_team": "Manchester United",
                            "away_team": "Liverpool",
                            "home_score": 2,
                            "away_score": 1,
                            "match_status": "Finished",
                            "match_date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                        },
                        {
                            "sport_name": "Basketball",
                            "home_team": "L.A. Lakers",
                            "away_team": "Boston Celtics",
                            "home_score": 110,
                            "away_score": 108,
                            "match_status": "Finished",
                            "match_date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
                        }
                    ]
                    
                    filtered_scores = []
                    for score in mock_scores:
                        match = True
                        if sport and score["sport_name"].lower() != sport:
                            match = False
                        if team and team not in score["home_team"].lower() and team not in score["away_team"].lower():
                            match = False
                        if date and score["match_date"] != date:
                            match = False
                        if match:
                            filtered_scores.append(score)

                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"data": filtered_scores}
                    return mock_response
                elif "/teams" in url:
                    team_name = params.get("team_name", "").lower()
                    sport = params.get("sport", "").lower()
                    
                    mock_teams = [
                        {
                            "name": "Golden State Warriors",
                            "sport": "Basketball",
                            "league": "NBA",
                            "city": "San Francisco",
                            "coach": "Steve Kerr",
                            "players": ["Stephen Curry"],
                            "titles": 7
                        },
                        {
                            "name": "Real Madrid CF",
                            "sport": "Football",
                            "league": "La Liga",
                            "city": "Madrid",
                            "coach": "Carlo Ancelotti",
                            "players": ["Vinicius Jr."],
                            "titles": 36
                        }
                    ]
                    
                    filtered_teams = []
                    for team_data in mock_teams:
                        match = True
                        if team_name and team_name not in team_data["name"].lower():
                            match = False
                        if sport and team_data["sport"].lower() != sport:
                            match = False
                        if match:
                            filtered_teams.append(team_data)

                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"data": filtered_teams}
                    return mock_response
                elif "/players" in url:
                    player_name = params.get("player_name", "").lower()
                    sport = params.get("sport", "").lower()

                    mock_players = [
                        {
                            "name": "Stephen Curry",
                            "current_team": "Golden State Warriors",
                            "sport": "Basketball",
                            "position": "Point Guard",
                            "stats": {"points_per_game": 27.3},
                            "achievements": ["4x NBA Champion"]
                        },
                        {
                            "name": "Lionel Messi",
                            "current_team": "Inter Miami CF",
                            "sport": "Football",
                            "position": "Forward",
                            "stats": {"goals_scored": 834},
                            "achievements": ["8x Ballon d'Or"]
                        }
                    ]

                    filtered_players = []
                    for player_data in mock_players:
                        match = True
                        if player_name and player_name not in player_data["name"].lower():
                            match = False
                        if sport and player_data["sport"].lower() != sport:
                            match = False
                        if match:
                            filtered_players.append(player_data)

                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"data": filtered_players}
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
                mock_response.text = f"<html><body><h1>Search results for {params.get('q', 'sports')}</h1><p>Some sports related content from web search.</p></body></html>"
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

        # Patch QueryUploadedDocs and summarize_document in the sports_tool module
        original_QueryUploadedDocs = sys.modules['domain_tools.sports_tools.sports_tool'].QueryUploadedDocs
        original_summarize_document = sys.modules['domain_tools.sports_tools.sports_tool'].summarize_document
        sys.modules['domain_tools.sports_tools.sports_tool'].QueryUploadedDocs = MockQueryUploadedDocs
        sys.modules['domain_tools.sports_tools.sports_tool'].summarize_document = MockSummarizeDocument()

        async def run_sports_tests():
            print("\n--- Testing sports_tool functions with Analytics ---")

            # Test get_sports_scores (success)
            print("\n--- Test 1: get_sports_scores (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock() # Reset mock call count
            result_scores = await get_sports_scores(sport="Football", user_token=test_user_pro)
            print(f"Sports Scores: {result_scores}")
            assert "Sports Scores:" in result_scores
            assert "Manchester United vs Liverpool" in result_scores
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "sports_get_sports_scores"
            assert logged_data["success"] is True
            print("Test 1 Passed (and analytics logged success).")

            # Test get_team_info (API failure - no data found)
            print("\n--- Test 2: get_team_info (API Failure) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_team_info = await get_team_info("NonExistent Team", user_token=test_user_pro)
            print(f"Team Info (API Error): {result_team_info}")
            assert "Could not retrieve complete live team information for 'NonExistent Team'." in result_team_info
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "sports_get_team_info"
            assert logged_data["success"] is False
            assert "Response path 'data.0' not found" in logged_data["error_message"] or "incomplete" in logged_data["error_message"]
            print("Test 2 Passed (and analytics logged failure).")

            # Test search_player_stats (RBAC denied)
            print("\n--- Test 3: search_player_stats (RBAC Denied) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_player_stats_rbac_denied = await search_player_stats("Cristiano Ronaldo", user_token=test_user_free)
            print(f"Player Stats (Free User, RBAC Denied): {result_player_stats_rbac_denied}")
            assert "Error: Access to sports tools is not enabled for your current tier." in result_player_stats_rbac_denied
            # No analytics log expected here because RBAC check happens before _make_dynamic_api_request
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 3 Passed (RBAC correctly prevented call and no analytics logged).")

            # Test sports_search_web (generic tool, not using _make_dynamic_api_request)
            print("\n--- Test 4: sports_search_web (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_web_search = await sports_search_web("history of basketball", user_token=test_user_pro)
            print(f"Web Search Result: {result_web_search[:100]}...")
            assert "Search results for history of basketball" in result_web_search
            # Analytics for generic tools like scrape_web or summarize_document
            # would need to be integrated within those shared_tools themselves,
            # or wrapped by a higher-level agent logging.
            # For now, we are focusing on _make_dynamic_api_request.
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 4 Passed (no analytics expected for generic tool directly).")

            # Test 5: sports_query_uploaded_docs (generic tool)
            print("\n--- Test 5: sports_query_uploaded_docs (Generic Tool) ---")
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
            original_QueryUploadedDocs_in_test = sys.modules['domain_tools.sports_tools.sports_tool'].QueryUploadedDocs
            sys.modules['domain_tools.sports_tools.sports_tool'].QueryUploadedDocs = MockQueryUploadedDocs

            result_doc_query = await sports_query_uploaded_docs("my fantasy football league rules", user_token=test_user_pro)
            print(f"Document Query Result: {result_doc_query}")
            assert "Mocked document query results for 'my fantasy football league rules' in section 'sports'." in result_doc_query
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 5 Passed (no analytics expected for generic tool directly, will be logged by DocumentTools).")
            sys.modules['domain_tools.sports_tools.sports_tool'].QueryUploadedDocs = original_QueryUploadedDocs_in_test # Restore original

            # Test 6: sports_summarize_document_by_path (generic tool)
            print("\n--- Test 6: sports_summarize_document_by_path (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            # Create a dummy file for summarization test
            dummy_file_path = Path("uploads") / test_user_pro / "sports" / "team_strategy.pdf"
            dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_file_path.write_text("This is a dummy team strategy document for testing summarization.")

            result_summarize = await sports_summarize_document_by_path(str(dummy_file_path))
            print(f"Summarize Result: {result_summarize}")
            assert "Mocked summary of team_strategy.pdf" in result_summarize
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 6 Passed (no analytics expected for generic tool directly).")


            print("\nAll sports_tool tests with analytics considerations completed.")

        await run_sports_tests()

        # Restore original requests.get
        requests.get = original_requests_get

        # Clean up dummy files and directories
        test_user_dirs = [Path("uploads") / test_user_pro, BASE_VECTOR_DIR / test_user_pro]
        for d in test_user_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print(f"Cleaned up {d}")
