# domain_tools/crypto_tools/crypto_tool.py

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
            else:
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
_mock_crypto_data = {
    "crypto_price": {
        "bitcoin": {"price": 65000.00, "currency": "USD", "last_updated": datetime.now().isoformat(), "market_cap": 1280000000000, "vol_24hr": 35000000000, "change_24hr": 2.5},
        "ethereum": {"price": 3500.00, "currency": "USD", "last_updated": datetime.now().isoformat(), "market_cap": 420000000000, "vol_24hr": 15000000000, "change_24hr": 1.8}
    },
    "crypto_info": {
        "bitcoin": {
            "name": "Bitcoin",
            "symbol": "BTC",
            "description": "Bitcoin is a decentralized digital currency, without a central bank or single administrator, that can be sent from user to user on the peer-to-peer bitcoin network without the need for intermediaries.",
            "genesis_date": "2009-01-03",
            "market_cap_rank": 1,
            "hashing_algorithm": "SHA-256",
            "website": "https://bitcoin.org/en/"
        },
        "ethereum": {
            "name": "Ethereum",
            "symbol": "ETH",
            "description": "Ethereum is a decentralized, open-source blockchain with smart contract functionality. Ether (ETH) is the native cryptocurrency of the Ethereum platform.",
            "genesis_date": "2015-07-30",
            "market_cap_rank": 2,
            "hashing_algorithm": "Ethash",
            "website": "https://ethereum.org/"
        }
    },
    "historical_crypto_price": {
        "bitcoin": {
            (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"): {"price": 64500.00, "market_cap": 1270000000000, "volume": 34000000000},
            (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"): {"price": 63000.00, "market_cap": 1250000000000, "volume": 32000000000}
        }
    }
}

@tool
def get_crypto_price(crypto_id: str, vs_currencies: str = "usd", user_token: str = "default") -> str:
    """
    Retrieves the current price of a cryptocurrency in one or more specified fiat currencies or other cryptocurrencies.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
        vs_currencies (str, optional): A comma-separated string of currency symbols to compare against (e.g., "usd", "eur", "jpy"). Defaults to "usd".
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of the cryptocurrency price, or an error/fallback message.
    """
    logger.info(f"Tool: get_crypto_price called for crypto_id: '{crypto_id}', vs_currencies: '{vs_currencies}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'crypto_tool_access', False):
        return "Error: Access to crypto tools is not enabled for your current tier."
    
    params = {"ids": crypto_id.lower(), "vs_currencies": vs_currencies.lower()}
    api_data = asyncio.run(_make_dynamic_api_request("crypto", "get_crypto_price", params, user_token))

    if api_data:
        try:
            # CoinGecko simple price returns a flat dict with price, market_cap, vol_24hr, change_24hr
            price = api_data.get("price")
            market_cap = api_data.get("market_cap")
            vol_24hr = api_data.get("vol_24hr")
            change_24hr = api_data.get("change_24hr")
            last_updated = api_data.get("last_updated")

            if price is not None:
                response_str = f"Current price of {crypto_id.capitalize()}: {price} {vs_currencies.upper()}"
                if market_cap is not None:
                    response_str += f"\n  Market Cap: {market_cap:,} {vs_currencies.upper()}"
                if vol_24hr is not None:
                    response_str += f"\n  24hr Volume: {vol_24hr:,} {vs_currencies.upper()}"
                if change_24hr is not None:
                    response_str += f"\n  24hr Change: {change_24hr:.2f}%"
                if last_updated:
                    # CoinGecko's last_updated_at is a Unix timestamp
                    try:
                        last_updated_dt = datetime.fromtimestamp(last_updated)
                        response_str += f"\n  Last Updated: {last_updated_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                    except (ValueError, TypeError):
                        response_str += f"\n  Last Updated: {last_updated}" # Fallback if not a timestamp
                return response_str
            else:
                logger.warning(f"Live API data for {crypto_id} is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live crypto price for {crypto_id}. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live crypto price data for {crypto_id}: {e}")
            return f"Error parsing live data for {crypto_id}. Falling back to mock data."

    # Fallback to mock data
    mock_data = _mock_crypto_data.get("crypto_price", {}).get(crypto_id.lower())
    if mock_data:
        response_str = f"Current price of {crypto_id.capitalize()}: {mock_data['price']} {mock_data['currency']} (Mock Data Fallback)"
        if mock_data.get('market_cap') is not None:
            response_str += f"\n  Market Cap: {mock_data['market_cap']:,} {mock_data['currency']}"
        if mock_data.get('vol_24hr') is not None:
            response_str += f"\n  24hr Volume: {mock_data['vol_24hr']:,} {mock_data['currency']}"
        if mock_data.get('change_24hr') is not None:
            response_str += f"\n  24hr Change: {mock_data['change_24hr']:.2f}%"
        if mock_data.get('last_updated'):
            try:
                last_updated_dt = datetime.fromisoformat(mock_data['last_updated'])
                response_str += f"\n  Last Updated: {last_updated_dt.strftime('%Y-%m-%d %H:%M:%S')}"
            except ValueError:
                response_str += f"\n  Last Updated: {mock_data['last_updated']}"
        return response_str
    else:
        return f"Cryptocurrency price for {crypto_id} not found. (API/Mock Fallback Failed)"


@tool
def get_crypto_info(crypto_id: str, user_token: str = "default") -> str:
    """
    Retrieves general information about a cryptocurrency, such as its description, genesis date, and market cap rank.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of cryptocurrency information, or an error/fallback message.
    """
    logger.info(f"Tool: get_crypto_info called for crypto_id: '{crypto_id}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'crypto_tool_access', False):
        return "Error: Access to crypto tools is not enabled for your current tier."

    params = {"id": crypto_id.lower()}
    api_data = asyncio.run(_make_dynamic_api_request("crypto", "get_crypto_info", params, user_token))

    if api_data:
        try:
            name = api_data.get("name")
            symbol = api_data.get("symbol")
            description = api_data.get("description")
            genesis_date = api_data.get("genesis_date")
            market_cap_rank = api_data.get("market_cap_rank")
            hashing_algorithm = api_data.get("hashing_algorithm")
            website = api_data.get("website")

            if name and description:
                response_str = (
                    f"Information for {name} ({symbol.upper()}):\n"
                    f"  Description: {description}\n"
                )
                if genesis_date:
                    response_str += f"  Genesis Date: {genesis_date}\n"
                if market_cap_rank is not None:
                    response_str += f"  Market Cap Rank: {market_cap_rank}\n"
                if hashing_algorithm:
                    response_str += f"  Hashing Algorithm: {hashing_algorithm}\n"
                if website:
                    response_str += f"  Website: {website}\n"
                return response_str
            else:
                logger.warning(f"Live API data for {crypto_id} is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live crypto information for {crypto_id}. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live crypto info data for {crypto_id}: {e}")
            return f"Error parsing live data for {crypto_id}. Falling back to mock data."

    # Fallback to mock data
    mock_data = _mock_crypto_data.get("crypto_info", {}).get(crypto_id.lower())
    if mock_data:
        response_str = (
            f"Information for {mock_data['name']} ({mock_data['symbol'].upper()}) (Mock Data Fallback):\n"
            f"  Description: {mock_data['description']}\n"
        )
        if mock_data.get('genesis_date'):
            response_str += f"  Genesis Date: {mock_data['genesis_date']}\n"
        if mock_data.get('market_cap_rank') is not None:
            response_str += f"  Market Cap Rank: {mock_data['market_cap_rank']}\n"
        if mock_data.get('hashing_algorithm'):
            response_str += f"  Hashing Algorithm: {mock_data['hashing_algorithm']}\n"
        if mock_data.get('website'):
            response_str += f"  Website: {mock_data['website']}\n"
        return response_str
    else:
        return f"Cryptocurrency information for {crypto_id} not found. (API/Mock Fallback Failed)"


@tool
def get_historical_crypto_price(crypto_id: str, date: str, vs_currency: str = "usd", user_token: str = "default") -> str:
    """
    Retrieves the historical price of a cryptocurrency for a specific date.
    Dates can be in various formats (e.g., 'YYYY-MM-DD', 'MM/DD/YYYY', 'January 1, 2023').
    Falls back to mock data if API key is missing or API call fails.

    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
        date (str): The specific date for which to retrieve historical data.
        vs_currency (str, optional): The currency to compare against (e.g., "usd", "eur"). Defaults to "usd".
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of historical cryptocurrency data, or an error/fallback message.
    """
    logger.info(f"Tool: get_historical_crypto_price called for crypto_id: '{crypto_id}', date: '{date}', vs_currency: '{vs_currency}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'crypto_tool_access', False):
        return "Error: Access to crypto tools is not enabled for your current tier."

    parsed_date = parse_date_to_yyyymmdd(date)
    if not parsed_date:
        return "Error: Could not parse the provided date. Please ensure the date is valid."

    params = {"id": crypto_id.lower(), "date": parsed_date, "vs_currency": vs_currency.lower()}
    api_data = asyncio.run(_make_dynamic_api_request("crypto", "get_historical_crypto_price", params, user_token))

    if api_data:
        try:
            # CoinGecko historical data returns price, market_cap, total_volume
            price = api_data.get("price")
            market_cap = api_data.get("market_cap")
            volume = api_data.get("volume")

            if price is not None:
                response_str = (
                    f"Historical Price for {crypto_id.capitalize()} on {parsed_date}:\n"
                    f"  Price: {price} {vs_currency.upper()}\n"
                )
                if market_cap is not None:
                    response_str += f"  Market Cap: {market_cap:,} {vs_currency.upper()}\n"
                if volume is not None:
                    response_str += f"  24hr Volume: {volume:,} {vs_currency.upper()}\n"
                return response_str
            else:
                logger.warning(f"Live API data for historical crypto price of {crypto_id} on {date} is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live historical crypto price for {crypto_id} on {date}. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live historical crypto price data for {crypto_id} on {date}: {e}")
            return f"Error parsing live data for {crypto_id} on {date}. Falling back to mock data."

    # Fallback to mock data
    mock_data_for_crypto = _mock_crypto_data.get("historical_crypto_price", {}).get(crypto_id.lower(), {})
    mock_daily_data = mock_data_for_crypto.get(parsed_date)
    if mock_daily_data:
        response_str = (
            f"Historical Price for {crypto_id.capitalize()} on {parsed_date} (Mock Data Fallback):\n"
            f"  Price: {mock_daily_data['price']} {vs_currency.upper()}\n"
            f"  Market Cap: {mock_daily_data['market_cap']:,} {vs_currency.upper()}\n"
            f"  24hr Volume: {mock_daily_data['volume']:,} {vs_currency.upper()}\n"
        )
        return response_str
    else:
        return f"Historical cryptocurrency price for {crypto_id} on {date} not found. (API/Mock Fallback Failed)"


# --- Existing Generic Tools (not directly using external APIs, but can be used in crypto context) ---

@tool
def crypto_search_web(query: str, user_token: str = "default", max_chars: int = 2000) -> str:
    """
    Searches the web for cryptocurrency-related information using a smart search fallback mechanism.
    This tool wraps the generic `scrape_web` tool, providing a crypto-specific interface.
    
    Args:
        query (str): The crypto-related search query (e.g., "latest news on Ethereum 2.0", "how to buy Solana").
        user_token (str): The unique identifier for the user. Defaults to "default".
        max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
    
    Returns:
        str: A string containing relevant information from the web.
    """
    logger.info(f"Tool: crypto_search_web called with query: '{query}' for user: '{user_token}'")
    return scrape_web(query=query, user_token=user_token, max_chars=max_chars)

@tool
def crypto_query_uploaded_docs(query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
    """
    Queries previously uploaded and indexed cryptocurrency documents for a user using vector similarity search.
    This tool wraps the generic `QueryUploadedDocs` tool, fixing the section to "crypto".
    
    Args:
        query (str): The search query to find relevant crypto documents (e.g., "whitepaper for project X", "my crypto portfolio balance").
        user_token (str): The unique identifier for the user. Defaults to "default".
        export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
        k (int): The number of top relevant documents to retrieve. Defaults to 5.
    
    Returns:
        str: A string containing the combined content of the relevant document chunks,
             or a message indicating no data/results found, or the export path if exported.
    """
    logger.info(f"Tool: crypto_query_uploaded_docs called with query: '{query}' for user: '{user_token}'")
    return QueryUploadedDocs(query=query, user_token=user_token, section="crypto", export=export, k=k)

@tool
def crypto_summarize_document_by_path(file_path_str: str) -> str:
    """
    Summarizes a document related to cryptocurrency or blockchain located at the given file path.
    The file path should be accessible by the system (e.g., in the 'uploads' directory).
    This tool wraps the generic `summarize_document` tool.
    
    Args:
        file_path_str (str): The full path to the document file to be summarized.
                              Example: "uploads/default/crypto/bitcoin_whitepaper.pdf"
    
    Returns:
        str: A concise summary of the document content.
    """
    logger.info(f"Tool: crypto_summarize_document_by_path called for file: '{file_path_str}'")
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
            self.coingecko_api_key = "MOCK_COINGECKO_API_KEY"
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
                    'crypto': 'coingecko'
                },
                'analytics': { # Mock analytics settings
                    'enabled': True,
                    'log_tool_usage': True,
                    'log_query_failures': True
                }
            }
            self._api_providers_data = { # Mock api_providers_data for crypto
                "crypto": {
                    "coingecko": {
                        "base_url": "https://api.coingecko.com/api/v3",
                        "api_key_name": "coingecko_api_key",
                        "api_key_param_name": "x_cg_demo_api_key", # For CoinGecko's demo key
                        "functions": {
                            "get_crypto_price": {
                                "endpoint": "/simple/price",
                                "required_params": ["ids", "vs_currencies"],
                                "optional_params": ["include_market_cap", "include_24hr_vol", "include_24hr_change", "include_last_updated_at"],
                                "response_path": [], # Root is the data, special handling in _make_dynamic_api_request
                                "data_map": {} # Special handling in _make_dynamic_api_request
                            },
                            "get_crypto_info": {
                                "endpoint": "/coins/{id}", # Path parameter
                                "path_params": ["id"],
                                "required_params": [],
                                "response_path": [], # Root is the data
                                "data_map": {
                                    "name": "name",
                                    "symbol": "symbol",
                                    "description": "description.en", # Nested path
                                    "genesis_date": "genesis_date",
                                    "market_cap_rank": "market_cap_rank",
                                    "hashing_algorithm": "hashing_algorithm",
                                    "website": "links.homepage.0" # Nested path, first item in list
                                }
                            },
                            "get_historical_crypto_price": {
                                "endpoint": "/coins/{id}/history", # Path parameter
                                "path_params": ["id"],
                                "required_params": ["date", "vs_currency"],
                                "response_path": [], # Root is the data
                                "data_map": {
                                    "price": "market_data.current_price.{vs_currency}", # Dynamic key
                                    "market_cap": "market_data.market_cap.{vs_currency}",
                                    "volume": "market_data.total_volumes.{vs_currency}"
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
                'crypto_tool_access': {
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
            # Simulate CoinGecko responses
            if "api.coingecko.com/api/v3" in url:
                if "/simple/price" in url:
                    ids = params.get("ids", "").lower()
                    vs_currencies = params.get("vs_currencies", "").lower()
                    if ids == "bitcoin" and vs_currencies == "usd":
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "bitcoin": {
                                "usd": 65000.00,
                                "usd_market_cap": 1280000000000,
                                "usd_24hr_vol": 35000000000,
                                "usd_24hr_change": 2.5,
                                "last_updated_at": int(datetime.now().timestamp())
                            }
                        }
                        return mock_response
                    elif ids == "ethereum" and vs_currencies == "usd":
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "ethereum": {
                                "usd": 3500.00,
                                "usd_market_cap": 420000000000,
                                "usd_24hr_vol": 15000000000,
                                "usd_24hr_change": 1.8,
                                "last_updated_at": int(datetime.now().timestamp())
                            }
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {}
                        return mock_response
                elif "/coins/" in url and "/history" not in url: # get_crypto_info
                    crypto_id_from_url = url.split("/coins/")[1].split("/")[0].lower()
                    if crypto_id_from_url == "bitcoin":
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "id": "bitcoin", "symbol": "btc", "name": "Bitcoin",
                            "description": {"en": "Bitcoin is a decentralized digital currency..."},
                            "genesis_date": "2009-01-03", "market_cap_rank": 1,
                            "hashing_algorithm": "SHA-256",
                            "links": {"homepage": ["https://bitcoin.org/en/", "other.link"]}
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 404
                        mock_response.json.return_value = {"error": "coin not found"}
                        return mock_response
                elif "/coins/" in url and "/history" in url: # get_historical_crypto_price
                    crypto_id_from_url = url.split("/coins/")[1].split("/history")[0].lower()
                    date = params.get("date")
                    vs_currency = params.get("vs_currency", "usd").lower()
                    if crypto_id_from_url == "bitcoin" and date == (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"):
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "id": "bitcoin", "symbol": "btc", "name": "Bitcoin",
                            "market_data": {
                                "current_price": {vs_currency: 64500.00},
                                "market_cap": {vs_currency: 1270000000000},
                                "total_volume": {vs_currency: 34000000000}
                            }
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {} # No data for this date/crypto
                        return mock_response
            
            # Simulate scrape_web's internal requests.get if needed
            if "google.com/search" in url or "example.com" in url: # Mock for scrape_web
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.text = f"<html><body><h1>Search results for {params.get('q', 'crypto')}</h1><p>Some crypto related content from web search.</p></body></html>"
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

        # Patch QueryUploadedDocs and summarize_document in the crypto_tool module
        original_QueryUploadedDocs = sys.modules['domain_tools.crypto_tools.crypto_tool'].QueryUploadedDocs
        original_summarize_document = sys.modules['domain_tools.crypto_tools.crypto_tool'].summarize_document
        sys.modules['domain_tools.crypto_tools.crypto_tool'].QueryUploadedDocs = MockQueryUploadedDocs
        sys.modules['domain_tools.crypto_tools.crypto_tool'].summarize_document = MockSummarizeDocument()


        async def run_crypto_tests():
            print("\n--- Testing crypto_tool functions with Analytics ---")

            # Test get_crypto_price (success)
            print("\n--- Test 1: get_crypto_price (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock() # Reset mock call count
            result_crypto_price = await get_crypto_price("bitcoin", user_token=test_user_pro)
            print(f"Crypto Price: {result_crypto_price}")
            assert "Current price of Bitcoin: 65000.0 USD" in result_crypto_price
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "crypto_get_crypto_price"
            assert logged_data["success"] is True
            print("Test 1 Passed (and analytics logged success).")

            # Test get_crypto_info (API failure - coin not found)
            print("\n--- Test 2: get_crypto_info (API Failure) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_crypto_info = await get_crypto_info("nonexistentcoin", user_token=test_user_pro)
            print(f"Crypto Info (API Error): {result_crypto_info}")
            assert "Could not retrieve complete live crypto information for nonexistentcoin." in result_crypto_info
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "crypto_get_crypto_info"
            assert logged_data["success"] is False
            assert "coin not found" in logged_data["error_message"]
            print("Test 2 Passed (and analytics logged failure).")

            # Test get_historical_crypto_price (RBAC denied)
            print("\n--- Test 3: get_historical_crypto_price (RBAC Denied) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_historical_rbac_denied = await get_historical_crypto_price("ethereum", "2023-01-01", user_token=test_user_free)
            print(f"Historical Crypto Price (Free User, RBAC Denied): {result_historical_rbac_denied}")
            assert "Error: Access to crypto tools is not enabled for your current tier." in result_historical_rbac_denied
            # No analytics log expected here because RBAC check happens before _make_dynamic_api_request
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 3 Passed (RBAC correctly prevented call and no analytics logged).")

            # Test crypto_search_web (generic tool, not using _make_dynamic_api_request)
            print("\n--- Test 4: crypto_search_web (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_web_search = await crypto_search_web("best crypto wallets", user_token=test_user_pro)
            print(f"Web Search Result: {result_web_search[:100]}...")
            assert "Search results for best crypto wallets" in result_web_search
            # Analytics for generic tools like scrape_web or summarize_document
            # would need to be integrated within those shared_tools themselves,
            # or wrapped by a higher-level agent logging.
            # For now, we are focusing on _make_dynamic_api_request.
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 4 Passed (no analytics expected for generic tool directly).")

            # Test 5: crypto_query_uploaded_docs (generic tool)
            print("\n--- Test 5: crypto_query_uploaded_docs (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_doc_query = await crypto_query_uploaded_docs("whitepaper details", user_token=test_user_pro)
            print(f"Document Query Result: {result_doc_query}")
            assert "Mocked document query results for 'whitepaper details' in section 'crypto'." in result_doc_query
            # Analytics for generic tools like QueryUploadedDocs would be logged by DocumentTools
            # For now, we are focusing on _make_dynamic_api_request and this wrapper.
            # The actual analytics for the underlying query_uploaded_docs_internal will be logged by DocumentTools.
            # Here, we expect analytics for the wrapper `crypto_query_uploaded_docs` itself.
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called() # This tool will be refactored to use DocumentTools, so direct analytics here will be removed.
            print("Test 5 Passed (no analytics expected for generic tool directly, will be logged by DocumentTools).")

            # Test 6: crypto_summarize_document_by_path (generic tool)
            print("\n--- Test 6: crypto_summarize_document_by_path (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            # Create a dummy file for summarization test
            dummy_file_path = Path("uploads") / test_user_pro / "crypto" / "dummy_whitepaper.txt"
            dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_file_path.write_text("This is a dummy whitepaper content for testing summarization.")

            result_summarize = await crypto_summarize_document_by_path(str(dummy_file_path))
            print(f"Summarize Result: {result_summarize}")
            assert "Mocked summary of dummy_whitepaper.txt" in result_summarize
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called() # No analytics expected for generic tool directly
            print("Test 6 Passed (no analytics expected for generic tool directly).")

            print("\nAll crypto_tool tests with analytics considerations completed.")

        await run_crypto_tests()

        # Restore original requests.get
        requests.get = original_requests_get

        # Restore original QueryUploadedDocs and summarize_document
        sys.modules['domain_tools.crypto_tools.crypto_tool'].QueryUploadedDocs = original_QueryUploadedDocs
        sys.modules['domain_tools.crypto_tools.crypto_tool'].summarize_document = original_summarize_document

        # Clean up dummy files and directories
        test_user_dirs = [Path("uploads") / test_user_pro, BASE_VECTOR_DIR / test_user_pro]
        for d in test_user_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print(f"Cleaned up {d}")




**`domain_tools/crypto_tools/crypto_tool.py`**:
* It **does include** `from shared_tools.query_uploaded_docs_tool import QueryUploadedDocs` and it **is used** in the `crypto_query_uploaded_docs` tool function.
* It **does include** `from shared_tools.doc_summarizer import summarize_document` and it **is used** in the `crypto_summarize_document_by_path` tool function.

**`domain_tools/crypto_tools/__init__.py`**:
* This file currently imports `get_crypto_price`, `get_crypto_info`, and `get_historical_crypto_price` and exposes them as methods.
* It **does not yet** include `crypto_query_uploaded_docs` or `crypto_summarize_document_by_path` as methods, nor does its `__init__` accept `document_tools`.

'''

### Corrections for `crypto_tools`

We need to apply the same refactoring pattern as we did for `finance_tools` to `crypto_tools`.

Here are the updates for `crypto_tool.py` and `crypto_tools/__init__.py`:



### 1. Update `domain_tools/crypto_tools/crypto_tool.py`

Please update your `domain_tools/crypto_tools/crypto_tool.py` file by **removing** the line:

`from shared_tools.query_uploaded_docs_tool import QueryUploadedDocs`

The updated file content will be:


```python
# domain_tools/crypto_tools/crypto_tool.py
'''
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
            else:
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
_mock_crypto_data = {
    "crypto_price": {
        "bitcoin": {"price": 65000.00, "currency": "USD", "last_updated": datetime.now().isoformat(), "market_cap": 1280000000000, "vol_24hr": 35000000000, "change_24hr": 2.5},
        "ethereum": {"price": 3500.00, "currency": "USD", "last_updated": datetime.now().isoformat(), "market_cap": 420000000000, "vol_24hr": 15000000000, "change_24hr": 1.8}
    },
    "crypto_info": {
        "bitcoin": {
            "name": "Bitcoin",
            "symbol": "BTC",
            "description": "Bitcoin is a decentralized digital currency, without a central bank or single administrator, that can be sent from user to user on the peer-to-peer bitcoin network without the need for intermediaries.",
            "genesis_date": "2009-01-03",
            "market_cap_rank": 1,
            "hashing_algorithm": "SHA-256",
            "website": "[https://bitcoin.org/en/](https://bitcoin.org/en/)"
        },
        "ethereum": {
            "name": "Ethereum",
            "symbol": "ETH",
            "description": "Ethereum is a decentralized, open-source blockchain with smart contract functionality. Ether (ETH) is the native cryptocurrency of the Ethereum platform.",
            "genesis_date": "2015-07-30",
            "market_cap_rank": 2,
            "hashing_algorithm": "Ethash",
            "website": "[https://ethereum.org/](https://ethereum.org/)"
        }
    },
    "historical_crypto_price": {
        "bitcoin": {
            (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"): {"price": 64500.00, "market_cap": 1270000000000, "volume": 34000000000},
            (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"): {"price": 63000.00, "market_cap": 1250000000000, "volume": 32000000000}
        }
    }
}

@tool
def get_crypto_price(crypto_id: str, vs_currencies: str = "usd", user_token: str = "default") -> str:
    """
    Retrieves the current price of a cryptocurrency in one or more specified fiat currencies or other cryptocurrencies.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
        vs_currencies (str, optional): A comma-separated string of currency symbols to compare against (e.g., "usd", "eur", "jpy"). Defaults to "usd".
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of the cryptocurrency price, or an error/fallback message.
    """
    logger.info(f"Tool: get_crypto_price called for crypto_id: '{crypto_id}', vs_currencies: '{vs_currencies}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'crypto_tool_access', False):
        return "Error: Access to crypto tools is not enabled for your current tier."
    
    params = {"ids": crypto_id.lower(), "vs_currencies": vs_currencies.lower()}
    api_data = asyncio.run(_make_dynamic_api_request("crypto", "get_crypto_price", params, user_token))

    if api_data:
        try:
            # CoinGecko simple price returns a flat dict with price, market_cap, vol_24hr, change_24hr
            price = api_data.get("price")
            market_cap = api_data.get("market_cap")
            vol_24hr = api_data.get("vol_24hr")
            change_24hr = api_data.get("change_24hr")
            last_updated = api_data.get("last_updated")

            if price is not None:
                response_str = f"Current price of {crypto_id.capitalize()}: {price} {vs_currencies.upper()}"
                if market_cap is not None:
                    response_str += f"\n  Market Cap: {market_cap:,} {vs_currencies.upper()}"
                if vol_24hr is not None:
                    response_str += f"\n  24hr Volume: {vol_24hr:,} {vs_currencies.upper()}"
                if change_24hr is not None:
                    response_str += f"\n  24hr Change: {change_24hr:.2f}%"
                if last_updated:
                    # CoinGecko's last_updated_at is a Unix timestamp
                    try:
                        last_updated_dt = datetime.fromtimestamp(last_updated)
                        response_str += f"\n  Last Updated: {last_updated_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                    except (ValueError, TypeError):
                        response_str += f"\n  Last Updated: {last_updated}" # Fallback if not a timestamp
                return response_str
            else:
                logger.warning(f"Live API data for {crypto_id} is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live crypto price for {crypto_id}. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live crypto price data for {crypto_id}: {e}")
            return f"Error parsing live data for {crypto_id}. Falling back to mock data."

    # Fallback to mock data
    mock_data = _mock_crypto_data.get("crypto_price", {}).get(crypto_id.lower())
    if mock_data:
        response_str = f"Current price of {crypto_id.capitalize()}: {mock_data['price']} {mock_data['currency']} (Mock Data Fallback)"
        if mock_data.get('market_cap') is not None:
            response_str += f"\n  Market Cap: {mock_data['market_cap']:,} {mock_data['currency']}"
        if mock_data.get('vol_24hr') is not None:
            response_str += f"\n  24hr Volume: {mock_data['vol_24hr']:,} {mock_data['currency']}"
        if mock_data.get('change_24hr') is not None:
            response_str += f"\n  24hr Change: {mock_data['change_24hr']:.2f}%"
        if mock_data.get('last_updated'):
            try:
                last_updated_dt = datetime.fromisoformat(mock_data['last_updated'])
                response_str += f"\n  Last Updated: {last_updated_dt.strftime('%Y-%m-%d %H:%M:%S')}"
            except ValueError:
                response_str += f"\n  Last Updated: {mock_data['last_updated']}"
        return response_str
    else:
        return f"Cryptocurrency price for {crypto_id} not found. (API/Mock Fallback Failed)"


@tool
def get_crypto_info(crypto_id: str, user_token: str = "default") -> str:
    """
    Retrieves general information about a cryptocurrency, such as its description, genesis date, and market cap rank.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of cryptocurrency information, or an error/fallback message.
    """
    logger.info(f"Tool: get_crypto_info called for crypto_id: '{crypto_id}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'crypto_tool_access', False):
        return "Error: Access to crypto tools is not enabled for your current tier."

    params = {"id": crypto_id.lower()}
    api_data = asyncio.run(_make_dynamic_api_request("crypto", "get_crypto_info", params, user_token))

    if api_data:
        try:
            name = api_data.get("name")
            symbol = api_data.get("symbol")
            description = api_data.get("description")
            genesis_date = api_data.get("genesis_date")
            market_cap_rank = api_data.get("market_cap_rank")
            hashing_algorithm = api_data.get("hashing_algorithm")
            website = api_data.get("website")

            if name and description:
                response_str = (
                    f"Information for {name} ({symbol.upper()}):\n"
                    f"  Description: {description}\n"
                )
                if genesis_date:
                    response_str += f"  Genesis Date: {genesis_date}\n"
                if market_cap_rank is not None:
                    response_str += f"  Market Cap Rank: {market_cap_rank}\n"
                if hashing_algorithm:
                    response_str += f"  Hashing Algorithm: {hashing_algorithm}\n"
                if website:
                    response_str += f"  Website: {website}\n"
                return response_str
            else:
                logger.warning(f"Live API data for {crypto_id} is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live crypto information for {crypto_id}. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live crypto info data for {crypto_id}: {e}")
            return f"Error parsing live data for {crypto_id}. Falling back to mock data."

    # Fallback to mock data
    mock_data = _mock_crypto_data.get("crypto_info", {}).get(crypto_id.lower())
    if mock_data:
        response_str = (
            f"Information for {mock_data['name']} ({mock_data['symbol'].upper()}) (Mock Data Fallback):\n"
            f"  Description: {mock_data['description']}\n"
        )
        if mock_data.get('genesis_date'):
            response_str += f"  Genesis Date: {mock_data['genesis_date']}\n"
        if mock_data.get('market_cap_rank') is not None:
            response_str += f"  Market Cap Rank: {mock_data['market_cap_rank']}\n"
        if mock_data.get('hashing_algorithm'):
            response_str += f"  Hashing Algorithm: {mock_data['hashing_algorithm']}\n"
        if mock_data.get('website'):
            response_str += f"  Website: {mock_data['website']}\n"
        return response_str
    else:
        return f"Cryptocurrency information for {crypto_id} not found. (API/Mock Fallback Failed)"


@tool
def get_historical_crypto_price(crypto_id: str, date: str, vs_currency: str = "usd", user_token: str = "default") -> str:
    """
    Retrieves the historical price of a cryptocurrency for a specific date.
    Dates can be in various formats (e.g., 'YYYY-MM-DD', 'MM/DD/YYYY', 'January 1, 2023').
    Falls back to mock data if API key is missing or API call fails.

    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
        date (str): The specific date for which to retrieve historical data.
        vs_currency (str, optional): The currency to compare against (e.g., "usd", "eur"). Defaults to "usd".
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of historical cryptocurrency data, or an error/fallback message.
    """
    logger.info(f"Tool: get_historical_crypto_price called for crypto_id: '{crypto_id}', date: '{date}', vs_currency: '{vs_currency}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'crypto_tool_access', False):
        return "Error: Access to crypto tools is not enabled for your current tier."

    parsed_date = parse_date_to_yyyymmdd(date)
    if not parsed_date:
        return "Error: Could not parse the provided date. Please ensure the date is valid."

    params = {"id": crypto_id.lower(), "date": parsed_date, "vs_currency": vs_currency.lower()}
    api_data = asyncio.run(_make_dynamic_api_request("crypto", "get_historical_crypto_price", params, user_token))

    if api_data:
        try:
            # CoinGecko historical data returns price, market_cap, total_volume
            price = api_data.get("price")
            market_cap = api_data.get("market_cap")
            volume = api_data.get("volume")

            if price is not None:
                response_str = (
                    f"Historical Price for {crypto_id.capitalize()} on {parsed_date}:\n"
                    f"  Price: {price} {vs_currency.upper()}\n"
                )
                if market_cap is not None:
                    response_str += f"  Market Cap: {market_cap:,} {vs_currency.upper()}\n"
                if volume is not None:
                    response_str += f"  24hr Volume: {volume:,} {vs_currency.upper()}\n"
                return response_str
            else:
                logger.warning(f"Live API data for historical crypto price of {crypto_id} on {date} is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live historical crypto price for {crypto_id} on {date}. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live historical crypto price data for {crypto_id} on {date}: {e}")
            return f"Error parsing live data for {crypto_id} on {date}. Falling back to mock data."

    # Fallback to mock data
    mock_data_for_crypto = _mock_crypto_data.get("historical_crypto_price", {}).get(crypto_id.lower(), {})
    mock_daily_data = mock_data_for_crypto.get(parsed_date)
    if mock_daily_data:
        response_str = (
            f"Historical Price for {crypto_id.capitalize()} on {parsed_date} (Mock Data Fallback):\n"
            f"  Price: {mock_daily_data['price']} {vs_currency.upper()}\n"
            f"  Market Cap: {mock_daily_data['market_cap']:,} {vs_currency.upper()}\n"
            f"  24hr Volume: {mock_daily_data['volume']:,} {vs_currency.upper()}\n"
        )
        return response_str
    else:
        return f"Historical cryptocurrency price for {crypto_id} on {date} not found. (API/Mock Fallback Failed)"


# --- Existing Generic Tools (not directly using external APIs, but can be used in crypto context) ---

@tool
def crypto_search_web(query: str, user_token: str = "default", max_chars: int = 2000) -> str:
    """
    Searches the web for cryptocurrency-related information using a smart search fallback mechanism.
    This tool wraps the generic `scrape_web` tool, providing a crypto-specific interface.
    
    Args:
        query (str): The crypto-related search query (e.g., "latest news on Ethereum 2.0", "how to buy Solana").
        user_token (str): The unique identifier for the user. Defaults to "default".
        max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
    
    Returns:
        str: A string containing relevant information from the web.
    """
    logger.info(f"Tool: crypto_search_web called with query: '{query}' for user: '{user_token}'")
    return scrape_web(query=query, user_token=user_token, max_chars=max_chars)

@tool
def crypto_query_uploaded_docs(query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
    """
    Queries previously uploaded and indexed cryptocurrency documents for a user using vector similarity search.
    This tool wraps the generic `QueryUploadedDocs` tool, fixing the section to "crypto".
    
    Args:
        query (str): The search query to find relevant crypto documents (e.g., "whitepaper for project X", "my crypto portfolio balance").
        user_token (str): The unique identifier for the user. Defaults to "default".
        export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
        k (int): The number of top relevant documents to retrieve. Defaults to 5.
    
    Returns:
        str: A string containing the combined content of the relevant document chunks,
             or a message indicating no data/results found, or the export path if exported.
    """
    logger.info(f"Tool: crypto_query_uploaded_docs called with query: '{query}' for user: '{user_token}'")
    # This will be replaced by a call to self.document_tools.query_uploaded_docs
    # For now, keeping the original call for review purposes.
    return QueryUploadedDocs(query=query, user_token=user_token, section="crypto", export=export, k=k)

@tool
def crypto_summarize_document_by_path(file_path_str: str) -> str:
    """
    Summarizes a document related to cryptocurrency or blockchain located at the given file path.
    The file path should be accessible by the system (e.g., in the 'uploads' directory).
    This tool wraps the generic `summarize_document` tool.
    
    Args:
        file_path_str (str): The full path to the document file to be summarized.
                              Example: "uploads/default/crypto/bitcoin_whitepaper.pdf"
    
    Returns:
        str: A concise summary of the document content.
    """
    logger.info(f"Tool: crypto_summarize_document_by_path called for file: '{file_path_str}'")
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
            self.coingecko_api_key = "MOCK_COINGECKO_API_KEY"
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
                    'crypto': 'coingecko'
                },
                'analytics': { # Mock analytics settings
                    'enabled': True,
                    'log_tool_usage': True,
                    'log_query_failures': True
                }
            }
            self._api_providers_data = { # Mock api_providers_data for crypto
                "crypto": {
                    "coingecko": {
                        "base_url": "[https://api.coingecko.com/api/v3](https://api.coingecko.com/api/v3)",
                        "api_key_name": "coingecko_api_key",
                        "api_key_param_name": "x_cg_demo_api_key", # For CoinGecko's demo key
                        "functions": {
                            "get_crypto_price": {
                                "endpoint": "/simple/price",
                                "required_params": ["ids", "vs_currencies"],
                                "optional_params": ["include_market_cap", "include_24hr_vol", "include_24hr_change", "include_last_updated_at"],
                                "response_path": [], # Root is the data, special handling in _make_dynamic_api_request
                                "data_map": {} # Special handling in _make_dynamic_api_request
                            },
                            "get_crypto_info": {
                                "endpoint": "/coins/{id}", # Path parameter
                                "path_params": ["id"],
                                "required_params": [],
                                "response_path": [], # Root is the data
                                "data_map": {
                                    "name": "name",
                                    "symbol": "symbol",
                                    "description": "description.en", # Nested path
                                    "genesis_date": "genesis_date",
                                    "market_cap_rank": "market_cap_rank",
                                    "hashing_algorithm": "hashing_algorithm",
                                    "website": "links.homepage.0" # Nested path, first item in list
                                }
                            },
                            "get_historical_crypto_price": {
                                "endpoint": "/coins/{id}/history", # Path parameter
                                "path_params": ["id"],
                                "required_params": ["date", "vs_currency"],
                                "response_path": [], # Root is the data
                                "data_map": {
                                    "price": "market_data.current_price.{vs_currency}", # Dynamic key
                                    "market_cap": "market_data.market_cap.{vs_currency}",
                                    "volume": "market_data.total_volumes.{vs_currency}"
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
                'crypto_tool_access': {
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
            # Simulate CoinGecko responses
            if "[api.coingecko.com/api/v3](https://api.coingecko.com/api/v3)" in url:
                if "/simple/price" in url:
                    ids = params.get("ids", "").lower()
                    vs_currencies = params.get("vs_currencies", "").lower()
                    if ids == "bitcoin" and vs_currencies == "usd":
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "bitcoin": {
                                "usd": 65000.00,
                                "usd_market_cap": 1280000000000,
                                "usd_24hr_vol": 35000000000,
                                "usd_24hr_change": 2.5,
                                "last_updated_at": int(datetime.now().timestamp())
                            }
                        }
                        return mock_response
                    elif ids == "ethereum" and vs_currencies == "usd":
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "ethereum": {
                                "usd": 3500.00,
                                "usd_market_cap": 420000000000,
                                "usd_24hr_vol": 15000000000,
                                "usd_24hr_change": 1.8,
                                "last_updated_at": int(datetime.now().timestamp())
                            }
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {}
                        return mock_response
                elif "/coins/" in url and "/history" not in url: # get_crypto_info
                    crypto_id_from_url = url.split("/coins/")[1].split("/")[0].lower()
                    if crypto_id_from_url == "bitcoin":
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "id": "bitcoin", "symbol": "btc", "name": "Bitcoin",
                            "description": {"en": "Bitcoin is a decentralized digital currency..."},
                            "genesis_date": "2009-01-03", "market_cap_rank": 1,
                            "hashing_algorithm": "SHA-256",
                            "links": {"homepage": ["[https://bitcoin.org/en/](https://bitcoin.org/en/)", "other.link"]}
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 404
                        mock_response.json.return_value = {"error": "coin not found"}
                        return mock_response
                elif "/coins/" in url and "/history" in url: # get_historical_crypto_price
                    crypto_id_from_url = url.split("/coins/")[1].split("/history")[0].lower()
                    date = params.get("date")
                    vs_currency = params.get("vs_currency", "usd").lower()
                    if crypto_id_from_url == "bitcoin" and date == (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"):
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "id": "bitcoin", "symbol": "btc", "name": "Bitcoin",
                            "market_data": {
                                "current_price": {vs_currency: 64500.00},
                                "market_cap": {vs_currency: 1270000000000},
                                "total_volume": {vs_currency: 34000000000}
                            }
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {} # No data for this date/crypto
                        return mock_response
            
            # Simulate scrape_web's internal requests.get if needed
            if "[google.com/search](https://google.com/search)" in url or "example.com" in url: # Mock for scrape_web
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.text = f"<html><body><h1>Search results for {params.get('q', 'crypto')}</h1><p>Some crypto related content from web search.</p></body></html>"
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

        # Patch QueryUploadedDocs and summarize_document in the crypto_tool module
        original_QueryUploadedDocs = sys.modules['domain_tools.crypto_tools.crypto_tool'].QueryUploadedDocs
        original_summarize_document = sys.modules['domain_tools.crypto_tools.crypto_tool'].summarize_document
        sys.modules['domain_tools.crypto_tools.crypto_tool'].QueryUploadedDocs = MockQueryUploadedDocs
        sys.modules['domain_tools.crypto_tools.crypto_tool'].summarize_document = MockSummarizeDocument()


        async def run_crypto_tests():
            print("\n--- Testing crypto_tool functions with Analytics ---")

            # Test get_crypto_price (success)
            print("\n--- Test 1: get_crypto_price (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock() # Reset mock call count
            result_crypto_price = await get_crypto_price("bitcoin", user_token=test_user_pro)
            print(f"Crypto Price: {result_crypto_price}")
            assert "Current price of Bitcoin: 65000.0 USD" in result_crypto_price
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "crypto_get_crypto_price"
            assert logged_data["success"] is True
            print("Test 1 Passed (and analytics logged success).")

            # Test get_crypto_info (API failure - coin not found)
            print("\n--- Test 2: get_crypto_info (API Failure) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_crypto_info = await get_crypto_info("nonexistentcoin", user_token=test_user_pro)
            print(f"Crypto Info (API Error): {result_crypto_info}")
            assert "Could not retrieve complete live crypto information for nonexistentcoin." in result_crypto_info
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "crypto_get_crypto_info"
            assert logged_data["success"] is False
            assert "coin not found" in logged_data["error_message"]
            print("Test 2 Passed (and analytics logged failure).")

            # Test get_historical_crypto_price (RBAC denied)
            print("\n--- Test 3: get_historical_crypto_price (RBAC Denied) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_historical_rbac_denied = await get_historical_crypto_price("ethereum", "2023-01-01", user_token=test_user_free)
            print(f"Historical Crypto Price (Free User, RBAC Denied): {result_historical_rbac_denied}")
            assert "Error: Access to crypto tools is not enabled for your current tier." in result_historical_rbac_denied
            # No analytics log expected here because RBAC check happens before _make_dynamic_api_request
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 3 Passed (RBAC correctly prevented call and no analytics logged).")

            # Test crypto_search_web (generic tool, not using _make_dynamic_api_request)
            print("\n--- Test 4: crypto_search_web (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_web_search = await crypto_search_web("best crypto wallets", user_token=test_user_pro)
            print(f"Web Search Result: {result_web_search[:100]}...")
            assert "Search results for best crypto wallets" in result_web_search
            # Analytics for generic tools like scrape_web or summarize_document
            # would need to be integrated within those shared_tools themselves,
            # or wrapped by a higher-level agent logging.
            # For now, we are focusing on _make_dynamic_api_request.
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 4 Passed (no analytics expected for generic tool directly).")

            # Test 5: crypto_query_uploaded_docs (generic tool)
            print("\n--- Test 5: crypto_query_uploaded_docs (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_doc_query = await crypto_query_uploaded_docs("whitepaper details", user_token=test_user_pro)
            print(f"Document Query Result: {result_doc_query}")
            assert "Mocked document query results for 'whitepaper details' in section 'crypto'." in result_doc_query
            # Analytics for generic tools like QueryUploadedDocs would be logged by DocumentTools
            # For now, we are focusing on _make_dynamic_api_request and this wrapper.
            # The actual analytics for the underlying query_uploaded_docs_internal will be logged by DocumentTools.
            # Here, we expect analytics for the wrapper `crypto_query_uploaded_docs` itself.
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called() # This tool will be refactored to use DocumentTools, so direct analytics here will be removed.
            print("Test 5 Passed (no analytics expected for generic tool directly, will be logged by DocumentTools).")

            # Test 6: crypto_summarize_document_by_path (generic tool)
            print("\n--- Test 6: crypto_summarize_document_by_path (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            # Create a dummy file for summarization test
            dummy_file_path = Path("uploads") / test_user_pro / "crypto" / "dummy_whitepaper.txt"
            dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_file_path.write_text("This is a dummy whitepaper content for testing summarization.")

            result_summarize = await crypto_summarize_document_by_path(str(dummy_file_path))
            print(f"Summarize Result: {result_summarize}")
            assert "Mocked summary of dummy_whitepaper.txt" in result_summarize
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called() # No analytics expected for generic tool directly
            print("Test 6 Passed (no analytics expected for generic tool directly).")

            print("\nAll crypto_tool tests with analytics considerations completed.")

        await run_crypto_tests()

        # Restore original requests.get
        requests.get = original_requests_get

        # Restore original QueryUploadedDocs and summarize_document
        sys.modules['domain_tools.crypto_tools.crypto_tool'].QueryUploadedDocs = original_QueryUploadedDocs
        sys.modules['domain_tools.crypto_tools.crypto_tool'].summarize_document = original_summarize_document

        # Clean up dummy files and directories
        test_user_dirs = [Path("uploads") / test_user_pro, BASE_VECTOR_DIR / test_user_pro]
        for d in test_user_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print(f"Cleaned up {d}")


