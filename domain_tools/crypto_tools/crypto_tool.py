# domain_tools/crypto_tools/crypto_tool.py

import logging
import requests
import json
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
from datetime import datetime, timedelta

# Import config_manager for API keys and dynamic API provider configurations
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability

logger = logging.getLogger(__name__)

# --- Generic API Request Helper (re-using the one from finance_tool, or defining here if standalone) ---
# For simplicity and to avoid circular imports if tools are separate,
# we'll include a copy of the helper here. In a larger refactor, this
# helper might live in a shared 'utils/api_helper.py' or similar.

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
    path_params = function_details.get("path_params", []) # For CoinGecko style path params

    if not endpoint and not function_param:
        logger.error(f"Neither 'endpoint' nor 'function_param' defined for function '{function_name}'.")
        return None

    # Construct URL
    full_url = f"{base_url}{endpoint}" if endpoint else base_url

    # Add path parameters to URL if specified
    for p_param in path_params:
        if p_param in params:
            full_url = full_url.replace(f"{{{p_param}}}", str(params.pop(p_param)))
        else:
            logger.warning(f"Missing path parameter '{p_param}' for function '{function_name}'.")
            return None # Cannot construct URL without required path params

    # Construct query parameters
    query_params = {}
    if function_param:
        query_params["function"] = function_param # Alpha Vantage specific

    # Add API key if it's a query param (not in path or header)
    if api_key_name and active_provider_name != "amadeus": # Amadeus handled by headers
        param_name_in_url = provider_config.get("api_key_param_name", api_key_name.replace("_api_key", ""))
        if api_key: # Only add if key exists
            query_params[param_name_in_url] = api_key 

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


# --- Mock Data for Fallback (Simplified) ---
_mock_crypto_data = {
    "bitcoin": {
        "price": 65000.00,
        "currency": "USD",
        "market_cap": 1280000000000,
        "vol_24hr": 35000000000,
        "change_24hr": -1.5,
        "last_updated": datetime.now().timestamp()
    },
    "ethereum": {
        "price": 3500.00,
        "currency": "USD",
        "market_cap": 420000000000,
        "vol_24hr": 18000000000,
        "change_24hr": 2.1,
        "last_updated": datetime.now().timestamp()
    },
    "historical_bitcoin": [
        {"timestamp": (datetime.now() - timedelta(days=5)).timestamp() * 1000, "price": 60000},
        {"timestamp": (datetime.now() - timedelta(days=4)).timestamp() * 1000, "price": 61000},
        {"timestamp": (datetime.now() - timedelta(days=3)).timestamp() * 1000, "price": 62500},
        {"timestamp": (datetime.now() - timedelta(days=2)).timestamp() * 1000, "price": 63000},
        {"timestamp": (datetime.now() - timedelta(days=1)).timestamp() * 1000, "price": 64500},
        {"timestamp": datetime.now().timestamp() * 1000, "price": 65000}
    ],
    "id_lookup": {
        "btc": {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"},
        "eth": {"id": "ethereum", "symbol": "eth", "name": "Ethereum"},
        "sol": {"id": "solana", "symbol": "sol", "name": "Solana"}
    }
}

@tool
def get_crypto_price(crypto_id: str, vs_currency: str = "usd", user_token: str = "default") -> str:
    """
    Retrieves the current price of a cryptocurrency by its ID (e.g., 'bitcoin')
    against a specified fiat or crypto currency (e.g., 'usd', 'eur').
    Falls back to mock data if API key is missing or API call fails.

    Args:
        crypto_id (str): The CoinGecko ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
        vs_currency (str, optional): The currency to compare against (e.g., "usd", "eur"). Defaults to "usd".
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A string containing the current price and related information, or an error/fallback message.
    """
    logger.info(f"Tool: get_crypto_price called for {crypto_id} vs {vs_currency} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'crypto_tool_access', False):
        return "Error: Access to crypto tools is not enabled for your current tier."

    api_data = _make_dynamic_api_request(
        "crypto", "get_crypto_price",
        {"ids": crypto_id.lower(), "vs_currencies": vs_currency.lower(),
         "include_market_cap": "true", "include_24hr_vol": "true",
         "include_24hr_change": "true", "include_last_updated_at": "true"},
        user_token
    )

    if api_data:
        try:
            price = api_data.get("price")
            market_cap = api_data.get("market_cap")
            vol_24hr = api_data.get("vol_24hr")
            change_24hr = api_data.get("change_24hr")
            last_updated_timestamp = api_data.get("last_updated")

            if price is not None:
                last_updated_str = datetime.fromtimestamp(last_updated_timestamp).strftime("%Y-%m-%d %H:%M:%S") if last_updated_timestamp else "N/A"
                response_str = (
                    f"Current price for {crypto_id.capitalize()} ({vs_currency.upper()}):\n"
                    f"  Price: {price:,.2f} {vs_currency.upper()}\n"
                )
                if market_cap is not None:
                    response_str += f"  Market Cap: {market_cap:,.2f} {vs_currency.upper()}\n"
                if vol_24hr is not None:
                    response_str += f"  24hr Volume: {vol_24hr:,.2f} {vs_currency.upper()}\n"
                if change_24hr is not None:
                    response_str += f"  24hr Change: {change_24hr:+.2f}%\n"
                response_str += f"  Last Updated: {last_updated_str}"
                return response_str
            else:
                logger.warning(f"Live API data for {crypto_id} is missing price. Raw: {api_data}")
                return f"Could not retrieve live price for {crypto_id.capitalize()}. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live crypto price data for {crypto_id}: {e}")
            return f"Error parsing live data for {crypto_id}. Falling back to mock data."

    # Fallback to mock data
    mock_data = _mock_crypto_data.get(crypto_id.lower())
    if mock_data and mock_data.get("currency", "usd").lower() == vs_currency.lower():
        return (
            f"Current price for {crypto_id.capitalize()} ({vs_currency.upper()}) (Mock Data Fallback):\n"
            f"  Price: {mock_data['price']:,.2f} {mock_data['currency'].upper()}\n"
            f"  Market Cap: {mock_data['market_cap']:,.2f} {mock_data['currency'].upper()}\n"
            f"  24hr Change: {mock_data['change_24hr']:+.2f}%\n"
            f"  Last Updated (Mock): {datetime.fromtimestamp(mock_data['last_updated']).strftime('%Y-%m-%d %H:%M:%S')}"
        )
    else:
        return f"Cryptocurrency price information not found for '{crypto_id}' in '{vs_currency}'. (API/Mock Fallback Failed)"


@tool
def get_historical_crypto_prices(crypto_id: str, vs_currency: str = "usd", days: int = 7, user_token: str = "default") -> str:
    """
    Retrieves historical daily prices for a cryptocurrency over a specified number of days.
    Returns data in JSON format for easy plotting/analysis.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        crypto_id (str): The CoinGecko ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
        vs_currency (str, optional): The currency to compare against (e.g., "usd", "eur"). Defaults to "usd".
        days (int, optional): Number of days for historical data (e.g., 7, 30). Defaults to 7.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A JSON string containing historical daily prices, or an error/fallback message.
    """
    logger.info(f"Tool: get_historical_crypto_prices called for {crypto_id} vs {vs_currency} over {days} days by user: {user_token}")

    if not get_user_tier_capability(user_token, 'historical_data_access', False):
        return "Error: Access to historical data is not enabled for your current tier."
    
    # CoinGecko expects 'id' as a path parameter
    api_data = _make_dynamic_api_request(
        "crypto", "get_historical_crypto_prices",
        {"id": crypto_id.lower(), "vs_currency": vs_currency.lower(), "days": days},
        user_token
    )

    if api_data:
        prices_data = api_data.get("prices")
        if prices_data:
            historical_data_formatted = []
            for timestamp, price in prices_data:
                historical_data_formatted.append({
                    "date": datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d"), # Convert ms to s
                    "price": price
                })
            return json.dumps(historical_data_formatted, indent=2)
        else:
            return f"No live historical data found for {crypto_id.capitalize()} over {days} days. Falling back to mock data."

    # Fallback to mock data
    mock_key = f"historical_{crypto_id.lower()}"
    if mock_key in _mock_crypto_data:
        filtered_mock_data = []
        # Filter mock data to simulate 'days' parameter
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        for entry in _mock_crypto_data[mock_key]:
            entry_date = datetime.fromtimestamp(entry["timestamp"] / 1000)
            if start_date <= entry_date <= end_date:
                filtered_mock_data.append({
                    "date": entry_date.strftime("%Y-%m-%d"),
                    "price": entry["price"]
                })
        
        if filtered_mock_data:
            return json.dumps(filtered_mock_data, indent=2)
        else:
            return f"No mock historical data found for {crypto_id.capitalize()} over {days} days. (API/Mock Fallback Failed)"
    else:
        return f"Historical cryptocurrency price information not found for '{crypto_id}'. (API/Mock Fallback Failed)"

@tool
def get_crypto_id_by_symbol(symbol: str, user_token: str = "default") -> str:
    """
    Looks up the CoinGecko ID for a given cryptocurrency symbol (e.g., 'btc', 'eth').
    This is useful as many CoinGecko API calls require the full ID ('bitcoin', 'ethereum').
    Falls back to mock data if API key is missing or API call fails.

    Args:
        symbol (str): The common symbol of the cryptocurrency (e.g., "btc", "eth").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: The CoinGecko ID (e.g., "bitcoin"), or an error/fallback message.
    """
    logger.info(f"Tool: get_crypto_id_by_symbol called for symbol: {symbol} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'crypto_tool_access', False):
        return "Error: Access to crypto tools is not enabled for your current tier."
    
    api_data = _make_dynamic_api_request(
        "crypto", "get_crypto_id_by_symbol",
        {}, # No specific params needed for this endpoint, it returns a list
        user_token
    )

    if api_data and api_data.get("data"): # 'data' key because _make_dynamic_api_request wraps lists
        crypto_list = api_data["data"]
        # Find the first match (CoinGecko list is large, so iterate)
        found_id = None
        found_name = None
        for crypto in crypto_list:
            if crypto.get("symbol", "").lower() == symbol.lower():
                found_id = crypto.get("id")
                found_name = crypto.get("name")
                break
        
        if found_id and found_name:
            return f"Found CoinGecko ID for symbol '{symbol.upper()}': '{found_id}' (Name: {found_name})"
        else:
            return f"No live CoinGecko ID found for symbol '{symbol}'. Falling back to mock data."

    # Fallback to mock data
    mock_lookup = _mock_crypto_data.get("id_lookup", {})
    for key, details in mock_lookup.items():
        if details.get("symbol", "").lower() == symbol.lower():
            return f"Found CoinGecko ID for symbol '{symbol.upper()}' (Mock Data Fallback): '{details['id']}' (Name: {details['name']})"
    
    return f"CoinGecko ID not found for symbol '{symbol}'. (API/Mock Fallback Failed)"


# CLI Test (optional)
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.coingecko_api_key = "MOCK_COINGECKO_KEY" # CoinGecko free tier doesn't need a key, but good to have a placeholder
            self.openai_api_key = "sk-mock-openai-key-12345"
            self.google_api_key = "AIzaSy-mock-google-key"
            self.firebase_config = "{}"

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
                }
            }
            self._api_providers_data = { # Mock api_providers_data for crypto
                "crypto": {
                    "coingecko": {
                        "base_url": "https://api.coingecko.com/api/v3",
                        "api_key_name": "coingecko_api_key", # Optional for free tier
                        "functions": {
                            "get_crypto_price": {
                                "endpoint": "/simple/price",
                                "required_params": ["ids", "vs_currencies"],
                                "optional_params": ["include_market_cap", "include_24hr_vol", "include_24hr_change", "include_last_updated_at"],
                                "response_path": [] # Special handling in _make_dynamic_api_request
                            },
                            "get_historical_crypto_prices": {
                                "endpoint": "/coins/{id}/market_chart",
                                "path_params": ["id"],
                                "required_params": ["vs_currency", "days"],
                                "optional_params": ["interval"],
                                "response_path": [] # Special handling in tool
                            },
                            "get_crypto_id_by_symbol": {
                                "endpoint": "/coins/list",
                                "required_params": [],
                                "optional_params": ["include_platform"],
                                "response_path": [] # Root level response is a list
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
            return st.secrets.get(key, default)

        def set_secret(self, key, value):
            setattr(st.secrets, key, value)
        
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
                'historical_data_access': {
                    'default': False,
                    'roles': {'premium': True, 'admin': True}
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

            for role in user_roles:
                if role in capability_config.get('roles', {}):
                    return capability_config['roles'][role]
            
            return capability_config.get('default', default_value)

    # Patch the actual imports for testing
    import streamlit as st_mock
    if not hasattr(st_mock, 'secrets'):
        st_mock.secrets = MockSecrets()
    
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager
    sys.modules['utils.user_manager'] = MockUserManager()
    sys.modules['utils.user_manager']._RBAC_CAPABILITIES = MockUserManager()._rbac_capabilities
    sys.modules['utils.user_manager']._TIER_HIERARCHY = MockUserManager()._tier_hierarchy

    # Mock requests.get for external API calls
    original_requests_get = requests.get

    def mock_requests_get_dynamic(url, params, headers, timeout):
        # Simulate CoinGecko responses based on endpoint and params
        if "api.coingecko.com/api/v3" in url:
            if "/simple/price" in url:
                ids = params.get("ids")
                vs_currencies = params.get("vs_currencies")
                if ids == "bitcoin" and vs_currencies == "usd":
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "bitcoin": {
                            "usd": 68500.00,
                            "usd_market_cap": 1350000000000,
                            "usd_24hr_vol": 40000000000,
                            "usd_24hr_change": 3.2,
                            "last_updated_at": datetime.now().timestamp()
                        }
                    }
                    return mock_response
                elif ids == "ethereum" and vs_currencies == "usd":
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "ethereum": {
                            "usd": 3800.00,
                            "usd_market_cap": 450000000000,
                            "usd_24hr_vol": 20000000000,
                            "usd_24hr_change": 1.8,
                            "last_updated_at": datetime.now().timestamp()
                        }
                    }
                    return mock_response
            elif "/coins/bitcoin/market_chart" in url:
                vs_currency = params.get("vs_currency")
                days = int(params.get("days", 0))
                if vs_currency == "usd" and days > 0:
                    mock_prices = []
                    for i in range(days + 1):
                        date = datetime.now() - timedelta(days=days - i)
                        price = 60000 + i * 1000 + (i % 2) * 500 # Simulate some price movement
                        mock_prices.append([date.timestamp() * 1000, price])
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"prices": mock_prices}
                    return mock_response
            elif "/coins/list" in url:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = [
                    {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"},
                    {"id": "ethereum", "symbol": "eth", "name": "Ethereum"},
                    {"id": "solana", "symbol": "sol", "name": "Solana"},
                    {"id": "ripple", "symbol": "xrp", "name": "XRP"}
                ]
                return mock_response
            
            # Simulate CoinGecko error (e.g., rate limit, invalid ID)
            if "invalid" in ids or "invalid" in url:
                 mock_response = MagicMock()
                 mock_response.status_code = 400
                 mock_response.json.return_value = {"status": {"error_code": 400, "error_message": "invalid parameter"}}
                 return mock_response

        return original_requests_get(url, params=params, headers=headers, timeout=timeout)

    requests.get = mock_requests_get_dynamic

    test_user_pro = sys.modules['utils.user_manager']._mock_users["mock_pro_token"]['user_id']
    test_user_premium = sys.modules['utils.user_manager']._mock_users["mock_premium_token']['user_id']
    
    print("\n--- Testing get_crypto_price function (with API key) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro # Ensure user has access
    result1 = get_crypto_price("bitcoin", user_token=test_user_pro)
    print(f"Result for Bitcoin Price (Pro User, API):\n{result1[:200]}...")
    assert "Current price for Bitcoin (USD):" in result1
    assert "Price: 68,500.00 USD" in result1
    print("Test 1 Passed.")

    print("\n--- Testing get_crypto_price function (no API key - fallback) ---")
    # CoinGecko free tier doesn't need a key, so simulate a generic API failure
    # by making _make_dynamic_api_request return None
    with patch('domain_tools.crypto_tools.crypto_tool._make_dynamic_api_request', return_value=None):
        result2 = get_crypto_price("ethereum", user_token=test_user_pro)
        print(f"Result for Ethereum Price (Pro User, Fallback):\n{result2[:200]}...")
        assert "Current price for Ethereum (USD) (Mock Data Fallback):" in result2
    print("Test 2 Passed.")

    print("\n--- Testing get_historical_crypto_prices function (with API key) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_premium # Historical data is premium
    result3 = get_historical_crypto_prices("bitcoin", days=3, user_token=test_user_premium)
    print(f"Result for Bitcoin Historical (Premium User, API):\n{result3[:200]}...")
    assert "price" in result3
    assert "date" in result3
    print("Test 3 Passed.")

    print("\n--- Testing get_crypto_id_by_symbol function (with API key) ---")
    result4 = get_crypto_id_by_symbol("eth", user_token=test_user_pro)
    print(f"Result for ETH Symbol Lookup (Pro User, API): {result4}")
    assert "Found CoinGecko ID for symbol 'ETH': 'ethereum' (Name: Ethereum)" in result4
    print("Test 4 Passed.")

    print("\nAll crypto_tool tests passed (real API simulation with fallback).")

    # Restore original requests.get
    requests.get = original_requests_get
