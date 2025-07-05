# domain_tools/finance_tools/finance_tool.py

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
_mock_finance_data = {
    "stock_price": {
        "AAPL": {"price": 175.00, "currency": "USD", "last_updated": datetime.now().isoformat()},
        "GOOGL": {"price": 150.50, "currency": "USD", "last_updated": datetime.now().isoformat()}
    },
    "company_overview": {
        "AAPL": {
            "symbol": "AAPL",
            "asset_type": "Equity",
            "name": "Apple Inc.",
            "description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
            "exchange": "NASDAQ",
            "currency": "USD",
            "country": "USA",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "market_capitalization": "2.7T USD",
            "pe_ratio": "28.5",
            "dividend_yield": "0.5%",
            "52_week_high": "199.62",
            "52_week_low": "129.00",
            "address": "One Apple Park Way, Cupertino, California, 95014, United States"
        }
    },
    "forex_rate": {
        "USD/JPY": {"exchange_rate": 155.00, "last_refreshed": datetime.now().isoformat(), "from_currency": "USD", "to_currency": "JPY"},
        "EUR/USD": {"exchange_rate": 1.08, "last_refreshed": datetime.now().isoformat(), "from_currency": "EUR", "to_currency": "USD"}
    },
    "historical_data": {
        "AAPL": {
            (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"): {"open": 170.00, "high": 176.00, "low": 169.50, "close": 175.00, "volume": 80000000},
            (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"): {"open": 168.00, "high": 171.00, "low": 167.50, "close": 170.00, "volume": 75000000}
        }
    }
}

@tool
def get_stock_price(symbol: str, user_token: str = "default") -> str:
    """
    Retrieves the current stock price for a given stock symbol.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL" for Apple, "GOOGL" for Alphabet).
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of the stock price and last updated time, or an error/fallback message.
    """
    logger.info(f"Tool: get_stock_price called for symbol: '{symbol}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'finance_tool_access', False):
        return "Error: Access to finance tools is not enabled for your current tier."
    
    params = {"symbol": symbol.upper()}
    api_data = asyncio.run(_make_dynamic_api_request("finance", "get_stock_price", params, user_token))

    if api_data:
        try:
            price = api_data.get("price")
            currency = api_data.get("currency", "USD")
            last_updated = api_data.get("last_updated")

            if price is not None:
                response_str = f"Current price of {symbol.upper()}: {price} {currency}"
                if last_updated:
                    # Attempt to parse and format if it's an ISO string
                    try:
                        last_updated_dt = datetime.fromisoformat(last_updated)
                        response_str += f" (as of {last_updated_dt.strftime('%Y-%m-%d %H:%M')})"
                    except ValueError:
                        response_str += f" (last updated: {last_updated})" # Use as is if not ISO
                return response_str
            else:
                logger.warning(f"Live API data for {symbol} is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live stock price for {symbol}. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live stock price data for {symbol}: {e}")
            return f"Error parsing live data for {symbol}. Falling back to mock data."

    # Fallback to mock data
    mock_data = _mock_finance_data.get("stock_price", {}).get(symbol.upper())
    if mock_data:
        response_str = f"Current price of {symbol.upper()}: {mock_data['price']} {mock_data['currency']} (Mock Data Fallback)"
        if mock_data.get('last_updated'):
            try:
                last_updated_dt = datetime.fromisoformat(mock_data['last_updated'])
                response_str += f" (as of {last_updated_dt.strftime('%Y-%m-%d %H:%M')})"
            except ValueError:
                response_str += f" (last updated: {mock_data['last_updated']})"
        return response_str
    else:
        return f"Stock price for {symbol} not found. (API/Mock Fallback Failed)"


@tool
def get_company_overview(symbol: str, user_token: str = "default") -> str:
    """
    Retrieves a detailed overview of a company based on its stock symbol.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of company information, or an error/fallback message.
    """
    logger.info(f"Tool: get_company_overview called for symbol: '{symbol}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'finance_tool_access', False):
        return "Error: Access to finance tools is not enabled for your current tier."

    params = {"symbol": symbol.upper()}
    api_data = asyncio.run(_make_dynamic_api_request("finance", "get_company_overview", params, user_token))

    if api_data:
        try:
            name = api_data.get("name")
            description = api_data.get("description")
            sector = api_data.get("sector")
            industry = api_data.get("industry")
            market_cap = api_data.get("market_capitalization")
            pe_ratio = api_data.get("pe_ratio")
            dividend_yield = api_data.get("dividend_yield")

            if name and description:
                response_str = (
                    f"Company Overview for {name} ({symbol.upper()}):\n"
                    f"  Description: {description}\n"
                    f"  Sector: {sector}\n"
                    f"  Industry: {industry}\n"
                    f"  Market Cap: {market_cap}\n"
                    f"  P/E Ratio: {pe_ratio}\n"
                    f"  Dividend Yield: {dividend_yield}\n"
                )
                return response_str
            else:
                logger.warning(f"Live API data for company overview of {symbol} is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live company overview for {symbol}. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live company overview data for {symbol}: {e}")
            return f"Error parsing live data for {symbol}. Falling back to mock data."

    # Fallback to mock data
    mock_data = _mock_finance_data.get("company_overview", {}).get(symbol.upper())
    if mock_data:
        response_str = (
            f"Company Overview for {mock_data['name']} ({symbol.upper()}) (Mock Data Fallback):\n"
            f"  Description: {mock_data['description']}\n"
            f"  Sector: {mock_data['sector']}\n"
            f"  Industry: {mock_data['industry']}\n"
            f"  Market Cap: {mock_data['market_capitalization']}\n"
            f"  P/E Ratio: {mock_data['pe_ratio']}\n"
            f"  Dividend Yield: {mock_data['dividend_yield']}\n"
        )
        return response_str
    else:
        return f"Company overview for {symbol} not found. (API/Mock Fallback Failed)"


@tool
def get_forex_exchange_rate(from_currency: str, to_currency: str, user_token: str = "default") -> str:
    """
    Retrieves the current exchange rate between two currencies.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        from_currency (str): The symbol of the currency to convert from (e.g., "USD", "EUR").
        to_currency (str): The symbol of the currency to convert to (e.g., "JPY", "GBP").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of the exchange rate, or an error/fallback message.
    """
    logger.info(f"Tool: get_forex_exchange_rate called for {from_currency}/{to_currency} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'finance_tool_access', False):
        return "Error: Access to finance tools is not enabled for your current tier."

    params = {"from_currency": from_currency.upper(), "to_currency": to_currency.upper()}
    api_data = asyncio.run(_make_dynamic_api_request("finance", "get_forex_exchange_rate", params, user_token))

    if api_data:
        try:
            exchange_rate = api_data.get("exchange_rate")
            last_refreshed = api_data.get("last_refreshed")

            if exchange_rate is not None:
                response_str = f"1 {from_currency.upper()} = {exchange_rate} {to_currency.upper()}"
                if last_refreshed:
                    try:
                        last_refreshed_dt = datetime.fromisoformat(last_refreshed)
                        response_str += f" (as of {last_refreshed_dt.strftime('%Y-%m-%d %H:%M')})"
                    except ValueError:
                        response_str += f" (last refreshed: {last_refreshed})"
                return response_str
            else:
                logger.warning(f"Live API data for forex {from_currency}/{to_currency} is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live exchange rate for {from_currency}/{to_currency}. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live forex data for {from_currency}/{to_currency}: {e}")
            return f"Error parsing live data for {from_currency}/{to_currency}. Falling back to mock data."

    # Fallback to mock data
    mock_key = f"{from_currency.upper()}/{to_currency.upper()}"
    mock_data = _mock_finance_data.get("forex_rate", {}).get(mock_key)
    if mock_data:
        response_str = f"1 {mock_data['from_currency']} = {mock_data['exchange_rate']} {mock_data['to_currency']} (Mock Data Fallback)"
        if mock_data.get('last_refreshed'):
            try:
                last_refreshed_dt = datetime.fromisoformat(mock_data['last_refreshed'])
                response_str += f" (as of {last_refreshed_dt.strftime('%Y-%m-%d %H:%M')})"
            except ValueError:
                response_str += f" (last refreshed: {mock_data['last_refreshed']})"
        return response_str
    else:
        return f"Exchange rate for {from_currency}/{to_currency} not found. (API/Mock Fallback Failed)"


@tool
def get_historical_stock_prices(symbol: str, date: str, user_token: str = "default") -> str:
    """
    Retrieves the historical stock prices (open, high, low, close, volume) for a given symbol on a specific date.
    Dates can be in various formats (e.g., 'YYYY-MM-DD', 'MM/DD/YYYY', 'January 1, 2023').
    Falls back to mock data if API key is missing or API call fails.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL").
        date (str): The specific date for which to retrieve historical data.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of historical stock data, or an error/fallback message.
    """
    logger.info(f"Tool: get_historical_stock_prices called for symbol: '{symbol}', date: '{date}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'finance_tool_access', False):
        return "Error: Access to finance tools is not enabled for your current tier."

    parsed_date = parse_date_to_yyyymmdd(date)
    if not parsed_date:
        return "Error: Could not parse the provided date. Please ensure the date is valid."

    params = {"symbol": symbol.upper(), "date": parsed_date}
    api_data = asyncio.run(_make_dynamic_api_request("finance", "get_historical_stock_prices", params, user_token))

    if api_data and api_data.get("data"): # Note: api_data is wrapped in {"data": ...} for lists/dicts where keys are dates
        # For historical data, the 'data' key holds a dict where keys are dates
        daily_data = api_data["data"].get(parsed_date)
        if daily_data:
            try:
                response_str = (
                    f"Historical Stock Data for {symbol.upper()} on {parsed_date}:\n"
                    f"  Open: {daily_data.get('open', 'N/A')}\n"
                    f"  High: {daily_data.get('high', 'N/A')}\n"
                    f"  Low: {daily_data.get('low', 'N/A')}\n"
                    f"  Close: {daily_data.get('close', 'N/A')}\n"
                    f"  Volume: {daily_data.get('volume', 'N/A')}\n"
                )
                return response_str
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing live historical data for {symbol} on {date}: {e}")
                return f"Error parsing live data for {symbol} on {date}. Falling back to mock data."
        else:
            logger.warning(f"No live historical data found for {symbol} on {date}. Raw: {api_data}")
            return f"No live historical data found for {symbol} on {date}. Falling back to mock data."

    # Fallback to mock data
    mock_data_for_symbol = _mock_finance_data.get("historical_data", {}).get(symbol.upper(), {})
    mock_daily_data = mock_data_for_symbol.get(parsed_date)
    if mock_daily_data:
        response_str = (
            f"Historical Stock Data for {symbol.upper()} on {parsed_date} (Mock Data Fallback):\n"
            f"  Open: {mock_daily_data['open']}\n"
            f"  High: {mock_daily_data['high']}\n"
            f"  Low: {mock_daily_data['low']}\n"
            f"  Close: {mock_daily_data['close']}\n"
            f"  Volume: {mock_daily_data['volume']}\n"
        )
        return response_str
    else:
        return f"Historical stock data for {symbol} on {date} not found. (API/Mock Fallback Failed)"


# --- Existing Generic Tools (not directly using external APIs, but can be used in finance context) ---

@tool
def finance_search_web(query: str, user_token: str = "default", max_chars: int = 2000) -> str:
    """
    Searches the web for finance-related information using a smart search fallback mechanism.
    This tool wraps the generic `scrape_web` tool, providing a finance-specific interface.
    
    Args:
        query (str): The finance-related search query (e.g., "impact of inflation on stock market", "best investment strategies 2024").
        user_token (str): The unique identifier for the user. Defaults to "default".
        max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
    
    Returns:
        str: A string containing relevant information from the web.
    """
    logger.info(f"Tool: finance_search_web called with query: '{query}' for user: '{user_token}'")
    return scrape_web(query=query, user_token=user_token, max_chars=max_chars)

@tool
def finance_query_uploaded_docs(query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
    """
    Queries previously uploaded and indexed financial documents for a user using vector similarity search.
    This tool wraps the generic `QueryUploadedDocs` tool, fixing the section to "finance".
    
    Args:
        query (str): The search query to find relevant financial documents (e.g., "annual report of company X", "my investment portfolio details").
        user_token (str): The unique identifier for the user. Defaults to "default".
        export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
        k (int): The number of top relevant documents to retrieve. Defaults to 5.
    
    Returns:
        str: A string containing the combined content of the relevant document chunks,
             or a message indicating no data/results found, or the export path if exported.
    """
    logger.info(f"Tool: finance_query_uploaded_docs called with query: '{query}' for user: '{user_token}'")
    return QueryUploadedDocs(query=query, user_token=user_token, section="finance", export=export, k=k)

@tool
def finance_summarize_document_by_path(file_path_str: str) -> str:
    """
    Summarizes a document related to finance or economics located at the given file path.
    The file path should be accessible by the system (e.g., in the 'uploads' directory).
    This tool wraps the generic `summarize_document` tool.
    
    Args:
        file_path_str (str): The full path to the document file to be summarized.
                              Example: "uploads/default/finance/Q3_earnings.pdf"
    
    Returns:
        str: A concise summary of the document content.
    """
    logger.info(f"Tool: finance_summarize_document_by_path called for file: '{file_path_str}'")
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
    from unittest.mock import MagicMock, AsyncMock, patch
    import shutil
    import os
    from shared_tools.vector_utils import BASE_VECTOR_DIR # For cleanup
    from shared_tools.python_interpreter_tool import python_interpreter_with_rbac # For testing REPL
    import asyncio # Import asyncio for running async functions

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.alphavantage_api_key = "MOCK_ALPHAVANTAGE_API_KEY"
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
                    'finance': 'alphavantage'
                },
                'analytics': { # Mock analytics settings
                    'enabled': True,
                    'log_tool_usage': True,
                    'log_query_failures': True
                }
            }
            self._api_providers_data = { # Mock api_providers_data for finance
                "finance": {
                    "alphavantage": {
                        "base_url": "https://www.alphavantage.co/query",
                        "api_key_name": "alphavantage_api_key",
                        "api_key_param_name": "apikey",
                        "functions": {
                            "get_stock_price": {
                                "function_param": "GLOBAL_QUOTE",
                                "required_params": ["symbol"],
                                "response_path": ["Global Quote"],
                                "data_map": {
                                    "price": "05. price",
                                    "currency": "08. previous close", # Mocking currency here, real API has no direct currency
                                    "last_updated": "07. latest trading day" # Mocking as last updated
                                }
                            },
                            "get_company_overview": {
                                "function_param": "OVERVIEW",
                                "required_params": ["symbol"],
                                "response_path": [], # Root is the data
                                "data_map": {
                                    "name": "Name",
                                    "description": "Description",
                                    "sector": "Sector",
                                    "industry": "Industry",
                                    "market_capitalization": "MarketCapitalization",
                                    "pe_ratio": "PERatio",
                                    "dividend_yield": "DividendYield"
                                }
                            },
                            "get_forex_exchange_rate": {
                                "function_param": "CURRENCY_EXCHANGE_RATE",
                                "required_params": ["from_currency", "to_currency"],
                                "response_path": ["Realtime Currency Exchange Rate"],
                                "data_map": {
                                    "from_currency": "1. From_Currency Code",
                                    "to_currency": "3. To_Currency Code",
                                    "exchange_rate": "5. Exchange Rate",
                                    "last_refreshed": "6. Last Refreshed"
                                }
                            },
                            "get_historical_stock_prices": {
                                "function_param": "TIME_SERIES_DAILY",
                                "required_params": ["symbol"],
                                "optional_params": ["outputsize"], # full or compact
                                "response_path": ["Time Series (Daily)"], # This will return a dict of dates
                                "data_map": {
                                    "open": "1. open",
                                    "high": "2. high",
                                    "low": "3. low",
                                    "close": "4. close",
                                    "volume": "5. volume"
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
                'finance_tool_access': {
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
            # Simulate Alpha Vantage responses
            if "alphavantage.co/query" in url:
                function = params.get("function")
                symbol = params.get("symbol")
                from_currency = params.get("from_currency")
                to_currency = params.get("to_currency")
                date = params.get("date")

                if function == "GLOBAL_QUOTE" and symbol == "AAPL":
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "Global Quote": {
                            "01. symbol": "AAPL", "02. open": "170.0000", "03. high": "176.0000",
                            "04. low": "169.5000", "05. price": "175.0000", "06. volume": "80000000",
                            "07. latest trading day": "2024-07-04", "08. previous close": "170.0000",
                            "09. change": "5.0000", "10. change percent": "2.9412%"
                        }
                    }
                    return mock_response
                elif function == "OVERVIEW" and symbol == "AAPL":
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "Symbol": "AAPL", "AssetType": "Equity", "Name": "Apple Inc.",
                        "Description": "Apple Inc. designs, manufactures, and markets smartphones...",
                        "Exchange": "NASDAQ", "Currency": "USD", "Country": "USA",
                        "Sector": "Technology", "Industry": "Consumer Electronics",
                        "MarketCapitalization": "2700000000000", "PERatio": "28.5",
                        "DividendYield": "0.005"
                    }
                    return mock_response
                elif function == "CURRENCY_EXCHANGE_RATE" and from_currency == "USD" and to_currency == "JPY":
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "Realtime Currency Exchange Rate": {
                            "1. From_Currency Code": "USD", "2. From_Currency Name": "United States Dollar",
                            "3. To_Currency Code": "JPY", "4. To_Currency Name": "Japanese Yen",
                            "5. Exchange Rate": "155.0000", "6. Last Refreshed": "2024-07-04 10:00:00",
                            "7. Time Zone": "UTC", "8. Bid Price": "154.9500", "9. Ask Price": "155.0500"
                        }
                    }
                    return mock_response
                elif function == "TIME_SERIES_DAILY" and symbol == "AAPL":
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "Meta Data": {},
                        "Time Series (Daily)": {
                            "2024-07-04": {"1. open": "170.00", "2. high": "176.00", "3. low": "169.50", "4. close": "175.00", "5. volume": "80000000"},
                            "2024-07-03": {"1. open": "168.00", "2. high": "171.00", "3. low": "167.50", "4. close": "170.00", "5. volume": "75000000"}
                        }
                    }
                    return mock_response
                else:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"Error Message": "Invalid API call or symbol."}
                    return mock_response
            
            # Simulate scrape_web's internal requests.get if needed
            if "google.com/search" in url or "example.com" in url: # Mock for scrape_web
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.text = f"<html><body><h1>Search results for {params.get('q', 'finance')}</h1><p>Some finance related content from web search.</p></body></html>"
                return mock_response

            return original_requests_get(url, params=params, headers=headers, timeout=timeout)

        requests.get = mock_requests_get_dynamic

        test_user_pro = "mock_pro_token"
        test_user_free = "mock_free_token"

        async def run_finance_tests():
            print("\n--- Testing finance_tool functions with Analytics ---")

            # Test get_stock_price (success)
            print("\n--- Test 1: get_stock_price (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock() # Reset mock call count
            result_stock_price = await get_stock_price("AAPL", user_token=test_user_pro)
            print(f"Stock Price: {result_stock_price}")
            assert "Current price of AAPL: 175.0000 USD" in result_stock_price
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "finance_get_stock_price"
            assert logged_data["success"] is True
            print("Test 1 Passed (and analytics logged success).")

            # Test get_company_overview (failure - mock API returns error)
            print("\n--- Test 2: get_company_overview (API Failure) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            # Temporarily modify mock_requests_get_dynamic for this specific call to simulate API error
            original_mock_get = requests.get
            def mock_get_error(url, params, headers, timeout):
                if "alphavantage.co/query" in url and params.get("function") == "OVERVIEW":
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"Error Message": "Invalid API call or symbol."}
                    return mock_response
                return original_mock_get(url, params=params, headers=headers, timeout=timeout)
            requests.get = mock_get_error

            result_company_overview = await get_company_overview("INVALID", user_token=test_user_pro)
            print(f"Company Overview (API Error): {result_company_overview}")
            assert "Could not retrieve complete live company overview for INVALID." in result_company_overview
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "finance_get_company_overview"
            assert logged_data["success"] is False
            assert "Invalid API call or symbol." in logged_data["error_message"]
            print("Test 2 Passed (and analytics logged failure).")
            requests.get = original_mock_get # Restore original mock

            # Test get_forex_exchange_rate (RBAC denied)
            print("\n--- Test 3: get_forex_exchange_rate (RBAC Denied) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_forex_rbac_denied = await get_forex_exchange_rate("USD", "JPY", user_token=test_user_free)
            print(f"Forex Rate (Free User, RBAC Denied): {result_forex_rbac_denied}")
            assert "Error: Access to finance tools is not enabled for your current tier." in result_forex_rbac_denied
            # No analytics log expected here because RBAC check happens before _make_dynamic_api_request
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 3 Passed (RBAC correctly prevented call and no analytics logged).")

            # Test finance_search_web (generic tool, not using _make_dynamic_api_request)
            print("\n--- Test 4: finance_search_web (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_web_search = await finance_search_web("latest stock market news", user_token=test_user_pro)
            print(f"Web Search Result: {result_web_search[:100]}...")
            assert "Search results for latest stock market news" in result_web_search
            # This tool uses scrape_web, which is a generic tool.
            # Analytics for generic tools like scrape_web or summarize_document
            # would need to be integrated within those shared_tools themselves,
            # or wrapped by a higher-level agent logging.
            # For now, we are focusing on _make_dynamic_api_request.
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 4 Passed (no analytics expected for generic tool directly).")

            print("\nAll finance_tool tests with analytics considerations completed.")

        await run_finance_tests()

        # Restore original requests.get
        requests.get = original_requests_get

        # Clean up dummy files and directories
        test_user_dirs = [Path("uploads") / test_user_pro, BASE_VECTOR_DIR / test_user_pro]
        for d in test_user_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print(f"Cleaned up {d}")
