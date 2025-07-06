# domain_tools/finance_tools/finance_tool.py

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
_mock_finance_data = {
    "stock_prices": {
        "GOOG": {
            "date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            "open": 150.00,
            "high": 152.50,
            "low": 149.80,
            "close": 151.20,
            "volume": 1000000
        },
        "AAPL": {
            "date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            "open": 170.00,
            "high": 171.50,
            "low": 169.80,
            "close": 170.80,
            "volume": 1500000
        }
    },
    "company_overview": {
        "GOOG": {
            "symbol": "GOOG",
            "name": "Alphabet Inc.",
            "exchange": "NASDAQ",
            "sector": "Technology",
            "industry": "Internet Content & Information",
            "description": "Alphabet Inc. is an American multinational technology conglomerate holding company...",
            "market_cap": "2.0T",
            "pe_ratio": 28.5
        },
        "AAPL": {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "exchange": "NASDAQ",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
            "market_cap": "3.0T",
            "pe_ratio": 32.1
        }
    },
    "currency_exchange_rates": {
        "USD_EUR": {
            "from_currency": "USD",
            "to_currency": "EUR",
            "exchange_rate": 0.92,
            "last_refreshed": datetime.now().isoformat()
        },
        "GBP_USD": {
            "from_currency": "GBP",
            "to_currency": "USD",
            "exchange_rate": 1.27,
            "last_refreshed": datetime.now().isoformat()
        }
    },
    "economic_indicators": {
        "unemployment_rate_us": {
            "indicator": "Unemployment Rate (US)",
            "date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "value": 3.9,
            "unit": "%",
            "source": "Bureau of Labor Statistics"
        },
        "gdp_growth_us": {
            "indicator": "GDP Growth Rate (US)",
            "date": (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
            "value": 2.5,
            "unit": "%",
            "source": "Bureau of Economic Analysis"
        }
    }
}

@tool
def get_stock_price(symbol: str, date: Optional[str] = None, user_token: str = "default") -> str:
    """
    Retrieves the daily stock price (Open, High, Low, Close, Volume) for a given stock symbol
    on a specific date. If no date is provided, it fetches the latest available daily price.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL", "GOOG").
        date (str, optional): The specific date in YYYY-MM-DD format (e.g., "2023-01-15").
                              If not provided, the latest available data is returned.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of stock price information, or an error/fallback message.
    """
    logger.info(f"Tool: get_stock_price called for symbol: '{symbol}', date: '{date}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'finance_tool_access', False):
        return "Error: Access to finance tools is not enabled for your current tier."
    
    params = {"symbol": symbol}
    parsed_date = None
    if date:
        parsed_date = parse_date_to_yyyymmdd(date)
        if not parsed_date:
            return "Error: Could not parse the provided date. Please ensure the date is valid."
        params["date"] = parsed_date

    api_data = asyncio.run(_make_dynamic_api_request("finance", "get_stock_price", params, user_token))

    if api_data:
        try:
            # Alpha Vantage returns a dictionary where keys are dates. We need to handle this.
            # If a specific date was requested, try to find it. Otherwise, get the first (latest) entry.
            if parsed_date:
                price_data = api_data.get("data", {}).get(parsed_date)
            else:
                # Get the first (latest) entry if no specific date was requested
                all_dates = list(api_data.get("data", {}).keys())
                if all_dates:
                    latest_date = all_dates[0] # Assuming API returns in reverse chronological order
                    price_data = api_data["data"][latest_date]
                    parsed_date = latest_date # Update parsed_date for the response string
                else:
                    price_data = None

            if price_data:
                open_price = price_data.get("open")
                high_price = price_data.get("high")
                low_price = price_data.get("low")
                close_price = price_data.get("close")
                volume = price_data.get("volume")

                response_str = (
                    f"Stock Price for {symbol} on {parsed_date or 'latest available date'}:\n"
                    f"  Open: {open_price}\n"
                    f"  High: {high_price}\n"
                    f"  Low: {low_price}\n"
                    f"  Close: {close_price}\n"
                    f"  Volume: {volume}"
                )
                return response_str
            else:
                logger.warning(f"Live API data for stock price for '{symbol}' on '{parsed_date}' not found or incomplete. Raw: {api_data}")
                return f"Could not retrieve live stock price for '{symbol}' on '{parsed_date or 'latest available date'}'. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live stock price data for '{symbol}': {e}")
            return f"Error parsing live data for '{symbol}'. Falling back to mock data."

    # Fallback to mock data
    mock_data = _mock_finance_data.get("stock_prices", {}).get(symbol.upper())
    if mock_data and (not parsed_date or mock_data.get("date") == parsed_date):
        response_str = (
            f"Stock Price for {symbol} on {mock_data['date']} (Mock Data Fallback):\n"
            f"  Open: {mock_data['open']}\n"
            f"  High: {mock_data['high']}\n"
            f"  Low: {mock_data['low']}\n"
            f"  Close: {mock_data['close']}\n"
            f"  Volume: {mock_data['volume']}"
        )
        return response_str
    else:
        return f"Stock price information not found for '{symbol}' on '{date or 'latest available date'}'. (API/Mock Fallback Failed)"


@tool
def get_company_overview(symbol: str, user_token: str = "default") -> str:
    """
    Retrieves a detailed overview of a company based on its stock symbol,
    including sector, industry, description, and market capitalization.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL", "GOOG").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of company information, or an error/fallback message.
    """
    logger.info(f"Tool: get_company_overview called for symbol: '{symbol}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'finance_tool_access', False):
        return "Error: Access to finance tools is not enabled for your current tier."
    
    params = {"symbol": symbol}
    api_data = asyncio.run(_make_dynamic_api_request("finance", "get_company_overview", params, user_token))

    if api_data:
        try:
            name = api_data.get("name")
            exchange = api_data.get("exchange")
            sector = api_data.get("sector")
            industry = api_data.get("industry")
            description = api_data.get("description")
            market_cap = api_data.get("market_cap")
            pe_ratio = api_data.get("pe_ratio")

            if name and description:
                response_str = (
                    f"Company Overview for {name} ({symbol}):\n"
                    f"  Exchange: {exchange}\n"
                    f"  Sector: {sector}\n"
                    f"  Industry: {industry}\n"
                    f"  Market Cap: {market_cap}\n"
                    f"  P/E Ratio: {pe_ratio}\n"
                    f"  Description: {description}"
                )
                return response_str
            else:
                logger.warning(f"Live API data for company overview for '{symbol}' is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live company overview for '{symbol}'. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live company overview data for '{symbol}': {e}")
            return f"Error parsing live data for '{symbol}'. Falling back to mock data."

    # Fallback to mock data
    mock_data = _mock_finance_data.get("company_overview", {}).get(symbol.upper())
    if mock_data:
        response_str = (
            f"Company Overview for {mock_data['name']} ({mock_data['symbol']}) (Mock Data Fallback):\n"
            f"  Exchange: {mock_data['exchange']}\n"
            f"  Sector: {mock_data['sector']}\n"
            f"  Industry: {mock_data['industry']}\n"
            f"  Market Cap: {mock_data['market_cap']}\n"
            f"  P/E Ratio: {mock_data['pe_ratio']}\n"
            f"  Description: {mock_data['description']}"
        )
        return response_str
    else:
        return f"Company overview information not found for '{symbol}'. (API/Mock Fallback Failed)"


@tool
def get_currency_exchange_rate(from_currency: str, to_currency: str, user_token: str = "default") -> str:
    """
    Retrieves the current exchange rate between two specified currencies.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        from_currency (str): The currency to convert from (e.g., "USD", "EUR").
        to_currency (str): The currency to convert to (e.g., "GBP", "JPY").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of the exchange rate, or an error/fallback message.
    """
    logger.info(f"Tool: get_currency_exchange_rate called for {from_currency} to {to_currency} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'finance_tool_access', False):
        return "Error: Access to finance tools is not enabled for your current tier."
    
    params = {"from_currency": from_currency.upper(), "to_currency": to_currency.upper()}
    api_data = asyncio.run(_make_dynamic_api_request("finance", "get_currency_exchange_rate", params, user_token))

    if api_data:
        try:
            rate = api_data.get("exchange_rate")
            last_refreshed = api_data.get("last_refreshed")
            if rate is not None:
                response_str = (
                    f"Current Exchange Rate:\n"
                    f"  1 {from_currency.upper()} = {rate} {to_currency.upper()}\n"
                )
                if last_refreshed:
                    try:
                        last_refreshed_dt = datetime.fromisoformat(last_refreshed)
                        response_str += f"  Last Refreshed: {last_refreshed_dt.strftime('%Y-%m-%d %H:%M')}"
                    except ValueError:
                        response_str += f"  Last Refreshed: {last_refreshed}"
                return response_str
            else:
                logger.warning(f"Live API data for exchange rate {from_currency}/{to_currency} is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live exchange rate for {from_currency} to {to_currency}. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live currency exchange rate data for {from_currency}/{to_currency}: {e}")
            return f"Error parsing live data for {from_currency} to {to_currency}. Falling back to mock data."

    # Fallback to mock data
    mock_key = f"{from_currency.upper()}_{to_currency.upper()}"
    mock_data = _mock_finance_data.get("currency_exchange_rates", {}).get(mock_key)
    if mock_data:
        response_str = (
            f"Current Exchange Rate (Mock Data Fallback):\n"
            f"  1 {mock_data['from_currency']} = {mock_data['exchange_rate']} {mock_data['to_currency']}\n"
        )
        if mock_data.get('last_refreshed'):
            try:
                last_refreshed_dt = datetime.fromisoformat(mock_data['last_refreshed'])
                response_str += f"  Last Refreshed: {last_refreshed_dt.strftime('%Y-%m-%d %H:%M')}"
            except ValueError:
                response_str += f"  Last Refreshed: {mock_data['last_refreshed']}"
        return response_str
    else:
        return f"Exchange rate information not found for {from_currency} to {to_currency}. (API/Mock Fallback Failed)"


@tool
def get_economic_indicator(indicator_name: str, country: Optional[str] = None, user_token: str = "default") -> str:
    """
    Retrieves the latest value for a specified economic indicator, optionally filtered by country.
    Examples of indicators: "Unemployment Rate", "GDP Growth".
    Falls back to mock data if API key is missing or API call fails.

    Args:
        indicator_name (str): The name of the economic indicator (e.g., "Unemployment Rate", "GDP Growth").
        country (str, optional): The country for which to retrieve the indicator (e.g., "US", "Germany").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of economic indicator information, or an error/fallback message.
    """
    logger.info(f"Tool: get_economic_indicator called for indicator: '{indicator_name}', country: '{country}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'finance_tool_access', False):
        return "Error: Access to finance tools is not enabled for your current tier."
    
    params = {"indicator_name": indicator_name}
    if country: params["country"] = country

    api_data = asyncio.run(_make_dynamic_api_request("finance", "get_economic_indicator", params, user_token))

    if api_data:
        try:
            indicator = api_data.get("indicator")
            date = api_data.get("date")
            value = api_data.get("value")
            unit = api_data.get("unit")
            source = api_data.get("source")

            if indicator and value is not None:
                response_str = (
                    f"Economic Indicator: {indicator}\n"
                    f"  Value: {value}{unit or ''}\n"
                    f"  Date: {date or 'N/A'}\n"
                    f"  Source: {source or 'N/A'}"
                )
                return response_str
            else:
                logger.warning(f"Live API data for economic indicator '{indicator_name}' is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live economic indicator for '{indicator_name}'. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live economic indicator data for '{indicator_name}': {e}")
            return f"Error parsing live data for '{indicator_name}'. Falling back to mock data."

    # Fallback to mock data
    mock_key_prefix = indicator_name.lower().replace(" ", "_")
    if country:
        mock_key_prefix += f"_{country.lower()}"
    
    mock_data = None
    for key, entry in _mock_finance_data.get("economic_indicators", {}).items():
        if mock_key_prefix in key:
            mock_data = entry
            break

    if mock_data:
        response_str = (
            f"Economic Indicator: {mock_data['indicator']} (Mock Data Fallback)\n"
            f"  Value: {mock_data['value']}{mock_data.get('unit', '')}\n"
            f"  Date: {mock_data['date']}\n"
            f"  Source: {mock_data['source']}"
        )
        return response_str
    else:
        return f"Economic indicator information not found for '{indicator_name}'. (API/Mock Fallback Failed)"


@tool
def finance_search_web(query: str, user_token: str = "default", max_chars: int = 2000) -> str:
    """
    Searches the web for general finance-related information using a smart search fallback mechanism.
    This tool is suitable for queries that cannot be answered by specific structured finance APIs.
    It leverages a web scraping tool to get information from the internet.

    Args:
        query (str): The search query (e.g., "latest financial news", "explanation of quantitative easing").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
        max_chars (int, optional): The maximum number of characters to return from the scraped content.
                                    Defaults to 2000.

    Returns:
        str: A summary of search results or an error message.
    """
    logger.info(f"Tool: finance_search_web called for query: '{query}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'finance_tool_access', False):
        return "Error: Access to finance tools is not enabled for your current tier."
    
    # Log the generic web search tool usage
    asyncio.create_task(analytics_tracker.log_tool_usage(
        tool_name="finance_search_web",
        tool_params={"query": query, "max_chars": max_chars},
        user_token=user_token,
        success=True, # Assume success for logging purposes here, actual success handled by scrape_web
        error_message=None
    ))

    # Use the generic scrape_web tool for web search
    # scrape_web is already designed to handle its own logging for success/failure
    return asyncio.run(scrape_web(query=query, user_token=user_token, max_chars=max_chars))

@tool
def finance_query_uploaded_docs(query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
    """
    Queries previously uploaded and indexed finance documents for a user using vector similarity search.
    This tool wraps the generic `QueryUploadedDocs` tool, fixing the section to "finance".
    
    Args:
        query (str): The search query to find relevant finance documents (e.g., "my investment portfolio", "company annual report").
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
    Summarizes a document related to finance (e.g., financial reports, market analysis) located at the given file path.
    The file path should be accessible by the system (e.g., in the 'uploads' directory).
    This tool wraps the generic `summarize_document` tool.
    
    Args:
        file_path_str (str): The full path to the document file to be summarized.
                              Example: "uploads/default/finance/annual_report.pdf"
    
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


# --- Test Functions (for direct execution of this file) ---
async def run_finance_tests():
    """Runs a series of tests for the finance tools."""
    print("--- Running Finance Tool Tests ---")
    test_user_pro = "test_user_pro_finance_123" # A dummy user token for testing

    # Mock config_manager and analytics_tracker for testing context
    class MockConfigManager:
        def get(self, key, default=None):
            if key == "analytics.log_tool_usage":
                return True # Enable logging for tests
            if key == "web_scraping.timeout_seconds":
                return 5
            return default
        
        def get_secret(self, key):
            # Simulate fetching a secret (e.g., from secrets.toml)
            if "alphavantage_api_key" in key:
                return "YOUR_ALPHAVANTAGE_API_KEY_HERE" # Use a dummy key for testing
            if "exchangerate_api_key" in key:
                return "YOUR_EXCHANGERATE_API_KEY_HERE" # Use a dummy key for testing
            return None

        def get_api_provider_config(self, domain, provider_name):
            # Simplified mock for Alpha Vantage and ExchangeRate-API
            if domain == "finance" and provider_name == "alphavantage":
                return {
                    "base_url": "https://www.alphavantage.co/query",
                    "api_key_name": "alphavantage_api_key",
                    "api_key_param_name": "apikey",
                    "functions": {
                        "get_stock_price": {
                            "function_param": "TIME_SERIES_DAILY",
                            "required_params": ["symbol"],
                            "optional_params": [],
                            "response_path": ["Time Series (Daily)"],
                            "data_map": {
                                "open": "1. open",
                                "high": "2. high",
                                "low": "3. low",
                                "close": "4. close",
                                "volume": "5. volume"
                            }
                        },
                        "get_company_overview": {
                            "function_param": "OVERVIEW",
                            "required_params": ["symbol"],
                            "optional_params": [],
                            "response_path": [], # Root of the response is the data
                            "data_map": {
                                "name": "Name",
                                "exchange": "Exchange",
                                "sector": "Sector",
                                "industry": "Industry",
                                "description": "Description",
                                "market_cap": "MarketCapitalization",
                                "pe_ratio": "PERatio"
                            }
                        }
                    }
                }
            elif domain == "finance" and provider_name == "exchangerate_api":
                return {
                    "base_url": "https://open.er-api.com/v6/latest/",
                    "api_key_name": "exchangerate_api_key", # This is actually part of the path, but we'll include it here
                    "path_params": ["from_currency"], # The 'from_currency' is part of the URL path
                    "functions": {
                        "get_currency_exchange_rate": {
                            "endpoint": "/{from_currency}", # Placeholder for from_currency
                            "required_params": ["to_currency"],
                            "response_path": ["rates"],
                            "data_map": {
                                "exchange_rate": "to_currency_placeholder", # This will be dynamically replaced
                                "last_refreshed": "time_last_update_utc" # Example path
                            }
                        }
                    }
                }
            # Add other provider configs as needed for testing
            return None

    class MockAnalyticsTracker:
        def __init__(self):
            self.logged_events = []
            self.db = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: self.logged_events.append(data)})()})()

        async def log_tool_usage(self, tool_name, tool_params, user_token, success, error_message=None):
            event_data = {
                "event_type": "tool_usage",
                "tool_name": tool_name,
                "tool_params": tool_params,
                "user_token": user_token,
                "timestamp": datetime.now().isoformat(),
                "success": success
            }
            if error_message:
                event_data["error_message"] = error_message
            self.logged_events.append(event_data)
            print(f"MockAnalyticsTracker: Logged tool usage for {tool_name}, success: {success}")

        async def log_event(self, event_type, event_details, user_id, success, error_message=None):
            event_data = {
                "event_type": event_type,
                "event_details": event_details,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "success": success
            }
            if error_message:
                event_data["error_message"] = error_message
            self.logged_events.append(event_data)
            print(f"MockAnalyticsTracker: Logged event {event_type}, success: {success}")


    # Temporarily replace global instances with mocks for testing
    original_config_manager = config_manager
    original_analytics_tracker = analytics_tracker
    original_requests_get = requests.get

    # Mock requests.get for external API calls
    def mock_requests_get(*args, **kwargs):
        url = args[0]
        params = kwargs.get('params', {})
        headers = kwargs.get('headers', {})

        # Mock for Alpha Vantage stock price
        if "alphavantage.co/query" in url and params.get("function") == "TIME_SERIES_DAILY":
            symbol = params.get("symbol", "").upper()
            if symbol == "GOOG":
                return MockResponse({
                    "Meta Data": {"2. Symbol": "GOOG"},
                    "Time Series (Daily)": {
                        "2025-07-05": {"1. open": "160.00", "2. high": "162.00", "3. low": "159.50", "4. close": "161.50", "5. volume": "1200000"},
                        "2025-07-04": {"1. open": "158.00", "2. high": "160.00", "3. low": "157.50", "4. close": "159.00", "5. volume": "1100000"}
                    }
                }, 200)
            elif symbol == "AAPL":
                return MockResponse({
                    "Meta Data": {"2. Symbol": "AAPL"},
                    "Time Series (Daily)": {
                        "2025-07-05": {"1. open": "180.00", "2. high": "181.00", "3. low": "179.50", "4. close": "180.50", "5. volume": "2000000"}
                    }
                }, 200)
            else:
                return MockResponse({"Error Message": "Invalid API call. Please retry or visit the documentation."}, 200)
        
        # Mock for Alpha Vantage company overview
        if "alphavantage.co/query" in url and params.get("function") == "OVERVIEW":
            symbol = params.get("symbol", "").upper()
            if symbol == "GOOG":
                return MockResponse({
                    "Symbol": "GOOG", "AssetType": "Common Stock", "Name": "Alphabet Inc.",
                    "Description": "Alphabet Inc. is an American multinational technology conglomerate holding company...",
                    "Exchange": "NASDAQ", "Sector": "Technology", "Industry": "Internet Content & Information",
                    "MarketCapitalization": "2000000000000", "PERatio": "28.5"
                }, 200)
            else:
                return MockResponse({"Error Message": "Invalid API call. Please retry or visit the documentation."}, 200)

        # Mock for ExchangeRate-API
        if "open.er-api.com/v6/latest/" in url:
            from_currency = url.split('/')[-1]
            if from_currency == "USD":
                return MockResponse({
                    "result": "success",
                    "documentation": "https://www.exchangerate-api.com/docs/v6",
                    "terms_of_use": "https://www.exchangerate-api.com/terms",
                    "time_last_update_unix": 1678886400,
                    "time_last_update_utc": "Fri, 17 Mar 2023 00:00:00 +0000",
                    "time_next_update_unix": 1678972800,
                    "time_next_update_utc": "Sat, 18 Mar 2023 00:00:00 +0000",
                    "base_code": "USD",
                    "rates": {
                        "EUR": 0.92, "GBP": 0.82, "JPY": 133.00, "USD": 1.0
                    }
                }, 200)
            elif from_currency == "GBP":
                return MockResponse({
                    "result": "success",
                    "base_code": "GBP",
                    "rates": {
                        "USD": 1.27, "EUR": 1.15, "GBP": 1.0
                    }
                }, 200)
            else:
                return MockResponse({"result": "error", "error-type": "unsupported-code"}, 200)
        
        # Default to a generic error for unmocked requests
        return MockResponse({}, 404)

    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code
            self.ok = status_code == 200

        def json(self):
            return self._json_data

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.exceptions.HTTPError(f"HTTP Error: {self.status_code}", response=self)

    config_manager = MockConfigManager()
    analytics_tracker = MockAnalyticsTracker()
    requests.get = mock_requests_get

    try:
        # Test 1: get_stock_price - latest
        print("\n--- Test 1: get_stock_price (Latest) ---")
        result = await get_stock_price("GOOG", user_token=test_user_pro)
        print(f"Result: {result}")
        assert "Stock Price for GOOG" in result and "Open: 160.00" in result
        assert any(e['tool_name'] == 'finance_get_stock_price' and e['success'] for e in analytics_tracker.logged_events)
        print("Test 1 Passed.")

        # Test 2: get_stock_price - specific date
        print("\n--- Test 2: get_stock_price (Specific Date) ---")
        analytics_tracker.logged_events.clear() # Clear logs for next test
        result = await get_stock_price("GOOG", date="2025-07-04", user_token=test_user_pro)
        print(f"Result: {result}")
        assert "Stock Price for GOOG on 2025-07-04" in result and "Open: 158.00" in result
        assert any(e['tool_name'] == 'finance_get_stock_price' and e['success'] for e in analytics_tracker.logged_events)
        print("Test 2 Passed.")

        # Test 3: get_company_overview
        print("\n--- Test 3: get_company_overview ---")
        analytics_tracker.logged_events.clear()
        result = await get_company_overview("AAPL", user_token=test_user_pro)
        print(f"Result: {result}")
        assert "Company Overview for Apple Inc. (AAPL)" in result and "Consumer Electronics" in result
        assert any(e['tool_name'] == 'finance_get_company_overview' and e['success'] for e in analytics_tracker.logged_events)
        print("Test 3 Passed.")

        # Test 4: get_currency_exchange_rate
        print("\n--- Test 4: get_currency_exchange_rate ---")
        analytics_tracker.logged_events.clear()
        result = await get_currency_exchange_rate("USD", "EUR", user_token=test_user_pro)
        print(f"Result: {result}")
        assert "1 USD = 0.92 EUR" in result
        assert any(e['tool_name'] == 'finance_get_currency_exchange_rate' and e['success'] for e in analytics_tracker.logged_events)
        print("Test 4 Passed.")

        # Test 5: get_economic_indicator (mocked fallback)
        print("\n--- Test 5: get_economic_indicator (Mocked Fallback) ---")
        analytics_tracker.logged_events.clear()
        # This will hit mock data as we don't have a live API mock for it
        result = await get_economic_indicator("Unemployment Rate", "US", user_token=test_user_pro)
        print(f"Result: {result}")
        assert "Economic Indicator: Unemployment Rate (US) (Mock Data Fallback)" in result and "Value: 3.9%" in result
        # No analytics logged for this as it's a mock fallback.
        print("Test 5 Passed (mock fallback expected).")

        # Test 6: finance_search_web (generic tool)
        print("\n--- Test 6: finance_search_web (Generic Tool) ---")
        analytics_tracker.logged_events.clear()
        result_web_search = await finance_search_web("impact of inflation on economy", user_token=test_user_pro)
        print(f"Web Search Result: {result_web_search[:100]}...")
        assert "Search results for impact of inflation on economy" in result_web_search
        # Analytics for generic tools like scrape_web or summarize_document
        # would need to be integrated within those shared_tools themselves,
        # or wrapped by a higher-level agent logging.
        # For now, we are focusing on _make_dynamic_api_request.
        assert any(e['tool_name'] == 'finance_search_web' and e['success'] for e in analytics_tracker.logged_events)
        print("Test 6 Passed (analytics expected for wrapper tool).")

        # Test 7: finance_query_uploaded_docs (generic tool)
        print("\n--- Test 7: finance_query_uploaded_docs (Generic Tool) ---")
        analytics_tracker.logged_events.clear()
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
        original_QueryUploadedDocs = QueryUploadedDocs
        QueryUploadedDocs = MockQueryUploadedDocs

        result_doc_query = await finance_query_uploaded_docs("my financial reports", user_token=test_user_pro)
        print(f"Document Query Result: {result_doc_query}")
        assert "Mocked document query results for 'my financial reports' in section 'finance'." in result_doc_query
        # Analytics for generic tools like QueryUploadedDocs would be logged by DocumentTools
        # For now, we are focusing on _make_dynamic_api_request and this wrapper.
        # The actual analytics for the underlying query_uploaded_docs_internal will be logged by DocumentTools.
        # Here, we expect analytics for the wrapper `finance_query_uploaded_docs` itself.
        assert any(e['tool_name'] == 'finance_query_uploaded_docs' and e['success'] for e in analytics_tracker.logged_events)
        print("Test 7 Passed (analytics expected for wrapper tool).")
        QueryUploadedDocs = original_QueryUploadedDocs # Restore original

        print("\nAll finance_tool tests with analytics considerations completed.")

    finally:
        # Restore original instances
        config_manager = original_config_manager
        analytics_tracker = original_analytics_tracker
        requests.get = original_requests_get
        print("Restored original config_manager, analytics_tracker, and requests.get.")

# Ensure tests are only run when the script is executed directly
if __name__ == "__main__":
    # Use asyncio.run to execute the async test function
    asyncio.run(run_finance_tests())

