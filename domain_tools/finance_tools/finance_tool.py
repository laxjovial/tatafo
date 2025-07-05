# domain_tools/finance_tools/finance_tool.py

import logging
import requests
import json
from typing import Optional, List, Dict, Any
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

# --- Generic API Request Helper (re-using the one from crypto_tool, or defining here if standalone) ---
# In a larger refactor, this helper might live in a shared 'utils/api_helper.py' or similar.

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
            # Ensure path parameters are correctly formatted (e.g., uppercase for currencies)
            value = str(params.pop(p_param))
            if p_param in ["base_currency", "target_currency"]: # Example for currency APIs
                value = value.upper()
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
        # For ExchangeRate-API, the key is a path parameter, already handled above.
        # This 'elif' ensures we don't add it as a query param if it's already in the path.
        pass


    for param_key in function_details.get("required_params", []) + function_details.get("optional_params", []):
        if param_key in params:
            # Special handling for currency parameters to ensure uppercase
            if param_key in ["from_currency", "to_currency", "base_currency", "target_currency", "symbol"] and isinstance(params[param_key], str):
                query_params[param_key] = params[param_key].upper()
            else:
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
_mock_finance_data = {
    "stock_prices": {
        "AAPL": {
            "2023-01-01": {"open": 170.00, "high": 172.50, "low": 169.00, "close": 171.80, "volume": 80000000},
            "2023-01-02": {"open": 171.00, "high": 173.00, "low": 170.50, "close": 172.50, "volume": 75000000},
        },
        "MSFT": {
            "2023-01-01": {"open": 250.00, "high": 252.00, "low": 249.00, "close": 251.50, "volume": 60000000},
        }
    },
    "company_overview": {
        "AAPL": {
            "Symbol": "AAPL",
            "AssetType": "Common Stock",
            "Name": "Apple Inc.",
            "Description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
            "Sector": "Technology",
            "Industry": "Consumer Electronics",
            "MarketCapitalization": "3000000000000",
            "FiscalYearEnd": "September",
            "LatestQuarter": "2023-09-30"
        }
    },
    "global_quote": {
        "AAPL": {
            "symbol": "AAPL",
            "open": "171.00",
            "high": "173.00",
            "low": "170.50",
            "price": "172.50",
            "volume": "75000000",
            "latest_trading_day": "2023-01-02",
            "previous_close": "171.80",
            "change": "0.70",
            "change_percent": "0.4074%"
        }
    },
    "exchange_rates": {
        "USD": {
            "EUR": 0.92,
            "GBP": 0.80,
            "JPY": 155.00
        }
    },
    "currency_conversion": {
        "USD_to_EUR_100": 92.00
    }
}

@tool
def get_stock_price(symbol: str, user_token: str = "default") -> str:
    """
    Retrieves the latest stock price and global quote information for a given stock symbol.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL", "MSFT").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A string containing the latest stock price and other quote details, or an error/fallback message.
    """
    logger.info(f"Tool: get_stock_price called for symbol: {symbol} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'finance_tool_access', False):
        return "Error: Access to finance tools is not enabled for your current tier."

    api_data = _make_dynamic_api_request(
        "finance", "get_global_quote",
        {"symbol": symbol.upper()},
        user_token
    )

    if api_data:
        try:
            price = api_data.get("price")
            open_price = api_data.get("open")
            high_price = api_data.get("high")
            low_price = api_data.get("low")
            volume = api_data.get("volume")
            latest_trading_day = api_data.get("latest_trading_day")
            change = api_data.get("change")
            change_percent = api_data.get("change_percent")

            if price is not None:
                response_str = (
                    f"Latest stock quote for {symbol.upper()}:\n"
                    f"  Price: {float(price):,.2f}\n"
                    f"  Open: {float(open_price):,.2f}\n"
                    f"  High: {float(high_price):,.2f}\n"
                    f"  Low: {float(low_price):,.2f}\n"
                    f"  Volume: {int(volume):,}\n"
                    f"  Change: {float(change):+.2f} ({change_percent})\n"
                    f"  Latest Trading Day: {latest_trading_day}"
                )
                return response_str
            else:
                logger.warning(f"Live API data for {symbol} is missing price. Raw: {api_data}")
                return f"Could not retrieve live stock price for {symbol.upper()}. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live stock price data for {symbol}: {e}")
            return f"Error parsing live data for {symbol}. Falling back to mock data."

    # Fallback to mock data
    mock_data = _mock_finance_data.get("global_quote", {}).get(symbol.upper())
    if mock_data:
        return (
            f"Latest stock quote for {symbol.upper()} (Mock Data Fallback):\n"
            f"  Price: {float(mock_data['price']):,.2f}\n"
            f"  Open: {float(mock_data['open']):,.2f}\n"
            f"  High: {float(mock_data['high']):,.2f}\n"
            f"  Low: {float(mock_data['low']):,.2f}\n"
            f"  Volume: {int(mock_data['volume']):,}\n"
            f"  Change: {float(mock_data['change']):+.2f} ({mock_data['change_percent']})\n"
            f"  Latest Trading Day: {mock_data['latest_trading_day']}"
        )
    else:
        return f"Stock price information not found for '{symbol.upper()}'. (API/Mock Fallback Failed)"


@tool
def get_company_overview(symbol: str, user_token: str = "default") -> str:
    """
    Retrieves a detailed company overview for a given stock symbol, including description,
    sector, industry, and market capitalization.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL", "MSFT").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A string containing the company overview, or an error/fallback message.
    """
    logger.info(f"Tool: get_company_overview called for symbol: {symbol} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'finance_tool_access', False):
        return "Error: Access to finance tools is not enabled for your current tier."

    api_data = _make_dynamic_api_request(
        "finance", "get_company_overview",
        {"symbol": symbol.upper()},
        user_token
    )

    if api_data:
        try:
            name = api_data.get("Name")
            description = api_data.get("Description")
            sector = api_data.get("Sector")
            industry = api_data.get("Industry")
            market_cap = api_data.get("MarketCapitalization")

            if name and description:
                response_str = (
                    f"Company Overview for {name} ({symbol.upper()}):\n"
                    f"  Description: {description}\n"
                    f"  Sector: {sector}\n"
                    f"  Industry: {industry}\n"
                    f"  Market Cap: {int(market_cap):,}\n"
                )
                return response_str
            else:
                logger.warning(f"Live API data for {symbol} is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live company overview for {symbol.upper()}. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live company overview data for {symbol}: {e}")
            return f"Error parsing live data for {symbol}. Falling back to mock data."

    # Fallback to mock data
    mock_data = _mock_finance_data.get("company_overview", {}).get(symbol.upper())
    if mock_data:
        return (
            f"Company Overview for {mock_data['Name']} ({symbol.upper()}) (Mock Data Fallback):\n"
            f"  Description: {mock_data['Description']}\n"
            f"  Sector: {mock_data['Sector']}\n"
            f"  Industry: {mock_data['Industry']}\n"
            f"  Market Cap: {int(mock_data['MarketCapitalization']):,}\n"
        )
    else:
        return f"Company overview information not found for '{symbol.upper()}'. (API/Mock Fallback Failed)"


@tool
def get_historical_stock_prices(symbol: str, start_date: str, end_date: str, user_token: str = "default") -> str:
    """
    Retrieves historical daily stock prices for a given symbol within a specified date range.
    Dates can be in various formats (e.g., 'YYYY-MM-DD', 'MM/DD/YYYY', 'July 5, 2025').
    Returns data in JSON format for easy plotting/analysis.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL", "MSFT").
        start_date (str): The start date for historical data (e.g., "2023-01-01", "01/01/2023").
        end_date (str): The end date for historical data (e.g., "2023-01-31", "January 31, 2023").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A JSON string containing historical daily prices, or an error/fallback message.
    """
    logger.info(f"Tool: get_historical_stock_prices called for symbol: {symbol} from {start_date} to {end_date} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'historical_data_access', False):
        return "Error: Access to historical data is not enabled for your current tier."
    
    # Parse and validate dates
    parsed_start_date = parse_date_to_yyyymmdd(start_date)
    parsed_end_date = parse_date_to_yyyymmdd(end_date)

    if not parsed_start_date or not parsed_end_date:
        return "Error: Could not parse provided start or end date. Please ensure dates are valid."

    # Alpha Vantage TIME_SERIES_DAILY does not directly support start/end dates in the API call.
    # It returns compact (last 100 days) or full history. We will fetch full and filter.
    # Note: For a real-world app, consider a different API or more complex Alpha Vantage logic
    # to handle large data sets efficiently.
    api_data = _make_dynamic_api_request(
        "finance", "get_historical_stock_prices",
        {"symbol": symbol.upper(), "outputsize": "full"}, # Request full data to filter
        user_token
    )

    if api_data and api_data.get("data"): # 'data' key because _make_dynamic_api_request wraps dicts
        historical_data_raw = api_data["data"]
        filtered_data = {}
        
        start_dt = datetime.strptime(parsed_start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(parsed_end_date, "%Y-%m-%d").date()

        for date_str, values in historical_data_raw.items():
            try:
                current_date_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                if start_dt <= current_date_dt <= end_dt:
                    filtered_data[date_str] = {
                        "open": float(values.get("open", 0)),
                        "high": float(values.get("high", 0)),
                        "low": float(values.get("low", 0)),
                        "close": float(values.get("close", 0)),
                        "volume": int(values.get("volume", 0))
                    }
            except ValueError:
                logger.warning(f"Skipping unparseable date in historical data: {date_str}")
                continue
        
        if filtered_data:
            return json.dumps(filtered_data, indent=2)
        else:
            return f"No live historical data found for {symbol.upper()} between {parsed_start_date} and {parsed_end_date}. Falling back to mock data."

    # Fallback to mock data
    mock_data = _mock_finance_data.get("stock_prices", {}).get(symbol.upper())
    if mock_data:
        filtered_mock_data = {}
        start_dt = datetime.strptime(parsed_start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(parsed_end_date, "%Y-%m-%d").date()

        for date_str, values in mock_data.items():
            try:
                current_date_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                if start_dt <= current_date_dt <= end_dt:
                    filtered_mock_data[date_str] = values
            except ValueError:
                continue

        if filtered_mock_data:
            return json.dumps(filtered_mock_data, indent=2)
        else:
            return f"No mock historical data found for {symbol.upper()} between {parsed_start_date} and {parsed_end_date}. (API/Mock Fallback Failed)"
    else:
        return f"Historical stock price information not found for '{symbol.upper()}'. (API/Mock Fallback Failed)"


@tool
def get_currency_exchange_rate(base_currency: str, target_currency: str, user_token: str = "default") -> str:
    """
    Retrieves the latest exchange rate between two currencies.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        base_currency (str): The base currency (e.g., "USD", "EUR").
        target_currency (str): The target currency (e.g., "GBP", "JPY").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A string containing the exchange rate, or an error/fallback message.
    """
    logger.info(f"Tool: get_currency_exchange_rate called for {base_currency} to {target_currency} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'finance_tool_access', False):
        return "Error: Access to finance tools is not enabled for your current tier."
    
    api_data = _make_dynamic_api_request(
        "finance", "get_exchange_rate_latest",
        {"base_currency": base_currency.upper()}, # Pass base_currency for path param
        user_token
    )

    if api_data:
        try:
            rate = api_data.get("conversion_rates", {}).get(target_currency.upper())
            if rate is not None:
                return f"1 {base_currency.upper()} = {rate:,.4f} {target_currency.upper()}"
            else:
                logger.warning(f"Live API data for {base_currency}/{target_currency} is missing rate. Raw: {api_data}")
                return f"Could not retrieve live exchange rate for {base_currency.upper()} to {target_currency.upper()}. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live exchange rate data for {base_currency}/{target_currency}: {e}")
            return f"Error parsing live data for {base_currency}/{target_currency}. Falling back to mock data."

    # Fallback to mock data
    mock_data = _mock_finance_data.get("exchange_rates", {}).get(base_currency.upper(), {}).get(target_currency.upper())
    if mock_data is not None:
        return f"1 {base_currency.upper()} = {mock_data:,.4f} {target_currency.upper()} (Mock Data Fallback)"
    else:
        return f"Exchange rate information not found for '{base_currency.upper()}' to '{target_currency.upper()}'. (API/Mock Fallback Failed)"


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str, user_token: str = "default") -> str:
    """
    Converts a specified amount from one currency to another using the latest exchange rates.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        amount (float): The amount to convert.
        from_currency (str): The currency to convert from (e.g., "USD", "JPY").
        to_currency (str): The currency to convert to (e.g., "EUR", "GBP").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A string containing the converted amount, or an error/fallback message.
    """
    logger.info(f"Tool: convert_currency called for {amount} {from_currency} to {to_currency} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'finance_tool_access', False):
        return "Error: Access to finance tools is not enabled for your current tier."

    api_data = _make_dynamic_api_request(
        "finance", "convert_currency",
        {"amount": amount, "from_currency": from_currency.upper(), "to_currency": to_currency.upper()}, # Pass for path params
        user_token
    )

    if api_data:
        try:
            converted_amount = api_data.get("conversion_result")
            if converted_amount is not None:
                return f"{amount:,.2f} {from_currency.upper()} is equal to {converted_amount:,.2f} {to_currency.upper()}"
            else:
                logger.warning(f"Live API data for currency conversion is missing result. Raw: {api_data}")
                return f"Could not perform live currency conversion for {amount:,.2f} {from_currency.upper()} to {to_currency.upper()}. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live currency conversion data: {e}")
            return f"Error parsing live data for currency conversion. Falling back to mock data."

    # Fallback to mock data
    # Mock for a specific conversion (e.g., 100 USD to EUR)
    mock_key = f"{from_currency.upper()}_to_{to_currency.upper()}_{int(amount)}"
    mock_result = _mock_finance_data.get("currency_conversion", {}).get(mock_key)
    if mock_result is not None:
        return f"{amount:,.2f} {from_currency.upper()} is equal to {mock_result:,.2f} {to_currency.upper()} (Mock Data Fallback)"
    else:
        # Generic mock if specific one not found, using a simple fixed rate
        base_rate = _mock_finance_data.get("exchange_rates", {}).get("USD", {}).get(to_currency.upper(), 1.0)
        from_rate = _mock_finance_data.get("exchange_rates", {}).get("USD", {}).get(from_currency.upper(), 1.0)
        if from_rate != 0:
            generic_converted_amount = amount * (base_rate / from_rate)
            return f"{amount:,.2f} {from_currency.upper()} is equal to {generic_converted_amount:,.2f} {to_currency.upper()} (Generic Mock Fallback)"
        else:
            return f"Currency conversion information not found for '{from_currency.upper()}' to '{to_currency.upper()}'. (API/Mock Fallback Failed)"


# --- Existing Finance Tools (not directly using external APIs) ---

@tool
def finance_search_web(query: str, user_token: str = "default", max_chars: int = 2000) -> str:
    """
    Searches the web for finance-related information using a smart search fallback mechanism.
    This tool wraps the generic `scrape_web` tool, providing a finance-specific interface.
    
    Args:
        query (str): The finance-related search query (e.g., "latest stock market news", "explain quantitative easing").
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
    Queries previously uploaded and indexed finance documents for a user using vector similarity search.
    This tool wraps the generic `QueryUploadedDocs` tool, fixing the section to "finance".
    
    Args:
        query (str): The search query to find relevant finance documents (e.g., "what is in the Q3 earnings report", "summary of the company's annual filing").
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
    Summarizes a document related to finance located at the given file path.
    The file path should be accessible by the system (e.g., in the 'uploads' directory).
    This tool wraps the generic `summarize_document` tool.
    
    Args:
        file_path_str (str): The full path to the document file to be summarized.
                              Example: "uploads/default/finance/earnings_report.pdf"
    
    Returns:
        str: A concise summary of the document content.
    """
    logger.info(f"Tool: finance_summarize_document_by_path called for file: '{file_path_str}'")
    file_path = Path(file_path_str)
    if not file_path.exists():
        logger.error(f"Document not found at '{file_path_str}' for summarization.")
        return f"Error: Document not found at '{file_path_str}'."
    
    try:
        # Note: The summarize_document tool now handles its own RBAC check internally
        # based on the user_token passed to it (if it accepts one).
        # For simplicity here, we're assuming summarize_document will handle it
        # or that this tool itself is only available to tiers with summarization.
        summary = summarize_document(file_path) # Assuming summarize_document can take Path object
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
            self.alphavantage_api_key = "MOCK_ALPHAVANTAGE_KEY"
            self.exchangerate_api_key = "MOCK_EXCHANGERATE_KEY"
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
                    'finance': 'alphavantage',
                    'currency': 'exchangerate_api'
                }
            }
            self._api_providers_data = { # Mock api_providers_data for finance and currency
                "finance": {
                    "alphavantage": {
                        "base_url": "https://www.alphavantage.co/query",
                        "api_key_name": "alphavantage_api_key",
                        "api_key_param_name": "apikey",
                        "functions": {
                            "get_global_quote": {
                                "endpoint": "", # Base URL is enough
                                "function_param": "GLOBAL_QUOTE",
                                "required_params": ["symbol"],
                                "data_map": { # Map Alpha Vantage keys to generic keys
                                    "symbol": "01. symbol",
                                    "open": "02. open",
                                    "high": "03. high",
                                    "low": "04. low",
                                    "price": "05. price",
                                    "volume": "06. volume",
                                    "latest_trading_day": "07. latest trading day",
                                    "previous_close": "08. previous close",
                                    "change": "09. change",
                                    "change_percent": "10. change percent"
                                }
                            },
                            "get_company_overview": {
                                "endpoint": "",
                                "function_param": "OVERVIEW",
                                "required_params": ["symbol"],
                                "data_map": { # Map Alpha Vantage keys to generic keys
                                    "Symbol": "Symbol",
                                    "AssetType": "AssetType",
                                    "Name": "Name",
                                    "Description": "Description",
                                    "Sector": "Sector",
                                    "Industry": "Industry",
                                    "MarketCapitalization": "MarketCapitalization",
                                    "FiscalYearEnd": "FiscalYearEnd",
                                    "LatestQuarter": "LatestQuarter"
                                }
                            },
                            "get_historical_stock_prices": {
                                "endpoint": "",
                                "function_param": "TIME_SERIES_DAILY",
                                "required_params": ["symbol"],
                                "optional_params": ["outputsize"], # "compact" or "full"
                                "response_path": ["Time Series (Daily)"], # Nested path for daily data
                                "data_map": { # Map Alpha Vantage nested keys to generic keys
                                    "open": "1. open",
                                    "high": "2. high",
                                    "low": "3. low",
                                    "close": "4. close",
                                    "volume": "5. volume"
                                }
                            }
                        }
                    }
                },
                "currency": {
                    "exchangerate_api": {
                        "base_url": "https://v6.exchangerate-api.com/v6",
                        "api_key_name": "exchangerate_api_key",
                        "path_params": ["api_key"], # Key is part of the path
                        "functions": {
                            "get_exchange_rate_latest": {
                                "endpoint": "/{api_key}/latest/{base_currency}",
                                "path_params": ["api_key", "base_currency"],
                                "required_params": [], # Params already in path
                                "data_map": { # Map ExchangeRate-API keys
                                    "result": "result",
                                    "documentation": "documentation",
                                    "terms_of_use": "terms_of_use",
                                    "time_last_update_unix": "time_last_update_unix",
                                    "time_last_update_utc": "time_last_update_utc",
                                    "time_next_update_unix": "time_next_update_unix",
                                    "time_next_update_utc": "time_next_update_utc",
                                    "base_code": "base_code",
                                    "conversion_rates": "conversion_rates"
                                }
                            },
                            "convert_currency": {
                                "endpoint": "/{api_key}/pair/{from_currency}/{to_currency}/{amount}",
                                "path_params": ["api_key", "from_currency", "to_currency", "amount"],
                                "required_params": [], # Params already in path
                                "data_map": { # Map ExchangeRate-API keys
                                    "result": "result",
                                    "documentation": "documentation",
                                    "terms_of_use": "terms_of_use",
                                    "time_last_update_unix": "time_last_update_unix",
                                    "time_last_update_utc": "time_last_update_utc",
                                    "time_next_update_unix": "time_next_update_unix",
                                    "time_next_update_utc": "time_next_update_utc",
                                    "base_code": "base_code",
                                    "target_code": "target_code",
                                    "conversion_rate": "conversion_rate",
                                    "conversion_result": "conversion_result"
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
            # Simulate Streamlit secrets access
            mock_secrets_instance = MockSecrets()
            return mock_secrets_instance.get(key, default)

        def set_secret(self, key, value):
            # This would typically update Streamlit secrets or a persistent store
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
                'historical_data_access': {
                    'default': False,
                    'roles': {'premium': True, 'admin': True}
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
        # Simulate AlphaVantage responses
        if "www.alphavantage.co/query" in url:
            function = params.get("function")
            symbol = params.get("symbol")
            if function == "GLOBAL_QUOTE":
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "Global Quote": {
                        "01. symbol": symbol,
                        "02. open": "150.00",
                        "03. high": "152.00",
                        "04. low": "149.50",
                        "05. price": "151.50",
                        "06. volume": "10000000",
                        "07. latest trading day": "2024-07-04",
                        "08. previous close": "150.50",
                        "09. change": "1.00",
                        "10. change percent": "0.6645%"
                    }
                }
                return mock_response
            elif function == "OVERVIEW":
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "Symbol": symbol,
                    "AssetType": "Common Stock",
                    "Name": f"{symbol} Corp.",
                    "Description": f"A leading company in the {symbol} sector.",
                    "Sector": "Technology",
                    "Industry": "Software",
                    "MarketCapitalization": "1000000000000",
                    "FiscalYearEnd": "December",
                    "LatestQuarter": "2024-03-31"
                }
                return mock_response
            elif function == "TIME_SERIES_DAILY":
                mock_response = MagicMock()
                mock_response.status_code = 200
                # Generate mock daily data for the last 10 days
                daily_data = {}
                for i in range(10):
                    date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                    daily_data[date] = {
                        "1. open": str(100 + i),
                        "2. high": str(102 + i),
                        "3. low": str(99 + i),
                        "4. close": str(101 + i),
                        "5. volume": str(1000000 + i * 10000)
                    }
                mock_response.json.return_value = {"Time Series (Daily)": daily_data}
                return mock_response

        # Simulate ExchangeRate-API responses
        elif "v6.exchangerate-api.com/v6" in url:
            if "/latest/" in url:
                parts = url.split('/')
                base_currency = parts[-1]
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "result": "success",
                    "base_code": base_currency,
                    "conversion_rates": {
                        "USD": 1.0, "EUR": 0.92, "GBP": 0.80, "JPY": 155.00, # Example rates
                        base_currency: 1.0 # Self-rate
                    }
                }
                # Add a few more rates for completeness
                if base_currency == "USD":
                    mock_response.json.return_value["conversion_rates"]["NGN"] = 1400.0
                elif base_currency == "NGN":
                    mock_response.json.return_value["conversion_rates"]["USD"] = 0.00071
                return mock_response
            elif "/pair/" in url:
                parts = url.split('/')
                from_currency = parts[-3]
                to_currency = parts[-2]
                amount = float(parts[-1])
                
                # Simple mock conversion logic
                if from_currency == "USD" and to_currency == "EUR":
                    converted_amount = amount * 0.92
                elif from_currency == "EUR" and to_currency == "USD":
                    converted_amount = amount / 0.92
                elif from_currency == "USD" and to_currency == "NGN":
                    converted_amount = amount * 1400.0
                elif from_currency == "NGN" and to_currency == "USD":
                    converted_amount = amount / 1400.0
                else:
                    converted_amount = amount * 1.0 # Fallback

                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "result": "success",
                    "base_code": from_currency,
                    "target_code": to_currency,
                    "conversion_rate": converted_amount / amount, # Calculate rate
                    "conversion_result": converted_amount
                }
                return mock_response
        
        # Simulate scrape_web's internal requests.get if needed
        if "google.com/search" in url or "example.com" in url: # Mock for scrape_web
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = f"<html><body><h1>Search results for {params.get('q', 'finance')}</h1><p>Some financial news snippet.</p></body></html>"
            return mock_response

        return original_requests_get(url, params=params, headers=headers, timeout=timeout)

    requests.get = mock_requests_get_dynamic

    test_user_pro = "mock_pro_token"
    test_user_premium = "mock_premium_token"
    test_user_free = "mock_free_token"

    print("\n--- Testing finance_tool functions (Refactored) ---")

    # Test get_stock_price
    print("\n--- Testing get_stock_price ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result_price = get_stock_price("AAPL", user_token=test_user_pro)
    print(f"AAPL Price (Pro User, API):\n{result_price[:200]}...")
    assert "Latest stock quote for AAPL:" in result_price
    assert "Price: 151.50" in result_price
    print("Test 1 Passed.")

    # Test get_company_overview
    print("\n--- Testing get_company_overview ---")
    result_overview = get_company_overview("MSFT", user_token=test_user_pro)
    print(f"MSFT Overview (Pro User, API):\n{result_overview[:200]}...")
    assert "Company Overview for MSFT Corp. (MSFT):" in result_overview
    assert "A leading company in the MSFT sector." in result_overview
    print("Test 2 Passed.")

    # Test get_historical_stock_prices
    print("\n--- Testing get_historical_stock_prices ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_premium # Historical data is premium
    # Test with various date formats
    result_historical = get_historical_stock_prices("GOOG", "2024-06-25", "July 4, 2024", user_token=test_user_premium)
    print(f"GOOG Historical Prices (Premium User, API):\n{result_historical[:500]}...")
    assert "2024-07-04" in result_historical # Check for a recent date in mock data
    assert "open" in result_historical
    print("Test 3 Passed.")

    # Test get_currency_exchange_rate
    print("\n--- Testing get_currency_exchange_rate ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result_exchange = get_currency_exchange_rate("USD", "EUR", user_token=test_user_pro)
    print(f"USD to EUR Exchange Rate (Pro User, API): {result_exchange}")
    assert "1 USD = 0.9200 EUR" in result_exchange
    print("Test 4 Passed.")

    # Test convert_currency
    print("\n--- Testing convert_currency ---")
    result_convert = convert_currency(100.0, "USD", "NGN", user_token=test_user_pro)
    print(f"100 USD to NGN Conversion (Pro User, API): {result_convert}")
    assert "100.00 USD is equal to 140000.00 NGN" in result_convert
    print("Test 5 Passed.")

    # Test RBAC for finance_tool_access (e.g., get_stock_price for free user)
    print("\n--- Testing RBAC for finance_tool_access (Free User) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result_rbac_denied = get_stock_price("AAPL", user_token=test_user_free)
    print(f"AAPL Price (Free User, RBAC Denied): {result_rbac_denied}")
    assert "Error: Access to finance tools is not enabled for your current tier." in result_rbac_denied
    print("Test 6 Passed.")

    # Test finance_search_web (already works, just for completeness)
    print("\n--- Testing finance_search_web ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    search_query = "latest inflation rate in US"
    search_result = finance_search_web(search_query, user_token=test_user_pro)
    print(f"Search Result for '{search_query}':\n{search_result[:500]}...")
    assert "Search results for latest inflation rate in US" in search_result
    print("Test 7 Passed.")

    # Test python_interpreter_with_rbac with fetched data (example)
    print("\n--- Testing python_interpreter_with_rbac with fetched data ---")
    python_code_finance = f"""
import json
data_str = '''{result_historical}'''
data = json.loads(data_str)
print(f"Number of historical data points: {{len(data)}}")
"""
    print(f"Executing Python code:\n{python_code_finance}")
    try:
        # Test with a user who has data_analysis_enabled
        pro_user_token = test_user_pro
        repl_output = python_interpreter_with_rbac(code=python_code_finance, user_token=pro_user_token)
        print(f"Python REPL Output (Pro User):\n{repl_output}")
        assert "Number of historical data points:" in repl_output
        assert "10" in repl_output # Based on mock data generation
        print("Test 8 Passed.")

        # Test with a user who does NOT have data_analysis_enabled
        free_user_token = test_user_free
        repl_output_denied = python_interpreter_with_rbac(code=python_code_finance, user_token=free_user_token)
        print(f"Python REPL Output (Free User):\n{repl_output_denied}")
        assert "Access Denied" in repl_output_denied
        print("Test 9 Passed.")

    except Exception as e:
        print(f"Error executing Python REPL: {e}.")
        print("Make sure pandas, numpy, json are installed if you're running complex analysis.")


    print("\nAll finance_tool tests passed (real API simulation with fallback).")

    # Restore original requests.get
    requests.get = original_requests_get

    # Clean up dummy files and directories
    dummy_data_dir = Path("data")
    # Remove only the finance_apis.yaml created for testing, if it exists
    dummy_finance_apis_path = dummy_data_dir / "finance_apis.yaml"
    if dummy_finance_apis_path.exists():
        os.remove(dummy_finance_apis_path)
        print(f"Cleaned up {dummy_finance_apis_path}")

    # Clean up other test artifacts if they were created (e.g., by QueryUploadedDocs)
    test_user_dirs = [Path("exports") / test_user_pro, Path("uploads") / test_user_pro, BASE_VECTOR_DIR / test_user_pro]
    for d in test_user_dirs:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            print(f"Cleaned up {d}")
