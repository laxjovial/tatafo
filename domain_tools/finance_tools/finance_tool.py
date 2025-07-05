# domain_tools/finance_tools/finance_tool.py

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

# --- Generic API Request Helper for Dynamic Providers ---
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
    # Get the default active API provider for the finance domain from config.yml
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
        # Use the key name from api_key_name, e.g., "alphavantage_api_key" -> "apikey"
        # This is a common pattern where the secret key name is different from the param name
        # We assume the param name is the api_key_name without "_api_key" suffix, or explicitly defined in config
        param_name_in_url = provider_config.get("api_key_param_name", api_key_name.replace("_api_key", ""))
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
_mock_finance_data = {
    "AAPL": {
        "price": "175.00 USD",
        "currency": "USD",
        "change": "+1.50 (+0.86%)",
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    "MSFT": {
        "price": "420.50 USD",
        "currency": "USD",
        "change": "-0.75 (-0.18%)",
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    "TSLA": {
        "price": "180.25 USD",
        "currency": "USD",
        "change": "+5.10 (+2.91%)",
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    "GOOG": { # For lookup_stock_symbol fallback
        "symbol": "GOOG",
        "name": "Alphabet Inc. (Class C)",
        "type": "Equity",
        "region": "United States",
        "currency": "USD"
    },
    "GOOGL": { # For lookup_stock_symbol fallback
        "symbol": "GOOGL",
        "name": "Alphabet Inc. (Class A)",
        "type": "Equity",
        "region": "United States",
        "currency": "USD"
    },
    "news_aapl": [
        {"title": "Apple Announces New iPhone Model", "source": "TechCrunch", "date": "2024-07-01"},
        {"title": "Apple Stock Rises Amidst Strong Sales", "source": "Bloomberg", "date": "2024-06-28"}
    ],
    "news_msft": [
        {"title": "Microsoft Launches New AI Initiative", "source": "Reuters", "date": "2024-07-03"},
        {"title": "Microsoft Cloud Revenue Beats Estimates", "source": "Wall Street Journal", "date": "2024-07-01"}
    ],
    "historical_goog": [
        {"date": "2023-01-01", "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 1000000},
        {"date": "2023-01-02", "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.8, "volume": 1100000},
        {"date": "2023-01-03", "open": 101.8, "high": 103.0, "low": 101.0, "close": 102.5, "volume": 1200000},
        {"date": "2023-01-04", "open": 102.5, "high": 104.0, "low": 102.0, "close": 103.8, "volume": 1300000},
        {"date": "2023-01-05", "open": 103.8, "high": 105.0, "low": 103.0, "close": 104.5, "volume": 1400000}
    ]
}

@tool
def get_stock_price(symbol: str, user_token: str = "default") -> str:
    """
    Retrieves the current stock price for a given stock symbol using the configured API.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL", "MSFT").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A string containing the current stock price and related information, or an error/fallback message.
    """
    logger.info(f"Tool: get_stock_price called for symbol: {symbol} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'finance_tool_access', False):
        return "Error: Access to financial tools is not enabled for your current tier."

    api_data = _make_dynamic_api_request("finance", "get_stock_price", {"symbol": symbol}, user_token)

    if api_data:
        try:
            price = float(api_data.get("price", 0))
            open_price = float(api_data.get("open", 0))
            high_price = float(api_data.get("high", 0))
            low_price = float(api_data.get("low", 0))
            volume = int(api_data.get("volume", 0))
            last_trading_day = api_data.get("last_trading_day", "N/A")
            change = float(api_data.get("change", 0))
            change_percent = api_data.get("change_percent", "0.00%")

            return (
                f"Current stock price for {symbol.upper()}:\n"
                f"  Price: ${price:,.2f}\n"
                f"  Open: ${open_price:,.2f}\n"
                f"  High: ${high_price:,.2f}\n"
                f"  Low: ${low_price:,.2f}\n"
                f"  Change: ${change:,.2f} ({change_percent})\n"
                f"  Volume: {volume:,}\n"
                f"  Last Trading Day: {last_trading_day}"
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live stock price data for {symbol}: {e}")
            return f"Error parsing live data for {symbol}. Falling back to mock data."

    # Fallback to mock data if API call failed or data is incomplete
    mock_data = _mock_finance_data.get(symbol.upper())
    if mock_data:
        return (
            f"Current stock price for {symbol.upper()} (Mock Data Fallback):\n"
            f"  Price: {mock_data['price']}\n"
            f"  Change: {mock_data['change']}\n"
            f"  Last Updated (Mock): {mock_data['last_updated']}"
        )
    else:
        return f"Stock price information not found for '{symbol}'. (API/Mock Fallback Failed)"

@tool
def get_company_news(symbol: str, from_date: str, to_date: str, user_token: str = "default") -> str:
    """
    Retrieves recent company news for a given stock symbol using the configured API.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL", "MSFT").
        from_date (str): Start date for news in YYYY-MM-DD format.
        to_date (str): End date for news in YYYY-MM-DD format.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A string containing recent news articles, or an error/fallback message.
    """
    logger.info(f"Tool: get_company_news called for symbol: {symbol} from {from_date} to {to_date} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'finance_tool_access', False):
        return "Error: Access to financial tools is not enabled for your current tier."

    # Alpha Vantage News & Sentiment API uses 'tickers' and 'time_from'/'time_to' in specific formats
    # 'time_from' and 'time_to' need to be in YYYYMMDDTHHMM format
    api_time_from = f"{from_date}T0000"
    api_time_to = f"{to_date}T2359"

    api_data = _make_dynamic_api_request(
        "finance", "get_company_news",
        {"tickers": symbol, "time_from": api_time_from, "time_to": api_time_to}, # Pass raw params
        user_token
    )

    if api_data and api_data.get("data"):
        news_articles = api_data["data"]
        if news_articles:
            formatted_news = [f"Recent news for {symbol.upper()} from {from_date} to {to_date}:"]
            for i, article in enumerate(news_articles[:3]): # Limit to 3 articles for brevity
                formatted_news.append(f"  {i+1}. {article.get('title', 'No Title')}")
                formatted_news.append(f"     Source: {article.get('source_name', 'N/A')}") # Use mapped key
                formatted_news.append(f"     URL: {article.get('url', 'N/A')}")
                # Alpha Vantage's time_published is ISO 8601, can format if needed
            return "\n".join(formatted_news)
        else:
            return f"No live news found for {symbol.upper()} from {from_date} to {to_date}. Falling back to mock data."

    # Fallback to mock data
    articles = _mock_finance_data.get(f"news_{symbol.lower()}", [])
    if articles:
        formatted_news = [f"Recent news for {symbol.upper()} from {from_date} to {to_date} (Mock Data Fallback):"]
        for i, article in enumerate(articles):
            if from_date <= article["date"] <= to_date: # Filter mock data by date
                formatted_news.append(f"  {i+1}. {article['title']} (Source: {article['source']}, Date: {article['date']})")
        if len(formatted_news) > 1:
            return "\n".join(formatted_news)
        else:
            return f"No mock news found for {symbol.upper()} in the specified date range."
    else:
        return f"Company news not found for '{symbol}'. (API/Mock Fallback Failed)"

@tool
def get_historical_stock_prices(symbol: str, start_date: str, end_date: str, user_token: str = "default") -> str:
    """
    Retrieves historical daily stock prices for a given symbol and date range using the configured API.
    Returns data in JSON format for easy plotting/analysis.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL", "MSFT").
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A JSON string containing historical daily prices, or an error/fallback message.
    """
    logger.info(f"Tool: get_historical_stock_prices called for symbol: {symbol} from {start_date} to {end_date} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'historical_data_access', False):
        return "Error: Access to historical data is not enabled for your current tier."

    api_data = _make_dynamic_api_request("finance", "get_historical_stock_prices", {"symbol": symbol}, user_token)

    if api_data and api_data.get("data"):
        time_series_raw = api_data["data"] # This will be the mapped data from API
        historical_data = []
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

            for date_str, values in time_series_raw.items(): # Alpha Vantage Time Series (Daily) is dict of dicts
                current_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                if start_dt <= current_dt <= end_dt:
                    historical_data.append({
                        "date": date_str,
                        "open": float(values.get("open")),
                        "high": float(values.get("high")),
                        "low": float(values.get("low")),
                        "close": float(values.get("close")),
                        "volume": int(values.get("volume"))
                    })
            
            historical_data.sort(key=lambda x: x['date']) # Sort by date
            if historical_data:
                return json.dumps(historical_data, indent=2)
            else:
                return f"No live historical data found for {symbol.upper()} in the range {start_date} to {end_date}. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live historical data for {symbol}: {e}")
            return f"Error parsing live historical data for {symbol}. Falling back to mock data."

    # Fallback to mock data
    filtered_mock_data = []
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        for entry in _mock_finance_data.get(f"historical_{symbol.lower()}", []):
            entry_date = datetime.strptime(entry["date"], "%Y-%m-%d").date()
            if start_dt <= entry_date <= end_dt:
                filtered_mock_data.append(entry)
        
        if filtered_mock_data:
            return json.dumps(filtered_mock_data, indent=2)
        else:
            return f"No mock historical data found for {symbol.upper()} in the range {start_date} to {end_date}. (API/Mock Fallback Failed)"
    except ValueError:
        return f"Invalid date format for historical data. Please use YYYY-MM-DD. (API/Mock Fallback Failed)"


@tool
def lookup_stock_symbol(company_name: str, user_token: str = "default") -> str:
    """
    Looks up the stock ticker symbol for a given company name using the configured API.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        company_name (str): The name of the company (e.g., "Apple Inc.", "Microsoft").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: The stock ticker symbol (e.g., "AAPL"), or an error/fallback message.
    """
    logger.info(f"Tool: lookup_stock_symbol called for company: {company_name} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'finance_tool_access', False):
        return "Error: Access to financial tools is not enabled for your current tier."

    api_data = _make_dynamic_api_request("finance", "lookup_stock_symbol", {"keywords": company_name}, user_token)

    if api_data and api_data.get("data"): # 'data' key because _make_dynamic_api_request wraps lists
        best_matches = api_data["data"]
        if best_matches:
            best_match = best_matches[0]
            symbol = best_match.get("symbol")
            name = best_match.get("name")
            if symbol and name:
                return f"Found symbol for '{name}': {symbol}"
            else:
                return f"Could not parse symbol from live data for '{company_name}'. Falling back to mock data."
        else:
            return f"No live matches found for '{company_name}'. Falling back to mock data."

    # Fallback to mock data
    norm_company_name = company_name.lower()
    for symbol_key, details in _mock_finance_data.items():
        if "name" in details and norm_company_name in details["name"].lower():
            return f"Found symbol for '{details['name']}' (Mock Data Fallback): {details['symbol']}"
    
    return f"Stock symbol not found for '{company_name}'. (API/Mock Fallback Failed)"


# CLI Test (optional)
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.alphavantage_api_key = "MOCK_ALPHA_VANTAGE_KEY" # Use a dummy key
            self.openai_api_key = "sk-mock-openai-key-12345"
            self.google_api_key = "AIzaSy-mock-google-key"
            self.firebase_config = "{}"
            self.amadeus_client_id = "MOCK_AMADEUS_ID"
            self.amadeus_client_secret = "MOCK_AMADEUS_SECRET"


        def get(self, key, default=None):
            return getattr(self, key, default) # Simple attribute access for mock
    
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
                }
            }
            self._api_providers_data = { # Mock api_providers_data for finance
                "finance": {
                    "alphavantage": {
                        "base_url": "https://www.alphavantage.co/query",
                        "api_key_name": "alphavantage_api_key",
                        "api_key_param_name": "apikey", # Explicitly define param name
                        "functions": {
                            "get_stock_price": {
                                "function_param": "GLOBAL_QUOTE",
                                "required_params": ["symbol"],
                                "optional_params": ["outputsize", "datatype"],
                                "response_path": ["Global Quote"],
                                "data_map": {
                                    "price": "05. price", "open": "02. open", "high": "03. high",
                                    "low": "04. low", "volume": "06. volume", "last_trading_day": "07. latest trading day",
                                    "change": "09. change", "change_percent": "10. change percent"
                                }
                            },
                            "get_company_news": {
                                "function_param": "NEWS_SENTIMENT",
                                "required_params": ["tickers", "time_from", "time_to"],
                                "optional_params": ["sort", "limit"],
                                "response_path": ["feed"],
                                "data_map": {
                                    "title": "title", "source_name": ["source", "name"], "url": "url", "time_published": "time_published"
                                }
                            },
                            "get_historical_stock_prices": {
                                "function_param": "TIME_SERIES_DAILY",
                                "required_params": ["symbol"],
                                "optional_params": ["outputsize", "datatype"],
                                "response_path": ["Time Series (Daily)"],
                                "data_map": { # These are nested under dates, handled by specific logic
                                    "open": "1. open", "high": "2. high", "low": "3. low", "close": "4. close", "volume": "5. volume"
                                }
                            },
                            "lookup_stock_symbol": {
                                "function_param": "SYMBOL_SEARCH",
                                "required_params": ["keywords"],
                                "optional_params": ["datatype"],
                                "response_path": ["bestMatches"],
                                "data_map": {
                                    "symbol": "1. symbol", "name": "2. name"
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
                'finance_tool_access': {
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
        # Simulate Alpha Vantage responses based on function and symbol
        if "alphavantage.co/query" in url:
            function = params.get("function")
            symbol = params.get("symbol")
            keywords = params.get("keywords")
            tickers = params.get("tickers")
            
            if function == "GLOBAL_QUOTE" and symbol == "AAPL":
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "Global Quote": {
                        "01. symbol": "AAPL", "02. open": "170.0000", "03. high": "175.5000",
                        "04. low": "169.5000", "05. price": "175.2500", "06. volume": "75000000",
                        "07. latest trading day": "2025-07-05", "08. previous close": "173.7500",
                        "09. change": "1.5000", "10. change percent": "0.8633%"
                    }
                }
                return mock_response
            elif function == "NEWS_SENTIMENT" and tickers == "MSFT":
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "feed": [
                        {"title": "Microsoft Q2 Earnings Beat Expectations", "source": {"name": "Reuters"}, "url": "http://reuters.com/msft-earnings", "time_published": "20240703T100000"},
                        {"title": "Microsoft Announces New Cloud Partnership", "source": {"name": "ZDNet"}, "url": "http://zdnet.com/msft-cloud", "time_published": "20240701T150000"}
                    ]
                }
                return mock_response
            elif function == "TIME_SERIES_DAILY" and symbol == "GOOG":
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "Meta Data": {"1. Information": "Daily Prices", "2. Symbol": "GOOG"},
                    "Time Series (Daily)": {
                        "2023-01-05": {"1. open": "100.00", "2. high": "101.00", "3. low": "99.50", "4. close": "100.50", "5. volume": "1000000"},
                        "2023-01-04": {"1. open": "99.00", "2. high": "100.00", "3. low": "98.50", "4. close": "99.80", "5. volume": "1200000"}
                    }
                }
                return mock_response
            elif function == "SYMBOL_SEARCH" and keywords == "Google":
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "bestMatches": [
                        {"1. symbol": "GOOGL", "2. name": "Alphabet Inc. (Class A)", "3. type": "Equity"},
                        {"1. symbol": "GOOG", "2. name": "Alphabet Inc. (Class C)", "3. type": "Equity"}
                    ]
                }
                return mock_response
            
            # Simulate API key missing or invalid
            if not params.get("apikey") or params.get("apikey") == "INVALID_KEY":
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"Error Message": "Invalid API key."}
                return mock_response

            # Default for other Alpha Vantage calls (e.g., rate limit)
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 500 calls per day."}
            return mock_response
        return original_requests_get(url, params=params, headers=headers, timeout=timeout)

    requests.get = mock_requests_get_dynamic

    test_user_pro = sys.modules['utils.user_manager']._mock_users["mock_pro_token"]['user_id']
    test_user_premium = sys.modules['utils.user_manager']._mock_users["mock_premium_token']['user_id']
    
    print("\n--- Testing get_stock_price function (with API key) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro # Ensure user has access
    result1 = get_stock_price("AAPL", user_token=test_user_pro)
    print(f"Result for AAPL (Pro User, API):\n{result1[:200]}...")
    assert "Current stock price for AAPL:" in result1
    assert "Price: $175.25" in result1
    print("Test 1 Passed.")

    print("\n--- Testing get_stock_price function (no API key - fallback) ---")
    sys.modules['config.config_manager'].config_manager.set_secret("alphavantage_api_key", None) # Simulate no API key
    result2 = get_stock_price("MSFT", user_token=test_user_pro)
    print(f"Result for MSFT (Pro User, Fallback):\n{result2[:200]}...")
    assert "Current stock price for MSFT: (Mock Data Fallback)" in result2
    sys.modules['config.config_manager'].config_manager.set_secret("alphavantage_api_key", "MOCK_ALPHA_VANTAGE_KEY") # Restore

    print("\n--- Testing get_company_news function (with API key) ---")
    result3 = get_company_news("MSFT", "2024-07-01", "2024-07-05", user_token=test_user_pro)
    print(f"Result for MSFT News (Pro User, API):\n{result3[:200]}...")
    assert "Recent news for MSFT from 2024-07-01 to 2024-07-05:" in result3
    assert "Microsoft Q2 Earnings Beat Expectations" in result3
    print("Test 3 Passed.")

    print("\n--- Testing get_historical_stock_prices function (with API key) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_premium # Historical data is premium
    result4 = get_historical_stock_prices("GOOG", "2023-01-01", "2023-01-05", user_token=test_user_premium)
    print(f"Result for GOOG Historical (Premium User, API):\n{result4[:200]}...")
    assert "2023-01-05" in result4
    assert "open" in result4
    print("Test 4 Passed.")

    print("\n--- Testing lookup_stock_symbol function (with API key) ---")
    result5 = lookup_stock_symbol("Google", user_token=test_user_pro)
    print(f"Result for Google Symbol (Pro User, API): {result5}")
    assert "Found symbol for 'Alphabet Inc. (Class A)': GOOGL" in result5
    print("Test 5 Passed.")

    print("\nAll finance_tool tests passed (real API simulation with fallback).")

    # Restore original requests.get
    requests.get = original_requests_get
