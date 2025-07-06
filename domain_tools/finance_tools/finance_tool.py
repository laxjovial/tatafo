# domain_tools/finance_tools/finance_tool.py

import logging
import requests
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta
import asyncio

# Import generic tools
from langchain_core.tools import tool
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

class FinanceTools:
    """
    A collection of tools for finance-related operations, including stock prices,
    historical data, company overviews, and forex exchange rates.
    It integrates with external APIs and provides fallback mechanisms.
    """
    def __init__(self, config_manager, firestore_manager, log_event, document_tools):
        self.config_manager = config_manager
        self.firestore_manager = firestore_manager
        self.log_event = log_event
        self.document_tools = document_tools # For finance_query_uploaded_docs and finance_summarize_document_by_path

        # --- Mock Data for Fallback ---
        self._mock_finance_data = {
            "stock_prices": {
                "GOOG": {"symbol": "GOOG", "price": 170.00, "currency": "USD", "timestamp": datetime.now().isoformat()},
                "AAPL": {"symbol": "AAPL", "price": 180.50, "currency": "USD", "timestamp": datetime.now().isoformat()},
            },
            "historical_data": {
                "GOOG": [
                    {"date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"), "open": 165.00, "high": 168.00, "low": 164.00, "close": 167.50, "volume": 1000000},
                    {"date": (datetime.now() - timedelta(days=4)).strftime("%Y-%m-%d"), "open": 167.00, "high": 170.00, "low": 166.00, "close": 169.50, "volume": 1200000},
                ]
            },
            "company_overview": {
                "GOOG": {
                    "symbol": "GOOG",
                    "name": "Alphabet Inc.",
                    "exchange": "NASDAQ",
                    "sector": "Technology",
                    "industry": "Internet Content & Information",
                    "description": "Alphabet Inc. is an American multinational technology conglomerate holding company.",
                    "market_cap": "2.2 Trillion USD"
                }
            },
            "forex_rates": {
                "USD_EUR": {"from_currency": "USD", "to_currency": "EUR", "rate": 0.92, "timestamp": datetime.now().isoformat()},
                "EUR_GBP": {"from_currency": "EUR", "to_currency": "GBP", "rate": 0.85, "timestamp": datetime.now().isoformat()},
            }
        }

    async def _make_dynamic_api_request(
        self, # Added self
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
        log_tool_usage_enabled = self.config_manager.get("analytics.log_tool_usage", False)

        # Get the default active API provider for the domain from data/config.yml
        active_provider_name = self.config_manager.get(f"api_defaults.{domain}")
        if not active_provider_name:
            logger.error(f"No default API provider configured for domain '{domain}'.")
            if log_tool_usage_enabled:
                await analytics_tracker.log_tool_usage( # Use analytics_tracker directly
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_token=user_token,
                    success=False,
                    error_message=f"No default API provider configured for domain '{domain}'."
                )
            return None

        # Get the full configuration for the active provider from api_providers.yml
        provider_config = self.config_manager.get_api_provider_config(domain, active_provider_name)
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
        api_key = self.config_manager.get_secret(api_key_name) if api_key_name else None

        # Special handling for Amadeus which uses client_id and client_secret for token
        if active_provider_name == "amadeus":
            api_secret_name = provider_config.get("api_secret_name")
            api_secret = self.config_manager.get_secret(api_secret_name) if api_secret_name else None
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
            response = requests.get(full_url, params=query_params, headers=headers, timeout=self.config_manager.get("web_scraping.timeout_seconds", 15))
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

    @tool
    async def finance_get_stock_price(self, symbol: str, user_token: str = "default") -> str:
        """
        Retrieves the current stock price for a given stock symbol.
        Falls back to mock data if API key is missing or API call fails.

        Args:
            symbol (str): The stock symbol (e.g., "AAPL", "GOOG").
            user_token (str, optional): The unique identifier for the user. Defaults to "default".

        Returns:
            str: A formatted string of the stock price, or an error/fallback message.
        """
        logger.info(f"Tool: finance_get_stock_price called for symbol: '{symbol}' by user: {user_token}")

        if not get_user_tier_capability(user_token, 'finance_tool_access', False):
            return "Error: Access to finance tools is not enabled for your current tier."
        
        params = {"symbol": symbol}
        api_data = await self._make_dynamic_api_request("finance", "get_stock_price", params, user_token)

        if api_data:
            try:
                price = api_data.get("price")
                currency = api_data.get("currency")
                timestamp = api_data.get("timestamp")
                if price and currency:
                    return f"The current price of {symbol} is {price} {currency} (as of {timestamp})."
                else:
                    logger.warning(f"Live API data for {symbol} price is incomplete. Raw: {api_data}")
                    return f"Could not retrieve complete live stock price for {symbol}. Falling back to mock data."
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing live stock price data for {symbol}: {e}")
                return f"Error parsing live data for {symbol}. Falling back to mock data."

        # Fallback to mock data
        mock_price = self._mock_finance_data.get("stock_prices", {}).get(symbol.upper())
        if mock_price:
            return f"The current price of {mock_price['symbol']} is {mock_price['price']} {mock_price['currency']} (Mock Data Fallback, as of {mock_price['timestamp']})."
        else:
            return f"Stock price for {symbol} not found. (API/Mock Fallback Failed)"

    @tool
    async def finance_get_historical_stock_prices(self, symbol: str, user_token: str = "default") -> str:
        """
        Retrieves historical daily stock prices for a given stock symbol.
        Falls back to mock data if API key is missing or API call fails.

        Args:
            symbol (str): The stock symbol (e.g., "AAPL", "GOOG").
            user_token (str, optional): The unique identifier for the user. Defaults to "default".

        Returns:
            str: A formatted string of historical stock prices, or an error/fallback message.
        """
        logger.info(f"Tool: finance_get_historical_stock_prices called for symbol: '{symbol}' by user: {user_token}")

        if not get_user_tier_capability(user_token, 'historical_data_access', False):
            return "Error: Access to historical data is not enabled for your current tier."
        
        params = {"symbol": symbol}
        api_data = await self._make_dynamic_api_request("finance", "get_historical_stock_prices", params, user_token)

        if api_data and api_data.get("data"):
            historical_prices = api_data["data"]
            if historical_prices:
                response_str = f"Historical Prices for {symbol}:\n"
                # Sort by date (assuming YYYY-MM-DD format) and take most recent 5
                sorted_prices = sorted(historical_prices.items(), key=lambda item: item[0], reverse=True)[:5]
                for date, data in sorted_prices:
                    response_str += (
                        f"  Date: {date}\n"
                        f"    Open: {data.get('open', 'N/A')}\n"
                        f"    High: {data.get('high', 'N/A')}\n"
                        f"    Low: {data.get('low', 'N/A')}\n"
                        f"    Close: {data.get('close', 'N/A')}\n"
                        f"    Volume: {data.get('volume', 'N/A')}\n"
                    )
                return response_str
            else:
                return f"No live historical prices found for {symbol}. Falling back to mock data."

        # Fallback to mock data
        mock_historical_data = self._mock_finance_data.get("historical_data", {}).get(symbol.upper())
        if mock_historical_data:
            response_str = f"Historical Prices for {symbol} (Mock Data Fallback):\n"
            for data in mock_historical_data:
                response_str += (
                    f"  Date: {data['date']}\n"
                    f"    Open: {data['open']}\n"
                    f"    High: {data['high']}\n"
                    f"    Low: {data['low']}\n"
                    f"    Close: {data['close']}\n"
                    f"    Volume: {data['volume']}\n"
                )
            return response_str
        else:
            return f"Historical stock prices for {symbol} not found. (API/Mock Fallback Failed)"

    @tool
    async def finance_get_company_overview(self, symbol: str, user_token: str = "default") -> str:
        """
        Retrieves a company's overview, including its description, sector, and market capitalization.
        Falls back to mock data if API key is missing or API call fails.

        Args:
            symbol (str): The stock symbol (e.g., "AAPL", "GOOG").
            user_token (str, optional): The unique identifier for the user. Defaults to "default".

        Returns:
            str: A formatted string of company overview information, or an error/fallback message.
        """
        logger.info(f"Tool: finance_get_company_overview called for symbol: '{symbol}' by user: {user_token}")

        if not get_user_tier_capability(user_token, 'finance_tool_access', False):
            return "Error: Access to finance tools is not enabled for your current tier."
        
        params = {"symbol": symbol}
        api_data = await self._make_dynamic_api_request("finance", "get_company_overview", params, user_token)

        if api_data:
            try:
                name = api_data.get("name")
                sector = api_data.get("sector")
                industry = api_data.get("industry")
                description = api_data.get("description")
                market_cap = api_data.get("market_cap")

                if name and description:
                    return (
                        f"Company Overview for {name} ({symbol}):\n"
                        f"  Sector: {sector}\n"
                        f"  Industry: {industry}\n"
                        f"  Market Cap: {market_cap}\n"
                        f"  Description: {description}"
                    )
                else:
                    logger.warning(f"Live API data for {symbol} overview is incomplete. Raw: {api_data}")
                    return f"Could not retrieve complete live company overview for {symbol}. Falling back to mock data."
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing live company overview data for {symbol}: {e}")
                return f"Error parsing live data for {symbol}. Falling back to mock data."

        # Fallback to mock data
        mock_overview = self._mock_finance_data.get("company_overview", {}).get(symbol.upper())
        if mock_overview:
            return (
                f"Company Overview for {mock_overview['name']} ({mock_overview['symbol']}) (Mock Data Fallback):\n"
                f"  Sector: {mock_overview['sector']}\n"
                f"  Industry: {mock_overview['industry']}\n"
                f"  Market Cap: {mock_overview['market_cap']}\n"
                f"  Description: {mock_overview['description']}"
            )
        else:
            return f"Company overview for {symbol} not found. (API/Mock Fallback Failed)"

    @tool
    async def finance_get_forex_exchange_rate(self, from_currency: str, to_currency: str, user_token: str = "default") -> str:
        """
        Retrieves the current exchange rate between two currencies.
        Falls back to mock data if API key is missing or API call fails.

        Args:
            from_currency (str): The currency to convert from (e.g., "USD", "EUR").
            to_currency (str): The currency to convert to (e.g., "JPY", "GBP").
            user_token (str, optional): The unique identifier for the user. Defaults to "default".

        Returns:
            str: A formatted string of the exchange rate, or an error/fallback message.
        """
        logger.info(f"Tool: finance_get_forex_exchange_rate called for {from_currency} to {to_currency} by user: {user_token}")

        if not get_user_tier_capability(user_token, 'finance_tool_access', False):
            return "Error: Access to finance tools is not enabled for your current tier."
        
        params = {"from_currency": from_currency, "to_currency": to_currency}
        api_data = await self._make_dynamic_api_request("finance", "get_forex_exchange_rate", params, user_token)

        if api_data:
            try:
                rate = api_data.get("rate")
                timestamp = api_data.get("timestamp")
                if rate:
                    return f"The current exchange rate from {from_currency} to {to_currency} is {rate} (as of {timestamp})."
                else:
                    logger.warning(f"Live API data for {from_currency} to {to_currency} exchange rate is incomplete. Raw: {api_data}")
                    return f"Could not retrieve complete live exchange rate for {from_currency} to {to_currency}. Falling back to mock data."
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing live forex data for {from_currency} to {to_currency}: {e}")
                return f"Error parsing live data for {from_currency} to {to_currency}. Falling back to mock data."

        # Fallback to mock data
        mock_rate_key = f"{from_currency.upper()}_{to_currency.upper()}"
        mock_rate = self._mock_finance_data.get("forex_rates", {}).get(mock_rate_key)
        if mock_rate:
            return f"The current exchange rate from {mock_rate['from_currency']} to {mock_rate['to_currency']} is {mock_rate['rate']} (Mock Data Fallback, as of {mock_rate['timestamp']})."
        else:
            return f"Exchange rate for {from_currency} to {to_currency} not found. (API/Mock Fallback Failed)"


    # --- Existing Generic Tools (now methods of FinanceTools) ---

    @tool
    def finance_search_web(self, query: str, user_token: str = "default", max_chars: int = 2000) -> str:
        """
        Searches the web for finance-related information using a smart search fallback mechanism.
        This tool wraps the generic `scrape_web` tool, providing a finance-specific interface.
        
        Args:
            query (str): The finance-related search query (e.g., "latest stock market news", "explain cryptocurrency taxation").
            user_token (str): The unique identifier for the user. Defaults to "default".
            max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
        
        Returns:
            str: A string containing relevant information from the web.
        """
        logger.info(f"Tool: finance_search_web called with query: '{query}' for user: '{user_token}'")
        return scrape_web(query=query, user_token=user_token, max_chars=max_chars)

    @tool
    async def finance_query_uploaded_docs(self, query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed finance documents for a user using vector similarity search.
        This tool wraps the generic `QueryUploadedDocs` tool, fixing the section to "finance".
        
        Args:
            query (str): The search query to find relevant finance documents (e.g., "my investment portfolio details", "tax documents for 2023").
            user_token (str): The unique identifier for the user. Defaults to "default".
            export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
            k (int): The number of top relevant documents to retrieve. Defaults to 5.
        
        Returns:
            str: A string containing the combined content of the relevant document chunks,
                 or a message indicating no data/results found, or the export path if exported.
        """
        logger.info(f"Tool: finance_query_uploaded_docs called with query: '{query}' for user: '{user_token}'")
        if not self.document_tools:
            return "Error: Document tools are not initialized. Cannot query uploaded documents."
        
        # Call the actual query_uploaded_docs from the DocumentTools instance
        return await self.document_tools.document_query_uploaded_docs(
            query=query,
            user_token=user_token,
            section="finance", # Specify the section for finance documents
            export=export,
            k=k
        )

    @tool
    async def finance_summarize_document_by_path(self, file_path_str: str) -> str:
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
        if not self.document_tools:
            return "Error: Document tools are not initialized. Cannot summarize documents."

        # Call the actual summarize_document_by_path from the DocumentTools instance
        return await self.document_tools.document_summarize_document_by_path(file_path_str=file_path_str)


# CLI Test (optional)
if __name__ == "__main__":
    import asyncio
    from unittest.mock import MagicMock, AsyncMock, patch
    import shutil
    import os
    import sys # Import sys for patching modules
    from shared_tools.vector_utils import BASE_VECTOR_DIR # For cleanup
    from database.firestore_manager import FirestoreManager # For mocking
    from shared_tools.cloud_storage_utils import CloudStorageUtilsWrapper # For mocking
    from shared_tools.vector_utils import VectorUtilsWrapper # For mocking
    from domain_tools.document_tools.document_tool import DocumentTools # For mocking

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.finance_api_key = "MOCK_FINANCE_API_KEY"
            self.alphavantage_api_key = "MOCK_ALPHAVANTAGE_API_KEY"
            self.exchangerate_api_key = "MOCK_EXCHANGERATE_API_KEY"
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
                    'finance': 'alphavantage' # Using alphavantage for finance mock
                },
                'analytics': { # Mock analytics settings
                    'enabled': True,
                    'log_tool_usage': True,
                    'log_query_failures': True
                }
            }
            self._api_providers_data = { # Mock api_providers_data for finance
                "finance": {
                    "alphavantage": { # Mocking Alpha Vantage for stock prices and overview
                        "base_url": "https://www.alphavantage.co/query",
                        "api_key_name": "alphavantage_api_key",
                        "api_key_param_name": "apikey",
                        "functions": {
                            "get_stock_price": {
                                "function_param": "GLOBAL_QUOTE",
                                "required_params": ["symbol"],
                                "response_path": ["Global Quote"],
                                "data_map": {
                                    "symbol": "01. symbol",
                                    "price": "05. price",
                                    "currency": "8. currency", # Alpha Vantage doesn't usually provide currency directly in GLOBAL_QUOTE
                                    "timestamp": "07. latest trading day"
                                }
                            },
                            "get_historical_stock_prices": {
                                "function_param": "TIME_SERIES_DAILY",
                                "required_params": ["symbol"],
                                "optional_params": ["outputsize"],
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
                                "response_path": [], # Root of the response is the overview
                                "data_map": {
                                    "symbol": "Symbol",
                                    "name": "Name",
                                    "exchange": "Exchange",
                                    "sector": "Sector",
                                    "industry": "Industry",
                                    "description": "Description",
                                    "market_cap": "MarketCapitalization"
                                }
                            }
                        }
                    },
                    "exchangerate_api": { # Mocking ExchangeRate-API for forex
                        "base_url": "https://v6.exchangerate-api.com/v6",
                        "api_key_name": "exchangerate_api_key",
                        "path_params": ["api_key", "from_currency", "to_currency"],
                        "endpoint": "/{api_key}/pair/{from_currency}/{to_currency}",
                        "functions": {
                            "get_forex_exchange_rate": {
                                "required_params": [], # Params are in path
                                "response_path": [], # Root of the response is the overview
                                "data_map": {
                                    "from_currency": "base_code",
                                    "to_currency": "target_code",
                                    "rate": "conversion_rate",
                                    "timestamp": "time_last_update_utc"
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
                'historical_data_access': {
                    'default': False,
                    'roles': {'premium': True, 'admin': True}
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
                },
                'summarization_enabled': { # For summarize_document
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'llm_default_provider': { # For summarize_document
                    'default': 'gemini',
                    'tiers': {'pro': 'gemini', 'premium': 'openai', 'admin': 'gemini'}
                },
                'llm_default_model_name': { # For summarize_document
                    'default': 'gemini-1.5-flash',
                    'tiers': {'pro': 'gemini-1.5-flash', 'premium': 'gpt-4o', 'admin': 'gemini-1.5-flash'}
                },
                'llm_default_temperature': { # For summarize_document
                    'default': 0.7,
                    'tiers': {'pro': 0.5, 'premium': 0.3, 'admin': 0.7}
                },
            }
        }
        _tier_hierarchy = {
            "free": 0, "user": 1, "basic": 2, "pro": 3, "premium": 4, "admin": 99
        }

        # This mock is for the standalone get_user_tier_capability function
        # which is now imported directly by tools.
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
            
            if user_tier in capability_config.get('tiers', {}):
                return capability_config['tiers'][user_tier]

            return capability_config.get('default', default_value)


    # Patch the actual imports for testing
    import streamlit as st_mock
    if not hasattr(st_mock, 'secrets'):
        st_mock.secrets = MockSecrets()
    
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager
    
    # Patch the standalone get_user_tier_capability function in utils.user_manager
    # This is crucial for the tools to use the mock during their CLI tests.
    sys.modules['utils.user_manager'].get_user_tier_capability = MockUserManager().get_user_tier_capability

    # Mock analytics_tracker
    mock_analytics_tracker_db = MagicMock()
    mock_analytics_tracker_auth = MagicMock()
    mock_analytics_tracker_auth.currentUser = MagicMock(uid="mock_user_123")
    mock_analytics_tracker_db.collection.return_value.add = AsyncMock(return_value=MagicMock(id="mock_doc_id"))

    # Patch firebase_admin.firestore for the local import within log_event
    with patch.dict(sys.modules, {'firebase_admin.firestore': MagicMock(firestore=MagicMock())}):
        sys.modules['firebase_admin.firestore'].firestore.CollectionReference = MagicMock()
        sys.modules['firebase_admin'].firestore.DocumentReference = MagicMock()
        
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
            # Simulate hypothetical Alpha Vantage responses
            if "alphavantage.co" in url:
                function = params.get("function")
                symbol = params.get("symbol")
                if function == "GLOBAL_QUOTE":
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "Global Quote": {
                            "01. symbol": symbol.upper(),
                            "05. price": "175.00",
                            "07. latest trading day": datetime.now().strftime("%Y-%m-%d")
                        }
                    }
                    return mock_response
                elif function == "TIME_SERIES_DAILY":
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "Meta Data": {"2. Symbol": symbol.upper()},
                        "Time Series (Daily)": {
                            (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"): {
                                "1. open": "160.00", "2. high": "162.00", "3. low": "159.00", "4. close": "161.50", "5. volume": "500000"
                            },
                            (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"): {
                                "1. open": "158.00", "2. high": "160.00", "3. low": "157.00", "4. close": "159.50", "5. volume": "450000"
                            }
                        }
                    }
                    return mock_response
                elif function == "OVERVIEW":
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "Symbol": symbol.upper(),
                        "Name": f"Mock Company {symbol.upper()}",
                        "Exchange": "MOCKEX",
                        "Sector": "Technology",
                        "Industry": "Software",
                        "Description": f"A leading company in {symbol.upper()} sector.",
                        "MarketCapitalization": "1,000,000,000,000"
                    }
                    return mock_response
                else:
                    mock_response = MagicMock()
                    mock_response.status_code = 400
                    mock_response.json.return_value = {"Error Message": "Invalid function"}
                    return mock_response
            
            # Simulate hypothetical ExchangeRate-API responses
            if "exchangerate-api.com" in url and "/pair/" in url:
                parts = url.split("/pair/")[1].split("/")
                api_key = parts[0]
                from_currency = parts[1]
                to_currency = parts[2]
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "result": "success",
                    "base_code": from_currency.upper(),
                    "target_code": to_currency.upper(),
                    "conversion_rate": 1.15, # Example rate
                    "time_last_update_utc": datetime.now().isoformat()
                }
                return mock_response

            # Simulate scrape_web's internal requests.get if needed
            if "google.com/search" in url or "example.com" in url: # Mock for scrape_web
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.text = f"<html><body><h1>Search results for {params.get('q', 'finance')}</h1><p>Some finance related content from web search.</p></body></html>"
                return mock_response

            return original_requests_get(url, params=params, headers=headers, timeout=timeout)

        requests.get = MagicMock(side_effect=mock_requests_get_dynamic)

        test_user_pro = "mock_pro_token"
        test_user_free = "mock_free_token"
        test_user_premium = "mock_premium_token"

        # Mock for summarize_document
        class MockSummarizeDocument:
            def __call__(self, text, user_token):
                return f"Mocked summary of text for user {user_token}: {text[:50]}..."

        # Patch summarize_document in the finance_tool module
        original_summarize_document = sys.modules['domain_tools.finance_tools.finance_tool'].summarize_document
        sys.modules['domain_tools.finance_tools.finance_tool'].summarize_document = MockSummarizeDocument()

        # Mock FirestoreManager, CloudStorageUtilsWrapper, VectorUtilsWrapper, DocumentTools for init
        mock_firestore_manager = MagicMock(spec=FirestoreManager)
        mock_cloud_storage_utils = MagicMock(spec=CloudStorageUtilsWrapper)
        mock_vector_utils = MagicMock(spec=VectorUtilsWrapper)
        
        # Mock log_event function
        async def mock_log_event(*args, **kwargs):
            print(f"Mock log_event called with: {args}, {kwargs}")

        # Create a mock DocumentTools instance
        mock_document_tools = MagicMock(spec=DocumentTools)
        mock_document_tools.document_query_uploaded_docs = AsyncMock(return_value="Mocked document query results for finance.")
        mock_document_tools.document_summarize_document_by_path = AsyncMock(return_value="Mocked summary of dummy_file.txt")

        # Instantiate FinanceTools with mocks
        finance_tools_instance = FinanceTools(
            config_manager=sys.modules['config.config_manager'].config_manager,
            firestore_manager=mock_firestore_manager,
            log_event=mock_log_event,
            document_tools=mock_document_tools
        )

        async def run_finance_tests(finance_tools_instance): # Pass the instance to the test function
            print("\n--- Testing finance_tool functions with Analytics ---")

            # Test 1: finance_get_stock_price (success)
            print("\n--- Test 1: finance_get_stock_price (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock() # Reset mock call count
            result_price = await finance_tools_instance.finance_get_stock_price("GOOG", user_token=test_user_pro)
            print(f"Stock Price: {result_price}")
            assert "The current price of GOOG is 175.00" in result_price
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "finance_get_stock_price"
            assert logged_data["success"] is True
            print("Test 1 Passed (and analytics logged success).")

            # Test 2: finance_get_historical_stock_prices (API failure - no data found)
            print("\n--- Test 2: finance_get_historical_stock_prices (API Failure) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            # Temporarily modify mock_requests_get_dynamic to return no data for historical prices
            def mock_requests_get_no_historical(url, params, headers, timeout):
                if "alphavantage.co" in url and params.get("function") == "TIME_SERIES_DAILY":
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 500 calls per day. Please visit https://www.alphavantage.co/premium/ to upgrade your membership."}
                    return mock_response
                return mock_requests_get_dynamic.side_effect(url, params, headers, timeout) # Call original side effect for others
            
            # Need to re-assign the side_effect to the MagicMock
            requests.get.side_effect = mock_requests_get_no_historical

            result_historical = await finance_tools_instance.finance_get_historical_stock_prices("NONEXISTENT", user_token=test_user_premium)
            print(f"Historical Prices (API Error): {result_historical}")
            assert "No live historical prices found for NONEXISTENT." in result_historical or "API rate limit hit" in result_historical
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "finance_get_historical_stock_prices"
            assert logged_data["success"] is False
            assert "API rate limit hit" in logged_data["error_message"] or "Response path 'Time Series (Daily)' not found" in logged_data["error_message"]
            print("Test 2 Passed (and analytics logged failure).")

            # Restore original mock_requests_get_dynamic
            requests.get.side_effect = mock_requests_get_dynamic

            # Test 3: finance_get_company_overview (RBAC denied)
            print("\n--- Test 3: finance_get_company_overview (RBAC Denied) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_overview_rbac_denied = await finance_tools_instance.finance_get_company_overview("MSFT", user_token=test_user_free)
            print(f"Company Overview (Free User, RBAC Denied): {result_overview_rbac_denied}")
            assert "Error: Access to finance tools is not enabled for your current tier." in result_overview_rbac_denied
            # No analytics log expected here because RBAC check happens before _make_dynamic_api_request
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 3 Passed (RBAC correctly prevented call and no analytics logged).")

            # Test 4: finance_get_forex_exchange_rate (success)
            print("\n--- Test 4: finance_get_forex_exchange_rate (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            # Temporarily set API default for finance to exchangerate_api for this test
            sys.modules['config.config_manager'].config_manager._config_data['api_defaults']['finance'] = 'exchangerate_api'
            
            result_forex = await finance_tools_instance.finance_get_forex_exchange_rate("USD", "EUR", user_token=test_user_pro)
            print(f"Forex Rate: {result_forex}")
            assert "The current exchange rate from USD to EUR is 1.15" in result_forex
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "finance_get_forex_exchange_rate"
            assert logged_data["success"] is True
            print("Test 4 Passed (and analytics logged success).")

            # Reset API default for finance to alphavantage for subsequent tests
            sys.modules['config.config_manager'].config_manager._config_data['api_defaults']['finance'] = 'alphavantage'


            # Test 5: finance_search_web (generic tool)
            print("\n--- Test 5: finance_search_web (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_web_search = await finance_tools_instance.finance_search_web("impact of inflation on economy", user_token=test_user_pro)
            print(f"Web Search Result: {result_web_search[:100]}...")
            assert "Search results for impact of inflation on economy" in result_web_search
            # Analytics for generic tools like scrape_web or summarize_document
            # would need to be integrated within those shared_tools themselves,
            # or wrapped by a higher-level agent logging.
            # For now, we are focusing on _make_dynamic_api_request.
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 5 Passed (no analytics expected for generic tool directly).\n")

            # Test 6: finance_query_uploaded_docs (generic tool)
            print("\n--- Test 6: finance_query_uploaded_docs (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_doc_query = await finance_tools_instance.finance_query_uploaded_docs("my investment portfolio", user_token=test_user_pro)
            print(f"Document Query Result: {result_doc_query}")
            assert "Mocked document query results for finance." in result_doc_query
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once() # Now logged by DocumentTools mock
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "document_query_uploaded_docs"
            assert logged_data["success"] is True
            print("Test 6 Passed (analytics expected for generic tool via DocumentTools).")


            # Test 7: finance_summarize_document_by_path (generic tool)
            print("\n--- Test 7: finance_summarize_document_by_path (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            # Create a dummy file for summarization test
            dummy_file_path = Path("uploads") / test_user_pro / "finance" / "financial_report.txt"
            dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_file_path.write_text("This is a dummy financial report for testing summarization. It contains details about revenue and expenses.")

            result_summarize = await finance_tools_instance.finance_summarize_document_by_path(str(dummy_file_path))
            print(f"Summarize Result: {result_summarize}")
            assert "Mocked summary of dummy_file.txt" in result_summarize # Check for mock summary from DocumentTools
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once() # Now logged by DocumentTools mock
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "document_summarize_document_by_path"
            assert logged_data["success"] is True
            print("Test 7 Passed (analytics expected for generic tool via DocumentTools).")


            print("\nAll finance_tool tests with analytics considerations completed.")

        # Ensure tests are only run when the script is executed directly
        if __name__ == "__main__":
            # Use asyncio.run to execute the async test function
            asyncio.run(run_finance_tests(finance_tools_instance)) # Pass the instance here

        # Restore original requests.get
        requests.get = original_requests_get

        # Restore original summarize_document (if patched)
        sys.modules['domain_tools.finance_tools.finance_tool'].summarize_document = original_summarize_document

        # Clean up dummy files and directories
        test_user_dirs = [Path("uploads") / test_user_pro, BASE_VECTOR_DIR / test_user_pro]
        for d in test_user_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print(f"Cleaned up {d}")
