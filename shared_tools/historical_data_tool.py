# shared_tools/historical_data_tool.py

import logging
import requests
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Import generic tools
from langchain_core.tools import tool

# Import config_manager to access API configurations and secrets
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability
# Import date_parser for date format flexibility
from utils.date_parser import parse_date_to_yyyymmdd
# Import analytics_tracker
from utils import analytics_tracker # Import the module

# Import UserProfile for type hinting
from backend.models.user_models import UserProfile

logger = logging.getLogger(__name__)

# --- Generic API Request Helper (adapted for historical data, similar to domain tools) ---

def _get_nested_value(data: Dict[str, Any], path: List[str]):
    """Helper to get a value from a nested dictionary using a list of keys."""
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and isinstance(key, str) and key.isdigit(): # Handle list indices
            try:
                current = current[int(key)]
            except (IndexError, ValueError):
                return None
        else:
            return None
    return current

class HistoricalDataTools:
    """
    A collection of tools for retrieving and potentially visualizing historical data
    across various domains (finance, crypto, weather, etc.).
    It leverages dynamic API configurations and integrates with the Python interpreter
    for advanced analysis.
    """
    def __init__(self, config_manager, log_event):
        self.config_manager = config_manager
        self.log_event = log_event

    async def _make_dynamic_api_request_historical(
        self,
        domain: str,
        function_name: str,
        params: Dict[str, Any],
        user_context: UserProfile
    ) -> Optional[Dict[str, Any]]:
        """
        Makes an API request to the dynamically configured provider for historical data.
        Handles API key retrieval, request construction, and basic error handling.
        Returns parsed JSON data or None on failure. Logs tool usage analytics for failures.
        """
        user_id = user_context.user_id

        # Historical data APIs will likely have their own domain in api_defaults (e.g., 'historical_weather')
        active_provider_name = self.config_manager.get(f"api_defaults.{domain}")
        if not active_provider_name:
            logger.error(f"No default API provider configured for historical domain '{domain}'.")
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=f"No default API provider configured for historical domain '{domain}'."
            )
            return None

        provider_config = self.config_manager.get_api_provider_config(domain, active_provider_name)
        if not provider_config:
            logger.error(f"Configuration for API provider '{active_provider_name}' in domain '{domain}' not found.")
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=f"API provider config '{active_provider_name}' not found for domain '{domain}'."
            )
            return None

        base_url = provider_config.get("base_url")
        api_key_name = provider_config.get("api_key_name")
        api_key = self.config_manager.get_secret(api_key_name) if api_key_name else None

        headers = {}
        # Special handling for APIs that use client_id/secret for token (e.g., Amadeus if used for travel historical)
        # Or APIs that require API key in headers (e.g., some premium data providers)
        if provider_config.get("api_key_in_header"):
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}" # Or "X-API-Key" etc. as per API docs
            else:
                error_msg = f"API key for {active_provider_name} is missing for header authentication."
                logger.error(error_msg)
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_id=user_id,
                    success=False,
                    error_message=error_msg
                )
                return None

        if not base_url:
            logger.error(f"Base URL not configured for API provider '{active_provider_name}' in domain '{domain}'.")
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=f"Base URL not configured for '{active_provider_name}'."
            )
            return None

        function_details = provider_config.get("functions", {}).get(function_name)
        if not function_details:
            logger.error(f"Function '{function_name}' not configured for API provider '{active_provider_name}' in domain '{domain}'.")
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=f"Function '{function_name}' not configured for '{active_provider_name}'."
            )
            return None

        endpoint = function_details.get("endpoint")
        path_params_config = function_details.get("path_params", [])

        full_url = f"{base_url}{endpoint}" if endpoint else base_url

        for p_param in path_params_config:
            if p_param in params:
                value = str(params.pop(p_param))
                full_url = full_url.replace(f"{{{p_param}}}", value)
            else:
                error_msg = f"Missing path parameter '{p_param}' for function '{function_name}'."
                logger.warning(error_msg)
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_id=user_id,
                    success=False,
                    error_message=error_msg
                )
                return None

        query_params = {}

        # Add API key if it's a query param (and not in header)
        if api_key_name and api_key and not provider_config.get("api_key_in_header"):
            param_name_in_url = provider_config.get("api_key_param_name", api_key_name.replace("_api_key", ""))
            query_params[param_name_in_url] = api_key 

        for param_key in function_details.get("required_params", []) + function_details.get("optional_params", []):
            if param_key in params:
                query_params[param_key] = params[param_key]
            elif param_key in function_details.get("required_params", []):
                error_msg = f"Missing required parameter '{param_key}' for function '{function_name}'."
                logger.warning(error_msg)
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_id=user_id,
                    success=False,
                    error_message=error_msg
                )
                return None

        try:
            logger.info(f"Making historical data API call to: {full_url} with params: {query_params}")
            response = requests.get(full_url, params=query_params, headers=headers, timeout=self.config_manager.get("web_scraping.timeout_seconds", 15))
            response.raise_for_status()
            raw_data = response.json()
            
            api_error_message = None
            if raw_data.get("Error Message"): # Alpha Vantage specific
                api_error_message = f"API Error from {active_provider_name}: {raw_data['Error Message']}"
            elif raw_data.get("Note") and "Thank you for using Alpha Vantage!" in raw_data["Note"]: # Alpha Vantage rate limit
                api_error_message = f"API rate limit hit for {active_provider_name}: {raw_data['Note']}"
            elif raw_data.get("status") == "error": # Generic error status
                api_error_message = f"API Error from {active_provider_name}: {raw_data.get('message', 'Unknown error')}"
            elif raw_data.get("Error"): # Generic error key (e.g., OMDB)
                api_error_message = f"API Error from {active_provider_name}: {raw_data.get('Error')}"
            elif raw_data.get("result") == "error": # ExchangeRate-API error
                api_error_message = f"API Error from {active_provider_name}: {raw_data.get('error-type', 'Unknown ExchangeRate-API error')}"
            elif raw_data.get("Response") == "False": # OMDBAPI specific
                api_error_message = f"API Error from {active_provider_name}: {raw_data.get('Error', 'Unknown error')}"


            if api_error_message:
                logger.error(api_error_message)
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_id=user_id,
                    success=False,
                    error_message=api_error_message
                )
                return None

            data_to_map = raw_data
            response_path = function_details.get("response_path")
            if response_path:
                data_to_map = _get_nested_value(raw_data, response_path)
                if data_to_map is None:
                    error_msg = f"Response path '{'.'.join(response_path)}' not found in API response from {active_provider_name}. Raw data: {raw_data}"
                    logger.warning(error_msg)
                    await analytics_tracker.log_tool_usage(
                        tool_name=f"{domain}_{function_name}",
                        tool_params=params,
                        user_id=user_id,
                        success=False,
                        error_message=error_msg
                    )
                    return None

            mapped_data = {}
            data_map = function_details.get("data_map", {})

            # Special handling for date-keyed time series (e.g., Alpha Vantage TIME_SERIES_DAILY)
            response_type = function_details.get("response_type")
            if response_type == "time_series_daily" and isinstance(data_to_map, dict):
                processed_data_list = []
                for date_key, values in data_to_map.items():
                    mapped_values = {"date": date_key}  # Add date to each entry
                    for mapped_key, original_key_path in data_map.items():
                        # Use the same _get_nested_value logic for consistency
                        if isinstance(original_key_path, list):
                            mapped_values[mapped_key] = _get_nested_value(values, original_key_path)
                        elif '.' in str(original_key_path):
                            mapped_values[mapped_key] = _get_nested_value(values, original_key_path.split('.'))
                        else:
                            mapped_values[mapped_key] = values.get(original_key_path)
                    processed_data_list.append(mapped_values)
                # Sort by date for consistency
                processed_data_list.sort(key=lambda x: x.get('date', ''))
                final_result = {"data": processed_data_list}
            elif isinstance(data_to_map, list):
                mapped_data_list = []
                for item in data_to_map:
                    mapped_item = {}
                    for mapped_key, original_key_path in data_map.items():
                        if isinstance(original_key_path, list):
                            mapped_item[mapped_key] = _get_nested_value(item, original_key_path)
                        elif '.' in str(original_key_path):
                            mapped_item[mapped_key] = _get_nested_value(item, original_key_path.split('.'))
                        else:
                            mapped_item[mapped_key] = item.get(original_key_path)
                    mapped_data_list.append(mapped_item)
                final_result = {"data": mapped_data_list}
            else:
                for mapped_key, original_key_path in data_map.items():
                    if isinstance(original_key_path, list):
                        mapped_data[mapped_key] = _get_nested_value(data_to_map, original_key_path)
                    elif isinstance(original_key_path, str) and '.' in original_key_path:
                        mapped_data[mapped_key] = _get_nested_value(data_to_map, original_key_path.split('.'))
                    else:
                        mapped_data[mapped_key] = data_to_map.get(original_key_path)
                final_result = mapped_data

            return final_result

        except requests.exceptions.Timeout:
            error_msg = f"API request to {active_provider_name} timed out for function '{function_name}'."
            logger.error(error_msg)
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=error_msg
            )
            return None
        except requests.exceptions.RequestException as e:
            error_msg = f"Error making API request to {active_provider_name} for function '{function_name}': {e}"
            logger.error(error_msg)
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=str(e)
            )
            return None
        except json.JSONDecodeError:
            error_msg = f"Failed to decode JSON response from {active_provider_name} for function '{function_name}'."
            logger.error(error_msg)
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=error_msg
            )
            return None
        except Exception as e:
            error_msg = f"An unexpected error occurred during API call to {active_provider_name} for '{function_name}': {e}"
            logger.error(error_msg, exc_info=True)
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=error_msg
            )
            return None

    @tool
    @staticmethod
    async def historical_get_data(**kwargs: Any) -> str:
        """
        Retrieves historical data for a given domain and identifier within a specified date range.
        This tool is generic and can fetch historical data for various domains (e.g., finance, crypto, weather).
        The dates should be provided in YYYY-MM-DD format.
        Args:
            kwargs (dict): A dictionary containing the following keys:
                - domain (str): The domain of the historical data (e.g., "historical_finance", "historical_crypto", "historical_weather").
                - identifier (str): The identifier for the data (e.g., stock ticker "AAPL", crypto ID "bitcoin", city name "London").
                - start_date (str): The start date for the historical data (YYYY-MM-DD).
                - end_date (str): The end date for the historical data (YYYY-MM-DD).
                - user_context (UserProfile): The user's profile for RBAC checks and logging.
                - data_type (str, optional): Specific type of data to retrieve (e.g., "daily" for weather, "price" for crypto).
                                           This maps to a specific function in api_providers.yml.
        Returns:
            str: A JSON string representation of the historical data, or an error message.
                 The data will be a list of dictionaries, each representing a data point.
        """
        domain = kwargs.get("domain")
        identifier = kwargs.get("identifier")
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")
        user_context = kwargs.get("user_context")
        data_type = kwargs.get("data_type")

        if user_context is None:
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: historical_get_data called for domain: '{domain}', identifier: '{identifier}', dates: {start_date} to {end_date} by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'historical_data_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to historical data tools is not enabled for your current tier."
        
        # Standardize dates
        parsed_start_date = parse_date_to_yyyymmdd(start_date)
        parsed_end_date = parse_date_to_yyyymmdd(end_date)

        if not parsed_start_date or not parsed_end_date:
            return f"Error: Could not parse start_date '{start_date}' or end_date '{end_date}'. Please use YYYY-MM-DD format."

        # Determine the function name based on domain and data_type
        function_name_map = {
            "historical_finance": "get_historical_stock_prices",
            "historical_crypto": "get_historical_crypto_prices",
            "historical_weather": "get_historical_weather",
            # Add more mappings as needed for other historical data types
        }
        api_function_name = function_name_map.get(domain)
        if not api_function_name:
            return f"Error: Unsupported historical data domain '{domain}'. Supported domains: {list(function_name_map.keys())}."

        params = {
            "identifier": identifier,
            "start_date": parsed_start_date,
            "end_date": parsed_end_date,
            **kwargs
        }
        if data_type:
            params["data_type"] = data_type # Pass data_type if the API config uses it as a param

        api_data = await HistoricalDataTools(config_manager, analytics_tracker.log_event)._make_dynamic_api_request_historical(domain, api_function_name, params, user_context)

        if api_data and api_data.get("data"):
            # Return the data as a JSON string for the LLM to process or pass to interpreter
            return json.dumps(api_data["data"])
        else:
            return f"Could not retrieve historical data for '{identifier}' in domain '{domain}' from {start_date} to {end_date}. The API call failed or returned no data. Please ensure your API key is valid and parameters are correct."

    @tool
    @staticmethod
    async def historical_plot_chart(**kwargs: Any) -> str:
        """
        Generates a textual description of a chart from historical data.
        This tool is intended to be used after `historical_get_data` to visualize the retrieved data.
        It does not generate an actual image, but describes what the chart would look like.
        Args:
            kwargs (dict): A dictionary containing the following keys:
                - data_json (str): A JSON string representing the historical data (list of dictionaries).
                                 This is typically the output of `historical_get_data`.
                - x_axis_key (str): The key in the data dictionaries to use for the X-axis (e.g., "date").
                - y_axis_key (str): The key in the data dictionaries to use for the Y-axis (e.g., "close_price", "temperature").
                - chart_type (str, optional): The type of chart to describe (e.g., "line", "bar", "scatter"). Defaults to "line".
                - title (str, optional): The title of the chart. Defaults to "Historical Data Chart".
                - user_context (UserProfile): The user's profile for RBAC checks and logging.
        Returns:
            str: A textual description of the requested chart, or an error message.
        """
        data_json = kwargs.get("data_json")
        x_axis_key = kwargs.get("x_axis_key")
        y_axis_key = kwargs.get("y_axis_key")
        chart_type = kwargs.get("chart_type", "line")
        title = kwargs.get("title", "Historical Data Chart")
        user_context = kwargs.get("user_context")

        if user_context is None:
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: historical_plot_chart called for chart type: '{chart_type}', x: '{x_axis_key}', y: '{y_axis_key}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'historical_data_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to historical data tools is not enabled for your current tier."

        try:
            data = json.loads(data_json)
            if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
                return "Error: Invalid data format. Expected a JSON list of dictionaries."
            if not data:
                return "No data provided to plot the chart."

            # Basic validation of keys
            if not all(x_axis_key in item for item in data):
                return f"Error: X-axis key '{x_axis_key}' not found in all data points."
            if not all(y_axis_key in item for item in data):
                return f"Error: Y-axis key '{y_axis_key}' not found in all data points."

            # Extract data points for description
            x_values = [item[x_axis_key] for item in data]
            y_values = [item[y_axis_key] for item in data]

            # Provide a textual description of the chart
            description = (
                f"A {chart_type} chart titled '{title}' would be generated.\n"
                f"The X-axis represents '{x_axis_key}' with values ranging from '{min(x_values)}' to '{max(x_values)}'.\n"
                f"The Y-axis represents '{y_axis_key}' with values ranging from '{min(y_values)}' to '{max(y_values)}'.\n"
                f"There are {len(data)} data points."
            )
            if len(data) > 5:
                description += f"\nFirst 5 data points: {data[:5]}"
                description += f"\nLast 5 data points: {data[-5:]}"
            else:
                description += f"\nAll data points: {data}"

            return description

        except json.JSONDecodeError:
            return "Error: Invalid JSON string provided for data."
        except Exception as e:
            logger.error(f"Error describing chart: {e}", exc_info=True)
            return f"An unexpected error occurred while trying to describe the chart: {e}"


# Instantiate the HistoricalDataTools as a singleton (or provide through dependency injection)
# For CLI testing, we'll instantiate it directly.
# historical_data_tools = HistoricalDataTools(config_manager, analytics_tracker.log_event)


# CLI Test (optional)
if __name__ == "__main__":
    import asyncio
    from unittest.mock import MagicMock, AsyncMock, patch, ANY
    import shutil
    import os
    import sys

    logging.basicConfig(level=logging.INFO)

    # Mock UserProfile for testing
    mock_user_pro_profile = UserProfile(user_id="mock_pro_token", username="ProUser", email="pro@example.com", tier="pro", roles=["user"])
    mock_user_free_profile = UserProfile(user_id="mock_free_token", username="FreeUser", email="free@example.com", tier="free", roles=["user"])

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.alphavantage_api_key = "MOCK_ALPHAVANTAGE_API_KEY_LIVE"
            self.coingecko_api_key = "MOCK_COINGECKO_API_KEY_LIVE" # Note: CoinGecko Free API usually doesn't need a key for simple price/historical
            self.weather_api_key = "MOCK_WEATHER_API_KEY_LIVE"
            self.serpapi_api_key = "MOCK_SERPAPI_KEY_LIVE" # For scrape_web
            self.openai_api_key = "sk-mock-openai-key-12345" # For summarizer
            self.google_api_key = "AIzaSy-mock-google-key" # For summarizer

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
                    'historical_finance': 'alphavantage',
                    'historical_crypto': 'coingecko',
                    'historical_weather': 'mock_historical_weather_provider',
                    'web_search': 'serpapi',
                    'document_summarization_llm': 'openai'
                },
                'analytics': { # Mock analytics settings
                    'enabled': True,
                    'log_tool_usage': True,
                    'log_query_failures': True
                }
            }
            self._api_providers_data = { # Mock api_providers_data for historical domains
                "historical_finance": {
                    "alphavantage": {
                        "base_url": "https://www.alphavantage.co/query",
                        "api_key_name": "alphavantage_api_key",
                        "api_key_param_name": "apikey",
                        "functions": {
                            "get_historical_stock_prices": {
                                "function_param": "TIME_SERIES_DAILY", # Alpha Vantage specific
                                "required_params": ["symbol"],
                                "optional_params": ["outputsize"],
                                "response_path": ["Time Series (Daily)"], # Path to the actual data
                                "data_map": { # Mapping for each daily entry
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
                "historical_crypto": {
                    "coingecko": {
                        "base_url": "https://api.coingecko.com/api/v3",
                        "api_key_name": "coingecko_api_key", # Although free API might not need it, keep for consistency
                        "api_key_param_name": "x_cg_demo_api_key", # Example for paid tiers
                        "functions": {
                            "get_historical_crypto_prices": {
                                "endpoint": "/coins/{id}/market_chart", # Path param {id}
                                "path_params": ["id"],
                                "required_params": ["vs_currency", "days"],
                                "optional_params": ["interval"],
                                "response_path": ["prices"], # Path to the prices array
                                "data_map": { # Each item in 'prices' is [timestamp, price]
                                    "timestamp": 0,
                                    "price": 1
                                }
                            }
                        }
                    }
                },
                "historical_weather": {
                    "mock_historical_weather_provider": {
                        "base_url": "http://mock-historical-weather-api.com/v1",
                        "api_key_name": "weather_api_key",
                        "api_key_param_name": "apiKey",
                        "functions": {
                            "get_historical_weather": {
                                "endpoint": "/history",
                                "required_params": ["location", "start_date", "end_date"],
                                "optional_params": ["unit"],
                                "response_path": ["history", "daily"], # Example path
                                "data_map": {
                                    "date": "date",
                                    "avg_temp_celsius": "avg_temp_c",
                                    "avg_temp_fahrenheit": "avg_temp_f",
                                    "condition": "condition.text",
                                    "precipitation_mm": "totalprecip_mm"
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


    # Mock user_manager.get_user_tier_capability for testing RBAC
    class MockUserManager:
        _mock_users = {
            "mock_free_token": {"user_id": "mock_free_token", "username": "FreeUser", "email": "free@example.com", "tier": "free", "roles": ["user"]},
            "mock_pro_token": {"user_id": "mock_pro_token", "username": "ProUser", "email": "pro@example.com", "tier": "pro", "roles": ["user"]},
            "mock_premium_token": {"user_id": "mock_premium_token", "username": "PremiumUser", "email": "premium@example.com", "tier": "premium", "roles": ["user"]},
            "mock_admin_token": {"user_id": "mock_admin_token", "username": "AdminUser", "email": "admin@example.com", "tier": "admin", "roles": ["user", "admin"]},
        }
        _rbac_capabilities = { # This now mirrors the _RBAC_CAPABILITIES_CONFIG in utils/user_manager.py
            'capabilities': {
                'historical_data_access': { # New capability
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'web_search_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
                'summarization_enabled': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'llm_default_provider': {
                    'default': 'gemini',
                    'tiers': {'pro': 'gemini', 'premium': 'openai', 'admin': 'gemini'}
                },
                'llm_default_model_name': {
                    'default': 'gemini-1.5-flash',
                    'tiers': {'pro': 'gemini-1.5-flash', 'premium': 'gpt-4o', 'admin': 'gemini-1.5-flash'}
                },
                'llm_default_temperature': {
                    'default': 0.7,
                    'tiers': {'pro': 0.5, 'premium': 0.3, 'admin': 0.7}
                },
            }
        }
        _tier_hierarchy = {
            "free": 0, "user": 1, "basic": 2, "pro": 3, "premium": 4, "admin": 99
        }

        def get_user_tier_capability(self, user_id: str, capability_key: str, default_value: Any = None, user_tier: Optional[str] = None, user_roles: Optional[List[str]] = None) -> Any:
            if user_tier is None or user_roles is None:
                user_info = self._mock_users.get(user_id, {})
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

        def mock_requests_get_dynamic(url, params=None, headers=None, timeout=None):
            # Simulate Alpha Vantage (historical_finance)
            if "alphavantage.co/query" in url and params.get("function") == "TIME_SERIES_DAILY":
                symbol = params.get("symbol", "").upper()
                if symbol == "IBM":
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "Meta Data": {"2. Symbol": "IBM"},
                        "Time Series (Daily)": {
                            "2023-01-03": {"1. open": "140.00", "2. high": "141.00", "3. low": "139.00", "4. close": "140.50", "5. volume": "1000000"},
                            "2023-01-02": {"1. open": "139.50", "2. high": "140.00", "3. low": "138.50", "4. close": "139.80", "5. volume": "950000"},
                            "2023-01-01": {"1. open": "138.00", "2. high": "139.00", "3. low": "137.50", "4. close": "138.50", "5. volume": "900000"},
                        }
                    }
                    return mock_response
                else:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"Error Message": "Invalid API call. Please retry or visit the documentation."}
                    return mock_response
            
            # Simulate CoinGecko (historical_crypto)
            elif "api.coingecko.com/api/v3/coins" in url and "/market_chart" in url:
                if "bitcoin" in url and params.get("vs_currency") == "usd":
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "prices": [
                            [1672531200000, 16500.00], # Jan 1, 2023
                            [1672617600000, 16550.00], # Jan 2, 2023
                            [1672704000000, 16600.00], # Jan 3, 2023
                        ],
                        "market_caps": [], "total_volumes": []
                    }
                    return mock_response
                else:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"status": {"error_code": 1000, "error_message": "Invalid ID or currency"}}
                    return mock_response

            # Simulate Mock Historical Weather API
            elif "mock-historical-weather-api.com/v1" in url and "/history" in url:
                location = params.get("location", "").lower()
                if "london" in location:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "history": {
                            "daily": [
                                {"date": "2023-01-01", "avg_temp_c": 5.0, "avg_temp_f": 41.0, "condition": {"text": "Cloudy"}, "totalprecip_mm": 2.5},
                                {"date": "2023-01-02", "avg_temp_c": 6.2, "avg_temp_f": 43.2, "condition": {"text": "Partly Cloudy"}, "totalprecip_mm": 0.0},
                                {"date": "2023-01-03", "avg_temp_c": 4.8, "avg_temp_f": 40.6, "condition": {"text": "Light Rain"}, "totalprecip_mm": 5.1},
                            ]
                        }
                    }
                    return mock_response
                else:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"history": {"daily": []}}
                    return mock_response
            
            # Default fallback for other requests
            return original_requests_get(url, params=params, headers=headers, timeout=timeout)

        requests.get = MagicMock(side_effect=mock_requests_get_dynamic)
        requests.post = MagicMock(side_effect=mock_requests_get_dynamic) # In case any mock uses POST

        # Instantiate HistoricalDataTools with mocks
        historical_data_tools_instance = HistoricalDataTools(
            config_manager=sys.modules['config.config_manager'].config_manager,
            log_event=analytics_tracker.log_event
        )

        async def run_historical_data_tests(historical_data_tools_instance):
            print("\n--- Testing HistoricalDataTools functions with Live API Simulation and Analytics ---")

            # Test 1: historical_get_data (Finance - Success)
            print("\n--- Test 1: historical_get_data (Finance - Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            finance_data_json = await historical_data_tools_instance.historical_get_data.ainvoke({
                "domain": "historical_finance",
                "identifier": "IBM",
                "start_date": "2023-01-01",
                "end_date": "2023-01-03",
                "user_context": mock_user_pro_profile
            })
            print(f"Historical Finance Data: {finance_data_json[:200]}...")
            assert "140.50" in finance_data_json # Check for a specific value
            assert "date" in finance_data_json # Ensure date is added
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called() # Success logged by LLMService wrapper
            print("Test 1 Passed.")

            # Test 2: historical_get_data (Crypto - Success)
            print("\n--- Test 2: historical_get_data (Crypto - Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            crypto_data_json = await historical_data_tools_instance.historical_get_data.ainvoke({
                "domain": "historical_crypto",
                "identifier": "bitcoin",
                "start_date": "2023-01-01",
                "end_date": "2023-01-03",
                "user_context": mock_user_pro_profile,
                "vs_currency": "usd",
                "days": 3 # CoinGecko uses 'days' for range
            })
            print(f"Historical Crypto Data: {crypto_data_json[:200]}...")
            assert "16500.0" in crypto_data_json
            assert "timestamp" in crypto_data_json
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 2 Passed.")

            # Test 3: historical_get_data (Weather - Success)
            print("\n--- Test 3: historical_get_data (Weather - Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            weather_data_json = await historical_data_tools_instance.historical_get_data.ainvoke({
                "domain": "historical_weather",
                "identifier": "London",
                "start_date": "2023-01-01",
                "end_date": "2023-01-03",
                "user_context": mock_user_pro_profile
            })
            print(f"Historical Weather Data: {weather_data_json[:200]}...")
            assert "avg_temp_celsius" in weather_data_json
            assert "5.0" in weather_data_json
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 3 Passed.")

            # Test 4: historical_get_data (RBAC Denied)
            print("\n--- Test 4: historical_get_data (RBAC Denied) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            rbac_denied_result = await historical_data_tools_instance.historical_get_data.ainvoke({
                "domain": "historical_finance",
                "identifier": "GOOG",
                "start_date": "2023-01-01",
                "end_date": "2023-01-05",
                "user_context": mock_user_free_profile
            })
            print(f"Historical Data (Free User, RBAC Denied): {rbac_denied_result}")
            assert "Error: Access to historical data tools is not enabled for your current tier." in rbac_denied_result
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 4 Passed.")

            # Test 5: historical_get_data (API Failure - No Data Found)
            print("\n--- Test 5: historical_get_data (API Failure - No Data Found) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            api_fail_result = await historical_data_tools_instance.historical_get_data.ainvoke({
                "domain": "historical_finance",
                "identifier": "NONEXISTENT",
                "start_date": "2023-01-01",
                "end_date": "2023-01-05",
                "user_context": mock_user_pro_profile
            })
            print(f"Historical Data (API Failure): {api_fail_result}")
            assert "Could not retrieve historical data for 'NONEXISTENT'" in api_fail_result
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "historical_finance_get_historical_stock_prices"
            assert logged_data["success"] is False
            assert "Invalid API call" in logged_data["error_message"]
            print("Test 5 Passed.")

            # Test 6: historical_plot_chart (Success)
            print("\n--- Test 6: historical_plot_chart (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            sample_data = [
                {"date": "2023-01-01", "value": 10},
                {"date": "2023-01-02", "value": 12},
                {"date": "2023-01-03", "value": 11},
            ]
            plot_description = await historical_data_tools_instance.historical_plot_chart.ainvoke({
                "data_json": json.dumps(sample_data),
                "x_axis_key": "date",
                "y_axis_key": "value",
                "chart_type": "line",
                "title": "Sample Trend",
                "user_context": mock_user_pro_profile
            })
            print(f"Chart Description: {plot_description}")
            assert "A line chart titled 'Sample Trend' would be generated." in plot_description
            assert "X-axis represents 'date'" in plot_description
            assert "Y-axis represents 'value'" in plot_description
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called() # Success logged by LLMService wrapper
            print("Test 6 Passed.")

            # Test 7: historical_plot_chart (Invalid Data)
            print("\n--- Test 7: historical_plot_chart (Invalid Data) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            invalid_data_json = "not a json string"
            plot_error = await historical_data_tools_instance.historical_plot_chart.ainvoke({
                "data_json": invalid_data_json,
                "x_axis_key": "date",
                "y_axis_key": "value",
                "user_context": mock_user_pro_profile
            })
            print(f"Chart Error: {plot_error}")
            assert "Error: Invalid JSON string provided for data." in plot_error
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called() # Error is handled internally, not an API call
            print("Test 7 Passed.")

            print("\nAll HistoricalDataTools tests completed.")

        # Ensure tests are only run when the script is executed directly
        if __name__ == "__main__":
            asyncio.run(run_historical_data_tests(historical_data_tools_instance))

        # Restore original requests.get
        requests.get = original_requests_get
        requests.post = original_requests_get # Restore post if it was patched to get
