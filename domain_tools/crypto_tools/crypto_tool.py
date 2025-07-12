# domain_tools/crypto_tools/crypto_tool.py

import logging
import requests
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta, timezone # Import timezone for consistent datetime objects

# Import generic tools
from langchain_core.tools import tool
from shared_tools.scrapper_tool import scrape_web # This tool is a standalone function
# from shared_tools.doc_summarizer import summarize_document # This will be accessed via DocumentTools
# from shared_tools.query_uploaded_docs_tool import query_uploaded_docs # This will be accessed via DocumentTools

# Import config_manager to access API configurations and secrets
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability
# Import date_parser for date format flexibility
from utils.date_parser import parse_date_to_yyyymmdd

# Import analytics_tracker (for logging failures within _make_dynamic_api_request)
from utils import analytics_tracker

# Import UserProfile for type hinting
from backend.models.user_models import UserProfile

logger = logging.getLogger(__name__)

# --- Generic API Request Helper (copied for standalone tool file, ideally in shared utils) ---
# This helper is designed to work with the structure defined in api_providers.yml

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

class CryptoTools:
    """
    A collection of tools for cryptocurrency-related operations, including prices,
    historical data, and general information.
    It integrates with external APIs and provides fallback mechanisms.
    """
    def __init__(self, config_manager, log_event, document_tools):
        self.config_manager = config_manager
        self.log_event = log_event # For direct logging if needed, but _make_dynamic_api_request handles tool usage
        self.document_tools = document_tools # For crypto_query_uploaded_docs and crypto_summarize_document_by_path

    async def _make_dynamic_api_request(
        self,
        domain: str,
        function_name: str,
        params: Dict[str, Any],
        user_context: UserProfile # Changed from user_token to user_context
    ) -> Optional[Dict[str, Any]]:
        """
        Makes an API request to the dynamically configured provider for a given domain and function.
        Handles API key retrieval, request construction, and basic error handling.
        Returns parsed JSON data or None on failure (triggering generic fallback message).
        Logs tool usage analytics for *failures* via analytics_tracker.
        Success logging is handled by LLMService's wrapped_tool_executor.
        """
        user_id = user_context.user_id # Get user_id from UserProfile

        # Get the default active API provider for the domain from data/config.yml
        active_provider_name = self.config_manager.get(f"api_defaults.{domain}")
        if not active_provider_name:
            logger.error(f"No default API provider configured for domain '{domain}'.")
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=f"No default API provider configured for domain '{domain}'."
            )
            return None

        # Get the full configuration for the active provider from api_providers.yml
        provider_config = self.config_manager.get_api_provider_config(domain, active_provider_name)
        if not provider_config:
            logger.error(f"Configuration for API provider '{active_provider_name}' in domain '{domain}' not found in api_providers.yml.")
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

        # CoinGecko specific: API key can be passed as a query parameter or not needed for demo tier
        # The `api_key_param_name` in config will handle this.
        headers = {} # No special headers by default for CoinGecko

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
        path_params_config = function_details.get("path_params", []) # For paths like /coins/{id}

        # Construct URL
        full_url = f"{base_url}{endpoint}" if endpoint else base_url

        # Add path parameters to URL if specified
        for p_param in path_params_config:
            if p_param in params:
                value = str(params.pop(p_param)) # Remove from params after using for path
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
                return None # Cannot construct URL without required path params

        # Construct query parameters
        query_params = {}

        # Add API key if it's a query param
        if api_key_name and api_key:
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
                return None # Missing required param, cannot proceed

        try:
            logger.info(f"Making API call to: {full_url} with params: {query_params}")
            response = requests.get(full_url, params=query_params, headers=headers, timeout=self.config_manager.get("web_scraping.timeout_seconds", 15))
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            raw_data = response.json()
            
            # Check for API-specific error messages in the response body
            api_error_message = None
            if raw_data.get("status") and raw_data["status"].get("error_code"): # CoinGecko error format
                api_error_message = f"API Error from {active_provider_name}: {raw_data['status'].get('error_message', 'Unknown CoinGecko error')}"
            elif raw_data.get("error"): # Generic error key
                api_error_message = f"API Error from {active_provider_name}: {raw_data['error']}"
            elif raw_data.get("message") == "Not Found": # Generic "Not Found" message
                api_error_message = f"API Error from {active_provider_name}: Resource not found."
            elif raw_data.get("status") == "error": # Generic error status
                api_error_message = f"API Error from {active_provider_name}: {raw_data.get('message', 'Unknown error')}"

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


            # Extract data based on response_path
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

            # Apply data mapping
            mapped_data = {}
            data_map = function_details.get("data_map", {})

            if function_name == "get_crypto_price" and active_provider_name == "coingecko":
                # CoinGecko simple price response structure: { "bitcoin": { "usd": 20000, "usd_market_cap": ... } }
                crypto_id_param = params.get("ids", "").lower() # Use 'ids' from original params
                vs_currency_param = params.get("vs_currencies", "").lower() # Use 'vs_currencies' from original params

                if crypto_id_param in raw_data and vs_currency_param in raw_data[crypto_id_param]:
                    mapped_data["price"] = raw_data[crypto_id_param][vs_currency_param]
                    # Dynamically map other fields based on vs_currency
                    for suffix in ["_market_cap", "_24hr_vol", "_24hr_change"]:
                        full_key = f"{vs_currency_param}{suffix}"
                        if full_key in raw_data[crypto_id_param]:
                            mapped_data[suffix.strip('_')] = raw_data[crypto_id_param][full_key]
                    if "last_updated_at" in raw_data[crypto_id_param]:
                        mapped_data["last_updated"] = raw_data[crypto_id_param]["last_updated_at"]
                    final_result = mapped_data
                else:
                    error_msg = f"CoinGecko simple price response unexpected for {crypto_id_param}/{vs_currency_param}: {raw_data}"
                    logger.warning(error_msg)
                    await analytics_tracker.log_tool_usage(
                        tool_name=f"{domain}_{function_name}",
                        tool_params=params,
                        user_id=user_id,
                        success=False,
                        error_message=error_msg
                    )
                    return None
            elif function_name == "get_historical_crypto_price" and active_provider_name == "coingecko":
                # CoinGecko historical data: market_data.current_price.usd, market_data.market_cap.usd, market_data.total_volume.usd
                vs_currency_param = params.get("vs_currency", "usd").lower()
                
                price_path = ["market_data", "current_price", vs_currency_param]
                market_cap_path = ["market_data", "market_cap", vs_currency_param]
                volume_path = ["market_data", "total_volume", vs_currency_param]

                mapped_data["price"] = _get_nested_value(raw_data, price_path)
                mapped_data["market_cap"] = _get_nested_value(raw_data, market_cap_path)
                mapped_data["volume"] = _get_nested_value(raw_data, volume_path)
                final_result = mapped_data
            elif function_name == "get_crypto_id_by_symbol" and active_provider_name == "coingecko":
                # CoinGecko /coins/list returns a list of {id, symbol, name}
                # We need to find the matching symbol and return its ID
                symbol_to_find = params.get("symbol", "").lower()
                found_id = None
                for item in raw_data:
                    if item.get("symbol", "").lower() == symbol_to_find:
                        found_id = item.get("id")
                        break
                if found_id:
                    final_result = {"id": found_id, "symbol": symbol_to_find}
                else:
                    error_msg = f"Symbol '{symbol_to_find}' not found in CoinGecko list."
                    logger.warning(error_msg)
                    await analytics_tracker.log_tool_usage(
                        tool_name=f"{domain}_{function_name}",
                        tool_params=params,
                        user_id=user_id,
                        success=False,
                        error_message=error_msg
                    )
                    return None
            else: # General single object mapping using data_map
                for mapped_key, original_key_path in data_map.items():
                    if isinstance(original_key_path, list):
                        mapped_data[mapped_key] = _get_nested_value(data_to_map, original_key_path)
                    elif isinstance(original_key_path, str) and '.' in original_key_path:
                        mapped_data[mapped_key] = _get_nested_value(data_to_map, original_key_path.split('.'))
                    else:
                        mapped_data[mapped_key] = data_to_map.get(original_key_path)
                final_result = mapped_data

            # Success logging is handled by LLMService's wrapped_tool_executor, not here.
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
                error_message=str(e) # Ensure error message is string
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
    async def crypto_get_crypto_price(self, crypto_id: str, vs_currencies: str = "usd", user_context: UserProfile = None) -> str:
        """
        Retrieves the current price of a cryptocurrency in one or more specified fiat currencies or other cryptocurrencies.
        Falls back to a generic message if API key is missing or API call fails.

        Args:
            crypto_id (str): The ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
            vs_currencies (str, optional): A comma-separated string of currency symbols to compare against (e.g., "usd", "eur", "jpy"). Defaults to "usd".
            user_context (UserProfile): The user's profile for RBAC checks and logging.

        Returns:
            str: A formatted string of the cryptocurrency price, or an error/fallback message.
        """
        if user_context is None: # For CLI testing without full UserProfile
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: crypto_get_crypto_price called for crypto_id: '{crypto_id}', vs_currencies: '{vs_currencies}' by user: {user_context.user_id}")

        # RBAC check is now performed by the LLMService's wrapped_tool_executor,
        # but a redundant check here can act as a safeguard or for direct calls.
        if not get_user_tier_capability(user_context.user_id, 'crypto_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to crypto tools is not enabled for your current tier."
        
        params = {"ids": crypto_id.lower(), "vs_currencies": vs_currencies.lower(), "include_market_cap": "true", "include_24hr_vol": "true", "include_24hr_change": "true", "include_last_updated_at": "true"}
        api_data = await self._make_dynamic_api_request("crypto", "get_crypto_price", params, user_context)

        if api_data:
            try:
                price = api_data.get("price")
                market_cap = api_data.get("market_cap")
                vol_24hr = api_data.get("vol_24hr")
                change_24hr = api_data.get("change_24hr")
                last_updated = api_data.get("last_updated")

                if price is not None:
                    response_str = f"Current price of {crypto_id.capitalize()}: {price} {vs_currencies.upper()}"
                    if market_cap is not None:
                        response_str += f"\n  Market Cap: {market_cap:,.2f} {vs_currencies.upper()}"
                    if vol_24hr is not None:
                        response_str += f"\n  24hr Volume: {vol_24hr:,.2f} {vs_currencies.upper()}"
                    if change_24hr is not None:
                        response_str += f"\n  24hr Change: {change_24hr:.2f}%"
                    if last_updated:
                        # CoinGecko's last_updated_at is a Unix timestamp
                        try:
                            last_updated_dt = datetime.fromtimestamp(last_updated, tz=timezone.utc)
                            response_str += f"\n  Last Updated: {last_updated_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                        except (ValueError, TypeError):
                            response_str += f"\n  Last Updated: {last_updated}" # Fallback if not a timestamp
                    return response_str
                else:
                    logger.warning(f"Live API data for {crypto_id} is incomplete. Raw: {api_data}")
                    return f"Could not retrieve complete live crypto price for {crypto_id.capitalize()}. Please try again or check the ID."
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing live crypto price data for {crypto_id}: {e}")
                return f"Error parsing live data for {crypto_id.capitalize()}. Please try again."
        else:
            return f"Could not retrieve live cryptocurrency price for {crypto_id.capitalize()}. The API call failed or returned no data. Please ensure your API key is valid and try again."


    @tool
    async def crypto_get_crypto_info(self, crypto_id: str, user_context: UserProfile = None) -> str:
        """
        Retrieves general information about a cryptocurrency, such as its description, genesis date, and market cap rank.
        Falls back to a generic message if API key is missing or API call fails.

        Args:
            crypto_id (str): The ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
            user_context (UserProfile): The user's profile for RBAC checks and logging.

        Returns:
            str: A formatted string of cryptocurrency information, or an error/fallback message.
        """
        if user_context is None: # For CLI testing without full UserProfile
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: crypto_get_crypto_info called for crypto_id: '{crypto_id}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'crypto_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to crypto tools is not enabled for your current tier."

        params = {"id": crypto_id.lower()} # 'id' is a path parameter, but we pass it in params for _make_dynamic_api_request
        api_data = await self._make_dynamic_api_request("crypto", "get_crypto_info", params, user_context)

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
                        f"Information for {name} ({symbol.upper() if symbol else 'N/A'}):\n"
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
                    return f"Could not retrieve complete live crypto information for {crypto_id.capitalize()}. Please try again or check the ID."
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing live crypto info data for {crypto_id}: {e}")
                return f"Error parsing live data for {crypto_id.capitalize()}. Please try again."
        else:
            return f"Could not retrieve live cryptocurrency information for {crypto_id.capitalize()}. The API call failed or returned no data. Please ensure your API key is valid and try again."


    @tool
    async def crypto_get_historical_crypto_price(self, crypto_id: str, date: str, vs_currency: str = "usd", user_context: UserProfile = None) -> str:
        """
        Retrieves the historical price of a cryptocurrency for a specific date.
        Dates can be in various formats (e.g., 'YYYY-MM-DD', 'MM/DD/YYYY', 'January 1, 2023').
        Falls back to a generic message if API key is missing or API call fails.

        Args:
            crypto_id (str): The ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
            date (str): The specific date for which to retrieve historical data.
            vs_currency (str, optional): The currency to compare against (e.g., "usd", "eur"). Defaults to "usd".
            user_context (UserProfile): The user's profile for RBAC checks and logging.

        Returns:
            str: A formatted string of historical cryptocurrency data, or an error/fallback message.
        """
        if user_context is None: # For CLI testing without full UserProfile
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: crypto_get_historical_crypto_price called for crypto_id: '{crypto_id}', date: '{date}', vs_currency: '{vs_currency}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'crypto_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to crypto tools is not enabled for your current tier."

        parsed_date = parse_date_to_yyyymmdd(date)
        if not parsed_date:
            return "Error: Could not parse the provided date. Please ensure the date is valid (e.g., YYYY-MM-DD)."

        params = {"id": crypto_id.lower(), "date": parsed_date, "vs_currency": vs_currency.lower()}
        api_data = await self._make_dynamic_api_request("crypto", "get_historical_crypto_price", params, user_context)

        if api_data:
            try:
                price = api_data.get("price")
                market_cap = api_data.get("market_cap")
                volume = api_data.get("volume")

                if price is not None:
                    response_str = (
                        f"Historical Price for {crypto_id.capitalize()} on {parsed_date}:\n"
                        f"  Price: {price} {vs_currency.upper()}\n"
                    )
                    if market_cap is not None:
                        response_str += f"  Market Cap: {market_cap:,.2f} {vs_currency.upper()}\n"
                    if volume is not None:
                        response_str += f"  24hr Volume: {volume:,.2f} {vs_currency.upper()}\n"
                    return response_str
                else:
                    logger.warning(f"Live API data for historical crypto price of {crypto_id} on {date} is incomplete. Raw: {api_data}")
                    return f"Could not retrieve complete live historical crypto price for {crypto_id.capitalize()} on {date}. Please try again or check the ID/date."
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing live historical crypto price data for {crypto_id} on {date}: {e}")
                return f"Error parsing live data for {crypto_id.capitalize()} on {date}. Please try again."
        else:
            return f"Could not retrieve live historical cryptocurrency price for {crypto_id.capitalize()} on {date}. The API call failed or returned no data. Please ensure your API key is valid and try again."

    @tool
    async def crypto_get_crypto_id_by_symbol(self, symbol: str, user_context: UserProfile = None) -> str:
        """
        Looks up the CoinGecko ID for a given cryptocurrency symbol.
        This ID is often required for other CoinGecko API calls.

        Args:
            symbol (str): The cryptocurrency symbol (e.g., "BTC", "ETH", "SOL").
            user_context (UserProfile): The user's profile for RBAC checks and logging.

        Returns:
            str: The CoinGecko ID for the symbol, or an error message if not found.
        """
        if user_context is None: # For CLI testing without full UserProfile
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: crypto_get_crypto_id_by_symbol called for symbol: '{symbol}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'crypto_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to crypto tools is not enabled for your current tier."
        
        # The /coins/list endpoint takes no params, but we pass symbol for internal logic
        params = {"symbol": symbol.lower()} 
        api_data = await self._make_dynamic_api_request("crypto", "get_crypto_id_by_symbol", params, user_context)

        if api_data and api_data.get("id"):
            return f"The CoinGecko ID for symbol {symbol.upper()} is: {api_data['id']}."
        else:
            return f"Could not find CoinGecko ID for symbol {symbol.upper()}. Please check the symbol and try again."


    # --- Existing Generic Tools (now methods of CryptoTools) ---
    # These functions wrap existing shared tools or DocumentTools methods.
    # They will pass the user_context down if the wrapped tool supports it.

    @tool
    async def crypto_search_web(self, query: str, user_context: UserProfile, max_chars: int = 2000) -> str:
        """
        Searches the web for cryptocurrency-related information using a smart search fallback mechanism.
        This tool wraps the generic `scrape_web` tool, providing a crypto-specific interface.
        
        Args:
            query (str): The crypto-related search query (e.g., "latest news on Ethereum 2.0", "how to buy Solana").
            user_context (UserProfile): The user's profile for RBAC checks and logging.
            max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
        
        Returns:
            str: A string containing relevant information from the web.
        """
        logger.info(f"Tool: crypto_search_web called with query: '{query}' for user: '{user_context.user_id}'")
        # scrape_web is a standalone function, ensure it handles its own RBAC/logging if applicable
        # For now, it's assumed LLMService wrapper handles its API limit check.
        return await scrape_web(query=query, user_token=user_context.user_id, max_chars=max_chars) # Pass user_token for scrape_web's internal logging

    @tool
    async def crypto_query_uploaded_docs(self, query: str, user_context: UserProfile, export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed cryptocurrency documents for a user using vector similarity search.
        This tool wraps the generic `DocumentTools.document_query_uploaded_docs` tool, fixing the section to "crypto".
        
        Args:
            query (str): The search query to find relevant crypto documents (e.g., "whitepaper for project X", "my crypto portfolio balance").
            user_context (UserProfile): The user's profile for RBAC checks and logging.
            export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
            k (int): The number of top relevant documents to retrieve. Defaults to 5.
        
        Returns:
            str: A string containing the combined content of the relevant document chunks,
                 or a message indicating no data/results found, or the export path if exported.
        """
        logger.info(f"Tool: crypto_query_uploaded_docs called with query: '{query}' for user: '{user_context.user_id}'")
        if not self.document_tools:
            return "Error: Document tools are not initialized. Cannot query uploaded documents."
        
        # Call the actual document_query_uploaded_docs from the DocumentTools instance
        return await self.document_tools.document_query_uploaded_docs(
            query=query,
            user_context=user_context, # Pass user_context directly
            section="crypto", # Specify the section for crypto documents
            export=export,
            k=k
        )

    @tool
    async def crypto_summarize_document_by_path(self, file_path_str: str, user_context: UserProfile) -> str:
        """
        Summarizes a document related to cryptocurrency or blockchain located at the given file path.
        The file path should be accessible by the system (e.g., in the 'uploads' directory).
        This tool wraps the generic `DocumentTools.document_summarize_document_by_path` tool.
        
        Args:
            file_path_str (str): The full path to the document file to be summarized.
                                Example: "uploads/default/crypto/bitcoin_whitepaper.pdf"
            user_context (UserProfile): The user's profile for RBAC checks and logging.
        
        Returns:
            str: A concise summary of the document content.
        """
        logger.info(f"Tool: crypto_summarize_document_by_path called for file: '{file_path_str}' by user: '{user_context.user_id}'")
        if not self.document_tools:
            return "Error: Document tools are not initialized. Cannot summarize documents."

        # Call the actual document_summarize_document_by_path from the DocumentTools instance
        return await self.document_tools.document_summarize_document_by_path(
            file_path_str=file_path_str,
            user_context=user_context # Pass user_context directly
        )


# CLI Test (optional)
if __name__ == "__main__":
    import asyncio
    from unittest.mock import MagicMock, AsyncMock, patch, ANY
    import shutil
    import os
    import sys
    from shared_tools.vector_utils import BASE_VECTOR_DIR # For cleanup
    from database.firestore_manager import FirestoreManager # For mocking
    from shared_tools.cloud_storage_utils import CloudStorageUtilsWrapper # For mocking
    from shared_tools.vector_utils import VectorUtilsWrapper # For mocking
    from domain_tools.document_tools.document_tool import DocumentTools # For mocking
    from backend.models.user_models import UserProfile # For mock user_context
    from langchain_core.messages import HumanMessage, AIMessage # For mocking LLM in summarizer

    logging.basicConfig(level=logging.INFO)

    # Mock UserProfile for testing
    mock_user_pro_profile = UserProfile(user_id="mock_pro_token", username="ProUser", email="pro@example.com", tier="pro", roles=["user"])
    mock_user_free_profile = UserProfile(user_id="mock_free_token", username="FreeUser", email="free@example.com", tier="free", roles=["user"])
    mock_user_premium_profile = UserProfile(user_id="mock_premium_token", username="PremiumUser", email="premium@example.com", tier="premium", roles=["user"])
    mock_user_admin_profile = UserProfile(user_id="mock_admin_token", username="AdminUser", email="admin@example.com", tier="admin", roles=["user", "admin"])


    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.coingecko_api_key = "MOCK_COINGECKO_API_KEY_LIVE"
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
                    'crypto': 'coingecko',
                    'web_search': 'serpapi',
                    'document_summarization_llm': 'openai'
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
                            },
                             "get_crypto_id_by_symbol": {
                                "endpoint": "/coins/list", # Endpoint for listing all coins
                                "required_params": [], # No required params for the list endpoint itself
                                "optional_params": [],
                                "response_path": [], # Root is a list, special handling in _make_dynamic_api_request
                                "data_map": {} # Special handling in _make_dynamic_api_request
                            }
                        }
                    }
                },
                "web_search": { # Mock for web search (SerpAPI)
                    "serpapi": {
                        "base_url": "https://serpapi.com/search",
                        "api_key_name": "serpapi_api_key",
                        "api_key_param_name": "api_key",
                        "functions": {
                            "scrape_web": { # This function name should match the tool name
                                "required_params": ["q"],
                                "optional_params": ["engine"],
                                "response_path": ["organic_results"], # Example path for search results
                                "data_map": { # Simplified mapping for search results
                                    "title": "title",
                                    "link": "link",
                                    "snippet": "snippet"
                                }
                            }
                        }
                    }
                },
                "document_summarization_llm": { # Mock for summarization LLM
                    "openai": {
                        "base_url": "https://api.openai.com/v1/chat/completions",
                        "api_key_name": "openai_api_key",
                        "functions": {
                            "summarize_document": { # This function name should match the tool name
                                "endpoint": "", # No specific endpoint for chat completions
                                "required_params": [],
                                "optional_params": [],
                                "response_path": ["choices", 0, "message", "content"],
                                "data_map": {} # No specific mapping needed for direct content
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
    # This mock is for the standalone get_user_tier_capability function
    # which is now imported directly by tools.
    class MockUserManager:
        _mock_users = {
            "mock_free_token": {"user_id": "mock_free_token", "username": "FreeUser", "email": "free@example.com", "tier": "free", "roles": ["user"]},
            "mock_pro_token": {"user_id": "mock_pro_token", "username": "ProUser", "email": "pro@example.com", "tier": "pro", "roles": ["user"]},
            "mock_premium_token": {"user_id": "mock_premium_token", "username": "PremiumUser", "email": "premium@example.com", "tier": "premium", "roles": ["user"]},
            "mock_admin_token": {"user_id": "mock_admin_token", "username": "AdminUser", "email": "admin@example.com", "tier": "admin", "roles": ["user", "admin"]},
        }
        _rbac_capabilities = { # This now mirrors the _RBAC_CAPABILITIES_CONFIG in utils/user_manager.py
            'capabilities': {
                'crypto_tool_access': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'document_upload_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
                'document_query_enabled': { # Added for document tool
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'web_search_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
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

        def get_user_tier_capability(self, user_id: str, capability_key: str, default_value: Any = None, user_tier: Optional[str] = None, user_roles: Optional[List[str]] = None) -> Any:
            # If user_tier/user_roles are provided, use them directly (from UserProfile)
            # Otherwise, try to look up from _mock_users
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
    
    # Patch config_manager and user_manager in their respective modules
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager # Also patch the class if needed by other modules
    
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

        def mock_requests_get_dynamic(url, params=None, headers=None, timeout=None):
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
                                "last_updated_at": int(datetime.now(timezone.utc).timestamp())
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
                                "last_updated_at": int(datetime.now(timezone.utc).timestamp())
                            }
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {} # Empty for unknown coins
                        return mock_response
                elif "/coins/list" in url: # get_crypto_id_by_symbol
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = [
                        {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"},
                        {"id": "ethereum", "symbol": "eth", "name": "Ethereum"},
                        {"id": "solana", "symbol": "sol", "name": "Solana"},
                        {"id": "dogecoin", "symbol": "doge", "name": "Dogecoin"},
                    ]
                    return mock_response
                elif "/coins/" in url and "/history" not in url: # get_crypto_info
                    crypto_id_from_url = url.split("/coins/")[1].split("/")[0].lower()
                    if crypto_id_from_url == "bitcoin":
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "id": "bitcoin", "symbol": "btc", "name": "Bitcoin",
                            "description": {"en": "Bitcoin is a decentralized digital currency, without a central bank or single administrator, that can be sent from user to user on the peer-to-peer bitcoin network without the need for intermediaries."},
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
                    if crypto_id_from_url == "bitcoin" and date == (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d"):
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
            
            # Simulate scrape_web's internal requests.get if needed (SerpAPI)
            if "serpapi.com/search" in url:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "organic_results": [
                        {"title": "Mock Search Result 1", "link": "http://example.com/1", "snippet": f"Snippet for {params.get('q', 'crypto')} result 1."},
                        {"title": "Mock Search Result 2", "link": "http://example.com/2", "snippet": f"Snippet for {params.get('q', 'crypto')} result 2."}
                    ]
                }
                return mock_response

            # Mock LLM for summarizer (if it uses requests.post for an API)
            if "api.openai.com/v1/chat/completions" in url:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Mocked LLM summary content."}}]
                }
                return mock_response

            return original_requests_get(url, params=params, headers=headers, timeout=timeout)

        requests.get = MagicMock(side_effect=mock_requests_get_dynamic)
        requests.post = MagicMock(side_effect=mock_requests_get_dynamic) # For OpenAI chat completions

        # Mock FirestoreManager, CloudStorageUtilsWrapper, VectorUtilsWrapper, DocumentTools for init
        mock_firestore_manager = MagicMock(spec=FirestoreManager)
        mock_cloud_storage_utils = MagicMock(spec=CloudStorageUtilsWrapper)
        mock_vector_utils = MagicMock(spec=VectorUtilsWrapper)
        
        # Create a mock DocumentTools instance
        mock_document_tools = MagicMock(spec=DocumentTools)
        mock_document_tools.document_query_uploaded_docs = AsyncMock(return_value="Mocked document query results for crypto.")
        mock_document_tools.document_summarize_document_by_path = AsyncMock(return_value="Mocked summary of dummy_file.txt")

        # Instantiate CryptoTools with mocks
        crypto_tools_instance = CryptoTools(
            config_manager=sys.modules['config.config_manager'].config_manager,
            log_event=analytics_tracker.log_event, # Pass the actual (mocked) log_event
            document_tools=mock_document_tools
        )

        async def run_crypto_tests(crypto_tools_instance):
            print("\n--- Testing crypto_tool functions with Live API Simulation and Analytics ---")

            # Test 1: crypto_get_crypto_price (success)
            print("\n--- Test 1: crypto_get_crypto_price (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock() # Reset mock call count
            result_crypto_price = await crypto_tools_instance.crypto_get_crypto_price("bitcoin", user_context=mock_user_pro_profile)
            print(f"Crypto Price: {result_crypto_price}")
            assert "Current price of Bitcoin: 65000.0 USD" in result_crypto_price
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called() # Analytics should NOT be logged for success here
            print("Test 1 Passed.")

            # Test 2: crypto_get_crypto_info (API failure - coin not found)
            print("\n--- Test 2: crypto_get_crypto_info (API Failure - Coin Not Found) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_crypto_info = await crypto_tools_instance.crypto_get_crypto_info("nonexistentcoin", user_context=mock_user_pro_profile)
            print(f"Crypto Info (API Error): {result_crypto_info}")
            assert "Could not retrieve complete live crypto information for Nonexistentcoin." in result_crypto_info
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once() # Analytics should be logged for failure
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "crypto_get_crypto_info"
            assert logged_data["success"] is False
            assert "coin not found" in logged_data["error_message"]
            print("Test 2 Passed.")

            # Test 3: crypto_get_historical_crypto_price (RBAC denied)
            print("\n--- Test 3: crypto_get_historical_crypto_price (RBAC Denied) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_historical_rbac_denied = await crypto_tools_instance.crypto_get_historical_crypto_price("ethereum", "2023-01-01", user_context=mock_user_free_profile)
            print(f"Historical Crypto Price (Free User, RBAC Denied): {result_historical_rbac_denied}")
            assert "Error: Access to crypto tools is not enabled for your current tier." in result_historical_rbac_denied
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called() # RBAC check happens before _make_dynamic_api_request
            print("Test 3 Passed.")

            # Test 4: crypto_get_crypto_id_by_symbol (success)
            print("\n--- Test 4: crypto_get_crypto_id_by_symbol (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_id = await crypto_tools_instance.crypto_get_crypto_id_by_symbol("btc", user_context=mock_user_pro_profile)
            print(f"Crypto ID: {result_id}")
            assert "The CoinGecko ID for symbol BTC is: bitcoin." in result_id
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called() # Analytics should NOT be logged for success here
            print("Test 4 Passed.")

            # Test 5: crypto_search_web (generic tool)
            print("\n--- Test 5: crypto_search_web (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_web_search = await crypto_tools_instance.crypto_search_web("best crypto wallets", user_context=mock_user_pro_profile)
            print(f"Web Search Result: {result_web_search[:100]}...")
            assert "Search results for best crypto wallets" in result_web_search
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called() # Analytics for scrape_web is handled by its own internal logging or LLMService wrapper
            print("Test 5 Passed.")

            # Test 6: crypto_query_uploaded_docs (generic tool via DocumentTools)
            print("\n--- Test 6: crypto_query_uploaded_docs (Generic Tool via DocumentTools) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_doc_query = await crypto_tools_instance.crypto_query_uploaded_docs("whitepaper details", user_context=mock_user_pro_profile)
            print(f"Document Query Result: {result_doc_query}")
            assert "Mocked document query results for crypto." in result_doc_query
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once() # Analytics logged by DocumentTools mock
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "document_query_uploaded_docs"
            assert logged_data["success"] is True
            print("Test 6 Passed.")

            # Test 7: crypto_summarize_document_by_path (generic tool via DocumentTools)
            print("\n--- Test 7: crypto_summarize_document_by_path (Generic Tool via DocumentTools) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            # Create a dummy file for summarization test
            test_user_pro_dir = Path("uploads") / mock_user_pro_profile.user_id
            dummy_file_path = test_user_pro_dir / "crypto" / "dummy_whitepaper.txt"
            dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_file_path.write_text("This is a dummy whitepaper content for testing summarization.")

            result_summarize = await crypto_tools_instance.crypto_summarize_document_by_path(str(dummy_file_path), user_context=mock_user_pro_profile)
            print(f"Summarize Result: {result_summarize}")
            assert "Mocked summary of dummy_file.txt" in result_summarize # Check for mock summary from DocumentTools
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once() # Now logged by DocumentTools mock
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "document_summarize_document_by_path"
            assert logged_data["success"] is True
            print("Test 7 Passed.")

            print("\nAll crypto_tool tests with live API simulation and analytics considerations completed.")

        # Ensure tests are only run when the script is executed directly
        if __name__ == "__main__":
            asyncio.run(run_crypto_tests(crypto_tools_instance))

        # Restore original requests.get and requests.post
        requests.get = original_requests_get
        requests.post = original_requests_get # Restore to original get if post was patched to get

        # Clean up dummy files and directories
        test_user_dirs = [Path("uploads") / mock_user_pro_profile.user_id, BASE_VECTOR_DIR / mock_user_pro_profile.user_id]
        for d in test_user_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print(f"Cleaned up {d}")

