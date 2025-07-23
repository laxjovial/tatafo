# domain_tools/crypto_tools/crypto_tool.py

import logging
from typing import Optional, Dict, Any
from langchain_core.tools import tool

# Import the new flexible API request function
from shared_tools.historical_data_tool import make_api_request

# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability

# Import UserProfile for type hinting
from backend.models.user_models import UserProfile

# Import config_manager and analytics_tracker for direct use in standalone tool functions
from config.config_manager import config_manager
from utils.analytics_tracker import log_event # Assuming log_event is a module-level function or singleton

logger = logging.getLogger(__name__)

# --- Standalone Tool Functions (Moved from CryptoTools class) ---

@tool
async def get_crypto_price(
    crypto_id: str,
    vs_currencies: str = "usd",
    user_context: Optional[UserProfile] = None,
    provider: str = "coingecko",
    user_api_keys: list = []
) -> str:
    """
    Retrieves the current price of a cryptocurrency.

    This tool fetches real-time cryptocurrency prices from specified providers.
    It supports multiple `vs_currencies` (e.g., "usd", "eur", "jpy") as a comma-separated string.
    The `provider` argument allows selection between supported API providers
    (e.g., "coingecko", "alphavantage").
    Requires 'crypto_price_query_enabled' capability.

    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
        vs_currencies (str, optional): A comma-separated string of currencies
                                       to compare against (e.g., "usd,eur"). Defaults to "usd".
        user_context (UserProfile, optional): The user's profile for RBAC checks. Defaults to None.
        provider (str, optional): The API provider to use. Defaults to "coingecko".
        user_api_keys (list, optional): List of user-provided API keys (e.g., from Streamlit secrets).

    Returns:
        str: A JSON string containing the cryptocurrency price information,
             or an error message.
    """
    if user_context is None:
        user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

    logger.info(f"Tool: get_crypto_price called for crypto_id: '{crypto_id}', vs_currencies: '{vs_currencies}', provider: '{provider}', user: '{user_context.user_id}'")

    # RBAC Check: Crypto price query access
    if not get_user_tier_capability(user_context.user_id, 'crypto_price_query_enabled', False):
        log_event(user_context.user_id, "tool_usage", "permission_denied",
                  {"tool_name": "get_crypto_price", "crypto_id": crypto_id, "provider": provider})
        return "Error: Cryptocurrency price querying is not enabled for your current tier. Please upgrade your plan."

    # Get API provider configuration from config_manager
    api_config = config_manager.get_api_provider_config("crypto_prices", provider)
    if not api_config:
        log_event(user_context.user_id, "tool_usage", "error",
                  {"tool_name": "get_crypto_price", "crypto_id": crypto_id, "provider": provider, "error": f"API provider '{provider}' not configured."})
        return f"Error: API provider '{provider}' is not configured for crypto price lookup."

    endpoint = api_config.get("endpoints", {}).get("get_price")
    base_url = api_config.get("base_url")
    # For CoinGecko, it's typically 'simple/price'
    # For Alpha Vantage, it's 'query' and function=CURRENCY_EXCHANGE_RATE

    if not endpoint or not base_url:
        log_event(user_context.user_id, "tool_usage", "error",
                  {"tool_name": "get_crypto_price", "crypto_id": crypto_id, "provider": provider, "error": "Missing endpoint or base_url in config."})
        return f"Error: Configuration for '{provider}' is incomplete."

    headers = api_config.get("headers", {})
    api_key_name = api_config.get("api_key_name")

    # Prioritize user_api_keys, then config_manager secrets, then environment variables
    api_key = next((key_val for key_name, key_val in user_api_keys if key_name == api_key_name), None)
    if not api_key and api_key_name:
        api_key = config_manager.get_secret(api_key_name)

    params = {}
    if provider == "coingecko":
        params = {
            "ids": crypto_id,
            "vs_currencies": vs_currencies
        }
    elif provider == "alphavantage":
        if api_key:
            params["apikey"] = api_key
        params.update({
            "function": "CURRENCY_EXCHANGE_RATE", # This is a placeholder, Alpha Vantage's crypto endpoint is different
            "from_currency": crypto_id.upper(), # Assuming crypto_id can be mapped to symbol
            "to_currency": vs_currencies.split(',')[0].upper() # Only supports one for this function
        })
        # Alpha Vantage uses 'query' as its base endpoint, function is a param
        if endpoint == "query":
            full_url = f"{base_url}"
        else:
            full_url = f"{base_url}/{endpoint}"
    else:
        # Default parameter handling for other providers
        params.update({"id": crypto_id, "vs_currencies": vs_currencies})
        if api_key and api_key_name:
            params[api_key_name] = api_key
        full_url = f"{base_url}/{endpoint}"

    try:
        response_data = await make_api_request(
            base_url=base_url,
            endpoint=endpoint,
            params=params,
            headers=headers,
            api_key=api_key,
            api_key_name=api_key_name,
            provider=provider
        )

        if response_data:
            # Basic parsing, depends on provider's JSON structure
            if provider == "coingecko" and crypto_id in response_data:
                price_info = response_data[crypto_id]
                result_str = f"Current price of {crypto_id}: "
                for currency, price in price_info.items():
                    result_str += f"{price} {currency.upper()}, "
                result_str = result_str.rstrip(', ') + "."
                log_event(user_context.user_id, "tool_usage", "success",
                          {"tool_name": "get_crypto_price", "crypto_id": crypto_id, "provider": provider, "price_data": price_info})
                return result_str
            elif provider == "alphavantage" and "Realtime Currency Exchange Rate" in response_data:
                exchange_rate_info = response_data["Realtime Currency Exchange Rate"]
                from_currency = exchange_rate_info.get("2. From_Currency Code")
                to_currency = exchange_rate_info.get("4. To_Currency Code")
                exchange_rate = exchange_rate_info.get("5. Exchange Rate")
                result_str = f"Alpha Vantage: 1 {from_currency} = {exchange_rate} {to_currency}."
                log_event(user_context.user_id, "tool_usage", "success",
                          {"tool_name": "get_crypto_price", "crypto_id": crypto_id, "provider": provider, "price_data": exchange_rate_info})
                return result_str
            else:
                # Fallback for other providers or unexpected structure
                log_event(user_context.user_id, "tool_usage", "warning",
                          {"tool_name": "get_crypto_price", "crypto_id": crypto_id, "provider": provider, "message": "Unexpected response structure."})
                return f"Successfully fetched data from {provider}, but couldn't parse price: {response_data}"
        else:
            log_event(user_context.user_id, "tool_usage", "no_data",
                      {"tool_name": "get_crypto_price", "crypto_id": crypto_id, "provider": provider})
            return f"No price data found for '{crypto_id}' from '{provider}'."
    except Exception as e:
        logger.error(f"Error fetching crypto price for {crypto_id} from {provider}: {e}", exc_info=True)
        log_event(user_context.user_id, "tool_usage", "error",
                  {"tool_name": "get_crypto_price", "crypto_id": crypto_id, "provider": provider, "error": str(e)})
        return f"Error fetching cryptocurrency price: {e}"


# --- CryptoTools Class (for other methods that might need internal state) ---
class CryptoTools:
    """
    A collection of tools for cryptocurrency-related operations, including historical data
    and general information. This class is now primarily for methods that require
    initialization with specific managers (firestore, document_tools).
    Standalone price lookup is moved to a top-level function.
    """
    def __init__(self, config_manager, firestore_manager, log_event, document_tools):
        self.config_manager = config_manager
        self.firestore_manager = firestore_manager
        self.log_event = log_event
        self.document_tools = document_tools

    @tool
    async def crypto_get_historical_crypto_prices(
        self,
        crypto_id: str,
        vs_currency: str = "usd",
        days: int = 7,
        interval: Optional[str] = None, # 'daily', 'hourly' etc. depends on provider
        user_context: Optional[UserProfile] = None,
        provider: str = "coingecko",
        user_api_keys: list = []
    ) -> str:
        """
        Retrieves historical price data for a cryptocurrency.

        This tool fetches historical cryptocurrency price data for a specified number of days.
        It supports different intervals based on the provider.
        Requires 'crypto_historical_query_enabled' capability.

        Args:
            crypto_id (str): The ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
            vs_currency (str, optional): The currency to compare against (e.g., "usd"). Defaults to "usd".
            days (int, optional): Number of days for historical data. Defaults to 7.
            interval (str, optional): The interval of the data (e.g., "daily", "hourly").
                                      Availability depends on the provider. Defaults to None.
            user_context (UserProfile, optional): The user's profile for RBAC checks. Defaults to None.
            provider (str, optional): The API provider to use. Defaults to "coingecko".
            user_api_keys (list, optional): List of user-provided API keys (e.g., from Streamlit secrets).

        Returns:
            str: A JSON string containing the historical price data,
                 or an error message.
        """
        if user_context is None:
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: crypto_get_historical_crypto_prices called for crypto_id: '{crypto_id}', days: {days}, provider: '{provider}', user: '{user_context.user_id}'")

        # RBAC Check: Historical crypto data access
        if not get_user_tier_capability(user_context.user_id, 'crypto_historical_query_enabled', False):
            self.log_event(user_context.user_id, "tool_usage", "permission_denied",
                           {"tool_name": "crypto_get_historical_crypto_prices", "crypto_id": crypto_id, "provider": provider})
            return "Error: Historical cryptocurrency data querying is not enabled for your current tier. Please upgrade your plan."

        api_config = self.config_manager.get_api_provider_config("historical_crypto", provider)
        if not api_config:
            self.log_event(user_context.user_id, "tool_usage", "error",
                           {"tool_name": "crypto_get_historical_crypto_prices", "crypto_id": crypto_id, "provider": provider, "error": f"API provider '{provider}' not configured for historical crypto data."})
            return f"Error: API provider '{provider}' is not configured for historical crypto data."

        endpoint = api_config.get("endpoints", {}).get("get_historical_prices")
        base_url = api_config.get("base_url")

        if not endpoint or not base_url:
            self.log_event(user_context.user_id, "tool_usage", "error",
                           {"tool_name": "crypto_get_historical_crypto_prices", "crypto_id": crypto_id, "provider": provider, "error": "Missing endpoint or base_url in config."})
            return f"Error: Configuration for '{provider}' historical data is incomplete."

        headers = api_config.get("headers", {})
        api_key_name = api_config.get("api_key_name")

        api_key = next((key_val for key_name, key_val in user_api_keys if key_name == api_key_name), None)
        if not api_key and api_key_name:
            api_key = self.config_manager.get_secret(api_key_name)

        params = {}
        if provider == "coingecko":
            params = {
                "vs_currency": vs_currency,
                "days": days,
                "interval": interval # 'daily', 'hourly' depending on days
            }
            full_url = f"{base_url}/coins/{crypto_id}/{endpoint}"
        elif provider == "alphavantage":
            # Alpha Vantage historical crypto is more complex, might need different function/params
            # This is a placeholder for Alpha Vantage specific historical crypto parameters
            if api_key:
                params["apikey"] = api_key
            params.update({
                "function": "DIGITAL_CURRENCY_DAILY", # Example
                "symbol": crypto_id.upper(), # Map crypto_id to symbol
                "market": vs_currency.upper() # Map vs_currency to market
            })
            full_url = f"{base_url}" # Alpha Vantage uses 'query' as base, function in params
        else:
            # Default parameter handling for other providers
            params.update({"id": crypto_id, "vs_currency": vs_currency, "days": days})
            if interval:
                params["interval"] = interval
            if api_key and api_key_name:
                params[api_key_name] = api_key
            full_url = f"{base_url}/{endpoint}"


        try:
            response_data = await make_api_request(
                base_url=base_url,
                endpoint=endpoint,
                params=params,
                headers=headers,
                api_key=api_key,
                api_key_name=api_key_name,
                provider=provider,
                full_url=full_url # Pass the constructed full_url for CoinGecko like direct endpoints
            )

            if response_data:
                # Basic parsing, depends on provider's JSON structure
                if provider == "coingecko" and "prices" in response_data:
                    # 'prices' is a list of [timestamp, price]
                    historical_data = response_data["prices"]
                    result_str = f"Historical prices for {crypto_id} ({vs_currency}) over {days} days: {len(historical_data)} data points."
                    # You might want to format this more nicely for display or export
                    self.log_event(user_context.user_id, "tool_usage", "success",
                                   {"tool_name": "crypto_get_historical_crypto_prices", "crypto_id": crypto_id, "provider": provider, "data_points": len(historical_data)})
                    return result_str + f"\\nRaw data preview: {historical_data[:5]}..."
                elif provider == "alphavantage" and "Time Series (Digital Currency Daily)" in response_data:
                    # Alpha Vantage specific parsing
                    time_series = response_data["Time Series (Digital Currency Daily)"]
                    result_str = f"Alpha Vantage historical data for {crypto_id} ({vs_currency}): {len(time_series)} data points."
                    self.log_event(user_context.user_id, "tool_usage", "success",
                                   {"tool_name": "crypto_get_historical_crypto_prices", "crypto_id": crypto_id, "provider": provider, "data_points": len(time_series)})
                    return result_str + f"\\nRaw data preview: {list(time_series.items())[:5]}..."
                else:
                    self.log_event(user_context.user_id, "tool_usage", "warning",
                                   {"tool_name": "crypto_get_historical_crypto_prices", "crypto_id": crypto_id, "provider": provider, "message": "Unexpected response structure for historical data."})
                    return f"Successfully fetched data from {provider}, but couldn't parse historical data: {response_data}"
            else:
                self.log_event(user_context.user_id, "tool_usage", "no_data",
                               {"tool_name": "crypto_get_historical_crypto_prices", "crypto_id": crypto_id, "provider": provider})
                return f"No historical data found for '{crypto_id}' from '{provider}'."
        except Exception as e:
            logger.error(f"Error fetching historical crypto price for {crypto_id} from {provider}: {e}", exc_info=True)
            self.log_event(user_context.user_id, "tool_usage", "error",
                           {"tool_name": "crypto_get_historical_crypto_prices", "crypto_id": crypto_id, "provider": provider, "error": str(e)})
            return f"Error fetching historical cryptocurrency data: {e}"

    @tool
    async def crypto_get_market_chart(
        self,
        crypto_id: str,
        vs_currency: str = "usd",
        days: int = 7,
        user_context: Optional[UserProfile] = None,
        provider: str = "coingecko",
        user_api_keys: list = []
    ) -> str:
        """
        Retrieves market chart data (price, market cap, total volumes) for a cryptocurrency.

        This tool provides comprehensive market data including price, market capitalization,
        and 24h total volumes for a given cryptocurrency and time period.
        Requires 'crypto_market_chart_query_enabled' capability.

        Args:
            crypto_id (str): The ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
            vs_currency (str, optional): The currency to compare against (e.g., "usd"). Defaults to "usd".
            days (int, optional): Number of days for data. Defaults to 7.
            user_context (UserProfile, optional): The user's profile for RBAC checks. Defaults to None.
            provider (str, optional): The API provider to use. Defaults to "coingecko".
            user_api_keys (list, optional): List of user-provided API keys (e.g., from Streamlit secrets).

        Returns:
            str: A JSON string containing the market chart data, or an error message.
        """
        if user_context is None:
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: crypto_get_market_chart called for crypto_id: '{crypto_id}', days: {days}, provider: '{provider}', user: '{user_context.user_id}'")

        # RBAC Check: Market chart access
        if not get_user_tier_capability(user_context.user_id, 'crypto_market_chart_query_enabled', False):
            self.log_event(user_context.user_id, "tool_usage", "permission_denied",
                           {"tool_name": "crypto_get_market_chart", "crypto_id": crypto_id, "provider": provider})
            return "Error: Cryptocurrency market chart querying is not enabled for your current tier. Please upgrade your plan."

        api_config = self.config_manager.get_api_provider_config("market_chart_crypto", provider)
        if not api_config:
            self.log_event(user_context.user_id, "tool_usage", "error",
                           {"tool_name": "crypto_get_market_chart", "crypto_id": crypto_id, "provider": provider, "error": f"API provider '{provider}' not configured for market chart data."})
            return f"Error: API provider '{provider}' is not configured for market chart data."

        endpoint = api_config.get("endpoints", {}).get("get_market_chart")
        base_url = api_config.get("base_url")

        if not endpoint or not base_url:
            self.log_event(user_context.user_id, "tool_usage", "error",
                           {"tool_name": "crypto_get_market_chart", "crypto_id": crypto_id, "provider": provider, "error": "Missing endpoint or base_url in config."})
            return f"Error: Configuration for '{provider}' market chart is incomplete."

        headers = api_config.get("headers", {})
        api_key_name = api_config.get("api_key_name")

        api_key = next((key_val for key_name, key_val in user_api_keys if key_name == api_key_name), None)
        if not api_key and api_key_name:
            api_key = self.config_manager.get_secret(api_key_name)

        params = {
            "vs_currency": vs_currency,
            "days": days
        }
        full_url = f"{base_url}/coins/{crypto_id}/{endpoint}"

        try:
            response_data = await make_api_request(
                base_url=base_url,
                endpoint=endpoint,
                params=params,
                headers=headers,
                api_key=api_key,
                api_key_name=api_key_name,
                provider=provider,
                full_url=full_url
            )

            if response_data:
                # CoinGecko's market_chart returns 'prices', 'market_caps', 'total_volumes'
                result_str = f"Market chart data for {crypto_id} ({vs_currency}) over {days} days."
                if "prices" in response_data:
                    result_str += f"\\nPrices data points: {len(response_data['prices'])}"
                if "market_caps" in response_data:
                    result_str += f", Market Cap data points: {len(response_data['market_caps'])}"
                if "total_volumes" in response_data:
                    result_str += f", Total Volumes data points: {len(response_data['total_volumes'])}"

                self.log_event(user_context.user_id, "tool_usage", "success",
                               {"tool_name": "crypto_get_market_chart", "crypto_id": crypto_id, "provider": provider, "data_summary": result_str})
                return result_str + f"\\nRaw data preview (prices): {response_data.get('prices', [])[:5]}..."
            else:
                self.log_event(user_context.user_id, "tool_usage", "no_data",
                               {"tool_name": "crypto_get_market_chart", "crypto_id": crypto_id, "provider": provider})
                return f"No market chart data found for '{crypto_id}' from '{provider}'."
        except Exception as e:
            logger.error(f"Error fetching market chart for {crypto_id} from {provider}: {e}", exc_info=True)
            self.log_event(user_context.user_id, "tool_usage", "error",
                           {"tool_name": "crypto_get_market_chart", "crypto_id": crypto_id, "provider": provider, "error": str(e)})
            return f"Error fetching market chart data: {e}"


    # Mock for document_tools.summarize_document_by_path
    @tool
    async def document_summarize_document_by_path(self, file_path: str, user_context: Optional[UserProfile] = None) -> str:
        """
        Summarizes the content of a document located at the given file path.
        This is a mock implementation for testing within CryptoTools.
        """
        if user_context is None:
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])
        
        logger.info(f"Mock Tool: document_summarize_document_by_path called for path: '{file_path}' by user: '{user_context.user_id}'")
        
        # Log the event through the instance's log_event
        self.log_event(user_context.user_id, "tool_usage", "success",
                       {"tool_name": "document_summarize_document_by_path", "file_path": file_path, "status": "mocked"})
        
        return f"Mocked summary of {file_path}: This document appears to be a financial report detailing market trends and investment opportunities in the crypto sector. Key points include..."


# CLI Test (optional)
if __name__ == "__main__":
    import asyncio
    import sys
    from unittest.mock import MagicMock, AsyncMock, patch
    from pathlib import Path
    import os
    import shutil

    logging.basicConfig(level=logging.INFO)

    # Mock dependencies for CLI testing
    class MockConfigManager:
        def __init__(self):
            self._secrets = {}
            self._api_providers_data = {
                "crypto_prices": {
                    "coingecko": {
                        "base_url": "https://api.coingecko.com/api/v3",
                        "endpoints": {"get_price": "simple/price"},
                        "api_key_name": None # CoinGecko V3 simple endpoints often don't require keys
                    },
                    "alphavantage": {
                        "base_url": "https://www.alphavantage.co",
                        "endpoints": {"get_price": "query"}, # Alpha Vantage uses 'query'
                        "api_key_name": "ALPHAVANTAGE_API_KEY"
                    }
                },
                "historical_crypto": {
                    "coingecko": {
                        "base_url": "https://api.coingecko.com/api/v3",
                        "endpoints": {"get_historical_prices": "market_chart"}, # CoinGecko uses market_chart for historical
                        "api_key_name": None
                    }
                },
                "market_chart_crypto": {
                    "coingecko": {
                        "base_url": "https://api.coingecko.com/api/v3",
                        "endpoints": {"get_market_chart": "market_chart"},
                        "api_key_name": None
                    }
                }
            }

        def get_secret(self, key: str) -> Optional[str]:
            return self._secrets.get(key)

        def set_secret(self, key: str, value: str):
            self._secrets[key] = value

        def get_api_provider_config(self, category: str, provider: str) -> Optional[Dict[str, Any]]:
            return self._api_providers_data.get(category, {}).get(provider)

    class MockFirestoreManager:
        def __init__(self):
            logger.info("Using MockFirestoreManager for CryptoTools tests")
            self._data = {}

        async def get_document_by_id(self, collection_path, document_id):
            return self._data.get(collection_path, {}).get(document_id)

        async def set_document(self, collection_path, document_id, data):
            if collection_path not in self._data:
                self._data[collection_path] = {}
            self._data[collection_path][document_id] = data
            return {"status": "success", "id": document_id}
        
        def collection(self, collection_path):
            # Allow chaining for mocks, just return self or a sub-mock
            return MagicMock(add=AsyncMock(return_value=MagicMock(id="mock_doc_id")), 
                             document=MagicMock(return_value=MagicMock(set_document=AsyncMock())))


    # Mock user_manager for RBAC
    class MockUserManager:
        _rbac_capabilities = {
            'capabilities': {
                'crypto_price_query_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
                'crypto_historical_query_enabled': {'default': False, 'roles': {'premium': True, 'admin': True}},
                'crypto_market_chart_query_enabled': {'default': False, 'roles': {'premium': True, 'admin': True}},
                'document_summarize_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
            }
        }
        def get_user_tier_capability(self, user_token: Optional[str], capability_key: str, default_value: Any = None, user_tier: Optional[str] = None, user_roles: Optional[List[str]] = None) -> Any:
            user_info = {"tier": user_tier if user_tier else "free", "roles": user_roles if user_roles else []}
            if user_token == "test_user_pro": user_info = {"tier": "pro", "roles": ["user"]}
            if user_token == "test_user_premium": user_info = {"tier": "premium", "roles": ["user"]}
            if user_token == "test_user_free": user_info = {"tier": "free", "roles": ["user"]}
            if user_token == "test_user_admin": user_info = {"tier": "admin", "roles": ["user", "admin"]}

            if "admin" in user_info["roles"]:
                return True

            capability_config = self._rbac_capabilities.get('capabilities', {}).get(capability_key)
            if not capability_config:
                return default_value
            
            for role in user_info["roles"]:
                if role in capability_config.get('roles', {}):
                    return capability_config['roles'][role]
            
            if user_info["tier"] in capability_config.get('tiers', {}):
                return capability_config['tiers'][user_info["tier"]]

            return capability_config.get('default', default_value)


    # Patch modules so our mocks are used
    sys.modules['config.config_manager'] = MockConfigManager()
    sys.modules['utils.user_manager'] = MockUserManager()
    # Mock specific functions imported at module level
    sys.modules['utils.user_manager'].get_user_tier_capability = MockUserManager().get_user_tier_capability
    
    # Use a real AsyncMock for log_event to track calls
    mock_analytics_tracker_db = MockFirestoreManager() # Use the mock firestore for analytics logs
    original_log_event = log_event # Store original
    async def mock_log_event(user_id: str, event_type: str, status: str, details: Dict[str, Any]):
        log_data = {
            "user_id": user_id,
            "event_type": event_type,
            "status": status,
            "details": details,
            "timestamp": "2025-07-23T09:00:00Z" # Fixed timestamp for testing
        }
        await mock_analytics_tracker_db.collection("analytics_logs").add(log_data)
    sys.modules['utils.analytics_tracker'].log_event = mock_log_event

    # Mock make_api_request to simulate API responses
    original_make_api_request = make_api_request
    async def mock_make_api_request(
        base_url: str,
        endpoint: str,
        params: Dict[str, Any],
        headers: Dict[str, str],
        api_key: Optional[str] = None,
        api_key_name: Optional[str] = None,
        provider: str = "coingecko",
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        full_url: Optional[str] = None
    ):
        logger.info(f"Mocking API request for {provider} - {endpoint} with params: {params}")
        if provider == "coingecko":
            if "simple/price" in endpoint:
                if params.get("ids") == "bitcoin" and params.get("vs_currencies") == "usd":
                    return {"bitcoin": {"usd": 65000.0}}
                if params.get("ids") == "ethereum" and params.get("vs_currencies") == "usd,eur":
                    return {"ethereum": {"usd": 3500.0, "eur": 3200.0}}
                if params.get("ids") == "nonexistent":
                    return {}
            elif "market_chart" in endpoint:
                 if params.get("vs_currency") == "usd" and params.get("days") == 7:
                    return {
                        "prices": [[1678886400000, 60000], [1678972800000, 61000]],
                        "market_caps": [[1678886400000, 1.2e12], [1678972800000, 1.22e12]],
                        "total_volumes": [[1678886400000, 3e10], [1678972800000, 3.1e10]]
                    }
        elif provider == "alphavantage":
            if params.get("function") == "CURRENCY_EXCHANGE_RATE" and params.get("from_currency") == "BTC":
                return {"Realtime Currency Exchange Rate": {"1. From_Currency Code": "BTC", "2. From_Currency Name": "Bitcoin", "3. To_Currency Code": "USD", "4. To_Currency Name": "United States Dollar", "5. Exchange Rate": "65123.45"}}
        return None
    sys.modules['shared_tools.historical_data_tool'].make_api_request = mock_make_api_request


    # Create test user profiles
    mock_user_pro_profile = UserProfile(user_id="test_user_pro", username="Pro User", email="pro@example.com", tier="pro", roles=["user"])
    mock_user_premium_profile = UserProfile(user_id="test_user_premium", username="Premium User", email="premium@example.com", tier="premium", roles=["user"])
    mock_user_free_profile = UserProfile(user_id="test_user_free", username="Free User", email="free@example.com", tier="free", roles=["user"])

    async def run_crypto_tests():
        print("Running CryptoTools tests...")

        # Test 1: get_crypto_price - Pro user (allowed)
        print("\n--- Test 1: get_crypto_price (Pro user, allowed) ---")
        result1 = await get_crypto_price(crypto_id="bitcoin", vs_currencies="usd", user_context=mock_user_pro_profile)
        print(f"Result 1: {result1}")
        assert "65000.0 USD" in result1
        assert mock_analytics_tracker_db.collection.return_value.add.call_args[0][0]["event_type"] == "tool_usage"
        assert mock_analytics_tracker_db.collection.return_value.add.call_args[0][0]["status"] == "success"
        mock_analytics_tracker_db.collection.return_value.add.reset_mock()
        print("Test 1 Passed.")

        # Test 2: get_crypto_price - Free user (permission denied)
        print("\n--- Test 2: get_crypto_price (Free user, denied) ---")
        result2 = await get_crypto_price(crypto_id="ethereum", vs_currencies="usd", user_context=mock_user_free_profile)
        print(f"Result 2: {result2}")
        assert "Error: Cryptocurrency price querying is not enabled" in result2
        assert mock_analytics_tracker_db.collection.return_value.add.call_args[0][0]["status"] == "permission_denied"
        mock_analytics_tracker_db.collection.return_value.add.reset_mock()
        print("Test 2 Passed.")

        # Test 3: get_crypto_price - Multiple vs_currencies
        print("\n--- Test 3: get_crypto_price (Multiple currencies) ---")
        result3 = await get_crypto_price(crypto_id="ethereum", vs_currencies="usd,eur", user_context=mock_user_pro_profile)
        print(f"Result 3: {result3}")
        assert "3500.0 USD" in result3 and "3200.0 EUR" in result3
        mock_analytics_tracker_db.collection.return_value.add.reset_mock()
        print("Test 3 Passed.")

        # Test 4: get_crypto_price - Non-existent crypto
        print("\n--- Test 4: get_crypto_price (Non-existent crypto) ---")
        result4 = await get_crypto_price(crypto_id="nonexistent", vs_currencies="usd", user_context=mock_user_pro_profile)
        print(f"Result 4: {result4}")
        assert "No price data found" in result4
        mock_analytics_tracker_db.collection.return_value.add.reset_mock()
        print("Test 4 Passed.")

        # Test 5: get_crypto_price - Alpha Vantage provider
        print("\n--- Test 5: get_crypto_price (Alpha Vantage provider) ---")
        # Ensure Alpha Vantage API key is set for testing purposes
        sys.modules['config.config_manager'].set_secret("ALPHAVANTAGE_API_KEY", "mock_alphavantage_key")
        result5 = await get_crypto_price(crypto_id="BTC", vs_currencies="USD", user_context=mock_user_pro_profile, provider="alphavantage")
        print(f"Result 5: {result5}")
        assert "Alpha Vantage: 1 BTC = 65123.45 USD." in result5
        mock_analytics_tracker_db.collection.return_value.add.reset_mock()
        print("Test 5 Passed.")


        # Instantiate CryptoTools for methods that remain class-bound
        crypto_tools_instance = CryptoTools(
            config_manager=sys.modules['config.config_manager'],
            firestore_manager=mock_analytics_tracker_db, # Using mock firestore for logs
            log_event=sys.modules['utils.analytics_tracker'].log_event,
            document_tools=MagicMock() # Mock document_tools as it's a dependency
        )
        # Manually assign the mock for document_summarize_document_by_path
        crypto_tools_instance.document_tools.summarize_document_by_path = AsyncMock(side_effect=crypto_tools_instance.document_summarize_document_by_path)


        # Test 6: crypto_get_historical_crypto_prices (Premium user, allowed)
        print("\n--- Test 6: crypto_get_historical_crypto_prices (Premium user, allowed) ---")
        result6 = await crypto_tools_instance.crypto_get_historical_crypto_prices(crypto_id="bitcoin", days=7, user_context=mock_user_premium_profile)
        print(f"Result 6: {result6}")
        assert "Historical prices for bitcoin (usd) over 7 days" in result6
        assert mock_analytics_tracker_db.collection.return_value.add.call_args[0][0]["status"] == "success"
        mock_analytics_tracker_db.collection.return_value.add.reset_mock()
        print("Test 6 Passed.")
        
        # Test 7: document_summarize_document_by_path via CryptoTools instance
        print("\n--- Test 7: document_summarize_document_by_path (Pro user, allowed) ---")
        test_file_path = "path/to/dummy_file.txt"
        result_summarize = await crypto_tools_instance.document_summarize_document_by_path(test_file_path, user_context=mock_user_pro_profile)
        print(f"Summarize Result: {result_summarize}")
        assert "Mocked summary of dummy_file.txt" in result_summarize # Check for mock summary from DocumentTools
        assert mock_analytics_tracker_db.collection.return_value.add.call_args[0][0]["event_type"] == "tool_usage"
        assert mock_analytics_tracker_db.collection.return_value.add.call_args[0][0]["details"]["tool_name"] == "document_summarize_document_by_path"
        assert mock_analytics_tracker_db.collection.return_value.add.call_args[0][0]["status"] == "success"
        mock_analytics_tracker_db.collection.return_value.add.reset_mock()
        print("Test 7 Passed.")

        print("\nAll crypto_tool tests with live API simulation and analytics considerations completed.")

    # Ensure tests are only run when the script is executed directly
    if __name__ == "__main__":
        asyncio.run(run_crypto_tests())

        # Restore original modules/functions
        sys.modules['shared_tools.historical_data_tool'].make_api_request = original_make_api_request
        sys.modules['utils.analytics_tracker'].log_event = original_log_event
