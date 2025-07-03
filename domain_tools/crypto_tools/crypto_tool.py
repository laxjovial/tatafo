# domain_tools/crypto_tools/crypto_tool.py

import requests
import logging
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from langchain_core.tools import tool

# Import config_manager for API keys
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability

logger = logging.getLogger(__name__)

# --- Helper Function to get API Keys for Crypto APIs ---
def _get_crypto_api_key(api_name: str) -> Optional[str]:
    """
    Retrieves the API key for a given crypto API from secrets.
    CoinGecko's free tier typically doesn't require an API key for basic calls,
    but a paid tier would. We'll include a placeholder.
    """
    if api_name == "coingecko":
        # CoinGecko API key for paid plans, free tier often no key needed
        return config_manager.get_secret("coingecko_api_key")
    # Add other crypto API key retrieval logic here if needed
    return None

@tool
def get_crypto_price(coin_id: str, vs_currency: str = "usd", user_token: str = "default") -> str:
    """
    Retrieves the current price of a cryptocurrency.
    Uses CoinGecko API.

    Args:
        coin_id (str): The CoinGecko ID of the cryptocurrency (e.g., "bitcoin", "ethereum", "solana").
                       You can find IDs at https://api.coingecko.com/api/v3/coins/list
        vs_currency (str, optional): The currency to compare against (e.g., "usd", "eur", "gbp"). Defaults to "usd".
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks.

    Returns:
        str: A string containing the current crypto price, or an error message.
    """
    logger.info(f"Tool: get_crypto_price called for coin: {coin_id}, vs_currency: {vs_currency} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'crypto_tool_access', False):
        return "Error: Access to cryptocurrency tools is not enabled for your current tier."

    coingecko_api_key = _get_crypto_api_key("coingecko")
    headers = {"x-cg-pro-api-key": coingecko_api_key} if coingecko_api_key else {}

    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies={vs_currency}"
        response = requests.get(url, headers=headers, timeout=config_manager.get("web_scraping.timeout_seconds", 10))
        response.raise_for_status()
        data = response.json()

        if coin_id in data and vs_currency in data[coin_id]:
            price = data[coin_id][vs_currency]
            return f"Current price of {coin_id.capitalize()} is {price} {vs_currency.upper()}."
        elif not data:
            return f"No data found for {coin_id} against {vs_currency}. Check coin ID or currency."
        else:
            return f"Could not retrieve price for {coin_id} against {vs_currency}. Response: {data}"
    except requests.exceptions.RequestException as e:
        logger.error(f"CoinGecko price request failed for {coin_id}: {e}", exc_info=True)
        return f"Failed to fetch crypto price for {coin_id} due to a network error: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching crypto price for {coin_id}: {e}", exc_info=True)
        return f"An unexpected error occurred while fetching crypto price for {coin_id}: {e}"

@tool
def get_historical_crypto_prices(coin_id: str, vs_currency: str, days: int = 30, user_token: str = "default") -> str:
    """
    Retrieves historical daily prices for a cryptocurrency for a given number of days.
    The output is a JSON string suitable for chart generation.
    Uses CoinGecko API (market chart).

    Args:
        coin_id (str): The CoinGecko ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
        vs_currency (str): The currency to compare against (e.g., "usd", "eur").
        days (int, optional): Number of past days for historical data. Max 365 for daily data on free tier. Defaults to 30.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks.

    Returns:
        str: A JSON string representing the historical data (list of dicts),
             or an error message. Each dict contains 'date' (YYYY-MM-DD), 'price', 'market_cap', 'volume'.
    """
    logger.info(f"Tool: get_historical_crypto_prices called for coin: {coin_id}, vs_currency: {vs_currency}, days: {days} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'historical_data_access', False):
        return "Error: Access to historical data is not enabled for your current tier."
    
    if not get_user_tier_capability(user_token, 'crypto_tool_access', False):
        return "Error: Access to cryptocurrency tools is not enabled for your current tier."

    coingecko_api_key = _get_crypto_api_key("coingecko")
    headers = {"x-cg-pro-api-key": coingecko_api_key} if coingecko_api_key else {}

    # CoinGecko free API has limits for `days`:
    # 1 day: 5-minute intervals
    # 1-90 days: hourly intervals
    # >90 days: daily intervals (max 365 for basic historical data)
    # For simplicity, we'll request daily data for >1 day, limiting to 365.
    if days > 365:
        logger.warning(f"Requested days {days} exceeds CoinGecko free tier daily limit of 365. Capping to 365.")
        days = 365

    try:
        url = (
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?"
            f"vs_currency={vs_currency}&days={days}&interval=daily" # Request daily interval
        )
        response = requests.get(url, headers=headers, timeout=config_manager.get("web_scraping.timeout_seconds", 10))
        response.raise_for_status()
        data = response.json()

        if not data or "prices" not in data or not data["prices"]:
            return f"No historical data found for {coin_id} against {vs_currency} for the last {days} days."

        historical_data = []
        for price_point in data["prices"]:
            timestamp, price = price_point
            date_obj = datetime.fromtimestamp(timestamp / 1000) # Convert ms to s
            historical_data.append({
                "date": date_obj.strftime("%Y-%m-%d"),
                "price": price,
                "market_cap": data["market_caps"][data["prices"].index(price_point)][1] if "market_caps" in data and len(data["market_caps"]) > data["prices"].index(price_point) else None,
                "volume": data["total_volumes"][data["prices"].index(price_point)][1] if "total_volumes" in data and len(data["total_volumes"]) > data["prices"].index(price_point) else None,
            })
        
        # Ensure data is sorted by date (CoinGecko usually returns it sorted, but good practice)
        historical_data.sort(key=lambda x: x['date'])

        logger.info(f"Successfully fetched {len(historical_data)} historical data points for {coin_id}.")
        return json.dumps(historical_data)

    except requests.exceptions.RequestException as e:
        logger.error(f"CoinGecko historical price request failed for {coin_id}: {e}", exc_info=True)
        return f"Failed to fetch historical crypto prices for {coin_id} due to a network error: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching historical crypto prices for {coin_id}: {e}", exc_info=True)
        return f"An unexpected error occurred while fetching historical crypto prices for {coin_id}: {e}"

@tool
def get_crypto_id_by_symbol(symbol: str, user_token: str = "default") -> str:
    """
    Looks up the CoinGecko ID for a cryptocurrency given its common symbol (e.g., "btc", "eth", "sol").
    This is useful when the user provides a symbol instead of the full CoinGecko ID.

    Args:
        symbol (str): The common symbol of the cryptocurrency (e.g., "btc", "eth").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks.

    Returns:
        str: The CoinGecko ID of the cryptocurrency, or an error message.
    """
    logger.info(f"Tool: get_crypto_id_by_symbol called for symbol: {symbol} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'crypto_tool_access', False):
        return "Error: Access to cryptocurrency tools is not enabled for your current tier."

    coingecko_api_key = _get_crypto_api_key("coingecko")
    headers = {"x-cg-pro-api-key": coingecko_api_key} if coingecko_api_key else {}

    try:
        # CoinGecko's /coins/list endpoint provides id, symbol, and name
        url = "https://api.coingecko.com/api/v3/coins/list"
        response = requests.get(url, headers=headers, timeout=config_manager.get("web_scraping.timeout_seconds", 10))
        response.raise_for_status()
        coins_list = response.json()

        # Search for the symbol (case-insensitive)
        for coin in coins_list:
            if coin.get('symbol', '').lower() == symbol.lower():
                logger.info(f"Found CoinGecko ID '{coin['id']}' for symbol '{symbol}'.")
                return coin['id']
        
        return f"CoinGecko ID not found for symbol '{symbol}'. Please try a different symbol or the full coin name."

    except requests.exceptions.RequestException as e:
        logger.error(f"CoinGecko list coins request failed: {e}", exc_info=True)
        return f"Failed to lookup crypto ID for {symbol} due to a network error: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred while looking up crypto ID for {symbol}: {e}", exc_info=True)
        return f"An unexpected error occurred while looking up crypto ID for {symbol}: {e}"


# CLI Test (optional)
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.coingecko_api_key = "MOCK_COINGECKO_KEY"
            self.openai = {"api_key": "sk-mock-openai-key-12345"}
            self.google = {"api_key": "AIzaSy-mock-google-key"}
            self.firebase_config = "{}" # Mock empty config for Firebase if not set

        def get(self, key, default=None):
            parts = key.split('.')
            val = self
            for part in parts:
                if hasattr(val, part):
                    val = getattr(val, part)
                elif isinstance(val, dict) and part in val:
                    val = val[part]
                else:
                    return default
            return val
    
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
                'tiers': {}, # This will be overridden by tiers.yaml
                'default_user_tier': 'free',
                'default_user_roles': ['user'],
                'api_configs': [] # No need to load external API configs for this mock
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
            if key == "coingecko_api_key": return st.secrets.coingecko_api_key
            return st.secrets.get(key, default)

        def set_secret(self, key, value):
            setattr(st.secrets, key, value)


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

    class MockCoinGeckoResponse:
        def __init__(self, data, status_code=200):
            self._data = data
            self.status_code = status_code
        def json(self):
            return self._data
        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.exceptions.HTTPError(f"HTTP Error: {self.status_code}")

    def mock_requests_get_side_effect(url, params=None, headers=None, timeout=None):
        if "simple/price" in url:
            coin_id = url.split("ids=")[1].split("&")[0]
            vs_currency = url.split("vs_currencies=")[1].split("&")[0]
            if coin_id == "bitcoin" and vs_currency == "usd":
                return MockCoinGeckoResponse({"bitcoin": {"usd": 65000.0}})
            elif coin_id == "ethereum" and vs_currency == "usd":
                return MockCoinGeckoResponse({"ethereum": {"usd": 3500.0}})
            else:
                return MockCoinGeckoResponse({}) # No data
        elif "market_chart" in url:
            coin_id = url.split("coins/")[1].split("/market_chart")[0]
            vs_currency = url.split("vs_currency=")[1].split("&")[0]
            days = int(url.split("days=")[1].split("&")[0])
            
            mock_prices = []
            mock_market_caps = []
            mock_volumes = []
            for i in range(days):
                timestamp_ms = (datetime.now() - timedelta(days=days-1-i)).timestamp() * 1000
                price = 1000 + i * 10
                market_cap = 1000000000 + i * 10000000
                volume = 50000000 + i * 500000
                mock_prices.append([timestamp_ms, price])
                mock_market_caps.append([timestamp_ms, market_cap])
                mock_volumes.append([timestamp_ms, volume])
            return MockCoinGeckoResponse({"prices": mock_prices, "market_caps": mock_market_caps, "total_volumes": mock_volumes})
        elif "coins/list" in url:
            return MockCoinGeckoResponse([
                {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"},
                {"id": "ethereum", "symbol": "eth", "name": "Ethereum"},
                {"id": "solana", "symbol": "sol", "name": "Solana"},
                {"id": "dogecoin", "symbol": "doge", "name": "Dogecoin"},
            ])
        raise requests.exceptions.RequestException(f"Unexpected URL: {url}")

    requests.get = MagicMock(side_effect=mock_requests_get_side_effect)

    test_user_free = sys.modules['utils.user_manager']._mock_users["mock_free_token"]['user_id']
    test_user_pro = sys.modules['utils.user_manager']._mock_users["mock_pro_token"]['user_id']
    test_user_premium = sys.modules['utils.user_manager']._mock_users["mock_premium_token"]['user_id']
    test_user_admin = sys.modules['utils.user_manager']._mock_users["mock_admin_token"]['user_id']

    print("\n--- Testing get_crypto_price function ---")

    # Test 1: Pro user, valid coin (bitcoin)
    print("\n--- Test 1: Pro user, valid coin (bitcoin) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result1 = get_crypto_price("bitcoin", user_token=test_user_pro)
    print(f"Result for bitcoin (Pro user): {result1}")
    assert "Current price of Bitcoin is 65000.0 USD." in result1
    print("Test 1 Passed.")

    # Test 2: Free user, access denied
    print("\n--- Test 2: Free user, access denied ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result2 = get_crypto_price("ethereum", user_token=test_user_free)
    print(f"Result for ethereum (Free user): {result2}")
    assert "Error: Access to cryptocurrency tools is not enabled for your current tier." in result2
    print("Test 2 Passed.")

    # Test 3: Admin user, invalid coin ID
    print("\n--- Test 3: Admin user, invalid coin ID ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    result3 = get_crypto_price("nonexistentcoin", user_token=test_user_admin)
    print(f"Result for nonexistentcoin (Admin user): {result3}")
    assert "No data found for nonexistentcoin" in result3 or "Could not retrieve price for nonexistentcoin" in result3
    print("Test 3 Passed.")

    print("\n--- Testing get_historical_crypto_prices function ---")

    # Test 4: Premium user, 7 days historical data
    print("\n--- Test 4: Premium user, 7 days historical data ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_premium
    hist_data_premium = get_historical_crypto_prices("ethereum", "usd", 7, user_token=test_user_premium)
    print(f"Historical data for ethereum (Premium user):\n{hist_data_premium[:200]}...")
    hist_json = json.loads(hist_data_premium)
    assert len(hist_json) == 7
    assert "price" in hist_json[0]
    assert "date" in hist_json[0]
    print("Test 4 Passed.")

    # Test 5: Pro user, historical access denied
    print("\n--- Test 5: Pro user, historical access denied ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    hist_data_pro = get_historical_crypto_prices("bitcoin", "usd", 7, user_token=test_user_pro)
    print(f"Historical data for bitcoin (Pro user): {hist_data_pro}")
    assert "Error: Access to historical data is not enabled for your current tier." in hist_data_pro
    print("Test 5 Passed.")

    # Test 6: Admin user, days > 365 (should cap to 365)
    print("\n--- Test 6: Admin user, days > 365 ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    hist_data_capped = get_historical_crypto_prices("solana", "usd", 500, user_token=test_user_admin)
    print(f"Historical data for solana (Admin user, capped days):\n{hist_data_capped[:200]}...")
    hist_json_capped = json.loads(hist_data_capped)
    assert len(hist_json_capped) == 365 # Should be capped to 365
    print("Test 6 Passed.")

    print("\n--- Testing get_crypto_id_by_symbol function ---")

    # Test 7: Pro user, valid symbol
    print("\n--- Test 7: Pro user, valid symbol (btc) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result7 = get_crypto_id_by_symbol("btc", user_token=test_user_pro)
    print(f"Result for 'btc' (Pro user): {result7}")
    assert result7 == "bitcoin"
    print("Test 7 Passed.")

    # Test 8: Admin user, symbol not found
    print("\n--- Test 8: Admin user, symbol not found ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    result8 = get_crypto_id_by_symbol("xyz", user_token=test_user_admin)
    print(f"Result for 'xyz' (Admin user): {result8}")
    assert "CoinGecko ID not found for symbol 'xyz'." in result8
    print("Test 8 Passed.")

    # Test 9: Free user, access denied
    print("\n--- Test 9: Free user, access denied ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result9 = get_crypto_id_by_symbol("eth", user_token=test_user_free)
    print(f"Result for 'eth' (Free user): {result9}")
    assert "Error: Access to cryptocurrency tools is not enabled for your current tier." in result9
    print("Test 9 Passed.")

    print("\nAll crypto_tool tests passed (mocked APIs and RBAC).")

    # Restore original requests.get
    requests.get = original_requests_get
