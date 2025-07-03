# domain_tools/finance_tools/finance_tool.py

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

# --- Helper Function to get API Keys for Finance APIs ---
def _get_finance_api_key(api_name: str) -> Optional[str]:
    """
    Retrieves the API key for a given finance API from secrets.
    """
    if api_name == "alpha_vantage":
        return config_manager.get_secret("alpha_vantage_api_key")
    elif api_name == "finnhub":
        return config_manager.get_secret("finnhub_api_key")
    # Add other finance API key retrieval logic here if needed
    return None

@tool
def get_stock_price(symbol: str, user_token: str = "default") -> str:
    """
    Retrieves the current stock price for a given stock symbol.
    Uses Alpha Vantage or Finnhub API.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL", "MSFT").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks (e.g., rate limits).

    Returns:
        str: A string containing the current stock price and related information,
             or an error message.
    """
    logger.info(f"Tool: get_stock_price called for symbol: {symbol} by user: {user_token}")

    # RBAC Check: Example of a rate limit or feature access check
    # For simplicity, we'll assume a general finance tool access capability.
    if not get_user_tier_capability(user_token, 'finance_tool_access', True): # Assume basic access for now
        return "Error: Access to finance tools is not enabled for your current tier."

    # Try Alpha Vantage first
    alpha_vantage_api_key = _get_finance_api_key("alpha_vantage")
    if alpha_vantage_api_key:
        logger.info("Attempting to use Alpha Vantage for stock price.")
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={alpha_vantage_api_key}"
            response = requests.get(url, timeout=config_manager.get("web_scraping.timeout_seconds", 10))
            response.raise_for_status()
            data = response.json()

            if "Global Quote" in data and data["Global Quote"]:
                quote = data["Global Quote"]
                price = quote.get("05. price")
                open_price = quote.get("02. open")
                high_price = quote.get("03. high")
                low_price = quote.get("04. low")
                volume = quote.get("06. volume")
                last_trading_day = quote.get("07. latest trading day")
                change = quote.get("09. change")
                change_percent = quote.get("10. change percent")

                return (
                    f"Current Stock Price for {symbol.upper()}:\n"
                    f"Price: ${price}\n"
                    f"Open: ${open_price}\n"
                    f"High: ${high_price}\n"
                    f"Low: ${low_price}\n"
                    f"Volume: {volume}\n"
                    f"Last Trading Day: {last_trading_day}\n"
                    f"Change: {change} ({change_percent})"
                )
            elif "Error Message" in data:
                logger.warning(f"Alpha Vantage Error for {symbol}: {data['Error Message']}")
                return f"Error fetching stock price from Alpha Vantage: {data['Error Message']}. Trying Finnhub..."
            else:
                logger.warning(f"Alpha Vantage did not return expected data for {symbol}. Trying Finnhub...")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Alpha Vantage request failed for {symbol}: {e}. Trying Finnhub...")
        except Exception as e:
            logger.warning(f"Error processing Alpha Vantage response for {symbol}: {e}. Trying Finnhub...")

    # Fallback to Finnhub
    finnhub_api_key = _get_finance_api_key("finnhub")
    if finnhub_api_key:
        logger.info("Attempting to use Finnhub for stock price.")
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={finnhub_api_key}"
            response = requests.get(url, timeout=config_manager.get("web_scraping.timeout_seconds", 10))
            response.raise_for_status()
            data = response.json()

            if data and data.get("c"): # 'c' is current price
                current_price = data.get("c")
                open_price = data.get("o")
                high_price = data.get("h")
                low_price = data.get("l")
                previous_close = data.get("pc")
                
                change = current_price - previous_close if current_price and previous_close else "N/A"
                change_percent = (change / previous_close * 100) if current_price and previous_close and previous_close != 0 else "N/A"

                return (
                    f"Current Stock Price for {symbol.upper()} (Finnhub):\n"
                    f"Price: ${current_price}\n"
                    f"Open: ${open_price}\n"
                    f"High: ${high_price}\n"
                    f"Low: ${low_price}\n"
                    f"Previous Close: ${previous_close}\n"
                    f"Change: {change:.2f} ({change_percent:.2f}%)" if isinstance(change, (int, float)) and isinstance(change_percent, (int, float)) else f"Change: {change} ({change_percent})"
                )
            else:
                logger.warning(f"Finnhub did not return expected data for {symbol}: {data}")
                return f"Could not retrieve stock price for {symbol}. Data: {data}"
        except requests.exceptions.RequestException as e:
            logger.error(f"Finnhub request failed for {symbol}: {e}", exc_info=True)
            return f"Failed to fetch stock price for {symbol} due to a network error: {e}"
        except Exception as e:
            logger.error(f"Error processing Finnhub response for {symbol}: {e}", exc_info=True)
            return f"An unexpected error occurred while fetching stock price for {symbol}: {e}"

    return f"Error: No configured API key found for fetching stock prices, or all attempts failed for {symbol}."

@tool
def get_company_news(symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None, user_token: str = "default") -> str:
    """
    Retrieves recent company news for a given stock symbol within a specified date range.
    Uses Finnhub API. Date format: YYYY-MM-DD.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL", "MSFT").
        from_date (str, optional): Start date for news (YYYY-MM-DD). Defaults to 7 days ago.
        to_date (str, optional): End date for news (YYYY-MM-DD). Defaults to today.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks.

    Returns:
        str: A formatted string of recent news articles, or an error message.
    """
    logger.info(f"Tool: get_company_news called for symbol: {symbol}, from: {from_date}, to: {to_date} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'finance_tool_access', True):
        return "Error: Access to finance tools is not enabled for your current tier."

    finnhub_api_key = _get_finance_api_key("finnhub")
    if not finnhub_api_key:
        return "Error: Finnhub API key not configured for company news."

    today = datetime.now()
    if to_date:
        try:
            to_date_obj = datetime.strptime(to_date, "%Y-%m-%d")
        except ValueError:
            return "Error: Invalid 'to_date' format. Please use YYYY-MM-DD."
    else:
        to_date_obj = today

    if from_date:
        try:
            from_date_obj = datetime.strptime(from_date, "%Y-%m-%d")
        except ValueError:
            return "Error: Invalid 'from_date' format. Please use YYYY-MM-DD."
    else:
        from_date_obj = today - timedelta(days=7) # Default to last 7 days

    # Ensure from_date is not after to_date
    if from_date_obj > to_date_obj:
        return "Error: 'from_date' cannot be after 'to_date'."

    from_date_str = from_date_obj.strftime("%Y-%m-%d")
    to_date_str = to_date_obj.strftime("%Y-%m-%d")

    try:
        url = (
            f"https://finnhub.io/api/v1/company-news?"
            f"symbol={symbol}&from={from_date_str}&to={to_date_str}&token={finnhub_api_key}"
        )
        response = requests.get(url, timeout=config_manager.get("web_scraping.timeout_seconds", 10))
        response.raise_for_status()
        news_data = response.json()

        if not news_data:
            return f"No news found for {symbol.upper()} between {from_date_str} and {to_date_str}."

        formatted_news = [f"Recent News for {symbol.upper()} ({from_date_str} to {to_date_str}):"]
        for i, article in enumerate(news_data[:5]): # Limit to top 5 articles for brevity
            formatted_news.append(f"--- Article {i+1} ---")
            formatted_news.append(f"Headline: {article.get('headline', 'N/A')}")
            formatted_news.append(f"Source: {article.get('source', 'N/A')}")
            formatted_news.append(f"Summary: {article.get('summary', 'N/A')}")
            formatted_news.append(f"URL: {article.get('url', 'N/A')}")
        
        return "\n".join(formatted_news)

    except requests.exceptions.RequestException as e:
        logger.error(f"Finnhub news request failed for {symbol}: {e}", exc_info=True)
        return f"Failed to fetch company news for {symbol} due to a network error: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching company news for {symbol}: {e}", exc_info=True)
        return f"An unexpected error occurred while fetching company news for {symbol}: {e}"


# CLI Test (optional)
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.alpha_vantage_api_key = "MOCK_ALPHA_VANTAGE_KEY"
            self.finnhub_api_key = "MOCK_FINNHUB_KEY"
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
            if key == "alpha_vantage_api_key": return st.secrets.alpha_vantage_api_key
            if key == "finnhub_api_key": return st.secrets.finnhub_api_key
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
                'finance_tool_access': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
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

    class MockAlphaVantageResponse:
        def __init__(self, symbol, is_success=True, error_msg=None):
            self.status_code = 200
            self._symbol = symbol
            self._is_success = is_success
            self._error_msg = error_msg

        def json(self):
            if self._error_msg:
                return {"Error Message": self._error_msg}
            if not self._is_success:
                return {} # Empty or unexpected data
            return {
                "Global Quote": {
                    "01. symbol": self._symbol,
                    "02. open": "150.00",
                    "03. high": "155.00",
                    "04. low": "149.00",
                    "05. price": "154.25",
                    "06. volume": "1000000",
                    "07. latest trading day": "2023-10-26",
                    "08. previous close": "148.50",
                    "09. change": "5.75",
                    "10. change percent": "3.8700%"
                }
            }
        def raise_for_status(self):
            if not self._is_success and not self._error_msg:
                raise requests.exceptions.HTTPError("404 Not Found")

    class MockFinnhubQuoteResponse:
        def __init__(self, symbol, is_success=True):
            self.status_code = 200
            self._symbol = symbol
            self._is_success = is_success
        
        def json(self):
            if not self._is_success:
                return {}
            return {
                "c": 154.25,  # Current price
                "h": 155.00,  # High price of the day
                "l": 149.00,  # Low price of the day
                "o": 150.00,  # Open price of the day
                "pc": 148.50, # Previous close price
                "t": 1678912800 # Timestamp
            }
        def raise_for_status(self):
            pass

    class MockFinnhubNewsResponse:
        def __init__(self, symbol, num_articles=2):
            self.status_code = 200
            self._symbol = symbol
            self._num_articles = num_articles
        
        def json(self):
            articles = []
            for i in range(self._num_articles):
                articles.append({
                    "category": "company news",
                    "datetime": datetime.now().timestamp() - i * 3600, # Mock timestamp
                    "headline": f"Mock News Headline {i+1} for {self._symbol}",
                    "id": i + 1,
                    "image": "https://mock.image.url",
                    "related": self._symbol,
                    "source": "MockNewsSource",
                    "summary": f"This is a mock summary for news article {i+1} about {self._symbol}.",
                    "url": f"http://mocknews.com/article{i+1}"
                })
            return articles
        def raise_for_status(self):
            pass

    def mock_requests_get_side_effect(url, params=None, headers=None, timeout=None):
        if "alphavantage.co" in url:
            symbol = params.get("symbol")
            if symbol == "AV_ERROR":
                return MockAlphaVantageResponse(symbol, error_msg="Invalid API call")
            elif symbol == "AV_NO_DATA":
                return MockAlphaVantageResponse(symbol, is_success=False)
            return MockAlphaVantageResponse(symbol)
        elif "finnhub.io/api/v1/quote" in url:
            symbol = params.get("symbol")
            return MockFinnhubQuoteResponse(symbol)
        elif "finnhub.io/api/v1/company-news" in url:
            symbol = params.get("symbol")
            return MockFinnhubNewsResponse(symbol)
        raise requests.exceptions.RequestException(f"Unexpected URL: {url}")

    requests.get = MagicMock(side_effect=mock_requests_get_side_effect)

    test_user_free = sys.modules['utils.user_manager']._mock_users["mock_free_token"]['user_id']
    test_user_pro = sys.modules['utils.user_manager']._mock_users["mock_pro_token"]['user_id']
    test_user_admin = sys.modules['utils.user_manager']._mock_users["mock_admin_token"]['user_id']

    print("\n--- Testing get_stock_price function ---")

    # Test 1: Pro user, valid symbol (Alpha Vantage success)
    print("\n--- Test 1: Pro user, valid symbol (Alpha Vantage) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result1 = get_stock_price("AAPL", user_token=test_user_pro)
    print(f"Result for AAPL (Pro user, AV): {result1[:100]}...")
    assert "Current Stock Price for AAPL" in result1
    assert "$154.25" in result1
    print("Test 1 Passed.")

    # Test 2: Free user, access denied
    print("\n--- Test 2: Free user, access denied ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result2 = get_stock_price("GOOG", user_token=test_user_free)
    print(f"Result for GOOG (Free user): {result2}")
    assert "Error: Access to finance tools is not enabled for your current tier." in result2
    print("Test 2 Passed.")

    # Test 3: Admin user, valid symbol (Finnhub fallback if AV fails)
    print("\n--- Test 3: Admin user, valid symbol (Finnhub fallback) ---")
    # Temporarily disable Alpha Vantage key to force Finnhub fallback
    sys.modules['streamlit'].secrets.alpha_vantage_api_key = None
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    result3 = get_stock_price("MSFT", user_token=test_user_admin)
    print(f"Result for MSFT (Admin user, Finnhub): {result3[:100]}...")
    assert "Current Stock Price for MSFT (Finnhub)" in result3
    assert "$154.25" in result3
    # Restore Alpha Vantage key
    sys.modules['streamlit'].secrets.alpha_vantage_api_key = "MOCK_ALPHA_VANTAGE_KEY"
    print("Test 3 Passed.")

    # Test 4: Symbol not found (mocked API error)
    print("\n--- Test 4: Symbol not found (mocked API error) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result4 = get_stock_price("NONEXISTENT", user_token=test_user_pro)
    print(f"Result for NONEXISTENT: {result4}")
    assert "Could not retrieve stock price for NONEXISTENT." in result4 or "Error fetching stock price from Alpha Vantage" in result4
    print("Test 4 Passed.")

    print("\n--- Testing get_company_news function ---")

    # Test 5: Pro user, valid symbol, default dates
    print("\n--- Test 5: Pro user, valid symbol, default dates ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result5 = get_company_news("TSLA", user_token=test_user_pro)
    print(f"Result for TSLA news (Pro user):\n{result5[:200]}...")
    assert "Recent News for TSLA" in result5
    assert "Mock News Headline 1 for TSLA" in result5
    print("Test 5 Passed.")

    # Test 6: Premium user, custom dates
    print("\n--- Test 6: Premium user, custom dates ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_premium
    result6 = get_company_news("AMZN", from_date="2023-01-01", to_date="2023-01-05", user_token=test_user_premium)
    print(f"Result for AMZN news (Premium user, custom dates):\n{result6[:200]}...")
    assert "Recent News for AMZN (2023-01-01 to 2023-01-05)" in result6
    print("Test 6 Passed.")

    # Test 7: Free user, access denied
    print("\n--- Test 7: Free user, access denied ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result7 = get_company_news("NFLX", user_token=test_user_free)
    print(f"Result for NFLX news (Free user): {result7}")
    assert "Error: Access to finance tools is not enabled for your current tier." in result7
    print("Test 7 Passed.")

    # Test 8: Invalid date format
    print("\n--- Test 8: Invalid date format ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result8 = get_company_news("GOOG", from_date="invalid-date", user_token=test_user_pro)
    print(f"Result for invalid date: {result8}")
    assert "Error: Invalid 'from_date' format. Please use YYYY-MM-DD." in result8
    print("Test 8 Passed.")

    # Test 9: From date after to date
    print("\n--- Test 9: From date after to date ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result9 = get_company_news("GOOG", from_date="2023-01-05", to_date="2023-01-01", user_token=test_user_pro)
    print(f"Result for from_date after to_date: {result9}")
    assert "Error: 'from_date' cannot be after 'to_date'." in result9
    print("Test 9 Passed.")

    print("\nAll finance_tool tests passed (mocked APIs and RBAC).")

    # Restore original requests.get
    requests.get = original_requests_get
