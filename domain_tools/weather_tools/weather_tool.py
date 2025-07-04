# domain_tools/weather_tools/weather_tool.py

import logging
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
from datetime import datetime, timedelta

# Import config_manager for API keys
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability

logger = logging.getLogger(__name__)

# --- Helper Function to get API Keys for Weather APIs ---
def _get_weather_api_key(api_name: str) -> Optional[str]:
    """
    Retrieves the API key for a given weather API from secrets.
    """
    if api_name == "openweathermap": # Example placeholder for a real API
        return config_manager.get_secret("openweathermap_api_key")
    # Add other weather API key retrieval logic here if needed
    return None

@tool
def get_current_weather(location: str, user_token: str = "default") -> str:
    """
    Retrieves the current weather conditions for a specified location.
    Uses a mock weather API for demonstration.

    Args:
        location (str): The city or location for which to get weather (e.g., "London", "New York", "Tokyo").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks.

    Returns:
        str: A string containing current weather information, or an error message.
    """
    logger.info(f"Tool: get_current_weather called for location: {location} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'weather_tool_access', False):
        return "Error: Access to weather information tools is not enabled for your current tier."

    # In a real application, you would make an API call here (e.g., OpenWeatherMap API).
    # For demonstration, we'll use mock data.
    mock_weather_data = {
        "london": {
            "temperature": "15°C",
            "conditions": "Cloudy with light rain",
            "humidity": "85%",
            "wind_speed": "10 km/h"
        },
        "new york": {
            "temperature": "22°C",
            "conditions": "Sunny",
            "humidity": "60%",
            "wind_speed": "15 km/h"
        },
        "tokyo": {
            "temperature": "28°C",
            "conditions": "Partly cloudy",
            "humidity": "70%",
            "wind_speed": "5 km/h"
        },
        "ikorodu": {
            "temperature": "30°C",
            "conditions": "Hot and humid, scattered clouds",
            "humidity": "75%",
            "wind_speed": "8 km/h"
        }
    }

    weather_info = mock_weather_data.get(location.lower())

    if weather_info:
        formatted_info = (
            f"Current weather in {location.capitalize()}:\n"
            f"Temperature: {weather_info['temperature']}\n"
            f"Conditions: {weather_info['conditions']}\n"
            f"Humidity: {weather_info['humidity']}\n"
            f"Wind Speed: {weather_info['wind_speed']}"
        )
        return formatted_info
    else:
        return f"Current weather information not found for '{location}'. Please check the spelling or try a different location."

@tool
def get_weather_forecast(location: str, days: int, user_token: str = "default") -> str:
    """
    Retrieves the weather forecast for a specified location for a given number of days (up to 5).
    Uses a mock weather API for demonstration.

    Args:
        location (str): The city or location for which to get the forecast.
        days (int): The number of days for the forecast (1 to 5).
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks.

    Returns:
        str: A string containing weather forecast information, or an error message.
    """
    logger.info(f"Tool: get_weather_forecast called for location: {location}, days: {days} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'weather_tool_access', False):
        return "Error: Access to weather information tools is not enabled for your current tier."

    if not (1 <= days <= 5):
        return "Error: Forecast is only available for 1 to 5 days."

    # In a real application, you would make an API call here.
    # For demonstration, we'll use mock data.
    mock_forecast_data = {
        "london": [
            {"date": (datetime.now() + timedelta(days=0)).strftime("%Y-%m-%d"), "temp": "16°C", "conditions": "Light rain"},
            {"date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"), "temp": "17°C", "conditions": "Cloudy"},
            {"date": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"), "temp": "18°C", "conditions": "Partly sunny"},
            {"date": (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"), "temp": "17°C", "conditions": "Overcast"},
            {"date": (datetime.now() + timedelta(days=4)).strftime("%Y-%m-%d"), "temp": "19°C", "conditions": "Sunny intervals"}
        ],
        "new york": [
            {"date": (datetime.now() + timedelta(days=0)).strftime("%Y-%m-%d"), "temp": "23°C", "conditions": "Sunny"},
            {"date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"), "temp": "25°C", "conditions": "Sunny"},
            {"date": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"), "temp": "24°C", "conditions": "Partly cloudy"},
            {"date": (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"), "temp": "22°C", "conditions": "Thunderstorms"},
            {"date": (datetime.now() + timedelta(days=4)).strftime("%Y-%m-%d"), "temp": "20°C", "conditions": "Rain"}
        ],
        "ikorodu": [
            {"date": (datetime.now() + timedelta(days=0)).strftime("%Y-%m-%d"), "temp": "30°C", "conditions": "Scattered clouds"},
            {"date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"), "temp": "31°C", "conditions": "Sunny, hot"},
            {"date": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"), "temp": "29°C", "conditions": "Heavy rain"},
            {"date": (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"), "temp": "28°C", "conditions": "Moderate rain"},
            {"date": (datetime.now() + timedelta(days=4)).strftime("%Y-%m-%d"), "temp": "30°C", "conditions": "Partly cloudy"}
        ]
    }

    forecast_list = mock_forecast_data.get(location.lower())

    if forecast_list:
        forecast_output = [f"Weather forecast for {location.capitalize()} for the next {days} day(s):"]
        for i in range(min(days, len(forecast_list))):
            day_forecast = forecast_list[i]
            forecast_output.append(
                f"  {day_forecast['date']}: {day_forecast['temp']}, {day_forecast['conditions']}"
            )
        return "\n".join(forecast_output)
    else:
        return f"Weather forecast not found for '{location}'. Please check the spelling or try a different location."

# CLI Test (optional)
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.openweathermap_api_key = "MOCK_OWM_KEY"
            self.openai = {"api_key": "sk-mock-openai-key-12345"}
            self.google = {"api_key": "AIzaSy-mock-google-key"}
            self.firebase_config = "{}"

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
                'tiers': {},
                'default_user_tier': 'free',
                'default_user_roles': ['user'],
                'api_configs': []
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
            if key == "openweathermap_api_key": return st.secrets.openweathermap_api_key
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
                'weather_tool_access': {
                    'default': False,
                    'roles': {'user': True, 'pro': True, 'premium': True, 'admin': True} # Often basic users get access to weather
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

    # Mock requests.get for external API calls (not strictly needed for this mock, but good practice)
    original_requests_get = requests.get
    requests.get = MagicMock() # Mock all requests.get calls

    test_user_free = sys.modules['utils.user_manager']._mock_users["mock_free_token"]['user_id']
    test_user_user = sys.modules['utils.user_manager']._mock_users["mock_pro_token"]['user_id'] # Using pro as a 'user' for this test
    test_user_premium = sys.modules['utils.user_manager']._mock_users["mock_premium_token"]['user_id']
    test_user_admin = sys.modules['utils.user_manager']._mock_users["mock_admin_token"]['user_id']

    print("\n--- Testing get_current_weather function ---")

    # Test 1: User with access, valid location (London)
    print("\n--- Test 1: User with access, valid location (London) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_user
    result1 = get_current_weather("London", user_token=test_user_user)
    print(f"Result for London (User):\n{result1[:100]}...")
    assert "Current weather in London:" in result1
    assert "Temperature: 15°C" in result1
    print("Test 1 Passed.")

    # Test 2: Free user, access denied (as per mock RBAC for 'free')
    print("\n--- Test 2: Free user, access denied ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result2 = get_current_weather("New York", user_token=test_user_free)
    print(f"Result for New York (Free user): {result2}")
    assert "Error: Access to weather information tools is not enabled for your current tier." in result2
    print("Test 2 Passed.")

    # Test 3: Admin user, location not found
    print("\n--- Test 3: Admin user, location not found ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    result3 = get_current_weather("NonExistentCity", user_token=test_user_admin)
    print(f"Result for NonExistentCity (Admin user): {result3}")
    assert "Current weather information not found for 'NonExistentCity'." in result3
    print("Test 3 Passed.")

    print("\n--- Testing get_weather_forecast function ---")

    # Test 4: User with access, valid location and days (Tokyo, 3 days)
    print("\n--- Test 4: User with access, valid location and days (Tokyo, 3 days) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_user
    result4 = get_weather_forecast("Tokyo", 3, user_token=test_user_user)
    print(f"Result for Tokyo (User, 3 days):\n{result4[:100]}...")
    assert "Weather forecast for Tokyo for the next 3 day(s):" in result4
    assert "Partly cloudy" in result4
    print("Test 4 Passed.")

    # Test 5: Free user, access denied
    print("\n--- Test 5: Free user, access denied ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result5 = get_weather_forecast("Ikorodu", 2, user_token=test_user_free)
    print(f"Result for Ikorodu (Free user): {result5}")
    assert "Error: Access to weather information tools is not enabled for your current tier." in result5
    print("Test 5 Passed.")

    # Test 6: Pro user, invalid number of days
    print("\n--- Test 6: Pro user, invalid number of days ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result6 = get_weather_forecast("London", 7, user_token=test_user_pro)
    print(f"Result for London (Pro user, 7 days): {result6}")
    assert "Error: Forecast is only available for 1 to 5 days." in result6
    print("Test 6 Passed.")

    # Test 7: Admin user, location not found
    print("\n--- Test 7: Admin user, location not found ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    result7 = get_weather_forecast("Mars", 1, user_token=test_user_admin)
    print(f"Result for Mars (Admin user): {result7}")
    assert "Weather forecast not found for 'Mars'." in result7
    print("Test 7 Passed.")

    print("\nAll weather_tool tests passed (mocked data and RBAC).")

    # Restore original requests.get
    requests.get = original_requests_get
