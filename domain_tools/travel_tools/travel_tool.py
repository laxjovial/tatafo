# domain_tools/travel_tools/travel_tool.py

import logging
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
from datetime import datetime, timedelta

# Import config_manager for API keys
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability

logger = logging.getLogger(__name__)

# --- Helper Function to get API Keys for Travel APIs ---
def _get_travel_api_key(api_name: str) -> Optional[str]:
    """
    Retrieves the API key for a given travel API from secrets.
    """
    if api_name == "amadeus": # Example placeholder for a real API
        return config_manager.get_secret("amadeus_api_key")
    if api_name == "bookingcom": # Example placeholder for a real API
        return config_manager.get_secret("bookingcom_api_key")
    # Add other travel API key retrieval logic here if needed
    return None

@tool
def find_flights(origin: str, destination: str, date: str, user_token: str = "default") -> str:
    """
    Finds hypothetical flight information for a given origin, destination, and date.
    Uses a mock travel API for demonstration.

    Args:
        origin (str): The departure airport code or city (e.g., "LAX", "London").
        destination (str): The arrival airport code or city (e.g., "JFK", "Paris").
        date (str): The departure date in YYYY-MM-DD format (e.g., "2025-07-15").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks.

    Returns:
        str: A string containing flight information, or an error message.
    """
    logger.info(f"Tool: find_flights called for {origin} to {destination} on {date} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'travel_tool_access', False):
        return "Error: Access to travel planning tools is not enabled for your current tier."

    # Validate date format
    try:
        flight_date = datetime.strptime(date, "%Y-%m-%d").date()
        if flight_date < datetime.now().date():
            return "Error: Cannot search for flights in the past."
    except ValueError:
        return "Error: Invalid date format. Please use YYYY-MM-DD."

    # In a real application, you would make an API call here (e.g., Amadeus API, Skyscanner API).
    # For demonstration, we'll use mock data.
    mock_flight_data = {
        ("london", "new york", "2025-07-15"): {
            "flights": [
                {"flight_number": "BA177", "airline": "British Airways", "departure_time": "10:00", "arrival_time": "13:00", "price": "£550"},
                {"flight_number": "VS003", "airline": "Virgin Atlantic", "departure_time": "12:30", "arrival_time": "15:45", "price": "£600"}
            ]
        },
        ("lagos", "london", "2025-07-20"): {
            "flights": [
                {"flight_number": "WT101", "airline": "Wakanow Air", "departure_time": "22:00", "arrival_time": "05:00 (+1 day)", "price": "₦450,000"},
                {"flight_number": "BA075", "airline": "British Airways", "departure_time": "09:00", "arrival_time": "15:30", "price": "₦600,000"}
            ]
        },
        ("paris", "tokyo", "2025-08-01"): {
            "flights": [
                {"flight_number": "AF276", "airline": "Air France", "departure_time": "14:00", "arrival_time": "09:00 (+1 day)", "price": "€800"}
            ]
        }
    }

    # Normalize inputs for mock data lookup
    norm_origin = origin.lower()
    norm_destination = destination.lower()
    norm_date = date

    flights = mock_flight_data.get((norm_origin, norm_destination, norm_date))

    if flights and flights["flights"]:
        formatted_info = [f"Flights from {origin.capitalize()} to {destination.capitalize()} on {date}:"]
        for flight in flights["flights"]:
            formatted_info.append(
                f"  Flight {flight['flight_number']} ({flight['airline']}): "
                f"Departs {flight['departure_time']}, Arrives {flight['arrival_time']}, Price: {flight['price']}"
            )
        return "\n".join(formatted_info)
    else:
        return f"No direct flights found from '{origin}' to '{destination}' on '{date}'. Please try different dates or locations."

@tool
def find_hotels(location: str, check_in_date: str, check_out_date: str, user_token: str = "default") -> str:
    """
    Finds hypothetical hotel availability for a specified location and date range.
    Uses a mock travel API for demonstration.

    Args:
        location (str): The city or area for hotel search (e.g., "Paris", "Dubai").
        check_in_date (str): The check-in date in YYYY-MM-DD format.
        check_out_date (str): The check-out date in YYYY-MM-DD format.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks.

    Returns:
        str: A string containing hotel availability information, or an error message.
    """
    logger.info(f"Tool: find_hotels called for location: {location}, check-in: {check_in_date}, check-out: {check_out_date} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'travel_tool_access', False):
        return "Error: Access to travel planning tools is not enabled for your current tier."

    # Validate date formats and range
    try:
        ci_date = datetime.strptime(check_in_date, "%Y-%m-%d").date()
        co_date = datetime.strptime(check_out_date, "%Y-%m-%d").date()
        if ci_date < datetime.now().date():
            return "Error: Check-in date cannot be in the past."
        if co_date <= ci_date:
            return "Error: Check-out date must be after check-in date."
    except ValueError:
        return "Error: Invalid date format. Please use YYYY-MM-DD."

    # In a real application, you would make an API call here (e.g., Booking.com API, Expedia API).
    # For demonstration, we'll use mock data.
    mock_hotel_data = {
        ("paris", "2025-09-01", "2025-09-05"): {
            "hotels": [
                {"name": "Hotel Louvre", "stars": 4, "price_per_night": "€200", "availability": "High"},
                {"name": "Eiffel Tower Inn", "stars": 3, "price_per_night": "€120", "availability": "Medium"},
                {"name": "Luxury Paris Suite", "stars": 5, "price_per_night": "€450", "availability": "Limited"}
            ]
        },
        ("dubai", "2025-10-10", "2025-10-15"): {
            "hotels": [
                {"name": "Burj View Hotel", "stars": 5, "price_per_night": "AED 800", "availability": "High"},
                {"name": "Desert Oasis Resort", "stars": 4, "price_per_night": "AED 500", "availability": "Medium"}
            ]
        },
        ("ikorodu", "2025-07-10", "2025-07-12"): {
            "hotels": [
                {"name": "Ikorodu Grand Suites", "stars": 3, "price_per_night": "₦35,000", "availability": "High"},
                {"name": "Lagoon View Hotel", "stars": 2, "price_per_night": "₦20,000", "availability": "Medium"}
            ]
        }
    }

    # Normalize inputs for mock data lookup
    norm_location = location.lower()
    norm_check_in = check_in_date
    norm_check_out = check_out_date

    hotels = mock_hotel_data.get((norm_location, norm_check_in, norm_check_out))

    if hotels and hotels["hotels"]:
        formatted_info = [f"Hotels in {location.capitalize()} from {check_in_date} to {check_out_date}:"]
        for hotel in hotels["hotels"]:
            formatted_info.append(
                f"  {hotel['name']} ({hotel['stars']} Stars): "
                f"Price: {hotel['price_per_night']}/night, Availability: {hotel['availability']}"
            )
        return "\n".join(formatted_info)
    else:
        return f"No hotels found in '{location}' for the dates {check_in_date} to {check_out_date}. Please try different dates or locations."

# CLI Test (optional)
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.amadeus_api_key = "MOCK_AMADEUS_KEY"
            self.bookingcom_api_key = "MOCK_BOOKINGCOM_KEY"
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
            if key == "amadeus_api_key": return st.secrets.amadeus_api_key
            if key == "bookingcom_api_key": return st.secrets.bookingcom_api_key
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
                'travel_tool_access': {
                    'default': False,
                    'roles': {'premium': True, 'admin': True} # Travel planning is often a premium feature
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
    test_user_pro = sys.modules['utils.user_manager']._mock_users["mock_pro_token"]['user_id']
    test_user_premium = sys.modules['utils.user_manager']._mock_users["mock_premium_token"]['user_id']
    test_user_admin = sys.modules['utils.user_manager']._mock_users["mock_admin_token"]['user_id']

    print("\n--- Testing find_flights function ---")

    # Test 1: Premium user with access, valid flight (London to New York)
    print("\n--- Test 1: Premium user with access, valid flight (London to New York) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_premium
    result1 = find_flights("London", "New York", "2025-07-15", user_token=test_user_premium)
    print(f"Result for London to New York (Premium User):\n{result1[:100]}...")
    assert "Flights from London to New York on 2025-07-15:" in result1
    assert "Flight BA177 (British Airways)" in result1
    print("Test 1 Passed.")

    # Test 2: Pro user, access denied (as per mock RBAC)
    print("\n--- Test 2: Pro user, access denied ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result2 = find_flights("Lagos", "London", "2025-07-20", user_token=test_user_pro)
    print(f"Result for Lagos to London (Pro user): {result2}")
    assert "Error: Access to travel planning tools is not enabled for your current tier." in result2
    print("Test 2 Passed.")

    # Test 3: Admin user, flight not found
    print("\n--- Test 3: Admin user, flight not found ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    result3 = find_flights("Berlin", "Rome", "2025-08-10", user_token=test_user_admin)
    print(f"Result for Berlin to Rome (Admin user): {result3}")
    assert "No direct flights found from 'Berlin' to 'Rome' on '2025-08-10'." in result3
    print("Test 3 Passed.")

    # Test 4: Admin user, past date
    print("\n--- Test 4: Admin user, past date ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    result4 = find_flights("London", "New York", "2024-01-01", user_token=test_user_admin)
    print(f"Result for past date (Admin user): {result4}")
    assert "Error: Cannot search for flights in the past." in result4
    print("Test 4 Passed.")

    print("\n--- Testing find_hotels function ---")

    # Test 5: Premium user with access, valid hotel search (Paris)
    print("\n--- Test 5: Premium user with access, valid hotel search (Paris) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_premium
    result5 = find_hotels("Paris", "2025-09-01", "2025-09-05", user_token=test_user_premium)
    print(f"Result for Paris hotels (Premium user):\n{result5[:100]}...")
    assert "Hotels in Paris from 2025-09-01 to 2025-09-05:" in result5
    assert "Hotel Louvre (4 Stars)" in result5
    print("Test 5 Passed.")

    # Test 6: Free user, access denied
    print("\n--- Test 6: Free user, access denied ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result6 = find_hotels("Dubai", "2025-10-10", "2025-10-15", user_token=test_user_free)
    print(f"Result for Dubai hotels (Free user): {result6}")
    assert "Error: Access to travel planning tools is not enabled for your current tier." in result6
    print("Test 6 Passed.")

    # Test 7: Admin user, hotel not found
    print("\n--- Test 7: Admin user, hotel not found ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    result7 = find_hotels("Kyoto", "2025-11-01", "2025-11-05", user_token=test_user_admin)
    print(f"Result for Kyoto hotels (Admin user): {result7}")
    assert "No hotels found in 'Kyoto' for the dates 2025-11-01 to 2025-11-05." in result7
    print("Test 7 Passed.")

    # Test 8: Admin user, invalid date range
    print("\n--- Test 8: Admin user, invalid date range ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    result8 = find_hotels("Paris", "2025-09-05", "2025-09-01", user_token=test_user_admin)
    print(f"Result for invalid date range (Admin user): {result8}")
    assert "Error: Check-out date must be after check-in date." in result8
    print("Test 8 Passed.")

    print("\nAll travel_tool tests passed (mocked data and RBAC).")

    # Restore original requests.get
    requests.get = original_requests_get
