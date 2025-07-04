# domain_tools/education_tools/education_tool.py

import logging
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool

# Import config_manager for API keys
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability

logger = logging.getLogger(__name__)

# --- Helper Function to get API Keys for Education APIs ---
def _get_education_api_key(api_name: str) -> Optional[str]:
    """
    Retrieves the API key for a given education API from secrets.
    """
    if api_name == "wikipedia": # Example placeholder for a real API
        return config_manager.get_secret("wikipedia_api_key")
    # Add other education API key retrieval logic here if needed
    return None

@tool
def get_academic_definition(term: str, user_token: str = "default") -> str:
    """
    Retrieves the definition of an academic or scientific term.
    Uses a mock education API for demonstration.

    Args:
        term (str): The academic term to define (e.g., "photosynthesis", "democracy", "quantum mechanics").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks.

    Returns:
        str: A string containing the academic definition, or an error message.
    """
    logger.info(f"Tool: get_academic_definition called for term: {term} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'education_tool_access', False):
        return "Error: Access to educational information tools is not enabled for your current tier."

    # In a real application, you would make an API call here (e.g., Wikipedia API, dictionary API).
    # For demonstration, we'll use mock data.
    mock_definitions = {
        "photosynthesis": "The process used by plants, algae, and cyanobacteria to convert light energy into chemical energy, thereby converting carbon dioxide and water into sugars and oxygen.",
        "democracy": "A system of government by the whole population or all the eligible members of a state, typically through elected representatives.",
        "quantum mechanics": "A fundamental theory in physics that describes the behavior of matter and light at the atomic and subatomic scales."
    }

    definition = mock_definitions.get(term.lower())

    if definition:
        return f"Definition of '{term.capitalize()}': {definition}"
    else:
        return f"Academic definition not found for '{term}'. Please check the spelling or try a different term."

@tool
def get_historical_event_summary(event_name: str, user_token: str = "default") -> str:
    """
    Retrieves a summary of a significant historical event.
    Uses a mock education API for demonstration.

    Args:
        event_name (str): The name of the historical event (e.g., "World War II", "French Revolution").
                          For mock data, use "Moon Landing" or "Fall of the Berlin Wall".
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks.

    Returns:
        str: A string containing the event summary, or an error message.
    """
    logger.info(f"Tool: get_historical_event_summary called for event: {event_name} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'education_tool_access', False):
        return "Error: Access to educational information tools is not enabled for your current tier."

    # In a real application, you would make an API call here (e.g., Wikipedia API, historical events API).
    # For demonstration, we'll use mock data.
    mock_event_summaries = {
        "moon landing": {
            "summary": "The Apollo 11 mission achieved the first human landing on the Moon on July 20, 1969. Neil Armstrong and Buzz Aldrin were the first two humans to walk on the lunar surface, marking a pivotal moment in space exploration.",
            "date": "July 20, 1969",
            "significance": "Fulfilled President Kennedy's goal of landing a man on the Moon and returning him safely to Earth before the end of the 1960s. A monumental achievement for humanity."
        },
        "fall of the berlin wall": {
            "summary": "The Berlin Wall, a barrier that physically and ideologically divided East and West Berlin, fell on November 9, 1989. This event symbolized the collapse of communism in Eastern Europe and paved the way for German reunification.",
            "date": "November 9, 1989",
            "significance": "A key event in the end of the Cold War and the reunification of Germany."
        }
    }

    summary = mock_event_summaries.get(event_name.lower())

    if summary:
        formatted_summary = (
            f"Summary for historical event '{event_name}':\n"
            f"Summary: {summary['summary']}\n"
            f"Date: {summary['date']}\n"
            f"Significance: {summary['significance']}"
        )
        return formatted_summary
    else:
        return f"Historical event summary not found for '{event_name}'. Please check the name or try a different event."

# CLI Test (optional)
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.wikipedia_api_key = "MOCK_WIKIPEDIA_KEY"
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
            if key == "wikipedia_api_key": return st.secrets.wikipedia_api_key
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
                'education_tool_access': {
                    'default': False,
                    'roles': {'user': True, 'pro': True, 'premium': True, 'admin': True} # Often basic users get access to educational content
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

    print("\n--- Testing get_academic_definition function ---")

    # Test 1: User with access, valid term (Photosynthesis)
    print("\n--- Test 1: User with access, valid term (Photosynthesis) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_user
    result1 = get_academic_definition("Photosynthesis", user_token=test_user_user)
    print(f"Result for Photosynthesis (User):\n{result1[:100]}...")
    assert "Definition of 'Photosynthesis':" in result1
    assert "convert light energy into chemical energy" in result1
    print("Test 1 Passed.")

    # Test 2: Free user, access denied (as per mock RBAC for 'free')
    print("\n--- Test 2: Free user, access denied ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result2 = get_academic_definition("Democracy", user_token=test_user_free)
    print(f"Result for Democracy (Free user): {result2}")
    assert "Error: Access to educational information tools is not enabled for your current tier." in result2
    print("Test 2 Passed.")

    # Test 3: Admin user, term not found
    print("\n--- Test 3: Admin user, term not found ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    result3 = get_academic_definition("String Theory", user_token=test_user_admin)
    print(f"Result for String Theory (Admin user): {result3}")
    assert "Academic definition not found for 'String Theory'." in result3
    print("Test 3 Passed.")

    print("\n--- Testing get_historical_event_summary function ---")

    # Test 4: User with access, valid event (Moon Landing)
    print("\n--- Test 4: User with access, valid event (Moon Landing) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_user
    result4 = get_historical_event_summary("Moon Landing", user_token=test_user_user)
    print(f"Result for Moon Landing (User):\n{result4[:100]}...")
    assert "Summary for historical event 'Moon Landing':" in result4
    assert "first human landing on the Moon" in result4
    print("Test 4 Passed.")

    # Test 5: Free user, access denied (as per mock RBAC)
    print("\n--- Test 5: Free user, access denied ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result5 = get_historical_event_summary("Fall of the Berlin Wall", user_token=test_user_free)
    print(f"Result for Fall of the Berlin Wall (Free user): {result5}")
    assert "Error: Access to educational information tools is not enabled for your current tier." in result5
    print("Test 5 Passed.")

    # Test 6: Admin user, event not found
    print("\n--- Test 6: Admin user, event not found ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    result6 = get_historical_event_summary("Battle of Hastings", user_token=test_user_admin)
    print(f"Result for Battle of Hastings (Admin user): {result6}")
    assert "Historical event summary not found for 'Battle of Hastings'." in result6
    print("Test 6 Passed.")

    print("\nAll education_tool tests passed (mocked data and RBAC).")

    # Restore original requests.get
    requests.get = original_requests_get
