# domain_tools/legal_tools/legal_tool.py

import logging
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool

# Import config_manager for API keys
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability

logger = logging.getLogger(__name__)

# --- Helper Function to get API Keys for Legal APIs ---
def _get_legal_api_key(api_name: str) -> Optional[str]:
    """
    Retrieves the API key for a given legal API from secrets.
    """
    if api_name == "lexisnexis": # Example placeholder for a real API
        return config_manager.get_secret("lexisnexis_api_key")
    # Add other legal API key retrieval logic here if needed
    return None

@tool
def get_legal_definition(term: str, user_token: str = "default") -> str:
    """
    Retrieves the definition of a legal term.
    Uses a mock legal API for demonstration.

    Args:
        term (str): The legal term to define (e.g., "contract", "tort", "habeas corpus").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks.

    Returns:
        str: A string containing the legal definition, or an error message.
    """
    logger.info(f"Tool: get_legal_definition called for term: {term} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'legal_tool_access', False):
        return "Error: Access to legal information tools is not enabled for your current tier."

    # In a real application, you would make an API call here.
    # For demonstration, we'll use mock data.
    mock_definitions = {
        "contract": "A legally binding agreement between two or more parties that creates mutual obligations enforceable by law.",
        "tort": "A civil wrong that causes a claimant to suffer loss or harm, resulting in legal liability for the person who commits the tortious act.",
        "habeas corpus": "A writ requiring a person under arrest to be brought before a court or judge, especially to secure the person's release unless lawful grounds are shown for their detention.",
        "negligence": "Failure to exercise the care that a reasonably prudent person would exercise in like circumstances."
    }

    definition = mock_definitions.get(term.lower())

    if definition:
        return f"Definition of '{term.capitalize()}': {definition}"
    else:
        return f"Legal definition not found for '{term}'. Please check the spelling or try a different term."

@tool
def get_case_summary(case_name: str, user_token: str = "default") -> str:
    """
    Retrieves a summary of a hypothetical legal case.
    Uses a mock legal API for demonstration.

    Args:
        case_name (str): The name of the case (e.g., "Roe v. Wade", "Marbury v. Madison").
                         For mock data, use "Smith v. Jones" or "Doe v. Public".
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks.

    Returns:
        str: A string containing the case summary, or an error message.
    """
    logger.info(f"Tool: get_case_summary called for case: {case_name} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'legal_tool_access', False):
        return "Error: Access to legal information tools is not enabled for your current tier."

    # In a real application, you would make an API call here.
    # For demonstration, we'll use mock data.
    mock_case_summaries = {
        "smith v. jones": {
            "summary": "This hypothetical case involved a dispute over property boundaries. The plaintiff, Mr. Smith, claimed that Mr. Jones had encroached upon his land. The court ruled in favor of Mr. Smith, ordering Mr. Jones to remove the disputed fence.",
            "outcome": "Plaintiff (Smith) won.",
            "key_precedent": "Clarified aspects of adverse possession law."
        },
        "doe v. public": {
            "summary": "A class-action lawsuit concerning consumer privacy against a large tech company. Ms. Doe alleged the company mishandled user data. The case resulted in a significant settlement for the plaintiffs and stricter data handling regulations for the company.",
            "outcome": "Settlement reached in favor of plaintiffs.",
            "key_precedent": "Set new standards for data privacy in the tech industry."
        }
    }

    summary = mock_case_summaries.get(case_name.lower())

    if summary:
        formatted_summary = (
            f"Summary for case '{case_name}':\n"
            f"Summary: {summary['summary']}\n"
            f"Outcome: {summary['outcome']}\n"
            f"Key Precedent: {summary['key_precedent']}"
        )
        return formatted_summary
    else:
        return f"Case summary not found for '{case_name}'. Please check the name or try a different case."

# CLI Test (optional)
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.lexisnexis_api_key = "MOCK_LEXISNEXIS_KEY"
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
            if key == "lexisnexis_api_key": return st.secrets.lexisnexis_api_key
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
                'legal_tool_access': {
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

    # Mock requests.get for external API calls (not strictly needed for this mock, but good practice)
    original_requests_get = requests.get
    requests.get = MagicMock() # Mock all requests.get calls

    test_user_free = sys.modules['utils.user_manager']._mock_users["mock_free_token"]['user_id']
    test_user_pro = sys.modules['utils.user_manager']._mock_users["mock_pro_token"]['user_id']
    test_user_premium = sys.modules['utils.user_manager']._mock_users["mock_premium_token"]['user_id']
    test_user_admin = sys.modules['utils.user_manager']._mock_users["mock_admin_token"]['user_id']

    print("\n--- Testing get_legal_definition function ---")

    # Test 1: Premium user, valid term (Contract)
    print("\n--- Test 1: Premium user, valid term (Contract) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_premium
    result1 = get_legal_definition("Contract", user_token=test_user_premium)
    print(f"Result for Contract (Premium user):\n{result1[:100]}...")
    assert "Definition of 'Contract':" in result1
    assert "legally binding agreement" in result1
    print("Test 1 Passed.")

    # Test 2: Pro user, access denied (as per mock RBAC)
    print("\n--- Test 2: Pro user, access denied ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result2 = get_legal_definition("Tort", user_token=test_user_pro)
    print(f"Result for Tort (Pro user): {result2}")
    assert "Error: Access to legal information tools is not enabled for your current tier." in result2
    print("Test 2 Passed.")

    # Test 3: Admin user, term not found
    print("\n--- Test 3: Admin user, term not found ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    result3 = get_legal_definition("Quantum Meruit", user_token=test_user_admin)
    print(f"Result for Quantum Meruit (Admin user): {result3}")
    assert "Legal definition not found for 'Quantum Meruit'." in result3
    print("Test 3 Passed.")

    print("\n--- Testing get_case_summary function ---")

    # Test 4: Premium user, valid case (Smith v. Jones)
    print("\n--- Test 4: Premium user, valid case (Smith v. Jones) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_premium
    result4 = get_case_summary("Smith v. Jones", user_token=test_user_premium)
    print(f"Result for Smith v. Jones (Premium user):\n{result4[:100]}...")
    assert "Summary for case 'Smith v. Jones':" in result4
    assert "dispute over property boundaries" in result4
    print("Test 4 Passed.")

    # Test 5: Free user, access denied (as per mock RBAC)
    print("\n--- Test 5: Free user, access denied ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result5 = get_case_summary("Doe v. Public", user_token=test_user_free)
    print(f"Result for Doe v. Public (Free user): {result5}")
    assert "Error: Access to legal information tools is not enabled for your current tier." in result5
    print("Test 5 Passed.")

    # Test 6: Admin user, case not found
    print("\n--- Test 6: Admin user, case not found ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    result6 = get_case_summary("Brown v. Board of Education", user_token=test_user_admin)
    print(f"Result for Brown v. Board of Education (Admin user): {result6}")
    assert "Case summary not found for 'Brown v. Board of Education'." in result6
    print("Test 6 Passed.")

    print("\nAll legal_tool tests passed (mocked data and RBAC).")

    # Restore original requests.get
    requests.get = original_requests_get
