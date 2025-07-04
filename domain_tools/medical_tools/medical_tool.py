# domain_tools/medical_tools/medical_tool.py

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

# --- Helper Function to get API Keys for Medical APIs ---
def _get_medical_api_key(api_name: str) -> Optional[str]:
    """
    Retrieves the API key for a given medical API from secrets.
    """
    if api_name == "medlineplus": # Example placeholder for a real API
        return config_manager.get_secret("medlineplus_api_key")
    # Add other medical API key retrieval logic here if needed
    return None

@tool
def get_drug_info(drug_name: str, user_token: str = "default") -> str:
    """
    Retrieves information about a specific drug, including its uses, side effects,
    and dosage.
    Uses a mock medical API for demonstration.

    Args:
        drug_name (str): The name of the drug (e.g., "Aspirin", "Paracetamol").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks.

    Returns:
        str: A string containing drug information, or an error message.
    """
    logger.info(f"Tool: get_drug_info called for drug: {drug_name} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'medical_tool_access', False):
        return "Error: Access to medical information tools is not enabled for your current tier."

    # In a real application, you would make an API call here.
    # For demonstration, we'll use mock data.
    mock_drug_data = {
        "aspirin": {
            "uses": "Pain relief, fever reduction, anti-inflammatory, blood thinner.",
            "side_effects": "Stomach upset, heartburn, nausea, vomiting. Serious side effects include bleeding.",
            "dosage": "Adults: 325-650 mg every 4-6 hours as needed. Max 4000 mg/day.",
            "warnings": "Do not use in children/teenagers with flu-like symptoms or chickenpox due to Reye's syndrome risk."
        },
        "paracetamol": {
            "uses": "Pain relief, fever reduction.",
            "side_effects": "Rarely, skin rash, allergic reactions. Overdose can cause liver damage.",
            "dosage": "Adults: 500-1000 mg every 4-6 hours as needed. Max 4000 mg/day.",
            "warnings": "Do not exceed recommended dose. Avoid alcohol while taking."
        },
        "ibuprofen": {
            "uses": "Pain relief, fever reduction, anti-inflammatory (e.g., for arthritis, menstrual pain).",
            "side_effects": "Stomach upset, nausea, headache, dizziness. Can increase risk of heart attack or stroke.",
            "dosage": "Adults: 200-400 mg every 4-6 hours as needed. Max 1200 mg/day (OTC).",
            "warnings": "Long-term use may cause stomach bleeding or kidney problems. Consult doctor if you have heart conditions."
        }
    }

    drug_info = mock_drug_data.get(drug_name.lower())

    if drug_info:
        formatted_info = (
            f"Information for {drug_name.capitalize()}:\n"
            f"Uses: {drug_info['uses']}\n"
            f"Side Effects: {drug_info['side_effects']}\n"
            f"Dosage: {drug_info['dosage']}\n"
            f"Warnings: {drug_info['warnings']}"
        )
        return formatted_info
    else:
        return f"Drug information not found for '{drug_name}'. Please check the spelling or try a different drug."

@tool
def get_symptom_info(symptom_name: str, user_token: str = "default") -> str:
    """
    Retrieves information about a specific medical symptom, including common causes,
    associated conditions, and when to seek medical attention.
    Uses a mock medical API for demonstration.

    Args:
        symptom_name (str): The name of the symptom (e.g., "Headache", "Fever", "Cough").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks.

    Returns:
        str: A string containing symptom information, or an error message.
    """
    logger.info(f"Tool: get_symptom_info called for symptom: {symptom_name} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'medical_tool_access', False):
        return "Error: Access to medical information tools is not enabled for your current tier."

    # In a real application, you would make an API call here.
    # For demonstration, we'll use mock data.
    mock_symptom_data = {
        "headache": {
            "common_causes": "Stress, dehydration, lack of sleep, eye strain, caffeine withdrawal.",
            "associated_conditions": "Migraine, tension headache, sinus infection, common cold, flu.",
            "when_to_seek_medical_attention": "Sudden, severe headache; headache with fever, stiff neck, rash, confusion; headache after head injury; persistent headache."
        },
        "fever": {
            "common_causes": "Infections (viral or bacterial), inflammation, heat exhaustion, certain medications.",
            "associated_conditions": "Common cold, flu, pneumonia, urinary tract infection, strep throat.",
            "when_to_seek_medical_attention": "Fever over 103°F (39.4°C); fever with severe headache, stiff neck, rash, confusion; fever in infants under 3 months."
        },
        "cough": {
            "common_causes": "Common cold, flu, allergies, asthma, acid reflux, smoking.",
            "associated_conditions": "Bronchitis, pneumonia, whooping cough, COPD.",
            "when_to_seek_medical_attention": "Cough lasting more than a few weeks; cough with blood; severe chest pain; difficulty breathing; fever."
        }
    }

    symptom_info = mock_symptom_data.get(symptom_name.lower())

    if symptom_info:
        formatted_info = (
            f"Information for symptom: {symptom_name.capitalize()}:\n"
            f"Common Causes: {symptom_info['common_causes']}\n"
            f"Associated Conditions: {symptom_info['associated_conditions']}\n"
            f"When to Seek Medical Attention: {symptom_info['when_to_seek_medical_attention']}"
        )
        return formatted_info
    else:
        return f"Symptom information not found for '{symptom_name}'. Please check the spelling or try a different symptom."


# CLI Test (optional)
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.medlineplus_api_key = "MOCK_MEDLINEPLUS_KEY"
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
            if key == "medlineplus_api_key": return st.secrets.medlineplus_api_key
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
                'medical_tool_access': {
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

    print("\n--- Testing get_drug_info function ---")

    # Test 1: Premium user, valid drug (Aspirin)
    print("\n--- Test 1: Premium user, valid drug (Aspirin) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_premium
    result1 = get_drug_info("Aspirin", user_token=test_user_premium)
    print(f"Result for Aspirin (Premium user):\n{result1[:100]}...")
    assert "Information for Aspirin:" in result1
    assert "Pain relief, fever reduction" in result1
    print("Test 1 Passed.")

    # Test 2: Free user, access denied
    print("\n--- Test 2: Free user, access denied ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result2 = get_drug_info("Paracetamol", user_token=test_user_free)
    print(f"Result for Paracetamol (Free user): {result2}")
    assert "Error: Access to medical information tools is not enabled for your current tier." in result2
    print("Test 2 Passed.")

    # Test 3: Admin user, drug not found
    print("\n--- Test 3: Admin user, drug not found ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    result3 = get_drug_info("NonExistentDrug", user_token=test_user_admin)
    print(f"Result for NonExistentDrug (Admin user): {result3}")
    assert "Drug information not found for 'NonExistentDrug'." in result3
    print("Test 3 Passed.")

    print("\n--- Testing get_symptom_info function ---")

    # Test 4: Premium user, valid symptom (Headache)
    print("\n--- Test 4: Premium user, valid symptom (Headache) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_premium
    result4 = get_symptom_info("Headache", user_token=test_user_premium)
    print(f"Result for Headache (Premium user):\n{result4[:100]}...")
    assert "Information for symptom: Headache:" in result4
    assert "Stress, dehydration, lack of sleep" in result4
    print("Test 4 Passed.")

    # Test 5: Pro user, access denied (as per mock RBAC)
    print("\n--- Test 5: Pro user, access denied (as per mock RBAC) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result5 = get_symptom_info("Fever", user_token=test_user_pro)
    print(f"Result for Fever (Pro user): {result5}")
    assert "Error: Access to medical information tools is not enabled for your current tier." in result5
    print("Test 5 Passed.")

    # Test 6: Admin user, symptom not found
    print("\n--- Test 6: Admin user, symptom not found ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    result6 = get_symptom_info("UnknownSymptom", user_token=test_user_admin)
    print(f"Result for UnknownSymptom (Admin user): {result6}")
    assert "Symptom information not found for 'UnknownSymptom'." in result6
    print("Test 6 Passed.")

    print("\nAll medical_tool tests passed (mocked data and RBAC).")

    # Restore original requests.get
    requests.get = original_requests_get
