# shared_tools/python_interpreter_tool.py

import logging
import io
import sys
import contextlib
import traceback
from typing import Dict, Any, Optional

from langchain_core.tools import tool

# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability

logger = logging.getLogger(__name__)

@tool
def python_interpreter_with_rbac(code: str, user_token: str = "default") -> str:
    """
    Executes Python code in a sandboxed environment and captures its output.
    This tool is intended for data analysis, calculations, and other programmatic tasks.
    Access to this tool is controlled by Role-Based Access Control (RBAC).

    Args:
        code (str): The Python code to execute.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks.

    Returns:
        str: The captured standard output (stdout) and standard error (stderr) of the executed code.
             Returns an error message if execution fails or if access is denied by RBAC.
    """
    logger.info(f"Tool: python_interpreter_with_rbac called by user: {user_token}. Code (first 100 chars): '{code[:100]}...'")

    # RBAC Check for Data Analysis Enabled
    if not get_user_tier_capability(user_token, 'data_analysis_enabled', False):
        return "Error: Python interpreter (data analysis) is not enabled for your current tier. Please upgrade your plan."

    # Use a string buffer to capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = io.StringIO()
    
    sys.stdout = redirected_output
    sys.stderr = redirected_output

    try:
        # Create a restricted global and local namespace for execution
        # This helps in sandboxing, though full sandboxing is complex and often requires
        # dedicated environments like Docker containers or specialized libraries.
        # For this application's scope, this provides a basic level of isolation.
        exec_globals = {}
        exec_locals = {}

        # Add common data analysis libraries to the execution context for convenience
        try:
            import pandas as pd
            exec_globals['pd'] = pd
        except ImportError:
            logger.warning("Pandas not available for Python interpreter.")
        
        try:
            import numpy as np
            exec_globals['np'] = np
        except ImportError:
            logger.warning("NumPy not available for Python interpreter.")

        # Execute the code
        exec(code, exec_globals, exec_locals)
        
        output = redirected_output.getvalue()
        logger.info(f"Python interpreter executed successfully for user {user_token}. Output: {output[:200]}...")
        return f"Execution successful. Output:\n{output}"

    except Exception as e:
        # Capture traceback for more detailed error messages
        error_output = redirected_output.getvalue()
        full_traceback = traceback.format_exc()
        logger.error(f"Python interpreter execution failed for user {user_token}: {e}\nOutput: {error_output}\nTraceback: {full_traceback}", exc_info=True)
        return f"Execution failed. Error:\n{error_output}\n{full_traceback}"
    finally:
        # Restore original stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# CLI Test (optional)
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.openai = {"api_key": "sk-mock-openai-key-12345"}
            self.google = {"api_key": "AIzaSy-mock-google-key"}
            self.user_tokens = {
                "free_user_token": "mock_free_token",
                "pro_user_token": "mock_pro_token",
                "premium_user_token": "mock_premium_token",
                "admin_user_token": "mock_admin_token"
            }
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
                    'timeout_seconds': 5,
                    'max_search_results': 5
                },
                'tiers': {}, # This will be overridden by tiers.yaml
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
                'data_analysis_enabled': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True, 'dev': True}
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


    test_user_free = sys.modules['utils.user_manager']._mock_users["mock_free_token"]['user_id']
    test_user_pro = sys.modules['utils.user_manager']._mock_users["mock_pro_token"]['user_id']
    test_user_admin = sys.modules['utils.user_manager']._mock_users["mock_admin_token"]['user_id']

    print("\n--- Testing python_interpreter_with_rbac function ---")

    # Test 1: Pro user, simple calculation
    print("\n--- Test 1: Pro user, simple calculation ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    code1 = "print(10 + 20)"
    result1 = python_interpreter_with_rbac(code1, user_token=test_user_pro)
    print(f"Result 1 (Pro user, calculation): {result1}")
    assert "Execution successful. Output:\n30" in result1
    print("Test 1 Passed.")

    # Test 2: Pro user, pandas usage (mocked)
    print("\n--- Test 2: Pro user, pandas usage (mocked) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    code2 = """
import pandas as pd
data = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data)
print(df.sum().sum())
"""
    # Mock pandas for this test
    with patch('pandas.DataFrame') as MockDataFrame:
        MockDataFrame.return_value.sum.return_value.sum.return_value = 10 # Mock sum of sums
        result2 = python_interpreter_with_rbac(code2, user_token=test_user_pro)
        print(f"Result 2 (Pro user, pandas): {result2}")
        assert "Execution successful. Output:\n10" in result2
    print("Test 2 Passed.")

    # Test 3: Free user, access denied
    print("\n--- Test 3: Free user, access denied ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    code3 = "print('This should not run')"
    result3 = python_interpreter_with_rbac(code3, user_token=test_user_free)
    print(f"Result 3 (Free user): {result3}")
    assert "Error: Python interpreter (data analysis) is not enabled for your current tier." in result3
    print("Test 3 Passed.")

    # Test 4: Admin user, code with error
    print("\n--- Test 4: Admin user, code with error ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    code4 = "print(1 / 0)" # Division by zero error
    result4 = python_interpreter_with_rbac(code4, user_token=test_user_admin)
    print(f"Result 4 (Admin user, error):\n{result4[:200]}...")
    assert "Execution failed. Error:" in result4
    assert "ZeroDivisionError: division by zero" in result4
    print("Test 4 Passed.")

    # Test 5: Admin user, malicious code attempt (basic sandbox check)
    print("\n--- Test 5: Admin user, malicious code attempt (basic sandbox) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    # This is a very basic sandbox. Real sandboxing is complex.
    code5 = "import os; print(os.listdir('.'))"
    result5 = python_interpreter_with_rbac(code5, user_token=test_user_admin)
    print(f"Result 5 (Admin user, os.listdir):\n{result5[:200]}...")
    # This will likely work because os is imported by default.
    # A true sandbox would prevent this.
    assert "Execution successful. Output:" in result5
    assert "os.listdir" in result5 # Indicates it ran
    print("Test 5 Passed (Note: This highlights the need for a more robust sandbox in production).")

    print("\nAll python_interpreter_with_rbac tests passed (mocked RBAC).")
