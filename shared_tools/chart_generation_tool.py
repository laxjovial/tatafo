# shared_tools/chart_generation_tool.py

import logging
import json
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
import uuid
import os

from langchain_core.tools import tool

# Import config_manager and user_manager for RBAC checks (if needed within the tool itself)
from config.config_manager import config_manager
from utils.user_manager import get_user_tier_capability

logger = logging.getLogger(__name__)

# Base directory for storing generated charts
BASE_CHART_DIR = Path("charts")

@tool
def generate_and_save_chart(
    data_json: str,
    chart_type: str,
    x_column: str,
    y_column: str,
    title: str = "Generated Chart",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    user_token: str = "default", # User token for personalized chart storage/RBAC
    export_enabled: bool = False # Flag to check if chart export is allowed by RBAC
) -> str:
    """
    Generates a chart (e.g., line, bar, scatter) from JSON data and saves it as an image file.
    The chart image file path is returned. This tool is useful for visualizing data.

    Args:
        data_json (str): A JSON string representing the data to plot.
                         Expected format: a list of dictionaries, where each dictionary
                         is a row and keys are column names.
                         Example: '[{"date": "2023-01-01", "value": 10}, {"date": "2023-01-02", "value": 12}]'
        chart_type (str): The type of chart to generate. Supported: 'line', 'bar', 'scatter'.
        x_column (str): The name of the column to use for the X-axis.
        y_column (str): The name of the column to use for the Y-axis.
        title (str, optional): The title of the chart. Defaults to "Generated Chart".
        x_label (str, optional): Label for the X-axis. Defaults to x_column.
        y_label (str, optional): Label for the Y-axis. Defaults to y_column.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for creating user-specific chart directories.
        export_enabled (bool, optional): Flag indicating if chart export is enabled for the user's tier.
                                         This is checked by the calling agent/UI based on RBAC.

    Returns:
        str: The file path to the generated chart image, or an error message.
             The frontend can then display this image and provide a download link.
    """
    logger.info(f"Tool: generate_and_save_chart called for type '{chart_type}' with x='{x_column}', y='{y_column}'")

    # RBAC Check for Chart Generation (if called directly by an agent)
    # If this tool is called by an agent, the agent should already have checked this capability.
    # However, a redundant check here ensures no bypass.
    if not get_user_tier_capability(user_token, 'chart_generation_enabled', False):
        return "Error: Chart generation is not enabled for your current tier."
    
    # RBAC Check for Chart Export (if called directly by an agent)
    # The `export_enabled` flag passed to the tool should reflect the user's tier capability.
    # This tool itself doesn't directly check 'chart_export_enabled' from config,
    # it relies on the boolean flag passed to it.

    try:
        data = json.loads(data_json)
        if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
            raise ValueError("Input data_json must be a JSON array of objects.")
        
        if not data:
            return "Error: No data provided to generate chart."

        df = pd.DataFrame(data)

        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Specified columns '{x_column}' or '{y_column}' not found in data."

        # Create user-specific chart directory
        user_chart_dir = BASE_CHART_DIR / user_token
        user_chart_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        filename = f"chart_{uuid.uuid4().hex}.png"
        file_path = user_chart_dir / filename

        plt.figure(figsize=(10, 6))

        if chart_type == 'line':
            plt.plot(df[x_column], df[y_column])
        elif chart_type == 'bar':
            plt.bar(df[x_column], df[y_column])
        elif chart_type == 'scatter':
            plt.scatter(df[x_column], df[y_column])
        else:
            plt.close() # Close the figure to free memory
            return f"Error: Unsupported chart type '{chart_type}'. Supported types are 'line', 'bar', 'scatter'."

        plt.title(title)
        plt.xlabel(x_label if x_label else x_column)
        plt.ylabel(y_label if y_label else y_column)
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(file_path)
        plt.close() # Close the figure to free memory

        logger.info(f"Chart '{chart_type}' saved to: {file_path}")
        return str(file_path) # Return the string representation of the Path
    
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON data provided: {data_json}", exc_info=True)
        return "Error: Invalid JSON data provided for chart generation."
    except ValueError as ve:
        logger.error(f"Data error for chart generation: {ve}", exc_info=True)
        return f"Error: Data processing failed for chart generation: {ve}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during chart generation: {e}", exc_info=True)
        return f"An unexpected error occurred during chart generation: {e}"

# CLI Test (optional)
if __name__ == "__main__":
    import shutil
    from unittest.mock import MagicMock
    import sys

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
                'web_scraping': {'user_agent': 'Mozilla/5.0 (Test; Python)', 'timeout_seconds': 5, 'max_search_results': 5},
                'tiers': {}, # This will be overridden by tiers.yaml
                'default_user_tier': 'free',
                'default_user_roles': ['user']
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
                'chart_generation_enabled': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'chart_export_enabled': {
                    'default': False,
                    'roles': {'premium': True, 'admin': True}
                }
            }
        }
        _tier_hierarchy = {
            "free": 0, "user": 1, "basic": 2, "pro": 3, "premium": 4, "admin": 99
        }

        def get_current_user(self) -> Dict[str, Any]:
            # This mock needs to be set externally for specific tests
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
    
    # Mock config_manager and user_manager
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager
    sys.modules['utils.user_manager'] = MockUserManager()
    sys.modules['utils.user_manager']._RBAC_CAPABILITIES = MockUserManager()._rbac_capabilities
    sys.modules['utils.user_manager']._TIER_HIERARCHY = MockUserManager()._tier_hierarchy


    print("\n--- Testing generate_and_save_chart tool ---")
    test_user_pro = sys.modules['utils.user_manager']._mock_users["mock_pro_token"]['user_id']
    test_user_premium = sys.modules['utils.user_manager']._mock_users["mock_premium_token"]['user_id']
    test_user_free = sys.modules['utils.user_manager']._mock_users["mock_free_token"]['user_id']
    test_user_admin = sys.modules['utils.user_manager']._mock_users["mock_admin_token"]['user_id']


    sample_data = [
        {"month": "Jan", "sales": 100, "expenses": 50},
        {"month": "Feb", "sales": 120, "expenses": 60},
        {"month": "Mar", "sales": 90, "expenses": 55},
        {"month": "Apr", "sales": 130, "expenses": 70},
    ]
    sample_data_json = json.dumps(sample_data)

    # Clean up charts directory from previous runs
    if BASE_CHART_DIR.exists():
        shutil.rmtree(BASE_CHART_DIR)
    BASE_CHART_DIR.mkdir(exist_ok=True)


    # Test 1: Pro user - Chart generation enabled
    print("\n--- Test 1: Pro user, Line chart ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro # Set current user for mock
    chart_path_pro = generate_and_save_chart(
        data_json=sample_data_json,
        chart_type="line",
        x_column="month",
        y_column="sales",
        title="Monthly Sales (Pro User)",
        user_token=test_user_pro,
        export_enabled=True # Assume export is allowed for Pro tier
    )
    print(f"Chart path (Pro user): {chart_path_pro}")
    assert isinstance(chart_path_pro, str) and Path(chart_path_pro).exists()
    assert Path(chart_path_pro).parent.name == test_user_pro
    print("Test 1 Passed: Chart generated for Pro user.")

    # Test 2: Premium user - Chart generation and export enabled
    print("\n--- Test 2: Premium user, Bar chart ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_premium # Set current user for mock
    chart_path_premium = generate_and_save_chart(
        data_json=sample_data_json,
        chart_type="bar",
        x_column="month",
        y_column="expenses",
        title="Monthly Expenses (Premium User)",
        user_token=test_user_premium,
        export_enabled=True # Assume export is allowed for Premium tier
    )
    print(f"Chart path (Premium user): {chart_path_premium}")
    assert isinstance(chart_path_premium, str) and Path(chart_path_premium).exists()
    assert Path(chart_path_premium).parent.name == test_user_premium
    print("Test 2 Passed: Chart generated for Premium user.")

    # Test 3: Free user - Chart generation disabled
    print("\n--- Test 3: Free user, Chart generation disabled ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free # Set current user for mock
    error_message_free = generate_and_save_chart(
        data_json=sample_data_json,
        chart_type="line",
        x_column="month",
        y_column="sales",
        title="Monthly Sales (Free User)",
        user_token=test_user_free,
        export_enabled=False # Export not allowed for Free tier
    )
    print(f"Result (Free user): {error_message_free}")
    assert "Error: Chart generation is not enabled for your current tier." in error_message_free
    print("Test 3 Passed: Chart generation correctly denied for Free user.")

    # Test 4: Admin user - Chart generation enabled (via admin override)
    print("\n--- Test 4: Admin user, Scatter chart ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin # Set current user for mock
    chart_path_admin = generate_and_save_chart(
        data_json=sample_data_json,
        chart_type="scatter",
        x_column="sales",
        y_column="expenses",
        title="Sales vs Expenses (Admin User)",
        user_token=test_user_admin,
        export_enabled=True # Export allowed for Admin tier
    )
    print(f"Chart path (Admin user): {chart_path_admin}")
    assert isinstance(chart_path_admin, str) and Path(chart_path_admin).exists()
    assert Path(chart_path_admin).parent.name == test_user_admin
    print("Test 4 Passed: Chart generated for Admin user.")


    # Test 5: Invalid chart type
    print("\n--- Test 5: Invalid chart type ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro # Set current user for mock
    invalid_chart_type_result = generate_and_save_chart(
        data_json=sample_data_json,
        chart_type="pie", # Invalid type
        x_column="month",
        y_column="sales",
        user_token=test_user_pro
    )
    print(f"Result (Invalid chart type): {invalid_chart_type_result}")
    assert "Error: Unsupported chart type 'pie'" in invalid_chart_type_result
    print("Test 5 Passed: Invalid chart type handled.")

    # Test 6: Invalid JSON data
    print("\n--- Test 6: Invalid JSON data ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro # Set current user for mock
    invalid_json_result = generate_and_save_chart(
        data_json="not a json string",
        chart_type="line",
        x_column="month",
        y_column="sales",
        user_token=test_user_pro
    )
    print(f"Result (Invalid JSON): {invalid_json_result}")
    assert "Error: Invalid JSON data provided" in invalid_json_result
    print("Test 6 Passed: Invalid JSON data handled.")

    # Test 7: Missing columns
    print("\n--- Test 7: Missing columns ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro # Set current user for mock
    missing_columns_result = generate_and_save_chart(
        data_json=sample_data_json,
        chart_type="line",
        x_column="non_existent_col_x",
        y_column="non_existent_col_y",
        user_token=test_user_pro
    )
    print(f"Result (Missing columns): {missing_columns_result}")
    assert "Error: Specified columns 'non_existent_col_x' or 'non_existent_col_y' not found in data." in missing_columns_result
    print("Test 7 Passed: Missing columns handled.")

    print("\nAll chart generation tests passed (mocked RBAC).")

    # Clean up generated chart files and directories
    if BASE_CHART_DIR.exists():
        shutil.rmtree(BASE_CHART_DIR)
        print(f"\nCleaned up chart directory: {BASE_CHART_DIR}")
