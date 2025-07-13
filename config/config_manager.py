# config/config_manager.py

import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import toml # Import the toml library

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages application configuration, loading from config.yml, api_providers.yml, and secrets.toml.
    Implemented as a singleton to ensure a single, consistent configuration throughout the app.
    This manager now focuses on static application settings and secrets.
    Dynamic configurations like RBAC capabilities and tier hierarchy are handled
    by UserManager loading directly from Firestore.
    """
    _instance = None
    _is_loaded = False
    _config_data: Dict[str, Any] = {}
    _api_providers_data: Dict[str, Any] = {} # To store API provider configurations
    _secrets_data: Dict[str, Any] = {} # To store secrets if not using st.secrets directly

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            if not cls._instance._is_loaded:
                cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """
        Loads configuration from data/config.yml, data/api_providers.yml, and attempts to load secrets.
        This method is called only once when the singleton instance is created.
        """
        if self._is_loaded:
            return

        # Load config.yml
        config_path = Path("data/config.yml")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    self._config_data = yaml.safe_load(f)
                logger.info(f"Loaded config from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config.yml: {e}")
                self._config_data = {} # Ensure it's a dict even on failure
        else:
            logger.warning(f"config.yml not found at {config_path}. Using default empty config.")
            self._config_data = {}

        # Load api_providers.yml
        api_providers_path = Path("data/api_providers.yml")
        if api_providers_path.exists():
            try:
                with open(api_providers_path, 'r') as f:
                    self._api_providers_data = yaml.safe_load(f)
                logger.info(f"Loaded API providers from {api_providers_path}")
            except Exception as e:
                logger.error(f"Error loading api_providers.yml: {e}")
                self._api_providers_data = {} # Ensure it's a dict even on failure
        else:
            logger.warning(f"api_providers.yml not found at {api_providers_path}. Using default empty API providers.")
            self._api_providers_data = {}

        # Attempt to load secrets from secrets.toml if not using Streamlit's st.secrets
        secrets_path = Path(".streamlit/secrets.toml")
        if secrets_path.exists():
            try:
                with open(secrets_path, 'r') as f:
                    self._secrets_data = toml.load(f)
                logger.info(f"Loaded secrets from {secrets_path}")
            except Exception as e:
                logger.warning(f"Error loading secrets.toml: {e}. Secrets will be accessed via st.secrets if available.")
                self._secrets_data = {}
        else:
            logger.info(f"secrets.toml not found at {secrets_path}. Secrets will be accessed via st.secrets if available.")
            self._secrets_data = {}

        self._is_loaded = True

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value using a dot-separated key (e.g., "llm.default_model_name").
        """
        parts = key.split('.')
        current_config = self._config_data
        for part in parts:
            if isinstance(current_config, dict) and part in current_config:
                current_config = current_config[part]
            else:
                return default
        return current_config

    def get_secret(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a secret value.
        Prioritizes loaded Streamlit secrets (if available), otherwise looks in _secrets_data.
        Note: For secrets loaded directly from .toml, keys will be flattened (e.g., 'openai_api_key').
        """
        # In a Streamlit environment, st.secrets is the primary source
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and key in st.secrets:
                return st.secrets[key]
        except ImportError:
            pass # Not in a Streamlit environment

        if key in self._secrets_data:
            return self._secrets_data[key]
        
        logger.warning(f"Secret '{key}' not found in loaded secrets or st.secrets. Returning default.")
        return default

    def set_secret(self, key: str, value: Any):
        """
        Sets a secret value in the in-memory secrets data.
        This is primarily for mocking or dynamic testing in environments
        where st.secrets is not available or for future backend management.
        It does NOT persist to secrets.toml or a database.
        """
        self._secrets_data[key] = value
        logger.info(f"Secret '{key}' set in-memory.")

    def get_api_provider_config(self, domain: str, provider_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the full configuration for a specific API provider within a domain.
        """
        return self._api_providers_data.get(domain, {}).get(provider_name)

    def get_domain_api_providers(self, domain: str) -> Dict[str, Any]:
        """
        Retrieves all configured API providers for a given domain.
        """
        return self._api_providers_data.get(domain, {})

# Instantiate the ConfigManager as a singleton
config_manager = ConfigManager()


# CLI Test (optional)
if __name__ == "__main__":
    import os
    import shutil
    import json
    from unittest.mock import MagicMock, patch

    logging.basicConfig(level=logging.INFO)

    # Clean up old config/secrets files for a clean test run
    if Path("data/config.yml").exists():
        os.remove("data/config.yml")
    if Path("data/api_providers.yml").exists():
        os.remove("data/api_providers.yml")
    if Path(".streamlit/secrets.toml").exists():
        os.remove(".streamlit/secrets.toml")
    if Path(".streamlit").exists():
        shutil.rmtree(".streamlit")
    
    # Create dummy config files for testing
    Path("data").mkdir(exist_ok=True)
    Path(".streamlit").mkdir(exist_ok=True)

    # Dummy config.yml content
    dummy_config_yml_content = """
llm:
  default_provider: gemini
  default_model_name: gemini-1.5-flash
  default_temperature: 0.7
  max_summary_input_tokens: 128000
rag:
  chunk_size: 500
  chunk_overlap: 50
  max_query_results_k: 5
web_scraping:
  user_agent: 'Mozilla/5.0 (Test; Python)'
  timeout_seconds: 15
  max_search_results: 10
"""
    with open("data/config.yml", "w") as f:
        f.write(dummy_config_yml_content)

    # Dummy api_providers.yml content (reflecting new historical data APIs)
    dummy_api_providers_yml_content = """
historical_finance:
  alphavantage:
    base_url: "https://www.alphavantage.co/query"
    api_key_name: "alphavantage_api_key"
    api_key_param_name: "apikey"
    functions:
      get_historical_stock_prices:
        required_params: ["symbol", "function"]
        response_path: ["Time Series (Daily)"]
        data_map:
          open: "1. open"
          close: "4. close"
historical_crypto:
  coingecko:
    base_url: "https://api.coingecko.com/api/v3"
    api_key_name: "coingecko_api_key"
    functions:
      get_historical_crypto_prices:
        endpoint: "/coins/{id}/market_chart"
        path_params: ["id"]
        required_params: ["vs_currency", "days"]
        response_path: ["prices"]
        data_map:
          timestamp: 0
          price: 1
historical_weather:
  mock_historical_weather_provider:
    base_url: "http://mock-historical-weather-api.com/v1"
    api_key_name: "weather_api_key"
    functions:
      get_historical_weather:
        endpoint: "/history"
        required_params: ["location", "start_date", "end_date"]
        response_path: ["history", "daily"]
        data_map:
          date: "date"
          avg_temp_celsius: "avg_temp_c"
"""
    with open("data/api_providers.yml", "w") as f:
        f.write(dummy_api_providers_yml_content)

    # Dummy secrets.toml content
    dummy_secrets_toml_content = """
alphavantage_api_key = "test_alphavantage_key"
coingecko_api_key = "test_coingecko_key"
weather_api_key = "test_weather_key"
gemini_api_key = "test_gemini_key"
openai_api_key = "test_openai_key"
"""
    with open(".streamlit/secrets.toml", "w") as f:
        f.write(dummy_secrets_toml_content)

    # Re-instantiate ConfigManager to load the new dummy files
    # This simulates a fresh application start
    ConfigManager._instance = None
    ConfigManager._is_loaded = False
    config_manager = ConfigManager() # This will call _load_config()

    print("\n--- Testing ConfigManager ---")

    # Test 1: Retrieve general config values
    print("\n--- Test 1: Retrieve general config values ---")
    llm_model = config_manager.get("llm.default_model_name")
    rag_chunk_size = config_manager.get("rag.chunk_size")
    web_timeout = config_manager.get("web_scraping.timeout_seconds")
    print(f"LLM Model: {llm_model}")
    print(f"RAG Chunk Size: {rag_chunk_size}")
    print(f"Web Timeout: {web_timeout}")
    assert llm_model == "gemini-1.5-flash"
    assert rag_chunk_size == 500
    assert web_timeout == 15
    print("Test 1 Passed.")

    # Test 2: Retrieve secrets
    print("\n--- Test 2: Retrieve secrets ---")
    alpha_key = config_manager.get_secret("alphavantage_api_key")
    gemini_key = config_manager.get_secret("gemini_api_key")
    non_existent_key = config_manager.get_secret("non_existent_key", "default_value")
    print(f"Alpha Vantage Key: {alpha_key}")
    print(f"Gemini Key: {gemini_key}")
    print(f"Non-existent Key (default): {non_existent_key}")
    assert alpha_key == "test_alphavantage_key"
    assert gemini_key == "test_gemini_key"
    assert non_existent_key == "default_value"
    print("Test 2 Passed.")

    # Test 3: Retrieve API provider configurations
    print("\n--- Test 3: Retrieve API provider configurations ---")
    alpha_config = config_manager.get_api_provider_config("historical_finance", "alphavantage")
    coingecko_config = config_manager.get_api_provider_config("historical_crypto", "coingecko")
    print(f"Alpha Vantage Config: {json.dumps(alpha_config, indent=2)}")
    print(f"CoinGecko Config: {json.dumps(coingecko_config, indent=2)}")
    assert alpha_config is not None
    assert alpha_config["base_url"] == "https://www.alphavantage.co/query"
    assert "get_historical_crypto_prices" in coingecko_config["functions"]
    print("Test 3 Passed.")

    # Test 4: Set secret dynamically (in-memory)
    print("\n--- Test 4: Set secret dynamically (in-memory) ---")
    config_manager.set_secret("new_dynamic_key", "dynamic_value_123")
    retrieved_dynamic_key = config_manager.get_secret("new_dynamic_key")
    print(f"Dynamic Key: {retrieved_dynamic_key}")
    assert retrieved_dynamic_key == "dynamic_value_123"
    print("Test 4 Passed.")

    # Test 5: Missing config.yml or api_providers.yml
    print("\n--- Test 5: Missing config.yml or api_providers.yml ---")
    # Remove the files and re-instantiate
    os.remove("data/config.yml")
    os.remove("data/api_providers.yml")
    ConfigManager._instance = None
    ConfigManager._is_loaded = False
    new_config_manager = ConfigManager() # This should log warnings

    assert new_config_manager.get("llm.default_model_name") is None # Should be default None
    assert new_config_manager.get_api_provider_config("historical_finance", "alphavantage") is None
    print("Test 5 Passed: Handled missing config files gracefully.")

    print("\nAll ConfigManager tests completed.")

    # Clean up dummy files and directories
    if Path("data").exists():
        shutil.rmtree("data")
    if Path(".streamlit").exists():
        shutil.rmtree(".streamlit")
    print("\nCleaned up dummy config and secrets files.")
