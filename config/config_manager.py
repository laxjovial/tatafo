# config/config_manager.py

import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages application configuration loaded from config.yml and Streamlit secrets.
    Implemented as a singleton to ensure a single, consistent configuration instance.
    """
    _instance = None
    _is_loaded = False
    _config_data: Dict[str, Any] = {}
    _secrets_data: Dict[str, Any] = {} # To store secrets if not using st.secrets directly

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config() # Load configuration upon first instantiation
        return cls._instance

    def _load_config(self):
        """
        Loads configuration from data/config.yml and attempts to load secrets.
        This method is called only once when the singleton instance is created.
        """
        if self._is_loaded:
            return

        config_path = Path("data/config.yml")
        if not config_path.exists():
            logger.warning(f"Configuration file not found at {config_path}. Using default empty config.")
            self._config_data = {}
        else:
            try:
                with open(config_path, "r") as f:
                    self._config_data = yaml.safe_load(f) or {}
                logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config.yml: {e}")
                self._config_data = {} # Fallback to empty config on error

        # Attempt to load Streamlit secrets if available (for frontend)
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                # Convert Streamlit secrets to a standard dictionary for easier access
                # and to avoid direct dependency on st.secrets object structure
                self._secrets_data = {k: v for k, v in st.secrets.items()}
                logger.info("Streamlit secrets loaded.")
            else:
                logger.info("Streamlit secrets object not found. Running outside Streamlit context or secrets not configured.")
        except ImportError:
            logger.info("Streamlit not found. Assuming backend context or standalone script.")
        except Exception as e:
            logger.warning(f"Could not load Streamlit secrets: {e}")

        self._is_loaded = True

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value using a dot-separated key (e.g., "llm.model_name").
        
        Args:
            key (str): The dot-separated key for the configuration value.
            default (Any): The default value to return if the key is not found.
            
        Returns:
            Any: The configuration value or the default value.
        """
        parts = key.split('.')
        value = self._config_data
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def get_secret(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a secret value.
        Prioritizes loaded Streamlit secrets (if available), otherwise looks in _secrets_data.
        In a future backend-managed API key system, this method would be extended
        to fetch from the database if not found locally.
        
        Args:
            key (str): The key for the secret value.
            default (Any): The default value to return if the secret is not found.
            
        Returns:
            Any: The secret value or the default value.
        """
        # First, try to get from the internal _secrets_data (populated by st.secrets or mock)
        if key in self._secrets_data:
            return self._secrets_data[key]
        
        # In a real backend, if not found in _secrets_data, you might try to fetch from a database
        # Example (conceptual, requires database integration):
        # from database.firestore_manager import FirestoreManager
        # firestore_db = FirestoreManager().db
        # api_keys_ref = firestore_db.collection("api_keys").document(key)
        # api_key_doc = api_keys_ref.get()
        # if api_key_doc.exists:
        #     return api_key_doc.to_dict().get('value', default)

        logger.warning(f"Secret '{key}' not found in loaded secrets or Streamlit secrets.")
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

# Instantiate the ConfigManager as a singleton
config_manager = ConfigManager()

