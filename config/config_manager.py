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
        
        # Load api_providers.yml
        api_providers_path = Path("data/api_providers.yml")
        if not api_providers_path.exists():
            logger.warning(f"API Providers configuration file not found at {api_providers_path}. API integrations may not work.")
            self._api_providers_data = {}
        else:
            try:
                with open(api_providers_path, "r") as f:
                    self._api_providers_data = yaml.safe_load(f).get('api_providers', {}) or {}
                logger.info(f"API Providers configuration loaded from {api_providers_path}")
            except Exception as e:
                logger.error(f"Error loading api_providers.yml: {e}")
                self._api_providers_data = {} # Fallback to empty on error

        # Attempt to load Streamlit secrets first (for Streamlit apps)
        streamlit_secrets_loaded = False
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and st.secrets: # Check if st.secrets exists and is not empty
                self._secrets_data = {k: v for k, v in st.secrets.items()}
                logger.info("Streamlit secrets loaded.")
                streamlit_secrets_loaded = True
            else:
                logger.info("Streamlit secrets object not found or empty. Running outside Streamlit context or secrets not configured via Streamlit.")
        except ImportError:
            logger.info("Streamlit not found. Assuming backend context or standalone script.")
        except Exception as e:
            logger.warning(f"Could not load Streamlit secrets: {e}")

        # If Streamlit secrets didn't load, try loading from .streamlit/secrets.toml directly
        if not streamlit_secrets_loaded:
            secrets_toml_path = Path(".streamlit/secrets.toml")
            if secrets_toml_path.exists():
                try:
                    # Load secrets from the TOML file
                    parsed_secrets = toml.load(secrets_toml_path)
                    # Flatten the dictionary to match how st.secrets would present it
                    # This handles sections like [openai] and [google]
                    flattened_secrets = {}
                    for k, v in parsed_secrets.items():
                        if isinstance(v, dict):
                            for sub_k, sub_v in v.items():
                                flattened_secrets[f"{k}_{sub_k}"] = sub_v # e.g., openai_api_key
                        else:
                            flattened_secrets[k] = v
                    self._secrets_data.update(flattened_secrets)
                    logger.info(f"Secrets loaded directly from {secrets_toml_path}.")
                except Exception as e:
                    logger.error(f"Error loading secrets from {secrets_toml_path}: {e}")
            else:
                logger.warning(f"No secrets found. Valid paths for a secrets.toml file: {secrets_toml_path}")


        self._is_loaded = True

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value using a dot-separated key (e.g., "llm.model_name").
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
        Note: For secrets loaded directly from .toml, keys will be flattened (e.g., 'openai_api_key').
        """
        if key in self._secrets_data:
            return self._secrets_data[key]
        
        logger.warning(f"Secret '{key}' not found in loaded secrets. Returning default.")
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
