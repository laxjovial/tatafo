mplementing config/ Directory - Part 1
The config/ directory contains config_manager.py, which is responsible for loading application settings from data/config.yml and secrets from .streamlit/secrets.toml. This manager is designed as a singleton to ensure consistent access to configuration throughout the application.

1. config/config_manager.py
This file handles the loading and retrieval of configuration values. I've ensured it can handle both the Streamlit st.secrets context (for the frontend) and a more generic approach for the backend (where st.secrets might not be directly available, but we'd eventually connect to a secure backend for secrets).
config/config_manager.py
Jul 2, 7:08 AM

Open

Explanation of Changes and Role:

Singleton Pattern: Remains a singleton to ensure one consistent configuration.

_load_config(): Now explicitly attempts to import streamlit and access st.secrets. If streamlit is not found (e.g., when running the FastAPI backend or standalone scripts), it gracefully skips loading from st.secrets.

_secrets_data: A new internal dictionary _secrets_data is introduced to store secrets. When st.secrets is available, its contents are copied here. This provides a consistent interface for get_secret regardless of whether st.secrets is directly in scope.

get_secret(): This method now first checks _secrets_data. The commented-out section shows where future logic for fetching secrets from a backend database (like Firestore, once we implement that for API key management) would go.

set_secret(): A new method set_secret is added. This is useful for:

Testing: Allows tests to inject mock secret values without needing secrets.toml.

Future Backend Integration: If an admin UI allows setting API keys, the backend service could use this (or a similar mechanism) to update the in-memory cache of secrets after a database write. Crucially, this method does NOT persist the secret to secrets.toml or a database. It's for runtime use.

This updated config_manager.py is more robust and adaptable for both your Streamlit frontend and the future FastAPI backend.


Implementing Dynamic RBAC - Part 2: Refactor config/config_manager.py
The ConfigManager needs to be refactored. Its primary role will now be to manage general application settings (config.yml) and secrets (secrets.toml). The dynamic RBAC capabilities and tier hierarchy will be loaded and managed by utils/user_manager.py directly from Firestore, using the new methods in firestore_manager.py. This separation ensures that ConfigManager remains focused on static app configuration, while UserManager handles the dynamic, user-specific access controls.

1. config/config_manager.py (Refactored for Static Config Only)
I'm removing the direct loading of rbac_capabilities.yaml and tiers.yaml from this file. It will now focus solely on config.yml and st.secrets.
config/config_manager.py (Refactored for Static Config Only)
Jul 3, 2:26 AM
