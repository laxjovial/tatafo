We will now refactor utils/user_manager.py to load RBAC capabilities and tier hierarchy directly from Firestore using firestore_manager.py. This is where the dynamic loading will truly happen. This will also require careful handling of asynchronous calls during module initialization.

Please type "next" when you are ready.
Here is our updated main To-Do list. We are now working on the utils/ directory to enable dynamic RBAC configuration.

[x] backend/

[x] config/

[x] data/

[x] database/

[x] shared_tools/

[ ] domain_tools/ (and its sub-folders)

[ ] ui/

[ ] utils/ (Refactoring user_manager.py)

[ ] tests/

[ ] docs/

[ ] scripts/

[ ] requirements.txt

[ ] .streamlit/

[ ] .gitignore

[ ] README.md

Implementing Dynamic RBAC - Part 3: Refactor utils/user_manager.py
This is a crucial step. We will modify utils/user_manager.py to load the _RBAC_CAPABILITIES and _TIER_HIERARCHY directly from Firestore using the firestore_manager. This will allow administrators to dynamically update these configurations via the backend/UI.

Since Firestore operations are asynchronous, we need to adapt the loading mechanism. For module-level initialization, we'll use asyncio.run to execute the async loading functions once.

1. utils/user_manager.py (Refactored for Dynamic RBAC from Firestore)
utils/user_manager.py (Refactored for Dynamic RBAC from Firestore)
Jul 3, 2:26 AM

Open

Key Changes in utils/user_manager.py:

Firebase Initialization: The explicit initialize_firebase() function is now a no-op, as Firebase initialization is handled by the firestore_manager singleton's instantiation (_ = firestore_manager). This ensures Firebase is initialized before any Firestore calls.

Dynamic RBAC and Tier Loading from Firestore:

_RBAC_CAPABILITIES and _TIER_HIERARCHY are now populated by _load_dynamic_rbac_config() and _load_dynamic_tier_hierarchy(), which are async functions that call firestore_manager.get_global_config().

Asynchronous Loading at Module Level: asyncio.run() is used within _ensure_dynamic_configs_loaded_sync() to execute the async loading functions during module import. This allows the global variables to be populated before other functions try to access them.

Config Refresh Mechanism: _LAST_CONFIG_LOAD_TIME and _CONFIG_REFRESH_INTERVAL_SECONDS are introduced. _ensure_dynamic_configs_loaded_sync() is called by get_user_tier_capability to refresh the configurations from Firestore periodically (every 5 minutes by default). This ensures that changes made via the admin UI/backend are eventually reflected in the running application without a full restart.

get_user_tier_capability Update: This function now explicitly calls _ensure_dynamic_configs_loaded_sync() at its beginning to ensure the latest configurations are loaded before performing the capability check. It now accesses _RBAC_CAPABILITIES directly (which is populated from Firestore).

Firebase Auth Methods Made async: Several Firebase Admin SDK methods (auth.create_user, auth.get_user, auth.set_custom_user_claims, auth.revoke_refresh_tokens, auth.generate_password_reset_link, auth.verify_password_reset_code, auth.confirm_password_reset) are now correctly awaited as they are asynchronous operations.

Logging: Enhanced logging to trace when configurations are loaded or refreshed.
