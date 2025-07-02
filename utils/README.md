Implementing utils/ Directory - Part 1: user_manager.py
The utils/user_manager.py file is central to our user authentication, session management, and RBAC logic. We need to update it to load and utilize the new data/rbac_capabilities.yaml and data/tiers.yaml files. This will make our RBAC system much more flexible and configurable.

1. utils/user_manager.py (Updated to Load Capabilities and Tiers)
This update will modify get_user_tier_capability to check the rbac_capabilities.yaml file for granular permissions based on roles and tiers. It will also load the TIER_HIERARCHY from tiers.yaml.
utils/user_manager.py (Updated for RBAC Capabilities and Tiers)
Jul 2, 7:19 AM

Open

Key Changes in utils/user_manager.py:

Firebase Initialization: The initialize_firebase() function is now called directly when user_manager.py is imported. This ensures Firebase is set up early.

RBAC Config Loading:

_RBAC_CAPABILITIES and _TIER_HIERARCHY global variables are introduced.

_load_rbac_config() and _load_tier_hierarchy() functions are added to parse data/rbac_capabilities.yaml and data/tiers.yaml respectively. These are called on module import.

get_user_tier_capability Logic Update:

Now directly uses _RBAC_CAPABILITIES.

It first checks if the user has the "admin" role and, if so, grants maximum access (True for booleans, float('inf') for numbers). This implements your "super admin" override.

It then iterates through the user's roles to check for specific overrides defined in rbac_capabilities.yaml.

If no role-specific override is found, it falls back to the default value defined for that capability in the YAML.

Improved logging for debugging capability checks.

get_current_user(): Updated to correctly fetch custom claims (tier and roles) from Firebase Auth when a user's ID token is available in st.session_state. It also now correctly uses config_manager.get("default_user_tier", "free") and config_manager.get("default_user_roles", ["user"]) as fallbacks.

authenticate_user(): Added a warning that it's a server-side placeholder and client-side Firebase JS SDK handles primary login. It now also uses the mock user database from backend.api.auth_api for consistency in testing.

update_user_tier_and_roles(): Added auth.revoke_refresh_tokens(user_id) to ensure that changes to custom claims (tier/roles) take effect immediately on the client side by forcing a token refresh.

Now, we need to update ui/main_app.py to use the _TIER_HIERARCHY from utils/user_manager.py instead of its own hardcoded version.
