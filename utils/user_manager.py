# utils/user_manager.py

import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, firestore
import logging
from typing import Optional, Dict, Any, List
import json
import asyncio # For running async functions at module level

# Assume config_manager is available and initialized
from config.config_manager import config_manager
# Import firestore_manager for dynamic config loading
from database.firestore_manager import firestore_manager

logger = logging.getLogger(__name__)

# --- Firebase Initialization (moved here for centralized management) ---
def initialize_firebase():
    """Initializes Firebase Admin SDK if not already initialized."""
    if not firebase_admin._apps:
        try:
            # Use the environment variable for Firebase config if available, otherwise try secrets.toml
            # Note: firestore_manager's _initialize_firestore already handles this logic.
            # We just need to ensure it's called.
            # The firebase_admin.initialize_app() call is now managed by firestore_manager.
            # We only need to ensure firestore_manager is instantiated, which it is by importing.
            pass # Firebase initialization is now handled by firestore_manager's instantiation
        except Exception as e:
            logger.error(f"Error initializing Firebase Admin SDK (via firestore_manager): {e}")
            st.error("Failed to initialize Firebase. Please check your Firebase configuration.")
            st.stop()
    else:
        logger.debug("Firebase Admin SDK already initialized.")

# Initialize Firebase when user_manager is imported (by ensuring firestore_manager is instantiated)
# The firestore_manager is a singleton, so simply importing it ensures its __new__ method runs.
# We explicitly call it here to make sure it's initialized before dynamic config loading.
_ = firestore_manager # This line ensures the singleton is instantiated and firebase is initialized

# --- RBAC Capabilities and Tier Hierarchy Loading (NOW DYNAMIC FROM FIRESTORE) ---
_RBAC_CAPABILITIES: Dict[str, Any] = {}
_TIER_HIERARCHY: Dict[str, int] = {}
_LAST_CONFIG_LOAD_TIME: Optional[datetime] = None
_CONFIG_REFRESH_INTERVAL_SECONDS = 300 # Refresh config every 5 minutes (300 seconds)

async def _load_dynamic_rbac_config():
    """
    Loads RBAC capabilities from Firestore.
    """
    global _RBAC_CAPABILITIES, _LAST_CONFIG_LOAD_TIME
    try:
        rbac_doc = await firestore_manager.get_global_config("rbac_capabilities")
        if rbac_doc and rbac_doc.get('capabilities'):
            _RBAC_CAPABILITIES = rbac_doc['capabilities']
            logger.info("RBAC capabilities loaded from Firestore.")
        else:
            logger.warning("RBAC capabilities document not found or empty in Firestore. Using default empty config.")
            _RBAC_CAPABILITIES = {}
    except Exception as e:
        logger.error(f"Error loading RBAC capabilities from Firestore: {e}", exc_info=True)
        _RBAC_CAPABILITIES = {} # Fallback to empty config on error
    _LAST_CONFIG_LOAD_TIME = datetime.now()

async def _load_dynamic_tier_hierarchy():
    """
    Loads tier hierarchy from Firestore.
    """
    global _TIER_HIERARCHY
    try:
        tiers_doc = await firestore_manager.get_global_config("tiers")
        if tiers_doc and tiers_doc.get('tiers'):
            _TIER_HIERARCHY = {name: data.get('level', 0) for name, data in tiers_doc['tiers'].items()}
            logger.info("Tier hierarchy loaded from Firestore.")
        else:
            logger.warning("Tier hierarchy document not found or empty in Firestore. Using default empty config.")
            _TIER_HIERARCHY = {}
    except Exception as e:
        logger.error(f"Error loading tier hierarchy from Firestore: {e}", exc_info=True)
        _TIER_HIERARCHY = {} # Fallback to empty config on error

def _ensure_dynamic_configs_loaded_sync():
    """
    Synchronously ensures dynamic configs are loaded, refreshing if needed.
    This is called from synchronous contexts (e.g., get_user_tier_capability).
    """
    global _LAST_CONFIG_LOAD_TIME
    if _LAST_CONFIG_LOAD_TIME is None or (datetime.now() - _LAST_CONFIG_LOAD_TIME).total_seconds() > _CONFIG_REFRESH_INTERVAL_SECONDS:
        logger.info("Refreshing dynamic RBAC and Tier configurations from Firestore...")
        try:
            # Use asyncio.run to execute async loading functions in a synchronous context
            asyncio.run(_load_dynamic_rbac_config())
            asyncio.run(_load_dynamic_tier_hierarchy())
            logger.info("Dynamic RBAC and Tier configurations refreshed.")
        except RuntimeError as e:
            # Handle "cannot run a new event loop while a default loop is running" in Streamlit
            # This happens if asyncio.run is called within an existing event loop.
            # In Streamlit, this is usually fine for initial load, but can be tricky on reruns.
            # A more robust solution for Streamlit might involve st.cache_resource with an async function.
            logger.warning(f"Could not refresh dynamic configs synchronously (likely existing event loop): {e}")
            # Fallback for Streamlit: try to schedule it if possible, or rely on next full app load.
            # For now, we'll just log and use potentially stale config.
        except Exception as e:
            logger.error(f"Failed to refresh dynamic configurations: {e}", exc_info=True)

# Initial synchronous load when module is imported
_ensure_dynamic_configs_loaded_sync()

# --- User Session Management ---
def get_current_user() -> Dict[str, Any]:
    """
    Retrieves the current user's information from Streamlit session state.
    If not in session state, attempts to get from Firebase Auth.
    Returns a dictionary with user_id, email, username, tier, and roles.
    """
    if "user" not in st.session_state or not st.session_state.user:
        # Attempt to get user from Firebase Auth if not in session state
        try:
            # Check for __initial_auth_token (provided by Canvas for initial login)
            # This is a critical part of the Canvas environment setup
            if '__initial_auth_token' in st.session_state and st.session_state.__initial_auth_token:
                id_token = st.session_state.__initial_auth_token
                # Clear the token after use to prevent re-processing on rerun
                del st.session_state.__initial_auth_token 
            elif 'id_token' in st.session_state:
                id_token = st.session_state.id_token
            else:
                return {} # No user logged in

            decoded_token = auth.verify_id_token(id_token)
            user_id = decoded_token['uid']
            
            # Fetch custom claims for tier and roles from Firebase Auth
            user_record = auth.get_user(user_id)
            custom_claims = user_record.custom_claims or {}
            
            user_info = {
                "user_id": user_id,
                "email": decoded_token.get('email'),
                "username": decoded_token.get('name', decoded_token.get('email', user_id)),
                "tier": custom_claims.get('tier', config_manager.get("default_user_tier", "free")),
                "roles": custom_claims.get('roles', config_manager.get("default_user_roles", ["user"]))
            }
            st.session_state.user = user_info
            logger.info(f"User {user_id} loaded from Firebase Auth and set in session.")
            return user_info
        except ValueError as e:
            logger.warning(f"Invalid or expired ID token: {e}")
            st.session_state.user = {} # Clear invalid user
            return {}
        except Exception as e:
            logger.error(f"Error fetching user from Firebase Auth: {e}")
            st.session_state.user = {} # Clear on error
            return {}
    return st.session_state.user

def set_current_user(user_info: Dict[str, Any]):
    """Sets the current user's information in Streamlit session state."""
    st.session_state.user = user_info
    logger.info(f"User {user_info.get('user_id')} set in session state.")

def clear_current_user():
    """Clears the current user's information from Streamlit session state."""
    if "user" in st.session_state:
        del st.session_state.user
    if "id_token" in st.session_state:
        del st.session_state.id_token
    logger.info("User session cleared.")

# --- User Management (Firebase Auth & Firestore) ---
async def create_user(email, password, username, tier="free", roles=["user"]) -> str:
    """Creates a new user in Firebase Authentication and sets custom claims."""
    try:
        user = auth.create_user(email=email, password=password, display_name=username)
        # Set custom claims for tier and roles
        await auth.set_custom_user_claims(user.uid, {'tier': tier, 'roles': roles}) # Await custom claims
        logger.info(f"User '{username}' ({email}) created with UID: {user.uid}, Tier: {tier}, Roles: {roles}")
        return user.uid
    except Exception as e:
        logger.error(f"Error creating user {email}: {e}")
        raise ValueError(f"Failed to create user: {e}")

async def authenticate_user(email, password) -> Optional[str]:
    """
    Authenticates a user using email and password.
    Returns a Firebase ID token on success, None on failure.
    NOTE: This function is typically called from a backend service or client-side SDK.
    For Streamlit, we rely on the client-side Firebase JS SDK for direct authentication.
    This server-side function is more for admin-like operations or if a custom auth flow is built.
    """
    logger.warning("`authenticate_user` (server-side) is a placeholder. Client-side Firebase JS SDK handles primary user login.")
    try:
        # For now, we'll just return a mock token if email/password match mock data.
        # In a real scenario, this would involve more secure validation.
        # This function is primarily for testing custom auth flows or admin actions.
        # It does NOT use Firebase Auth's client-side password verification.
        from backend.api.auth_api import _mock_users_db # Access mock DB if still used for testing
        user_data = _mock_users_db.get(email)
        if user_data and user_data["password_hash"] == password:
            # Simulate ID token generation for mock
            return "mock_jwt_token" if email == "alice@example.com" else "mock_admin_token" if email == "bob@example.com" else "mock_pro_token" if email == "charlie@example.com" else "mock_jwt_token"
        return None
    except Exception as e:
        logger.error(f"Error authenticating user {email}: {e}")
        return None

async def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves user information by UID from Firebase Authentication."""
    try:
        user_record = await auth.get_user(user_id) # Await get_user
        custom_claims = user_record.custom_claims or {}
        return {
            "user_id": user_record.uid,
            "email": user_record.email,
            "username": user_record.display_name,
            "tier": custom_claims.get('tier', config_manager.get("default_user_tier", "free")),
            "roles": custom_claims.get('roles', config_manager.get("default_user_roles", ["user"]))
        }
    except auth.UserNotFoundError:
        logger.warning(f"User with UID {user_id} not found.")
        return None
    except Exception as e:
        logger.error(f"Error fetching user {user_id}: {e}")
        return None

async def get_all_users() -> Dict[str, Dict[str, Any]]:
    """Retrieves all users from Firebase Authentication (admin-only operation)."""
    users = {}
    try:
        # List all users in batches
        # auth.list_users().iterate_all() is synchronous, no await needed
        for user_record in auth.list_users().iterate_all():
            custom_claims = user_record.custom_claims or {}
            users[user_record.uid] = {
                "username": user_record.display_name,
                "email": user_record.email,
                "tier": custom_claims.get('tier', config_manager.get("default_user_tier", "free")),
                "roles": custom_claims.get('roles', config_manager.get("default_user_roles", ["user"]))
            }
        logger.info(f"Fetched {len(users)} users from Firebase Auth.")
        return users
    except Exception as e:
        logger.error(f"Error listing all users: {e}")
        return {}

async def update_user_tier_and_roles(user_id: str, new_tier: str, new_roles: List[str]):
    """Updates a user's custom claims (tier and roles) in Firebase Authentication."""
    try:
        await auth.set_custom_user_claims(user_id, {'tier': new_tier, 'roles': new_roles}) # Await set_custom_user_claims
        # Force token refresh on the client side for changes to take effect immediately
        await auth.revoke_refresh_tokens(user_id) # Await revoke_refresh_tokens
        logger.info(f"Updated user {user_id} to Tier: {new_tier}, Roles: {new_roles}. Refresh tokens revoked.")
    except auth.UserNotFoundError:
        logger.error(f"User with UID {user_id} not found for update.")
        raise ValueError(f"User with UID {user_id} not found.")
    except Exception as e:
        logger.error(f"Error updating user {user_id} claims: {e}")
        raise ValueError(f"Failed to update user claims: {e}")

# --- RBAC Capability Check ---
def get_user_tier_capability(user_token: Optional[str], capability_key: str, default_value: Any = None) -> Any:
    """
    Checks if a user has a specific capability based on their tier and roles.
    Admins (role 'admin') always have access.
    Ensures dynamic configs are loaded/refreshed before checking.
    
    Args:
        user_token (str): The user's unique identifier (e.g., Firebase UID).
                          Can be None if no user is logged in.
        capability_key (str): The key of the capability to check (e.g., "data_analysis_enabled").
        default_value (Any): The default value to return if the capability or user's tier/roles
                             are not explicitly defined for this capability.
                             
    Returns:
        Any: The value of the capability for the user, or the default_value.
    """
    # Ensure configs are loaded/refreshed before checking capabilities
    _ensure_dynamic_configs_loaded_sync()

    user_info = get_current_user() # Get current user info from session
    
    # If no user is logged in or user_info is empty, return the default value for the capability
    if not user_info or not user_info.get('user_id'):
        logger.debug(f"No active user for capability '{capability_key}'. Returning default: {default_value}")
        return default_value

    user_id = user_info['user_id']
    user_tier = user_info.get('tier', config_manager.get("default_user_tier", "free"))
    user_roles = user_info.get('roles', config_manager.get("default_user_roles", ["user"]))

    # Super Admin (role 'admin') bypasses all capability checks
    if "admin" in user_roles:
        logger.debug(f"User {user_id} is admin. Granting full capability for '{capability_key}'.")
        # If the default_value is a boolean, return True. If it's a number, return a very high number.
        if isinstance(default_value, bool): return True
        if isinstance(default_value, (int, float)): return float('inf') # Max numerical capability
        return default_value # For other types, just return the default (or consider always True for access)

    capability_config = _RBAC_CAPABILITIES.get(capability_key) # Access directly from global dict
    
    if not capability_config:
        logger.warning(f"Capability '{capability_key}' not defined in dynamic RBAC config. Returning default: {default_value}")
        return default_value

    # Check role-specific overrides first
    for role in user_roles:
        if role in capability_config.get('roles', {}):
            value = capability_config['roles'][role]
            logger.debug(f"Capability '{capability_key}' for user {user_id} (Role: {role}) overridden to: {value}")
            return value

    # If no role-specific override, return the default value for the capability
    value = capability_config.get('default', default_value)
    logger.debug(f"Capability '{capability_key}' for user {user_id} (Tier: {user_tier}) defaulting to: {value}")
    return value

# --- Password Reset Functions (for server-side use, e.g., by admin) ---
async def generate_password_reset_link(email: str) -> str:
    """Generates a password reset link for a given email."""
    try:
        link = await auth.generate_password_reset_link(email) # Await generate_password_reset_link
        logger.info(f"Generated password reset link for {email}")
        return link
    except auth.UserNotFoundError:
        logger.warning(f"Attempted to generate reset link for non-existent user: {email}")
        raise ValueError("User not found.")
    except Exception as e:
        logger.error(f"Error generating password reset link for {email}: {e}")
        raise ValueError(f"Failed to generate password reset link: {e}")

async def verify_password_reset_code(oob_code: str) -> str:
    """Verifies a password reset code and returns the email associated with it."""
    try:
        email = await auth.verify_password_reset_code(oob_code) # Await verify_password_reset_code
        logger.info(f"Verified password reset code. Email: {email}")
        return email
    except Exception as e:
        logger.error(f"Error verifying password reset code {oob_code}: {e}")
        raise ValueError(f"Invalid or expired password reset code: {e}")

async def confirm_password_reset(oob_code: str, new_password: str):
    """Confirms password reset with the new password."""
    try:
        await auth.confirm_password_reset(oob_code, new_password) # Await confirm_password_reset
        logger.info(f"Password successfully reset using code {oob_code}.")
    except Exception as e:
        logger.error(f"Error confirming password reset with code {oob_code}: {e}")
        raise ValueError(f"Failed to confirm password reset: {e}")

