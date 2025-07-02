# utils/user_manager.py

import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, firestore
import logging
from typing import Optional, Dict, Any, List
import yaml
from pathlib import Path

# Assume config_manager is available and initialized
from config.config_manager import config_manager

logger = logging.getLogger(__name__)

# --- Firebase Initialization (moved here for centralized management) ---
def initialize_firebase():
    """Initializes Firebase Admin SDK if not already initialized."""
    if not firebase_admin._apps:
        try:
            # Use the environment variable for Firebase config if available, otherwise try secrets.toml
            firebase_config_str = config_manager.get_secret("firebase_config")
            if firebase_config_str:
                cred = credentials.Certificate(json.loads(firebase_config_str))
                firebase_admin.initialize_app(cred)
                logger.info("Firebase Admin SDK initialized successfully from secrets.")
            else:
                logger.warning("Firebase config not found in secrets. Firebase Admin SDK not initialized.")
        except Exception as e:
            logger.error(f"Error initializing Firebase Admin SDK: {e}")
            st.error("Failed to initialize Firebase. Please check your Firebase configuration in .streamlit/secrets.toml.")
            st.stop()
    else:
        logger.debug("Firebase Admin SDK already initialized.")

# Initialize Firebase when user_manager is imported
initialize_firebase()

# --- RBAC Capabilities and Tier Hierarchy Loading ---
_RBAC_CAPABILITIES: Dict[str, Any] = {}
_TIER_HIERARCHY: Dict[str, int] = {}

def _load_rbac_config():
    """Loads RBAC capabilities from data/rbac_capabilities.yaml."""
    global _RBAC_CAPABILITIES
    rbac_path = Path("data/rbac_capabilities.yaml")
    if not rbac_path.exists():
        logger.warning(f"RBAC capabilities file not found at {rbac_path}. RBAC features may be limited.")
        _RBAC_CAPABILITIES = {}
        return
    try:
        with open(rbac_path, "r") as f:
            _RBAC_CAPABILITIES = yaml.safe_load(f).get('capabilities', {}) or {}
        logger.info(f"RBAC capabilities loaded from {rbac_path}")
    except Exception as e:
        logger.error(f"Error loading rbac_capabilities.yaml: {e}")
        _RBAC_CAPABILITIES = {}

def _load_tier_hierarchy():
    """Loads tier hierarchy from data/tiers.yaml."""
    global _TIER_HIERARCHY
    tiers_path = Path("data/tiers.yaml")
    if not tiers_path.exists():
        logger.warning(f"Tier hierarchy file not found at {tiers_path}. Tier-based access may be limited.")
        _TIER_HIERARCHY = {}
        return
    try:
        with open(tiers_path, "r") as f:
            tiers_config = yaml.safe_load(f).get('tiers', {}) or {}
            _TIER_HIERARCHY = {name: data.get('level', 0) for name, data in tiers_config.items()}
        logger.info(f"Tier hierarchy loaded from {tiers_path}")
    except Exception as e:
        logger.error(f"Error loading tiers.yaml: {e}")
        _TIER_HIERARCHY = {}

# Load configs on module import
_load_rbac_config()
_load_tier_hierarchy()

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
def create_user(email, password, username, tier="free", roles=["user"]) -> str:
    """Creates a new user in Firebase Authentication and sets custom claims."""
    try:
        user = auth.create_user(email=email, password=password, display_name=username)
        # Set custom claims for tier and roles
        auth.set_custom_user_claims(user.uid, {'tier': tier, 'roles': roles})
        logger.info(f"User '{username}' ({email}) created with UID: {user.uid}, Tier: {tier}, Roles: {roles}")
        return user.uid
    except Exception as e:
        logger.error(f"Error creating user {email}: {e}")
        raise ValueError(f"Failed to create user: {e}")

def authenticate_user(email, password) -> Optional[str]:
    """
    Authenticates a user using email and password.
    Returns a Firebase ID token on success, None on failure.
    NOTE: This function is typically called from a backend service or client-side SDK.
    For Streamlit, we rely on the client-side Firebase JS SDK for direct authentication.
    This server-side function is more for admin-like operations or if a custom auth flow is built.
    """
    # In a real server-side auth, you would verify credentials against Firebase Auth.
    # For client-side Streamlit, the JS SDK handles this and provides the ID token.
    # This function is a placeholder for server-side custom auth or admin actions.
    logger.warning("`authenticate_user` (server-side) is a placeholder. Client-side Firebase JS SDK handles primary user login.")
    try:
        # Example: If you were to use Firebase Admin SDK to verify password (not typical for client login)
        # This is more complex and usually not exposed directly.
        # Instead, the client sends id_token from successful client-side login.
        # For now, we'll just return a mock token if email/password match mock data.
        from backend.api.auth_api import _mock_users_db # Access mock DB
        user_data = _mock_users_db.get(email)
        if user_data and user_data["password_hash"] == password:
            # Simulate ID token generation for mock
            return "mock_jwt_token" if email == "alice@example.com" else "mock_admin_token" if email == "bob@example.com" else "mock_pro_token" if email == "charlie@example.com" else "mock_jwt_token"
        return None
    except Exception as e:
        logger.error(f"Error authenticating user {email}: {e}")
        return None

def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves user information by UID from Firebase Authentication."""
    try:
        user_record = auth.get_user(user_id)
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

def get_all_users() -> Dict[str, Dict[str, Any]]:
    """Retrieves all users from Firebase Authentication (admin-only operation)."""
    users = {}
    try:
        # List all users in batches
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

def update_user_tier_and_roles(user_id: str, new_tier: str, new_roles: List[str]):
    """Updates a user's custom claims (tier and roles) in Firebase Authentication."""
    try:
        auth.set_custom_user_claims(user_id, {'tier': new_tier, 'roles': new_roles})
        # Force token refresh on the client side for changes to take effect immediately
        auth.revoke_refresh_tokens(user_id)
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
    
    Args:
        user_token (str): The user's unique identifier (e.g., Firebase UID).
                          Can be None if no user is logged in.
        capability_key (str): The key of the capability to check (e.g., "data_analysis_enabled").
        default_value (Any): The default value to return if the capability or user's tier/roles
                             are not explicitly defined for this capability.
                             
    Returns:
        Any: The value of the capability for the user, or the default_value.
    """
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

    capability_config = _RBAC_CAPABILITIES.get('capabilities', {}).get(capability_key)
    
    if not capability_config:
        logger.warning(f"Capability '{capability_key}' not defined in rbac_capabilities.yaml. Returning default: {default_value}")
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
def generate_password_reset_link(email: str) -> str:
    """Generates a password reset link for a given email."""
    try:
        link = auth.generate_password_reset_link(email)
        logger.info(f"Generated password reset link for {email}")
        return link
    except auth.UserNotFoundError:
        logger.warning(f"Attempted to generate reset link for non-existent user: {email}")
        raise ValueError("User not found.")
    except Exception as e:
        logger.error(f"Error generating password reset link for {email}: {e}")
        raise ValueError(f"Failed to generate password reset link: {e}")

def verify_password_reset_code(oob_code: str) -> str:
    """Verifies a password reset code and returns the email associated with it."""
    try:
        email = auth.verify_password_reset_code(oob_code)
        logger.info(f"Verified password reset code. Email: {email}")
        return email
    except Exception as e:
        logger.error(f"Error verifying password reset code {oob_code}: {e}")
        raise ValueError(f"Invalid or expired password reset code: {e}")

def confirm_password_reset(oob_code: str, new_password: str):
    """Confirms password reset with the new password."""
    try:
        auth.confirm_password_reset(oob_code, new_password)
        logger.info(f"Password successfully reset using code {oob_code}.")
    except Exception as e:
        logger.error(f"Error confirming password reset with code {oob_code}: {e}")
        raise ValueError(f"Failed to confirm password reset: {e}")

