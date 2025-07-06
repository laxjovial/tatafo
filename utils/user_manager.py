# utils/user_manager.py

import streamlit as st
import httpx
import json
import logging
from typing import Optional, Dict, Any, List, Union
import datetime

# Import config_manager to get backend URL and RBAC configurations
from config.config_manager import config_manager
# Import analytics_tracker for logging events
from utils.analytics_tracker import log_event, initialize_analytics

logger = logging.getLogger(__name__)

# Backend API URL from config
BACKEND_API_URL = config_manager.get("backend_api_url", "http://localhost:8000")

# Initialize analytics tracker for frontend context (if not already done)
# This will ensure events from user_manager are logged
if 'analytics_initialized_frontend' not in st.session_state:
    try:
        # In a Streamlit app, we might not have direct access to firebase_admin.auth/firestore client
        # So we'll use a mock for the frontend analytics initialization, as the backend will handle persistence.
        # The log_event function itself will be responsible for making the API call to the backend.
        mock_db = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: None})()})()
        mock_auth = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': None})()})()
        app_id_for_analytics = config_manager.get("app_id", "default-streamlit-app-id")
        initialize_analytics(mock_db, mock_auth, app_id_for_analytics, "streamlit_frontend_user")
        st.session_state['analytics_initialized_frontend'] = True
        logger.info("Analytics tracker initialized for user_manager (frontend context).")
    except Exception as e:
        logger.error(f"Failed to initialize analytics in user_manager (frontend context): {e}")


# This function is used by the RBAC capabilities endpoint in backend/main.py and by tools.
# It should reflect the logic defined in config.yml for user tiers and roles.
def get_user_tier_capability(user_token: Optional[str], capability_key: str, default_value: Any = None) -> Any:
    """
    Determines user capabilities based on their tier and roles from config.
    This function is primarily for the backend to use when calculating capabilities,
    but also serves as a reference for frontend logic.
    """
    # In a real backend, user_info would come from an authenticated session/DB lookup
    # For this standalone function, we assume user_token can be used to get user_info
    # if it's called in a context where user_manager has access to a backend DB.
    # For frontend context, it will rely on st.session_state.user_profile and capabilities.
    
    # If running in Streamlit frontend, use session_state for capabilities
    if 'user_capabilities' in st.session_state and st.session_state.user_capabilities:
        return st.session_state.user_capabilities.get(capability_key, default_value)

    # Fallback for backend or local testing where session_state might not be available
    # This part should ideally be handled by a dedicated RBAC service in the backend
    # or by passing the full user object with tier/roles.
    # For now, we'll use a simplified mock for standalone testing if needed.
    
    # This is a simplified mock for local testing of tools outside of FastAPI/Streamlit context
    # In the actual FastAPI backend, the user object passed to tools will contain tier/roles.
    # This part should be aligned with how user_manager.py is used in the backend.
    
    # For the context of domain tools (like finance_tool.py) calling this,
    # they are passed the user_token, and this function needs to determine capabilities.
    # The actual implementation of how user_token maps to tier/roles here
    # would depend on whether this function directly queries a DB or relies on
    # a pre-fetched user object.
    
    # Given the traceback points to shared_tools/scrapper_tool.py, it's likely
    # this function is being called in a context where user_info is not readily available
    # from st.session_state, or it's called in the backend.

    # Let's assume for backend/tool context, we'd have a way to get user info.
    # For now, we'll use a simplified mock based on the `_rbac_capabilities`
    # and `_mock_users` that were used in the CLI tests of domain tools.
    
    # This mock data is usually part of a UserManager *instance* or a dedicated RBAC module.
    # Since this is a standalone function, we need to make it self-contained or
    # ensure the calling context provides the necessary user_info.

    # For now, let's assume this function is called with a user_token that
    # can be mapped to a tier/roles *within this function's scope* for testing.
    # In a real backend, `get_user_tier_capability` would likely be a method
    # of a `UserManager` class that has access to user data.

    # Replicating the mock RBAC logic from domain tool tests for this standalone function
    _mock_users = {
        "default": {"user_id": "default", "username": "DefaultUser", "email": "default@example.com", "tier": "free", "roles": ["user"]},
        "mock_free_token": {"user_id": "mock_free_token", "username": "FreeUser", "email": "free@example.com", "tier": "free", "roles": ["user"]},
        "mock_pro_token": {"user_id": "mock_pro_token", "username": "ProUser", "email": "pro@example.com", "tier": "pro", "roles": ["user"]},
        "mock_premium_token": {"user_id": "mock_premium_token", "username": "PremiumUser", "email": "premium@example.com", "tier": "premium", "roles": ["user"]},
        "mock_admin_token": {"user_id": "mock_admin_token", "username": "AdminUser", "email": "admin@example.com", "tier": "admin", "roles": ["user", "admin"]},
    }
    _rbac_capabilities = {
        'capabilities': {
            'finance_tool_access': {
                'default': False,
                'roles': {'pro': True, 'premium': True, 'admin': True}
            },
            'crypto_tool_access': {
                'default': False,
                'roles': {'pro': True, 'premium': True, 'admin': True}
            },
            'medical_tool_access': {
                'default': False,
                'roles': {'pro': True, 'premium': True, 'admin': True}
            },
            'news_tool_access': {
                'default': False,
                'roles': {'pro': True, 'premium': True, 'admin': True}
            },
            'legal_tool_access': {
                'default': False,
                'roles': {'pro': True, 'premium': True, 'admin': True}
            },
            'education_tool_access': {
                'default': False,
                'roles': {'pro': True, 'premium': True, 'admin': True}
            },
            'entertainment_tool_access': {
                'default': False,
                'roles': {'pro': True, 'premium': True, 'admin': True}
            },
            'weather_tool_access': {
                'default': False,
                'roles': {'pro': True, 'premium': True, 'admin': True}
            },
            'travel_tool_access': {
                'default': False,
                'roles': {'pro': True, 'premium': True, 'admin': True}
            },
            'sports_tool_access': {
                'default': False,
                'roles': {'pro': True, 'premium': True, 'admin': True}
            },
            'document_upload_enabled': {
                'default': False,
                'roles': {'pro': True, 'premium': True, 'admin': True}
            },
            'document_query_enabled': {
                'default': False,
                'roles': {'pro': True, 'premium': True, 'admin': True}
            },
            'web_search_enabled': {
                'default': False,
                'roles': {'pro': True, 'premium': True, 'admin': True}
            },
            'web_search_max_results': {
                'default': 2,
                'tiers': {'pro': 7, 'premium': 15}
            },
            'web_search_limit_chars': {
                'default': 500,
                'tiers': {'pro': 3000, 'premium': 10000}
            },
            'data_analysis_enabled': {
                'default': False,
                'roles': {'pro': True, 'premium': True, 'admin': True}
            },
            'summarization_enabled': {
                'default': False,
                'roles': {'pro': True, 'premium': True, 'admin': True}
            },
            'chart_generation_enabled': {
                'default': False,
                'roles': {'pro': True, 'premium': True, 'admin': True}
            },
            'sentiment_analysis_enabled': {
                'default': False,
                'roles': {'pro': True, 'premium': True, 'admin': True}
            },
            'analytics_access': {
                'default': False,
                'roles': {'admin': True}
            },
            'analytics_charts_enabled': {
                'default': False,
                'roles': {'admin': True}
            },
            'analytics_user_specific_access': {
                'default': False,
                'roles': {'admin': True}
            },
        }
    }

    user_info = _mock_users.get(user_token, _mock_users["default"]) # Use default if token not found
    user_id = user_info.get('user_id')
    user_tier = user_info.get('tier', 'free')
    user_roles = user_info.get('roles', [])

    if "admin" in user_roles:
        # Admins have all capabilities enabled or set to max/inf
        if capability_key in _rbac_capabilities['capabilities']:
            cap_config = _rbac_capabilities['capabilities'][capability_key]
            if isinstance(cap_config.get('default'), bool): return True
            if isinstance(cap_config.get('default'), (int, float)): return float('inf')
        return default_value # Return default if not a known boolean/numeric capability

    capability_config = _rbac_capabilities.get('capabilities', {}).get(capability_key)
    if not capability_config:
        return default_value

    # Check roles first
    for role in user_roles:
        if role in capability_config.get('roles', {}):
            return capability_config['roles'][role]
    
    # Then check tiers
    if user_tier in capability_config.get('tiers', {}):
        return capability_config['tiers'][user_tier]

    return capability_config.get('default', default_value)


class UserManager:
    def __init__(self, firestore_manager: Any, cloud_storage_utils: Any):
        self.firestore_manager = firestore_manager
        self.cloud_storage_utils = cloud_storage_utils
        # Ensure session state variables for user management exist (frontend context)
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'email' not in st.session_state:
            st.session_state.email = None
        if 'id_token' not in st.session_state:
            st.session_state.id_token = None
        if 'is_authenticated' not in st.session_state:
            st.session_state.is_authenticated = False
        if 'user_profile' not in st.session_state:
            st.session_state.user_profile = {}
        if 'user_capabilities' not in st.session_state:
            st.session_state.user_capabilities = {}

    def get_auth_headers(self) -> Dict[str, str]:
        """Returns authorization headers with the ID token if available."""
        if st.session_state.id_token:
            return {"Authorization": f"Bearer {st.session_state.id_token}"}
        return {}

    async def register_user(self, email, password, username) -> Dict[str, Any]:
        """Registers a new user via the backend API."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BACKEND_API_URL}/auth/register",
                    json={"email": email, "password": password, "username": username}
                )
                response.raise_for_status()  # Raise an exception for 4xx or 5xx responses
                result = response.json()
                if result.get("success"):
                    logger.info(f"User {email} registered successfully.")
                    await log_event('user_registration_frontend', {
                        'email': email,
                        'username': username,
                        'status': 'success'
                    }, user_id=result.get('user_id', 'N/A'), success=True)
                else:
                    logger.warning(f"User registration failed for {email}: {result.get('message')}")
                    await log_event('user_registration_frontend', {
                        'email': email,
                        'username': username,
                        'status': 'failure',
                        'reason': result.get('message', 'Unknown error')
                    }, user_id='N/A', success=False, error_message=result.get('message', 'Unknown error'))
                return result
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", str(e))
            logger.error(f"HTTP error during registration for {email}: {error_detail}")
            await log_event('user_registration_frontend', {
                'email': email,
                'username': username,
                'status': 'failure',
                'reason': f"HTTP error: {error_detail}"
            }, user_id='N/A', success=False, error_message=error_detail)
            return {"success": False, "message": error_detail}
        except httpx.RequestError as e:
            logger.error(f"Network error during registration for {email}: {e}")
            await log_event('user_registration_frontend', {
                'email': email,
                'username': username,
                'status': 'failure',
                'reason': f"Network error: {e}"
            }, user_id='N/A', success=False, error_message=str(e))
            return {"success": False, "message": f"Network error: {e}"}
        except Exception as e:
            logger.critical(f"Unexpected error during registration for {email}: {e}", exc_info=True)
            await log_event('user_registration_frontend', {
                'email': email,
                'username': username,
                'status': 'failure',
                'reason': f"Unexpected error: {e}"
            }, user_id='N/A', success=False, error_message=str(e))
            return {"success": False, "message": f"An unexpected error occurred: {e}"}

    async def login_user(self, email, password) -> Dict[str, Any]:
        """Logs in a user via the backend API and stores session state."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BACKEND_API_URL}/auth/login",
                    json={"email": email, "password": password}
                )
                response.raise_for_status()
                result = response.json()
                if result.get("success"):
                    st.session_state.user_id = result["user_id"]
                    st.session_state.id_token = result["id_token"]
                    st.session_state.email = email
                    # Fetch user profile immediately after login to get username, tier, roles
                    user_profile_data = await self.get_user_profile(st.session_state.user_id)
                    if user_profile_data.get("success"):
                        st.session_state.username = user_profile_data["profile"]["username"]
                        st.session_state.user_profile = user_profile_data["profile"]
                        # Also fetch capabilities
                        await self.fetch_user_capabilities(st.session_state.user_id)
                        st.session_state.is_authenticated = True
                        logger.info(f"User {email} logged in successfully. UID: {st.session_state.user_id}")
                        await log_event('user_login_frontend', {
                            'email': email,
                            'status': 'success'
                        }, user_id=st.session_state.user_id, success=True)
                        return {"success": True, "message": "Login successful."}
                    else:
                        # Even if profile fetch fails, we have ID token and UID
                        st.session_state.is_authenticated = True
                        logger.warning(f"User {email} logged in but profile fetch failed: {user_profile_data.get('message')}")
                        await log_event('user_login_frontend', {
                            'email': email,
                            'status': 'partial_success',
                            'reason': 'profile_fetch_failed'
                        }, user_id=st.session_state.user_id, success=True, error_message="Login successful but profile data could not be loaded.")
                        return {"success": True, "message": "Login successful, but profile data could not be loaded."}
                else:
                    logger.warning(f"User login failed for {email}: {result.get('message')}")
                    await log_event('user_login_frontend', {
                        'email': email,
                        'status': 'failure',
                        'reason': result.get('message', 'Unknown error')
                    }, user_id='N/A', success=False, error_message=result.get('message', 'Unknown error'))
                    return result
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", str(e))
            logger.error(f"HTTP error during login for {email}: {error_detail}")
            await log_event('user_login_frontend', {
                'email': email,
                'status': 'failure',
                'reason': f"HTTP error: {error_detail}"
            }, user_id='N/A', success=False, error_message=error_detail)
            return {"success": False, "message": error_detail}
        except httpx.RequestError as e:
            logger.error(f"Network error during login for {email}: {e}")
            await log_event('user_login_frontend', {
                'email': email,
                'status': 'failure',
                'reason': f"Network error: {e}"
            }, user_id='N/A', success=False, error_message=str(e))
            return {"success": False, "message": f"Network error: {e}"}
        except Exception as e:
            logger.critical(f"Unexpected error during login for {email}: {e}", exc_info=True)
            await log_event('user_login_frontend', {
                'email': email,
                'status': 'failure',
                'reason': f"Unexpected error: {e}"
            }, user_id='N/A', success=False, error_message=str(e))
            return {"success": False, "message": f"An unexpected error occurred: {e}"}

    def logout_user(self):
        """Clears user session state."""
        user_id = st.session_state.user_id # Capture before clearing
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.email = None
        st.session_state.id_token = None
        st.session_state.is_authenticated = False
        st.session_state.user_profile = {}
        st.session_state.user_capabilities = {}
        logger.info(f"User {user_id} logged out.")
        log_event('user_logout_frontend', {
            'status': 'success'
        }, user_id=user_id, success=True)
        # st.rerun() # Rerun to clear UI components related to logged-in state

    async def change_password(self, user_id: str, current_password: str, new_password: str) -> Dict[str, Any]:
        """Changes user password via the backend API."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BACKEND_API_URL}/auth/change_password/{user_id}",
                    json={"current_password": current_password, "new_password": new_password},
                    headers=self.get_auth_headers()
                )
                response.raise_for_status()
                result = response.json()
                if result.get("success"):
                    logger.info(f"Password changed for user {user_id}.")
                    await log_event('user_action_frontend', {
                        'action_type': 'password_change',
                        'status': 'success'
                    }, user_id=user_id, success=True)
                else:
                    logger.warning(f"Password change failed for user {user_id}: {result.get('message')}")
                    await log_event('user_action_frontend', {
                        'action_type': 'password_change',
                        'status': 'failure',
                        'reason': result.get('message', 'Unknown error')
                    }, user_id=user_id, success=False, error_message=result.get('message', 'Unknown error'))
                return result
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", str(e))
            logger.error(f"HTTP error changing password for {user_id}: {error_detail}")
            await log_event('user_action_frontend', {
                'action_type': 'password_change',
                'status': 'failure',
                'reason': f"HTTP error: {error_detail}"
            }, user_id=user_id, success=False, error_message=error_detail)
            return {"success": False, "message": error_detail}
        except httpx.RequestError as e:
            logger.error(f"Network error changing password for {user_id}: {e}")
            await log_event('user_action_frontend', {
                'action_type': 'password_change',
                'status': 'failure',
                'reason': f"Network error: {e}"
            }, user_id=user_id, success=False, error_message=str(e))
            return {"success": False, "message": f"Network error: {e}"}
        except Exception as e:
            logger.critical(f"Unexpected error changing password for {user_id}: {e}", exc_info=True)
            await log_event('user_action_frontend', {
                'action_type': 'password_change',
                'status': 'failure',
                'reason': f"Unexpected error: {e}"
            }, user_id=user_id, success=False, error_message=str(e))
            return {"success": False, "message": f"An unexpected error occurred: {e}"}

    async def forgot_password(self, email: str) -> Dict[str, Any]:
        """Requests a password reset email via the backend API."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BACKEND_API_URL}/auth/forgot_password",
                    json={"email": email}
                )
                response.raise_for_status()
                result = response.json()
                if result.get("success"):
                    logger.info(f"Forgot password request sent for {email}.")
                    await log_event('user_action_frontend', {
                        'action_type': 'forgot_password_request',
                        'email': email,
                        'status': 'success'
                    }, user_id='N/A', success=True)
                else:
                    logger.warning(f"Forgot password request failed for {email}: {result.get('message')}")
                    await log_event('user_action_frontend', {
                        'action_type': 'forgot_password_request',
                        'email': email,
                        'status': 'failure',
                        'reason': result.get('message', 'Unknown error')
                    }, user_id='N/A', success=False, error_message=result.get('message', 'Unknown error'))
                return result
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", str(e))
            logger.error(f"HTTP error during forgot password for {email}: {error_detail}")
            await log_event('user_action_frontend', {
                'action_type': 'forgot_password_request',
                'email': email,
                'status': 'failure',
                'reason': f"HTTP error: {error_detail}"
            }, user_id='N/A', success=False, error_message=error_detail)
            return {"success": False, "message": error_detail}
        except httpx.RequestError as e:
            logger.error(f"Network error during forgot password for {email}: {e}")
            await log_event('user_action_frontend', {
                'action_type': 'forgot_password_request',
                'email': email,
                'status': 'failure',
                'reason': f"Network error: {e}"
            }, user_id='N/A', success=False, error_message=str(e))
            return {"success": False, "message": f"Network error: {e}"}
        except Exception as e:
            logger.critical(f"Unexpected error during forgot password for {email}: {e}", exc_info=True)
            await log_event('user_action_frontend', {
                'action_type': 'forgot_password_request',
                'email': email,
                'status': 'failure',
                'reason': f"Unexpected error: {e}"
            }, user_id='N/A', success=False, error_message=str(e))
            return {"success": False, "message": f"An unexpected error occurred: {e}"}

    async def reset_password(self, oob_code: str, new_password: str) -> Dict[str, Any]:
        """Resets password using OOB code via the backend API."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BACKEND_API_URL}/auth/reset_password",
                    json={"oob_code": oob_code, "new_password": new_password}
                )
                response.raise_for_status()
                result = response.json()
                if result.get("success"):
                    logger.info(f"Password reset successfully with OOB code.")
                    await log_event('user_action_frontend', {
                        'action_type': 'password_reset_confirm',
                        'status': 'success'
                    }, user_id='N/A', success=True)
                else:
                    logger.warning(f"Password reset failed with OOB code: {result.get('message')}")
                    await log_event('user_action_frontend', {
                        'action_type': 'password_reset_confirm',
                        'status': 'failure',
                        'reason': result.get('message', 'Unknown error')
                    }, user_id='N/A', success=False, error_message=result.get('message', 'Unknown error'))
                return result
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", str(e))
            logger.error(f"HTTP error during password reset with OOB code: {error_detail}")
            await log_event('user_action_frontend', {
                'action_type': 'password_reset_confirm',
                'status': 'failure',
                'reason': f"HTTP error: {error_detail}"
            }, user_id='N/A', success=False, error_message=error_detail)
            return {"success": False, "message": error_detail}
        except httpx.RequestError as e:
            logger.error(f"Network error during password reset with OOB code: {e}")
            await log_event('user_action_frontend', {
                'action_type': 'password_reset_confirm',
                'status': 'failure',
                'reason': f"Network error: {e}"
            }, user_id='N/A', success=False, error_message=str(e))
            return {"success": False, "message": f"Network error: {e}"}
        except Exception as e:
            logger.critical(f"Unexpected error during password reset with OOB code: {e}", exc_info=True)
            await log_event('user_action_frontend', {
                'action_type': 'password_reset_confirm',
                'status': 'failure',
                'reason': f"Unexpected error: {e}"
            }, user_id='N/A', success=False, error_message=str(e))
            return {"success": False, "message": f"An unexpected error occurred: {e}"}

    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Fetches user profile from the backend API."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{BACKEND_API_URL}/users/{user_id}",
                    headers=self.get_auth_headers()
                )
                response.raise_for_status()
                profile_data = response.json()
                logger.debug(f"Fetched user profile for {user_id}: {profile_data}")
                await log_event('user_profile_frontend', {
                    'action': 'fetch',
                    'status': 'success'
                }, user_id=user_id, success=True)
                return {"success": True, "profile": profile_data}
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", str(e))
            logger.error(f"HTTP error fetching profile for {user_id}: {error_detail}")
            await log_event('user_profile_frontend', {
                'action': 'fetch',
                'status': 'failure',
                'reason': f"HTTP error: {error_detail}"
            }, user_id=user_id, success=False, error_message=error_detail)
            return {"success": False, "message": error_detail}
        except httpx.RequestError as e:
            logger.error(f"Network error fetching profile for {user_id}: {e}")
            await log_event('user_profile_frontend', {
                'action': 'fetch',
                'status': 'failure',
                'reason': f"Network error: {e}"
            }, user_id=user_id, success=False, error_message=str(e))
            return {"success": False, "message": f"Network error: {e}"}
        except Exception as e:
            logger.critical(f"Unexpected error fetching profile for {user_id}: {e}", exc_info=True)
            await log_event('user_profile_frontend', {
                'action': 'fetch',
                'status': 'failure',
                'reason': f"Unexpected error: {e}"
            }, user_id=user_id, success=False, error_message=str(e))
            return {"success": False, "message": f"An unexpected error occurred: {e}"}

    async def update_user_profile(self, user_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Updates user profile via the backend API."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{BACKEND_API_URL}/users/{user_id}",
                    json=update_data,
                    headers=self.get_auth_headers()
                )
                response.raise_for_status()
                result = response.json()
                if result.get("success"):
                    logger.info(f"User profile updated for {user_id}. Fields: {list(update_data.keys())}")
                    # Re-fetch profile to update session state with latest data
                    await self.get_user_profile(user_id)
                    await log_event('user_profile_frontend', {
                        'action': 'update',
                        'updated_fields': list(update_data.keys()),
                        'status': 'success'
                    }, user_id=user_id, success=True)
                else:
                    logger.warning(f"User profile update failed for {user_id}: {result.get('message')}")
                    await log_event('user_profile_frontend', {
                        'action': 'update',
                        'updated_fields': list(update_data.keys()),
                        'status': 'failure',
                        'reason': result.get('message', 'Unknown error')
                    }, user_id=user_id, success=False, error_message=result.get('message', 'Unknown error'))
                return result
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", str(e))
            logger.error(f"HTTP error updating profile for {user_id}: {error_detail}")
            await log_event('user_profile_frontend', {
                'action': 'update',
                'updated_fields': list(update_data.keys()),
                'status': 'failure',
                'reason': f"HTTP error: {error_detail}"
            }, user_id=user_id, success=False, error_message=error_detail)
            return {"success": False, "message": error_detail}
        except httpx.RequestError as e:
            logger.error(f"Network error updating profile for {user_id}: {e}")
            await log_event('user_profile_frontend', {
                'action': 'update',
                'updated_fields': list(update_data.keys()),
                'status': 'failure',
                'reason': f"Network error: {e}"
            }, user_id=user_id, success=False, error_message=str(e))
            return {"success": False, "message": f"An unexpected error occurred: {e}"}
        except Exception as e:
            logger.critical(f"Unexpected error updating profile for {user_id}: {e}", exc_info=True)
            await log_event('user_profile_frontend', {
                'action': 'update',
                'updated_fields': list(update_data.keys()),
                'status': 'failure',
                'reason': f"Unexpected error: {e}"
            }, user_id=user_id, success=False, error_message=str(e))
            return {"success": False, "message": f"An unexpected error occurred: {e}"}

    async def fetch_user_capabilities(self, user_id: str) -> Dict[str, Any]:
        """Fetches RBAC capabilities for the user from the backend API."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{BACKEND_API_URL}/rbac/capabilities/{user_id}",
                    headers=self.get_auth_headers()
                )
                response.raise_for_status()
                capabilities = response.json()
                st.session_state.user_capabilities = capabilities
                logger.debug(f"Fetched capabilities for {user_id}: {capabilities}")
                await log_event('rbac_frontend', {
                    'action': 'fetch_capabilities',
                    'status': 'success'
                }, user_id=user_id, success=True)
                return {"success": True, "capabilities": capabilities}
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", str(e))
            logger.error(f"HTTP error fetching capabilities for {user_id}: {error_detail}")
            await log_event('rbac_frontend', {
                'action': 'fetch_capabilities',
                'status': 'failure',
                'reason': f"HTTP error: {error_detail}"
            }, user_id=user_id, success=False, error_message=error_detail)
            return {"success": False, "message": error_detail}
        except httpx.RequestError as e:
            logger.error(f"Network error fetching capabilities for {user_id}: {e}")
            await log_event('rbac_frontend', {
                'action': 'fetch_capabilities',
                'status': 'failure',
                'reason': f"Network error: {e}"
            }, user_id=user_id, success=False, error_message=str(e))
            return {"success": False, "message": f"Network error: {e}"}
        except Exception as e:
            logger.critical(f"Unexpected error fetching capabilities for {user_id}: {e}", exc_info=True)
            await log_event('rbac_frontend', {
                'action': 'fetch_capabilities',
                'status': 'failure',
                'reason': f"Unexpected error: {e}"
            }, user_id=user_id, success=False, error_message=str(e))
            return {"success": False, "message": f"An unexpected error occurred: {e}"}

    # --- Admin Functions ---

    async def get_all_users_admin(self) -> Dict[str, Any]:
        """Fetches a list of all users from the backend API (admin access required)."""
        user_id = st.session_state.user_id if st.session_state.is_authenticated else 'N/A'
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{BACKEND_API_URL}/admin/users",
                    headers=self.get_auth_headers()
                )
                response.raise_for_status()
                users_list = response.json()
                logger.info(f"Admin {user_id} fetched {len(users_list)} users.")
                await log_event('admin_action_frontend', {
                    'action_type': 'get_all_users',
                    'num_users_fetched': len(users_list),
                    'status': 'success'
                }, user_id=user_id, success=True)
                return {"success": True, "users": users_list}
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", str(e))
            logger.error(f"HTTP error fetching all users for admin {user_id}: {error_detail}")
            await log_event('admin_action_frontend', {
                'action_type': 'get_all_users',
                'status': 'failure',
                'reason': f"HTTP error: {error_detail}"
            }, user_id=user_id, success=False, error_message=error_detail)
            return {"success": False, "message": error_detail}
        except httpx.RequestError as e:
            logger.error(f"Network error fetching all users for admin {user_id}: {e}")
            await log_event('admin_action_frontend', {
                'action_type': 'get_all_users',
                'status': 'failure',
                'reason': f"Network error: {e}"
            }, user_id=user_id, success=False, error_message=str(e))
            return {"success": False, "message": f"Network error: {e}"}
        except Exception as e:
            logger.critical(f"Unexpected error fetching all users for admin {user_id}: {e}", exc_info=True)
            await log_event('admin_action_frontend', {
                'action_type': 'get_all_users',
                'status': 'failure',
                'reason': f"Unexpected error: {e}"
            }, user_id=user_id, success=False, error_message=str(e))
            return {"success": False, "message": f"An unexpected error occurred: {e}"}

    async def update_user_roles_and_tier_admin(self, target_user_id: str, tier: str, roles: List[str]) -> Dict[str, Any]:
        """Updates a user's tier and roles via the backend API (admin access required)."""
        user_id = st.session_state.user_id if st.session_state.is_authenticated else 'N/A'
        try:
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{BACKEND_API_URL}/admin/users/{target_user_id}/roles_and_tier",
                    json={"tier": tier, "roles": roles},
                    headers=self.get_auth_headers()
                )
                response.raise_for_status()
                result = response.json()
                if result.get("success"):
                    logger.info(f"Admin {user_id} updated user {target_user_id} to Tier: {tier}, Roles: {roles}")
                    await log_event('admin_action_frontend', {
                        'action_type': 'update_user_roles_and_tier',
                        'target_user_uid': target_user_id,
                        'new_tier': tier,
                        'new_roles': roles,
                        'status': 'success'
                    }, user_id=user_id, success=True)
                else:
                    logger.warning(f"Admin {user_id} failed to update user {target_user_id}: {result.get('message')}")
                    await log_event('admin_action_frontend', {
                        'action_type': 'update_user_roles_and_tier',
                        'target_user_uid': target_user_id,
                        'new_tier': tier,
                        'new_roles': roles,
                        'status': 'failure',
                        'reason': result.get('message', 'Unknown error')
                    }, user_id=user_id, success=False, error_message=result.get('message', 'Unknown error'))
                return result
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", str(e))
            logger.error(f"HTTP error updating user {target_user_id} for admin {user_id}: {error_detail}")
            await log_event('admin_action_frontend', {
                'action_type': 'update_user_roles_and_tier',
                'target_user_uid': target_user_id,
                'new_tier': tier,
                'new_roles': roles,
                'status': 'failure',
                'reason': f"HTTP error: {error_detail}"
            }, user_id=user_id, success=False, error_message=error_detail)
            return {"success": False, "message": error_detail}
        except httpx.RequestError as e:
            logger.error(f"Network error updating user {target_user_id} for admin {user_id}: {e}")
            await log_event('admin_action_frontend', {
                'action_type': 'update_user_roles_and_tier',
                'target_user_uid': target_user_id,
                'new_tier': tier,
                'new_roles': roles,
                'status': 'failure',
                'reason': f"Network error: {e}"
            }, user_id=user_id, success=False, error_message=str(e))
            return {"success": False, "message": f"Network error: {e}"}
        except Exception as e:
            logger.critical(f"Unexpected error updating user {target_user_id} for admin {user_id}: {e}", exc_info=True)
            await log_event('admin_action_frontend', {
                'action_type': 'update_user_roles_and_tier',
                'target_user_uid': target_user_id,
                'new_tier': tier,
                'new_roles': roles,
                'status': 'failure',
                'reason': f"Unexpected error: {e}"
            }, user_id=user_id, success=False, error_message=str(e))
            return {"success": False, "message": f"An unexpected error occurred: {e}"}

# Global instance of UserManager (This line is for frontend Streamlit context)
# For backend FastAPI, we will pass explicit dependencies.
# user_manager = UserManager() # This line is commented out as it's not needed for the backend's UserManager instance.

