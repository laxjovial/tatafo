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


class UserManager:
    def __init__(self):
        # Ensure session state variables for user management exist
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
                        }, user_id=st.session_state.user_id, success=True, error_message="Login successful but profile could not be loaded.")
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
            return {"success": False, "message": f"Network error: {e}"}
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

# Global instance of UserManager
user_manager = UserManager()

# This function is used by the RBAC capabilities endpoint in backend/main.py.
# It should reflect the logic defined in config.yml for user tiers and roles.
# It's kept here for now but ideally, the backend would manage and serve these capabilities directly from its config.
def get_user_capabilities(user_tier: str, user_roles: List[str]) -> Dict[str, Any]:
    """
    Determines user capabilities based on their tier and roles from config.
    This function is primarily for the backend to use when calculating capabilities,
    but also serves as a reference for frontend logic.
    """
    capabilities = config_manager.get("rbac_capabilities", {})

    # Start with base capabilities for the user's tier
    tier_capabilities = capabilities.get("tiers", {}).get(user_tier, {})
    
    # Merge role-specific capabilities, roles override tier
    role_capabilities = {}
    for role in user_roles:
        role_caps = capabilities.get("roles", {}).get(role, {})
        for key, value in role_caps.items():
            # For boolean flags, if any role enables it, it's enabled
            if isinstance(value, bool):
                role_capabilities[key] = role_capabilities.get(key, False) or value
            # For numeric values (like max_k), take the highest/most permissive
            elif isinstance(value, (int, float)):
                role_capabilities[key] = max(role_capabilities.get(key, value), value)
            else: # For other types, roles simply override
                role_capabilities[key] = value

    # Combine tier and role capabilities, with roles taking precedence
    final_capabilities = {**tier_capabilities, **role_capabilities}

    # Apply any global overrides or default values if not explicitly set
    # Example: Ensure all expected keys are present with default False/0.7/etc.
    default_capabilities = {
        "llm_temperature_control_enabled": False,
        "llm_default_temperature": 0.7,
        "llm_max_temperature": 1.0,
        "llm_model_selection_enabled": False,
        "llm_default_provider": "gemini", # Default to Gemini if not specified
        "llm_default_model_name": "gemini-1.5-flash", # Default to Gemini-1.5-Flash
        "web_search_enabled": False,
        "data_analysis_enabled": False,
        "summarization_enabled": False,
        "chart_generation_enabled": False,
        "sentiment_analysis_enabled": False,
        "document_upload_enabled": False,
        "document_query_enabled": False,
        "document_query_max_results_k": 4,
        "chart_export_enabled": False,
        "finance_tool_access": False,
        "historical_data_access": False,
        "crypto_tool_access": False,
        "news_tool_access": False,
        "medical_tool_access": False,
        "legal_tool_access": False,
        "education_tool_access": False,
        "entertainment_tool_access": False,
        "weather_tool_access": False,
        "travel_tool_access": False,
        "sports_tool_access": False,
        "analytics_access": False,
        "analytics_charts_enabled": False,
        "analytics_user_specific_access": False,
    }
    # Merge defaults first, then final_capabilities
    return {**default_capabilities, **final_capabilities}

