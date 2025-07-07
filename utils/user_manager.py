# utils/user_manager.py

import logging
import httpx
import json
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone # Import timezone

# Import config_manager to get backend URL and RBAC configurations
from config.config_manager import config_manager
# Import analytics_tracker for logging events
from utils.analytics_tracker import log_event # Only import log_event, not initialize_analytics here

logger = logging.getLogger(__name__)

# Backend API URL from config
BACKEND_API_URL = config_manager.get("backend_api_url", "http://localhost:8000")

# --- Helper function for RBAC capabilities (used by tools and backend) ---
# This function is designed to be usable both in the backend (where user_token is passed)
# and potentially in a simplified frontend context (where user_capabilities might be pre-fetched).
# The mock data below is for standalone testing or if a full user object isn't available.
def get_user_tier_capability(user_token: Optional[str], capability_key: str, default_value: Any = None) -> Any:
    """
    Determines user capabilities based on their tier and roles.
    This function primarily serves the backend (e.g., tools) to check permissions.
    It uses mock data if a real user profile isn't available via the token.
    """
    # This mock data should ideally come from a central configuration or a dedicated RBAC service.
    # For demonstration, it's embedded here.
    _mock_users = {
        "default": {"user_id": "default", "username": "DefaultUser", "email": "default@example.com", "tier": "free", "roles": ["user"]},
        "mock_free_token": {"user_id": "mock_free_token", "username": "FreeUser", "email": "free@example.com", "tier": "free", "roles": ["user"]},
        "mock_pro_token": {"user_id": "mock_pro_token", "username": "ProUser", "email": "pro@example.com", "tier": "pro", "roles": ["user"]},
        "mock_premium_token": {"user_id": "mock_premium_token", "username": "PremiumUser", "email": "premium@example.com", "tier": "premium", "roles": ["user"]},
        "mock_admin_token": {"user_id": "mock_admin_token", "username": "AdminUser", "email": "admin@example.com", "tier": "admin", "roles": ["user", "admin"]},
    }
    _rbac_capabilities = {
        'capabilities': {
            'finance_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
            'crypto_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
            'medical_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
            'news_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
            'legal_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
            'education_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
            'entertainment_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
            'weather_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
            'travel_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
            'sports_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
            'document_upload_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
            'document_query_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
            'web_search_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
            'web_search_max_results': {'default': 2, 'tiers': {'pro': 7, 'premium': 15}},
            'web_search_limit_chars': {'default': 500, 'tiers': {'pro': 3000, 'premium': 10000}},
            'data_analysis_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
            'summarization_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
            'chart_generation_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
            'sentiment_analysis_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
            'analytics_access': {'default': False, 'roles': {'admin': True}},
            'analytics_charts_enabled': {'default': False, 'roles': {'admin': True}},
            'analytics_user_specific_access': {'default': False, 'roles': {'admin': True}},
        }
    }

    user_info = _mock_users.get(user_token, _mock_users["default"]) # Use default if token not found
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
    """
    Manages user profiles in Firestore, including creation, retrieval, and updates.
    Handles user tiers and roles.
    This class is intended for backend use (FastAPI).
    """
    def __init__(self, firestore_manager: Any, cloud_storage_utils: Any):
        self.firestore_manager = firestore_manager
        self.cloud_storage_utils = cloud_storage_utils # For future use, e.g., profile pictures
        logger.info("UserManager instantiated.")

    async def create_user_profile(self, uid: str, email: str, username: str, initial_tier: str = "free", initial_roles: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Creates a new user profile document in Firestore.
        """
        if initial_roles is None:
            initial_roles = ["user"] # Default role for new users

        user_profile_data = {
            "uid": uid,
            "email": email,
            "username": username,
            "tier": initial_tier,
            "roles": initial_roles,
            "created_at": datetime.now(timezone.utc),
            "last_login_at": datetime.now(timezone.utc),
            "profile_data": {} # Placeholder for additional profile fields
        }
        try:
            # Use set_doc with merge=True to create if not exists, or update if exists
            # Path: /users/{uid}
            await self.firestore_manager.set_doc(f"users/{uid}", user_profile_data, merge=True)
            logger.info(f"User profile created/updated for UID: {uid}")
            # Log analytics event
            await log_event(
                'user_profile_creation',
                {'uid': uid, 'email': email, 'username': username, 'tier': initial_tier},
                user_id=uid,
                success=True,
                log_from_backend=True
            )
            return {"success": True, "message": "User profile created successfully."}
        except Exception as e:
            logger.error(f"Error creating user profile for UID {uid}: {e}", exc_info=True)
            # Log analytics event for failure
            await log_event(
                'user_profile_creation',
                {'uid': uid, 'email': email, 'username': username, 'tier': initial_tier, 'error': str(e)},
                user_id=uid,
                success=False,
                error_message=str(e),
                log_from_backend=True
            )
            return {"success": False, "message": f"Failed to create user profile: {e}"}

    async def get_user(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a user's profile from Firestore.
        """
        try:
            user_data = await self.firestore_manager.get_doc(f"users/{uid}")
            if user_data:
                # Update last_login_at if the user is being retrieved (implying a login or active session)
                await self.update_user_profile(uid, {"last_login_at": datetime.now(timezone.utc)})
            return user_data
        except Exception as e:
            logger.error(f"Error retrieving user profile for UID {uid}: {e}", exc_info=True)
            return None

    async def get_all_users_admin(self) -> Dict[str, Any]:
        """
        Retrieves all user profiles (admin only).
        """
        try:
            users = await self.firestore_manager.get_collection("users")
            return {"success": True, "users": users}
        except Exception as e:
            logger.error(f"Error retrieving all user profiles: {e}", exc_info=True)
            return {"success": False, "message": f"Failed to retrieve users: {e}"}

    async def update_user_profile(self, uid: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates specific fields in a user's profile.
        """
        try:
            await self.firestore_manager.update_doc(f"users/{uid}", updates)
            logger.info(f"User profile updated for UID: {uid}. Fields: {list(updates.keys())}")
            # Log analytics event
            await log_event(
                'user_profile_update',
                {'uid': uid, 'updated_fields': list(updates.keys())},
                user_id=uid,
                success=True,
                log_from_backend=True
            )
            return {"success": True, "message": "User profile updated successfully."}
        except Exception as e:
            logger.error(f"Error updating user profile for UID {uid}: {e}", exc_info=True)
            # Log analytics event for failure
            await log_event(
                'user_profile_update',
                {'uid': uid, 'updated_fields': list(updates.keys()), 'error': str(e)},
                user_id=uid,
                success=False,
                error_message=str(e),
                log_from_backend=True
            )
            return {"success": False, "message": f"Failed to update user profile: {e}"}

    async def update_user_roles_and_tier(self, uid: str, new_tier: Optional[str] = None, new_roles: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Updates a user's tier and/or roles.
        """
        updates = {}
        if new_tier:
            updates["tier"] = new_tier
        if new_roles is not None: # Allow setting roles to an empty list
            updates["roles"] = new_roles
        
        if not updates:
            return {"success": False, "message": "No tier or roles provided for update."}

        result = await self.update_user_profile(uid, updates)
        # Log this specific admin action
        await log_event(
            'admin_user_roles_tier_update',
            {'target_uid': uid, 'new_tier': new_tier, 'new_roles': new_roles, 'admin_action_result': result.get('message')},
            user_id=None, # Admin user ID will be logged by the FastAPI endpoint calling this
            success=result['success'],
            error_message=result.get('message') if not result['success'] else None,
            log_from_backend=True
        )
        return result

