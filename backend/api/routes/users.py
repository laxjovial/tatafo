# backend/api/routes/users.py

import logging
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, List, Optional

# Import the user_service and user_manager instances
from backend.services.user_service import user_service
from utils.user_manager import get_user_tier_capability, _RBAC_CAPABILITIES

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/users/{user_id}", response_model=Dict[str, Any])
async def get_user_profile_route(user_id: str) -> Dict[str, Any]:
    """
    Retrieves a user's profile information by their user ID from Firestore.
    """
    logger.info(f"API: Request to retrieve user profile for user_id: {user_id}")
    user_profile = await user_service.get_user_profile(user_id) # Use user_service
    if not user_profile:
        logger.warning(f"API: User profile not found for user_id: {user_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    logger.info(f"API: Successfully retrieved user profile for user_id: {user_id}")
    return user_profile

@router.get("/rbac/capabilities/{user_token}", response_model=Dict[str, Any])
async def get_user_capabilities_route(user_token: str) -> Dict[str, Any]:
    """
    Retrieves all RBAC capabilities for a given user token.
    This endpoint provides the full map of capabilities with their effective access levels
    for the specified user, dynamically loaded from Firestore.
    """
    logger.info(f"API: Retrieving RBAC capabilities for user_token: {user_token}")
    
    user_capabilities = {}
    # Iterate through the _RBAC_CAPABILITIES (which is dynamically loaded from Firestore in user_manager)
    # and apply the get_user_tier_capability logic for each.
    # Note: _RBAC_CAPABILITIES is a global dict in user_manager, populated by _load_dynamic_rbac_config
    if not _RBAC_CAPABILITIES:
        logger.warning("RBAC capabilities not yet loaded in user_manager. Attempting to force load.")
        # This should ideally be handled by the user_manager's _ensure_dynamic_configs_loaded_sync
        # but adding a fallback here for robustness.
        from utils.user_manager import _ensure_dynamic_configs_loaded_sync
        _ensure_dynamic_configs_loaded_sync()

    for cap_key, cap_info_template in _RBAC_CAPABILITIES.get('capabilities', {}).items():
        # get_user_tier_capability already handles the logic of checking roles/defaults
        # It needs the user_token to determine the user's tier and roles.
        effective_value = get_user_tier_capability(user_token, cap_key, cap_info_template.get('default'))
        user_capabilities[cap_key] = effective_value
    
    logger.info(f"API: Successfully retrieved RBAC capabilities for user_token: {user_token}")
    return user_capabilities

# Note: In a real application, you'd have endpoints for:
# - User registration (POST /users/register)
# - User login (POST /users/login)
# - Password reset (POST /users/forgot-password, POST /users/reset-password)
# - User update (PUT /users/{user_id})
# - Admin-specific user management (GET/PUT/DELETE /admin/users/{user_id})
