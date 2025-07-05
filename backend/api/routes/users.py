# backend/api/routes/users.py

import logging
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, List, Optional

# Import the user_manager instance
from backend.services.user_service import user_manager, get_user_tier_capability, _RBAC_CAPABILITIES

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/users/{user_id}", response_model=Dict[str, Any])
async def get_user_profile(user_id: str) -> Dict[str, Any]:
    """
    Retrieves a user's profile information by their user ID.
    In a real application, this would be secured and only allow access to own profile
    or by admin roles.
    """
    logger.info(f"Attempting to retrieve user profile for user_id: {user_id}")
    user_profile = user_manager.get_user_by_token(user_id) # Use get_user_by_token to retrieve mock data
    if not user_profile:
        logger.warning(f"User profile not found for user_id: {user_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    logger.info(f"Successfully retrieved user profile for user_id: {user_id}")
    return user_profile

@router.get("/rbac/capabilities/{user_token}", response_model=Dict[str, Any])
async def get_user_capabilities(user_token: str) -> Dict[str, Any]:
    """
    Retrieves all RBAC capabilities for a given user token.
    This endpoint provides the full map of capabilities with their effective access levels
    for the specified user.
    """
    logger.info(f"Retrieving RBAC capabilities for user_token: {user_token}")
    
    # This is a simplified approach. In a real system, you might fetch
    # capabilities directly from Firestore based on the user's tier/roles.
    # For now, we'll iterate through the _RBAC_CAPABILITIES mock and apply
    # the get_user_tier_capability logic for each.
    
    user_capabilities = {}
    for cap_key, cap_info in _RBAC_CAPABILITIES.get('capabilities', {}).items():
        user_capabilities[cap_key] = get_user_tier_capability(user_token, cap_key, cap_info.get('default'))
    
    logger.info(f"Successfully retrieved RBAC capabilities for user_token: {user_token}")
    return user_capabilities

# Note: In a real application, you'd have endpoints for:
# - User registration (POST /users/register)
# - User login (POST /users/login)
# - Password reset (POST /users/forgot-password, POST /users/reset-password)
# - User update (PUT /users/{user_id})
# - Admin-specific user management (GET/PUT/DELETE /admin/users/{user_id})
