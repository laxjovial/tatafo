# backend/api/admin_api.py

import logging
from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Dict, Any, Annotated

# Import the AdminService
from backend.services.admin_service import admin_service

# Import authentication middleware for admin-only access
from backend.middleware.auth_middleware import get_current_admin_user

# Import Pydantic models for request/response validation
from backend.models.user_models import UserProfile # For returning user profiles
from backend.models.admin_models import UserUpdateAdmin, CapabilityUpdate, TierUpdate

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency to ensure only admin users can access these endpoints
# All endpoints in this router will automatically require admin privileges
@router.get("/users", response_model=List[UserProfile])
async def get_all_users_admin(current_admin: Annotated[dict, Depends(get_current_admin_user)]):
    """
    Retrieves a list of all user profiles. Requires admin privileges.
    """
    logger.info(f"Admin user {current_admin['user_id']} requesting all user profiles.")
    return await admin_service.get_all_user_profiles()

@router.put("/users/{user_id}", response_model=UserProfile)
async def update_user_profile_by_admin(
    user_id: str,
    user_update: UserUpdateAdmin,
    current_admin: Annotated[dict, Depends(get_current_admin_user)]
):
    """
    Updates a specific user's profile (including tier and roles) by an administrator.
    Requires admin privileges.
    """
    logger.info(f"Admin user {current_admin['user_id']} updating profile for user: {user_id}")
    updated_user = await admin_service.update_user_profile_admin(user_id, user_update)
    # Ensure roles are a list for the Pydantic model
    if isinstance(updated_user.get('roles'), str):
        updated_user['roles'] = updated_user['roles'].split(',')
    return UserProfile(**updated_user)

@router.get("/config/capabilities", response_model=Dict[str, Any])
async def get_rbac_capabilities_admin(current_admin: Annotated[dict, Depends(get_current_admin_user)]):
    """
    Retrieves the current RBAC capabilities configuration. Requires admin privileges.
    """
    logger.info(f"Admin user {current_admin['user_id']} requesting RBAC capabilities.")
    return await admin_service.get_rbac_capabilities()

@router.put("/config/capabilities", response_model=Dict[str, Any])
async def update_rbac_capabilities_admin(
    capability_update: CapabilityUpdate,
    current_admin: Annotated[dict, Depends(get_current_admin_user)]
):
    """
    Updates the RBAC capabilities configuration. Requires admin privileges.
    Can update a specific capability or replace the entire document.
    """
    logger.info(f"Admin user {current_admin['user_id']} updating RBAC capabilities.")
    return await admin_service.update_rbac_capabilities(capability_update)

@router.get("/config/tiers", response_model=Dict[str, Any])
async def get_tier_hierarchy_admin(current_admin: Annotated[dict, Depends(get_current_admin_user)]):
    """
    Retrieves the current tier hierarchy configuration. Requires admin privileges.
    """
    logger.info(f"Admin user {current_admin['user_id']} requesting tier hierarchy.")
    return await admin_service.get_tier_hierarchy()

@router.put("/config/tiers", response_model=Dict[str, Any])
async def update_tier_hierarchy_admin(
    tier_update: TierUpdate,
    current_admin: Annotated[dict, Depends(get_current_admin_user)]
):
    """
    Updates the tier hierarchy configuration. Requires admin privileges.
    Can update a specific tier or replace the entire document.
    """
    logger.info(f"Admin user {current_admin['user_id']} updating tier hierarchy.")
    return await admin_service.update_tier_hierarchy(tier_update)

