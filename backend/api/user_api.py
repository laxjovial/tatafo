# backend/api/user_api.py

import logging
from fastapi import APIRouter, HTTPException, status, Depends
from typing import Annotated, List, Dict, Any, Optional
from datetime import datetime, timezone # For setting last_login_at in login (if needed, though moved to auth_api)

# Import Pydantic models
from backend.models.user_models import UserProfile, UserUpdate

# Import middleware for authentication and authorization
from backend.middleware.auth_middleware import get_current_user, get_current_admin_user, get_user_manager_dependency

# Import UserManager (now the primary source for user data logic)
# Removed _RBAC_CAPABILITIES_CONFIG import as it's internal to UserManager
from utils.user_manager import UserManager

# Import Firebase Auth (for updating custom claims - if still used here, otherwise remove)
from firebase_admin import auth # Keep if you update custom claims here, otherwise can remove

# Project imports for analytics
from utils.analytics_tracker import log_event

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

router = APIRouter()

@router.get("/{user_id}", response_model=UserProfile)
async def get_user_profile(
    user_id: str,
    current_user: Annotated[UserProfile, Depends(get_current_user)],
    user_manager: UserManager = Depends(get_user_manager_dependency)
):
    """
    Retrieves a user's profile by ID from Firestore.
    Requires authentication. User can view their own profile; admin can view any.
    """
    # Authorization check: A user can only view their own profile unless they are an admin
    if current_user.id != user_id and "admin" not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this user's profile."
        )

    logger.info(f"User {current_user.id} requesting profile for {user_id}.")
    try:
        user_data = await user_manager.get_user(user_id)
        if not user_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found.")

        # Convert the Firestore data dictionary to a UserProfile Pydantic model
        # Assuming user_data from Firestore has keys matching UserProfile attributes
        # And handling datetime objects for Pydantic (UserProfile should handle ISO strings or datetime objects)
        user_profile = UserProfile(**user_data)

        # Log event
        await log_event(
            'user_profile_viewed',
            {'target_uid': user_id},
            user_id=current_user.id,
            success=True,
            log_from_backend=True
        )

        return user_profile
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving user profile {user_id}: {e}", exc_info=True)
        await log_event(
            'user_profile_view_failure',
            {'target_uid': user_id, 'error': str(e)},
            user_id=current_user.id,
            success=False,
            error_message=str(e),
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve user profile: {e}")

@router.put("/{user_id}", response_model=UserProfile)
async def update_user_profile(
    user_id: str,
    user_update: UserUpdate,
    current_user: Annotated[UserProfile, Depends(get_current_user)],
    user_manager: UserManager = Depends(get_user_manager_dependency)
):
    """
    Updates a user's profile.
    Only allows users to update their own profile, or admin to update any.
    """
    if current_user.id != user_id and "admin" not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this user's profile."
        )

    logger.info(f"User {current_user.id} attempting to update profile for {user_id}.")

    # Build update dictionary, excluding fields that should not be updated directly by user (e.g., uid, email, created_at)
    updates = user_update.model_dump(exclude_unset=True, exclude={'email', 'uid', 'created_at', 'tier', 'roles'})

    # RBAC checks
    if 'system_prompt' in updates and not user_manager.get_user_tier_capability(current_user.user_id, 'custom_system_prompt_enabled'):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Your tier does not have permission to set a custom system prompt.")

    if 'preferred_llm_provider' in updates and not user_manager.get_user_tier_capability(current_user.user_id, 'llm_provider_selection_enabled'):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Your tier does not have permission to select an LLM provider.")

    if not updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid fields provided for update.")

    try:
        # If username is updated, update display name in Firebase Auth too
        if 'username' in updates:
            try:
                auth.update_user(user_id, display_name=updates['username'])
                logger.info(f"Firebase Auth display_name updated for {user_id} to {updates['username']}.")
            except firebase_exceptions.FirebaseError as fe:
                logger.warning(f"Failed to update Firebase Auth display_name for {user_id}: {fe}")
                # Log this error but don't prevent Firestore update if Firebase Auth update isn't critical
                # You might want to raise HTTPException here if Firebase Auth update is mandatory
                pass # Continue to Firestore update

        result = await user_manager.update_user_profile(user_id, updates)
        if not result["success"]:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result["message"])

        # Re-fetch the updated profile to return the latest state
        updated_profile_data = await user_manager.get_user(user_id)
        if not updated_profile_data:
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Updated user profile not found after update.")

        # Convert to Pydantic model
        updated_user_profile = UserProfile(**updated_profile_data)

        # Log event
        await log_event(
            'user_profile_updated',
            {'target_uid': user_id, 'updated_fields': list(updates.keys())},
            user_id=current_user.id,
            success=True,
            log_from_backend=True
        )

        return updated_user_profile
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user profile {user_id}: {e}", exc_info=True)
        await log_event(
            'user_profile_update_failure',
            {'target_uid': user_id, 'error': str(e)},
            user_id=current_user.id,
            success=False,
            error_message=str(e),
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update user profile: {e}")


@router.get("/{user_id}/capabilities")
async def get_user_capabilities(
    user_id: str,
    current_user: Annotated[UserProfile, Depends(get_current_user)],
    user_manager: UserManager = Depends(get_user_manager_dependency)
) -> Dict[str, Any]:
    """
    Retrieves the specific capabilities (features, limits) for a given user based on their tier and roles.
    Allows user to see their own capabilities, or admin to see any user's.
    """
    if current_user.id != user_id and "admin" not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view capabilities for this user."
        )

    logger.info(f"User {current_user.id} requesting capabilities for user_id: {user_id}")

    # Fetch the target user's profile to get their tier and roles
    target_user_profile = await user_manager.get_user(user_id)
    if not target_user_profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target user not found")

    target_user_tier = target_user_profile.get('tier', 'free')
    target_user_roles = target_user_profile.get('roles', [])
    if isinstance(target_user_roles, str): # Ensure roles is a list if it somehow came as a string
        target_user_roles = target_user_roles.split(',')

    user_capabilities = {}

    # Iterate through the capabilities defined in UserManager's internal config
    # We retrieve the config for iteration, but use user_manager.get_user_tier_capability for values
    _RBAC_CAPABILITIES_CONFIG_INTERNAL = user_manager._RBAC_CAPABILITIES_CONFIG # Access internal config from UserManager

    for cap_key, cap_info_template in _RBAC_CAPABILITIES_CONFIG_INTERNAL.get('capabilities', {}).items():
        # --- UPDATED: Call get_user_tier_capability as a method on user_manager instance ---
        effective_value = await user_manager.get_user_tier_capability(
            capability_key=cap_key,
            user_tier=target_user_tier,
            user_roles=target_user_roles
            # default_value is handled internally by get_user_tier_capability or explicitly defaulted
            # if the key is not found in the capabilities config (which should be rare given iteration).
        )
        # --- END UPDATED ---
        user_capabilities[cap_key] = effective_value

    await log_event(
        'user_capabilities_viewed',
        {'target_uid': user_id},
        user_id=current_user.id,
        success=True,
        log_from_backend=True
    )

    return user_capabilities