# backend/api/user_api.py

import logging
from fastapi import APIRouter, HTTPException, status, Depends
from typing import Annotated, List, Dict, Any, Optional

# Import Pydantic models
from backend.models.user_models import UserProfile, UserUpdate

# Import middleware for authentication and authorization
from backend.middleware.auth_middleware import get_current_active_user, get_current_admin_user, get_user_manager_dependency

# Import UserManager (now the primary source for user data logic)
from utils.user_manager import UserManager, get_user_tier_capability, _RBAC_CAPABILITIES # Import _RBAC_CAPABILITIES for the capabilities endpoint

# Import Firebase Auth (for updating custom claims)
from firebase_admin import auth

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

router = APIRouter()

@router.get("/{user_id}", response_model=UserProfile)
async def get_user_profile(
    user_id: str,
    current_user: Annotated[UserProfile, Depends(get_current_active_user)], # Use UserProfile type hint
    user_manager: UserManager = Depends(get_user_manager_dependency) # Inject UserManager
):
    """
    Retrieves a user's profile by ID from Firestore.
    Requires authentication. User can view their own profile; admin can view any.
    """
    # Authorization check: A user can only view their own profile unless they are an admin
    if current_user.user_id != user_id and "admin" not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this user's profile"
        )

    try:
        # Use UserManager to get user data
        user_data = await user_manager.get_user(user_id)
        if not user_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        # Ensure roles are a list for the Pydantic model
        if isinstance(user_data.get('roles'), str):
            user_data['roles'] = user_data['roles'].split(',')
        
        # Ensure 'uid' from Firestore is mapped to 'user_id' for the Pydantic model
        user_data['user_id'] = user_data.get('uid')

        return UserProfile(**user_data)
    except Exception as e:
        logger.error(f"Error fetching user profile for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch user profile: {e}")

@router.put("/{user_id}", response_model=UserProfile)
async def update_user_profile(
    user_id: str,
    user_update: UserUpdate,
    current_user: Annotated[UserProfile, Depends(get_current_active_user)], # Use UserProfile type hint
    user_manager: UserManager = Depends(get_user_manager_dependency) # Inject UserManager
):
    """
    Updates a user's profile in Firestore and Firebase Auth custom claims.
    Requires authentication. User can update their own profile (limited fields); admin can update any.
    """
    is_admin = "admin" in current_user.roles
    if current_user.user_id != user_id and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this user's profile"
        )

    try:
        # Use UserManager to get existing user data
        existing_user_data = await user_manager.get_user(user_id)
        if not existing_user_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        update_data = user_update.model_dump(exclude_unset=True)

        # Restrict non-admin users from changing tier or roles
        if not is_admin:
            if 'tier' in update_data:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators can change user tiers.")
            if 'roles' in update_data:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators can change user roles.")
            # Optionally, restrict email changes for non-admins or require verification
            if 'email' in update_data and update_data['email'] != existing_user_data['email']:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Email change requires specific verification process or admin privileges.")

        # Use UserManager to update Firestore document
        await user_manager.update_user_profile(user_id, update_data)

        # If tier or roles are updated, also update Firebase Auth custom claims
        if 'tier' in update_data or 'roles' in update_data:
            # Fetch current claims to ensure we don't overwrite other claims
            user_record = auth.get_user(user_id)
            current_claims = user_record.custom_claims or {}
            
            new_claims = {**current_claims}
            if 'tier' in update_data:
                new_claims['tier'] = update_data['tier']
            if 'roles' in update_data:
                new_claims['roles'] = update_data['roles']
            
            auth.set_custom_user_claims(user_id, new_claims)
            # Invalidate user's refresh tokens to force re-authentication and claim update on client
            auth.revoke_refresh_tokens(user_id)
            logger.info(f"Firebase Auth custom claims updated for user {user_id}. Tokens revoked.")

        # Fetch updated user data to return the latest state
        updated_user_data = await user_manager.get_user(user_id)
        if isinstance(updated_user_data.get('roles'), str):
            updated_user_data['roles'] = updated_user_data['roles'].split(',')
        updated_user_data['user_id'] = user_id # Ensure user_id is set for the response model

        return UserProfile(**updated_user_data)
    except auth.UserNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found in Firebase Auth.")
    except Exception as e:
        logger.error(f"Error updating user profile for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update user profile: {e}")

@router.get("/", response_model=List[UserProfile])
async def get_all_users_api(
    current_user: Annotated[UserProfile, Depends(get_current_admin_user)], # Use UserProfile type hint
    user_manager: UserManager = Depends(get_user_manager_dependency) # Inject UserManager
):
    """
    Retrieves a list of all users from Firestore.
    Requires admin authorization.
    """
    # The `get_current_admin_user` dependency already ensures only admins can access this.
    try:
        # Use UserManager to get all user profiles
        all_users_data_response = await user_manager.get_all_users_admin()
        if not all_users_data_response.get("success"):
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=all_users_data_response.get("message", "Failed to retrieve all users."))
        
        all_users_data = all_users_data_response.get("users", [])
        
        users_list = []
        for user_data in all_users_data:
            # Ensure roles are a list for the Pydantic model
            if isinstance(user_data.get('roles'), str):
                user_data['roles'] = user_data['roles'].split(',')
            user_data['user_id'] = user_data.get('uid') # Map uid to user_id for UserProfile Pydantic model
            users_list.append(UserProfile(**user_data))
        return users_list
    except Exception as e:
        logger.error(f"Error fetching all user profiles: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch all user profiles: {e}")

# MERGED FROM backend/api/routes/users.py
@router.get("/capabilities/{user_id}", response_model=Dict[str, Any])
async def get_user_capabilities_route(
    user_id: str, # Changed from user_token to user_id as we have UserProfile now
    current_user: Annotated[UserProfile, Depends(get_current_active_user)], # Ensure user is authenticated
    user_manager: UserManager = Depends(get_user_manager_dependency) # Inject UserManager
) -> Dict[str, Any]:
    """
    Retrieves all RBAC capabilities for a given user.
    This endpoint provides the full map of capabilities with their effective access levels
    for the specified user, dynamically loaded from Firestore.
    Requires authentication. User can view their own capabilities; admin can view any.
    """
    # Authorization check: A user can only view their own capabilities unless they are an admin
    if current_user.user_id != user_id and "admin" not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this user's capabilities"
        )

    logger.info(f"API: Retrieving RBAC capabilities for user_id: {user_id}")
    
    user_capabilities = {}
    # Ensure _RBAC_CAPABILITIES is loaded in user_manager (it's loaded on UserManager init)
    
    # Fetch the target user's profile to get their tier and roles
    target_user_profile = await user_manager.get_user(user_id)
    if not target_user_profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target user not found")

    target_user_tier = target_user_profile.get('tier', 'free')
    target_user_roles = target_user_profile.get('roles', [])
    if isinstance(target_user_roles, str): # Ensure roles is a list
        target_user_roles = target_user_roles.split(',')

    # Iterate through the global RBAC capabilities configuration
    for cap_key, cap_info_template in _RBAC_CAPABILITIES.get('capabilities', {}).items():
        # get_user_tier_capability now directly uses tier and roles from the fetched profile
        effective_value = get_user_tier_capability(
            user_id=user_id, # Still pass user_id for logging/context within get_user_tier_capability
            capability_key=cap_key,
            default_value=cap_info_template.get('default'),
            user_tier=target_user_tier, # Pass the target user's tier
            user_roles=target_user_roles # Pass the target user's roles
        )
        user_capabilities[cap_key] = effective_value
    
    logger.info(f"API: Successfully retrieved RBAC capabilities for user_id: {user_id}")
    return user_capabilities

@router.get("/api-keys", response_model=List[Dict[str, str]])
async def get_user_api_keys(
    current_user: Annotated[UserProfile, Depends(get_current_active_user)],
    user_manager: UserManager = Depends(get_user_manager_dependency)
):
    """
    Retrieves a user's API keys from Firestore.
    """
    try:
        user_data = await user_manager.get_user(current_user.user_id)
        if not user_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        api_keys = user_data.get("api_keys", [])
        return api_keys
    except Exception as e:
        logger.error(f"Error fetching user API keys for {current_user.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch user API keys: {e}")

@router.post("/api-keys", response_model=Dict[str, str])
async def add_user_api_key(
    api_key: Dict[str, str],
    current_user: Annotated[UserProfile, Depends(get_current_active_user)],
    user_manager: UserManager = Depends(get_user_manager_dependency)
):
    """
    Adds a new API key to a user's profile in Firestore.
    """
    try:
        await user_manager.add_api_key(current_user.user_id, api_key)
        return api_key
    except Exception as e:
        logger.error(f"Error adding user API key for {current_user.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to add user API key: {e}")
