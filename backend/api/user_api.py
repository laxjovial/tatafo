# backend/api/user_api.py

from fastapi import APIRouter, HTTPException, status, Depends
from typing import Annotated, List, Dict, Any
import logging

# Import Pydantic models from our backend.models
from backend.models.user_models import UserProfile, UserUpdate

# Import middleware for authentication and authorization
# Now importing the dependency functions for current user, admin, UserManager and FirestoreManager
from backend.middleware.auth_middleware import get_current_user, get_current_admin_user, get_firestore_manager_dependency, get_user_manager_dependency

# Import Firebase Auth (for updating custom claims)
from firebase_admin import auth
from firebase_admin import exceptions as firebase_exceptions

# Project imports for analytics
from utils.analytics_tracker import log_event
from utils.user_manager import UserManager # For type hinting in Depends
from database.firestore_manager import FirestoreManager # For type hinting in Depends

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Set to DEBUG for detailed logging during development

router = APIRouter()

@router.get("/{user_id}", response_model=UserProfile)
async def get_user_profile(
    user_id: str,
    current_user: Annotated[UserProfile, Depends(get_current_user)], # Type-hint as UserProfile
    user_manager: UserManager = Depends(get_user_manager_dependency) # Inject UserManager
):
    """
    Retrieves a user's profile by ID from Firestore.
    Requires authentication. User can view their own profile; admin or creator can view any.
    """
    # Authorization check: A user can only view their own profile unless they are an admin or creator
    is_admin_or_creator = "admin" in current_user.roles or "creator" in current_user.roles
    if current_user.user_id != user_id and not is_admin_or_creator:
        await log_event(
            'authorization_failure',
            {'action': 'view_user_profile', 'target_user_id': user_id, 'reason': 'Not authorized'},
            user_id=current_user.user_id,
            success=False,
            error_message="Not authorized to view this user's profile",
            log_from_backend=True
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this user's profile"
        )

    logger.info(f"User {current_user.user_id} requesting profile for {user_id}.")
    try:
        user_data = await user_manager.get_user(user_id) # Use injected UserManager to get user data
        if not user_data:
            await log_event(
                'user_profile_retrieval',
                {'target_user_id': user_id, 'reason': 'User not found'},
                user_id=current_user.user_id,
                success=False,
                error_message="User profile not found",
                log_from_backend=True
            )
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        # Ensure roles are a list for the Pydantic model (UserManager should ideally handle this consistently)
        if isinstance(user_data.get('roles'), str):
            user_data['roles'] = user_data['roles'].split(',')
        
        # Ensure 'user_id' field is present for UserProfile Pydantic model
        user_data['user_id'] = user_id

        logger.info(f"Successfully retrieved profile for user {user_id}.")
        await log_event(
            'user_profile_retrieval',
            {'target_user_id': user_id},
            user_id=current_user.user_id,
            success=True,
            log_from_backend=True
        )
        return UserProfile(**user_data)
    except HTTPException:
        raise # Re-raise HTTPExceptions
    except Exception as e:
        logger.error(f"Error fetching user profile for {user_id}: {e}", exc_info=True)
        await log_event(
            'user_profile_retrieval',
            {'target_user_id': user_id, 'error': str(e)},
            user_id=current_user.user_id,
            success=False,
            error_message=f"Failed to fetch user profile: {e}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch user profile: {e}")

@router.put("/{user_id}", response_model=UserProfile)
async def update_user_profile(
    user_id: str,
    user_update: UserUpdate,
    current_user: Annotated[UserProfile, Depends(get_current_user)], # Type-hint as UserProfile
    user_manager: UserManager = Depends(get_user_manager_dependency) # Inject UserManager
):
    """
    Updates a user's profile in Firestore and Firebase Auth custom claims.
    Requires authentication. User can update their own profile (limited fields);
    admin or creator can update any user's profile (including tier/roles/status).
    """
    is_admin_or_creator = "admin" in current_user.roles or "creator" in current_user.roles

    # Authorization check: A user can only update their own profile unless they are an admin or creator
    if current_user.user_id != user_id and not is_admin_or_creator:
        await log_event(
            'authorization_failure',
            {'action': 'update_user_profile', 'target_user_id': user_id, 'reason': 'Not authorized'},
            user_id=current_user.user_id,
            success=False,
            error_message="Not authorized to update this user's profile",
            log_from_backend=True
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this user's profile"
        )

    logger.info(f"User {current_user.user_id} attempting to update profile for {user_id}.")
    update_data = user_update.model_dump(exclude_unset=True)

    if not update_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update.")

    try:
        existing_user_data = await user_manager.get_user(user_id) # Use injected UserManager
        if not existing_user_data:
            await log_event(
                'user_profile_update',
                {'target_user_id': user_id, 'reason': 'User not found'},
                user_id=current_user.user_id,
                success=False,
                error_message="User not found for update",
                log_from_backend=True
            )
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        # --- Tier, Roles, Status Update Restrictions ---
        # Only admins/creator can change tier, roles, or status
        if not is_admin_or_creator:
            if 'tier' in update_data:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators can change user tiers.")
            if 'roles' in update_data:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators can change user roles.")
            if 'status' in update_data:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators can change user account status.")
            
        # --- Email Update Handling ---
        # If email is being updated, UserManager should handle Firebase Auth update
        if 'email' in update_data and update_data['email'] != existing_user_data.get('email'):
            try:
                # UserManager's update_user_profile should handle Firebase Auth email update
                # This typically requires re-authentication on the client side for security.
                # For server-side update by admin, it's a direct call.
                auth.update_user(user_id, email=update_data['email'])
                logger.info(f"Firebase Auth email updated for {user_id}")
            except firebase_exceptions.FirebaseError as e:
                logger.error(f"Firebase Auth email update failed for {user_id}: {e}", exc_info=True)
                await log_event(
                    'user_profile_update',
                    {'target_user_id': user_id, 'field': 'email', 'error': str(e)},
                    user_id=current_user.user_id,
                    success=False,
                    error_message=f"Failed to update email in Firebase Auth: {e.code}",
                    log_from_backend=True
                )
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to update email in Firebase Auth: {e.code}")
            # Remove email from update_data so FirestoreManager doesn't try to update it directly if Firebase Auth handles it
            update_data.pop('email', None)

        # Update Firestore document using UserManager
        result = await user_manager.update_user_profile(user_id, update_data) # Use injected UserManager
        
        if not result["success"]:
            await log_event(
                'user_profile_update',
                {'target_user_id': user_id, 'fields': list(update_data.keys()), 'error': result["message"]},
                user_id=current_user.user_id,
                success=False,
                error_message=result["message"],
                log_from_backend=True
            )
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])

        # Fetch updated user data to return the latest state
        updated_user_data = await user_manager.get_user(user_id) # Use injected UserManager
        # Ensure roles is a list and user_id is present for Pydantic model
        if isinstance(updated_user_data.get('roles'), str):
            updated_user_data['roles'] = updated_user_data['roles'].split(',')
        updated_user_data['user_id'] = user_id

        logger.info(f"User {user_id} profile updated successfully by {current_user.user_id}.")
        await log_event(
            'user_profile_updated',
            {'target_user_id': user_id, 'fields': list(update_data.keys())},
            user_id=current_user.user_id,
            success=True,
            log_from_backend=True
        )
        return UserProfile(**updated_user_data)
    except HTTPException:
        raise # Re-raise HTTPExceptions
    except Exception as e:
        logger.error(f"Error updating user profile for {user_id}: {e}", exc_info=True)
        await log_event(
            'user_profile_update',
            {'target_user_id': user_id, 'fields': list(update_data.keys()), 'error': str(e)},
            user_id=current_user.user_id,
            success=False,
            error_message=f"Failed to update user profile: {e}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update user profile: {e}")

@router.get("/", response_model=List[UserProfile])
async def get_all_users_api(
    current_user: Annotated[UserProfile, Depends(get_current_admin_user)], # Type-hint as UserProfile
    user_manager: UserManager = Depends(get_user_manager_dependency) # Inject UserManager
):
    """
    Retrieves a list of all users from Firestore.
    Requires admin or creator authorization.
    """
    admin_uid = current_user.user_id
    logger.info(f"Admin user {admin_uid} requesting all user profiles.")
    try:
        all_users_data = await user_manager.get_all_user_profiles() # Use injected UserManager
        
        users_list = []
        for user_data in all_users_data:
            # Ensure roles are a list and user_id is present for Pydantic model
            if isinstance(user_data.get('roles'), str):
                user_data['roles'] = user_data['roles'].split(',')
            user_data['user_id'] = user_data.get('uid') # Map Firestore UID to user_id for Pydantic model
            users_list.append(UserProfile(**user_data))

        logger.info(f"Admin user {admin_uid} successfully retrieved all user profiles.")
        await log_event(
            'admin_action_get_all_users',
            {},
            user_id=admin_uid,
            success=True,
            log_from_backend=True
        )
        return users_list
    except Exception as e:
        logger.error(f"Error fetching all user profiles for admin {admin_uid}: {e}", exc_info=True)
        await log_event(
            'admin_action_get_all_users',
            {'error': str(e)},
            user_id=admin_uid,
            success=False,
            error_message=f"Failed to fetch all user profiles: {e}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch all user profiles: {e}")
