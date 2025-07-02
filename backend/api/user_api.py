# backend/api/user_api.py

from fastapi import APIRouter, HTTPException, status, Depends
from typing import Annotated, List, Dict, Any

# Import Pydantic models from our backend.models
from backend.models.user_models import UserProfile, UserUpdate

# Import middleware for authentication and authorization
from backend.middleware.auth_middleware import get_current_active_user, get_current_admin_user

# Import FirestoreManager
from database.firestore_manager import firestore_manager

# Import Firebase Auth (for updating custom claims)
from firebase_admin import auth

router = APIRouter()

@router.get("/{user_id}", response_model=UserProfile)
async def get_user_profile(user_id: str, current_user: Annotated[dict, Depends(get_current_active_user)]):
    """
    Retrieves a user's profile by ID from Firestore.
    Requires authentication. User can view their own profile; admin can view any.
    """
    # Authorization check: A user can only view their own profile unless they are an admin
    if current_user["user_id"] != user_id and "admin" not in current_user.get("roles", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this user's profile"
        )

    try:
        user_data = await firestore_manager.get_user_data(user_id)
        if not user_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        # Ensure roles are a list for the Pydantic model
        if isinstance(user_data.get('roles'), str):
            user_data['roles'] = user_data['roles'].split(',')
        
        return UserProfile(**user_data)
    except Exception as e:
        logger.error(f"Error fetching user profile for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch user profile: {e}")

@router.put("/{user_id}", response_model=UserProfile)
async def update_user_profile(user_id: str, user_update: UserUpdate, current_user: Annotated[dict, Depends(get_current_active_user)]):
    """
    Updates a user's profile in Firestore and Firebase Auth custom claims.
    Requires authentication. User can update their own profile (limited fields); admin can update any.
    """
    is_admin = "admin" in current_user.get("roles", [])
    if current_user["user_id"] != user_id and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this user's profile"
        )

    try:
        existing_user_data = await firestore_manager.get_user_data(user_id)
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

        # Update Firestore document
        await firestore_manager.update_user_data(user_id, update_data)

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
        updated_user_data = await firestore_manager.get_user_data(user_id)
        if isinstance(updated_user_data.get('roles'), str):
            updated_user_data['roles'] = updated_user_data['roles'].split(',')

        return UserProfile(**updated_user_data)
    except auth.UserNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found in Firebase Auth.")
    except Exception as e:
        logger.error(f"Error updating user profile for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update user profile: {e}")

@router.get("/", response_model=List[UserProfile])
async def get_all_users_api(current_user: Annotated[dict, Depends(get_current_admin_user)]):
    """
    Retrieves a list of all users from Firestore.
    Requires admin authorization.
    """
    # The `get_current_admin_user` dependency already ensures only admins can access this.
    try:
        all_users_data = await firestore_manager.get_all_user_profiles()
        
        users_list = []
        for user_data in all_users_data:
            # Ensure roles are a list for the Pydantic model
            if isinstance(user_data.get('roles'), str):
                user_data['roles'] = user_data['roles'].split(',')
            users_list.append(UserProfile(**user_data))
        return users_list
    except Exception as e:
        logger.error(f"Error fetching all user profiles: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch all user profiles: {e}")
