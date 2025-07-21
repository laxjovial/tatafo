# backend/middleware/auth_middleware.py

from fastapi import Header, HTTPException, status, Depends
from typing import Optional, Dict, Any
import logging

# Import Firebase Admin SDK components
from firebase_admin import auth
from firebase_admin import exceptions as firebase_exceptions

# Import Pydantic models
from backend.models.user_models import UserProfile

# Import project-specific utilities and managers
from utils.analytics_tracker import log_event
from database.firestore_manager import FirestoreManager
from utils.user_manager import UserManager
from backend.services.api_usage_service import ApiUsageService

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Set to DEBUG for detailed logging during development


# Dependency to provide FirestoreManager instance
# This function will be overridden in main.py to provide the actual instance
async def get_firestore_manager_dependency(
    firestore_manager: FirestoreManager = Depends(lambda: None) # Default to None, will be overridden
) -> FirestoreManager:
    """Dependency to get the FirestoreManager instance."""
    if firestore_manager is None:
        logger.error("FirestoreManager dependency not properly injected.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server configuration error: FirestoreManager not initialized.")
    return firestore_manager

# Dependency to provide UserManager instance
# This function will be overridden in main.py to provide the actual instance
async def get_user_manager_dependency(
    firestore_manager_dep: FirestoreManager = Depends(get_firestore_manager_dependency),
    user_manager: UserManager = Depends(lambda: None) # Default to None, will be overridden
) -> UserManager:
    """Dependency to get the UserManager instance."""
    if user_manager is None:
        logger.error("UserManager dependency not properly injected.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server configuration error: UserManager not initialized.")
    return user_manager

# Dependency to provide ApiUsageService instance
# This function will be overridden in main.py to provide the actual instance
async def get_api_usage_service_dependency(
    firestore_manager_dep: FirestoreManager = Depends(get_firestore_manager_dependency),
    api_usage_service: ApiUsageService = Depends(lambda: None) # Default to None, will be overridden
) -> ApiUsageService:
    """Dependency to get the ApiUsageService instance."""
    if api_usage_service is None:
        logger.error("ApiUsageService dependency not properly injected.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server configuration error: ApiUsageService not initialized.")
    return api_usage_service


async def get_current_user(
    id_token: Optional[str] = Header(None, alias="Authorization"),
    user_manager: UserManager = Depends(get_user_manager_dependency)
) -> UserProfile:
    """
    FastAPI dependency to authenticate a user using Firebase ID Token.
    Extracts the ID token from the Authorization header (Bearer token),
    verifies it with Firebase Auth, and retrieves the user's profile.
    """
    if not id_token:
        await log_event(
            'authentication_failure',
            {'reason': 'No ID token provided'},
            success=False,
            error_message="Authentication header missing",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication token missing or invalid.")

    # Remove "Bearer " prefix if present
    if id_token.startswith("Bearer "):
        id_token = id_token[len("Bearer "):]

    try:
        # Verify the ID token using Firebase Admin SDK
        decoded_token = auth.verify_id_token(id_token)
        user_id = decoded_token['uid']
        email = decoded_token.get('email')

        # Retrieve user profile from Firestore using UserManager
        user_profile = await user_manager.get_user(user_id)

        if not user_profile:
            logger.warning(f"User profile not found in Firestore for UID: {user_id}. Attempting to create basic profile.")
            # This scenario can happen if a user is created in Firebase Auth but not yet in Firestore.
            # Create a basic profile to avoid errors downstream.
            user_profile_data = {
                "user_id": user_id,
                "email": email,
                "username": decoded_token.get('name', email.split('@')[0] if email else f"user_{user_id[:8]}"),
                "tier": "free", # Default tier
                "roles": ["user"], # Default role
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_login_at": datetime.now(timezone.utc).isoformat(),
                "profile_data": {}
            }
            create_result = await user_manager.create_user_profile(user_id, user_profile_data)
            if create_result["success"]:
                user_profile = user_profile_data
                logger.info(f"Basic user profile created for new user: {user_id}")
            else:
                logger.error(f"Failed to create basic user profile for {user_id}: {create_result.get('message')}")
                # Even if profile creation failed, we proceed with basic info from decoded token
                # This could be problematic for capabilities, so ideally profile exists.
                # For robustness, we will create a UserProfile object from decoded token if Firestore failed.
                user_profile = {
                    "user_id": user_id,
                    "email": email,
                    "username": decoded_token.get('name', email.split('@')[0] if email else f"user_{user_id[:8]}"),
                    "tier": "free",
                    "roles": ["user"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "last_login_at": datetime.now(timezone.utc).isoformat(),
                    "profile_data": {}
                }

        # Update last_login_at
        await user_manager.update_user_last_login(user_id)

        # Increment API usage for authentication event
        api_usage_service: ApiUsageService = Depends(get_api_usage_service_dependency)() # Get instance to increment usage
        await api_usage_service.increment_api_call_count(user_id, "authentication_success")
        
        # Log successful authentication
        await log_event(
            'authentication_success',
            {'uid': user_id, 'email': email},
            user_id=user_id,
            success=True,
            log_from_backend=True
        )

        return UserProfile(**user_profile) # Convert dict to Pydantic model

    except firebase_exceptions.FirebaseError as e:
        logger.error(f"Firebase authentication error: {e}", exc_info=True)
        # Check specific Firebase error codes
        if e.code == 'auth/argument-error':
            detail = "Invalid ID token format."
        elif e.code == 'auth/invalid-id-token':
            detail = "Invalid or expired ID token."
        elif e.code == 'auth/user-not-found':
            detail = "User not found."
        else:
            detail = f"Firebase authentication error: {e.code}"

        await log_event(
            'authentication_failure',
            {'reason': detail, 'error_code': e.code},
            success=False,
            error_message=detail,
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)
    except Exception as e:
        logger.error(f"An unexpected authentication error occurred: {e}", exc_info=True)
        await log_event(
            'authentication_failure',
            {'reason': 'Unexpected error', 'error': str(e)},
            success=False,
            error_message=f"An unexpected authentication error occurred: {str(e)}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Authentication error: {str(e)}")


async def get_current_admin_user(current_user: UserProfile = Depends(get_current_user)) -> UserProfile:
    """
    FastAPI dependency to get the currently authenticated user with 'admin' or 'creator' role.
    Returns UserProfile if authorized, otherwise raises 403.
    """
    if "admin" not in current_user.roles and "creator" not in current_user.roles:
        await log_event(
            'authorization_failure',
            {'required_role': 'admin_or_creator', 'user_roles': current_user.roles},
            user_id=current_user.user_id,
            success=False,
            error_message="Not authorized: Admin or Creator access required",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized. Admin or Creator access required.")
    return current_user

# Other role-specific dependencies can be added here if needed, following the same pattern:
# async def get_current_customer_care_user(current_user: UserProfile = Depends(get_current_user)) -> UserProfile:
#     if "customer_care" not in current_user.roles:
#         await log_event(...); raise HTTPException(...)
#     return current_user