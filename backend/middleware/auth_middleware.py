# backend/middleware/auth_middleware.py

from fastapi import Header, HTTPException, status, Depends
from typing import Optional, Dict, Any
import logging

# Import Firebase Admin SDK components
from firebase_admin import auth
from firebase_admin import exceptions as firebase_exceptions

# Import Pydantic models
from backend.models.user_models import UserProfile

# Import project-specific utilities and managers (only type hints needed here)
from utils.analytics_tracker import log_event
from database.firestore_manager import FirestoreManager # For type hinting in Depends
from utils.user_manager import UserManager # For type hinting in Depends

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Set to DEBUG for detailed logging during development


# Dependency to provide FirestoreManager instance
# This function will be called by FastAPI's dependency injection system
# and main.py will need to provide the actual instance.
async def get_firestore_manager_dependency() -> FirestoreManager:
    """Dependency to get the FirestoreManager instance."""
    # This will be overridden by main.py's dependency override.
    # It's here to provide a type hint and a placeholder.
    raise NotImplementedError("FirestoreManager dependency must be provided by main.py")

# Dependency to provide UserManager instance
async def get_user_manager_dependency(
    firestore_manager_dep: FirestoreManager = Depends(get_firestore_manager_dependency)
) -> UserManager:
    """Dependency to get the UserManager instance."""
    # This will be overridden by main.py's dependency override.
    raise NotImplementedError("UserManager dependency must be provided by main.py")


# The main authentication dependency
async def get_current_user(
    authorization: Optional[str] = Header(None),
    user_manager: UserManager = Depends(get_user_manager_dependency) # Inject UserManager
) -> UserProfile:
    """
    FastAPI dependency to authenticate the user using a Firebase ID token.
    Extracts the token from the Authorization header (Bearer token).
    Verifies the token, fetches the user profile from Firestore, checks account status,
    updates last login time, and returns the UserProfile object.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, token = authorization.split()
    if scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme. Must be 'Bearer'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # Verify the Firebase ID token
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token['uid']
        
        # Retrieve user profile from Firestore using UserManager
        user_data = await user_manager.get_user(uid) 

        if not user_data:
            # Log specific failure for user profile not found
            await log_event(
                'authentication_failure',
                {'uid': uid, 'error_details': 'User profile not found in Firestore'},
                user_id=uid,
                success=False,
                error_message="User profile not found.",
                log_from_backend=True
            )
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User profile not found.")
        
        # Check if account is disabled/suspended
        # The 'status' field is now part of the UserProfile model
        if user_data.get('status') == 'disabled' or user_data.get('status') == 'suspended':
            await log_event(
                'authentication_failure',
                {'uid': uid, 'error_details': f"Account status: {user_data.get('status')}"},
                user_id=uid,
                success=False,
                error_message="Your account is currently disabled or suspended. Please contact support.",
                log_from_backend=True
            )
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Your account is disabled or suspended. Please contact support.")

        # Update last_login_at timestamp
        await user_manager.update_last_login(uid)

        logger.info(f"User {uid} authenticated successfully via Firebase ID Token.")
        await log_event(
            'user_authenticated',
            {'uid': uid},
            user_id=uid,
            success=True,
            log_from_backend=True
        )
        # Ensure 'user_id' field is present for UserProfile Pydantic model
        user_data['user_id'] = uid 
        return UserProfile(**user_data)
    except firebase_exceptions.AuthError as e:
        logger.error(f"Firebase ID Token verification failed: {e}", exc_info=True)
        await log_event(
            'authentication_failure',
            {'error_details': str(e), 'firebase_code': e.code if hasattr(e, 'code') else 'N/A'},
            user_id="unauthenticated",
            success=False,
            error_message=f"Invalid authentication credentials: {e.code}. Please log in again.",
            log_from_backend=True
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {e.code}. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during authentication: {e}", exc_info=True)
        await log_event(
            'authentication_failure',
            {'error_details': str(e)},
            user_id="unauthenticated",
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
    # Check for 'admin' role or 'creator' role (creator implies full admin access)
    if "admin" not in current_user.roles and "creator" not in current_user.roles:
        # Log authorization failure
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
#         await log_event(...)
#         raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized: Customer Care access required")
#     return current_user
  
