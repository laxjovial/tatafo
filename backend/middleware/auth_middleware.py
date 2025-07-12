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
# NOTE: firestore_manager and user_manager are initialized in main.py.
# For this middleware to access them, they need to be passed as dependencies,
# or main.py needs to ensure they are globally accessible after initialization.
# For now, we'll import and assume they are initialized elsewhere (e.g., in main.py)
# and accessed as global variables or through a singleton pattern.
# This is a temporary measure; proper dependency injection is preferred in a large app.
from database.firestore_manager import firestore_manager # Assuming this is globally available after main.py init
from utils.user_manager import UserManager # Assuming this is globally available after main.py init

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Set to DEBUG for detailed logging during development

# Re-initialize UserManager and FirestoreManager if they are not truly global singletons
# This is a workaround for the current structure; ideally, these would be passed via dependency injection.
# We'll assume they are initialized in main.py and accessible.
# If they are not truly global, this will cause issues.
# For the purpose of generating the correct logic, we'll assume they are available.
try:
    _firestore_manager = firestore_manager # Access the already initialized instance
    _user_manager = UserManager(_firestore_manager, None) # cloud_storage_utils is not needed here, pass None for now
except NameError:
    logger.warning("firestore_manager or UserManager not yet initialized globally. Middleware might fail if not properly injected.")
    # Fallback for local testing/initial setup if not globally initialized
    # In a real app, this would be a fatal error or use FastAPI's dependency injection.
    _firestore_manager = None
    _user_manager = None


# The main authentication dependency
async def get_current_user(authorization: Optional[str] = Header(None)) -> UserProfile:
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

    if _firestore_manager is None or _user_manager is None:
        logger.error("FirestoreManager or UserManager not initialized in auth_middleware.py. Cannot authenticate.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server authentication services not properly initialized."
        )

    try:
        # Verify the Firebase ID token
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token['uid']
        
        # Retrieve user profile from Firestore using UserManager
        user_data = await _user_manager.get_user(uid) 

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
        await _user_manager.update_last_login(uid)

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


# Renamed get_current_active_user to get_current_user as it's the primary authenticated user dependency
# This function is now the central point for user authentication.

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

