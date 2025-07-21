# backend/api/auth_api.py

from fastapi import APIRouter, HTTPException, status, Depends
from typing import Annotated, Dict, Any
import logging
from datetime import datetime, timezone

# Import Pydantic models from our backend.models
from backend.models.user_models import UserCreate, UserLogin, PasswordResetRequest, PasswordResetConfirm, ChangePassword, UserProfile # Added UserProfile

# Import middleware for protected routes (e.g., change password)
from backend.middleware.auth_middleware import get_current_user, get_firestore_manager_dependency, get_user_manager_dependency

# Import Firebase Auth (for creating users and setting custom claims)
from firebase_admin import auth
from firebase_admin import exceptions as firebase_exceptions

# Project imports for analytics and config
from utils.analytics_tracker import log_event
from config.config_manager import config_manager
from utils.user_manager import UserManager # For type hinting in Depends
from database.firestore_manager import FirestoreManager # For type hinting in Depends

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

router = APIRouter()

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    user_manager: UserManager = Depends(get_user_manager_dependency)
):
    """
    Registers a new user in Firebase Authentication and stores profile in Firestore.
    """
    if not user_data.email or not user_data.password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email and password are required.")

    logger.info(f"Attempting to register user: {user_data.email}")

    try:
        # 1. Create user in Firebase Authentication
        user = auth.create_user(
            email=user_data.email,
            password=user_data.password,
            display_name=user_data.username
        )
        uid = user.uid
        logger.info(f"Firebase Auth user created with UID: {uid}")

        # 2. Set initial custom claims for roles
        # Only set custom claims if `set_custom_claims_on_registration` is enabled in config
        if config_manager.get("auth.set_custom_claims_on_registration", True):
            initial_roles = ["user"]
            custom_claims = {"roles": initial_roles}
            auth.set_custom_user_claims(uid, custom_claims)
            logger.info(f"Custom claims set for UID {uid}: {custom_claims}")
        else:
            initial_roles = ["user"] # Still ensure default role for Firestore profile
            logger.info(f"Custom claims not set on registration as per config.")

        # 3. Create user profile in Firestore
        profile_result = await user_manager.create_user_profile(
            uid=uid,
            email=user_data.email,
            username=user_data.username,
            initial_tier="free", # Default initial tier
            initial_roles=initial_roles # Pass initial roles
        )

        if not profile_result["success"]:
            # If Firestore profile creation fails, attempt to delete Firebase Auth user
            logger.error(f"Failed to create Firestore profile for {uid}. Attempting to delete Firebase Auth user.")
            try:
                auth.delete_user(uid)
                logger.info(f"Firebase Auth user {uid} deleted due to Firestore profile creation failure.")
            except firebase_exceptions.FirebaseError as fe:
                logger.error(f"Failed to delete Firebase Auth user {uid} after Firestore error: {fe}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"User registration failed: {profile_result['message']}")

        await log_event(
            'user_registered',
            {'uid': uid, 'email': user_data.email},
            user_id=uid,
            success=True,
            log_from_backend=True
        )

        return {"message": "User registered successfully", "uid": uid, "email": user_data.email}

    except firebase_exceptions.FirebaseError as e:
        logger.error(f"Firebase registration error: {e}", exc_info=True)
        await log_event(
            'user_registration_failure',
            {'email': user_data.email, 'error': e.code},
            user_id=None,
            success=False,
            error_message=e.code,
            log_from_backend=True
        )
        if e.code == 'auth/email-already-exists':
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Registration failed: {e.code}")
    except Exception as e:
        logger.error(f"Unexpected error during registration: {e}", exc_info=True)
        await log_event(
            'user_registration_failure',
            {'email': user_data.email, 'error': str(e)},
            user_id=None,
            success=False,
            error_message=str(e),
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/login")
async def login_user(
    user_data: UserLogin,
    user_manager: UserManager = Depends(get_user_manager_dependency) # Inject UserManager
):
    """
    Exchanges Firebase ID token for verification and logs user in.
    Note: Actual token exchange happens on the client-side. This endpoint
    receives the ID token and verifies it using Firebase Admin SDK.
    """
    logger.info(f"Attempting to verify ID token.")

    try:
        # Verify the ID token using Firebase Admin SDK
        decoded_token = auth.verify_id_token(user_data.id_token)
        uid = decoded_token['uid']
        email = decoded_token.get('email')
        name = decoded_token.get('name', 'User')

        logger.info(f"ID token verified for UID: {uid}")

        # Retrieve or create user profile in Firestore
        user_profile = await user_manager.get_user(uid)
        if not user_profile:
            # If user profile doesn't exist (e.g., old user, or created directly in Firebase console)
            logger.warning(f"Firestore profile not found for {uid}. Creating one now.")
            # Get user record to ensure we have a display_name if available
            user_record = auth.get_user(uid)
            username = user_record.display_name if user_record.display_name else "Anonymous"
            # Extract roles from custom claims if available
            initial_roles = decoded_token.get('roles', ['user'])
            if not isinstance(initial_roles, list): # Ensure it's a list
                initial_roles = [initial_roles]
            if "user" not in initial_roles: initial_roles.append("user") # Ensure base 'user' role

            profile_result = await user_manager.create_user_profile(
                uid=uid,
                email=email,
                username=username,
                initial_tier="free", # Default tier for existing Firebase Auth users without profile
                initial_roles=initial_roles
            )
            if not profile_result["success"]:
                logger.error(f"Failed to create Firestore profile for existing Firebase user {uid}: {profile_result['message']}")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to initialize user profile.")
            user_profile = await user_manager.get_user(uid) # Re-fetch the newly created profile


        # --- NEW: Update last_login_at timestamp ---
        await user_manager.update_user_profile(uid, {"last_login_at": datetime.now(timezone.utc)})
        logger.info(f"User {uid} last_login_at updated.")
        # --- END NEW ---

        await log_event(
            'user_logged_in',
            {'uid': uid, 'email': email},
            user_id=uid,
            success=True,
            log_from_backend=True
        )

        return {"message": "Login successful", "uid": uid, "email": email}

    except firebase_exceptions.FirebaseError as e:
        logger.error(f"Firebase token verification error: {e}", exc_info=True)
        await log_event(
            'user_login_failure',
            {'error': e.code},
            user_id=None,
            success=False,
            error_message=e.code,
            log_from_backend=True
        )
        if e.code == 'auth/id-token-expired':
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired. Please log in again.")
        elif e.code == 'auth/argument-error':
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid ID token provided.")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Authentication failed: {e.code}")
    except Exception as e:
        logger.error(f"Unexpected error during login: {e}", exc_info=True)
        await log_event(
            'user_login_failure',
            {'error': str(e)},
            user_id=None,
            success=False,
            error_message=str(e),
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/reset-password-request", status_code=status.HTTP_200_OK)
async def request_password_reset(data: PasswordResetRequest):
    """
    Sends a password reset email to the user.
    """
    logger.info(f"Password reset request for: {data.email}")
    try:
        auth.generate_password_reset_link(data.email)
        logger.info(f"Password reset link sent to {data.email}")
        await log_event(
            'password_reset_request',
            {'email': data.email},
            user_id=None, # User ID unknown at this point
            success=True,
            log_from_backend=True
        )
        return {"message": "Password reset email sent successfully. Check your inbox."}
    except firebase_exceptions.FirebaseError as e:
        logger.error(f"Firebase error requesting password reset for {data.email}: {e}", exc_info=True)
        await log_event(
            'password_reset_request_failure',
            {'email': data.email, 'error': str(e)},
            user_id=None,
            success=False,
            error_message=str(e),
            log_from_backend=True
        )
        if e.code == 'auth/user-not-found':
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No user found with that email address.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Password reset request failed: {e.code}")
    except Exception as e:
        logger.error(f"Unexpected error requesting password reset for {data.email}: {e}", exc_info=True)
        await log_event(
            'password_reset_request_failure',
            {'email': data.email, 'error': str(e)},
            user_id=None,
            success=False,
            error_message=str(e),
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/change-password", status_code=status.HTTP_200_OK)
async def change_password(
    data: ChangePassword,
    current_user: Annotated[UserProfile, Depends(get_current_user)] # Changed from Dict[str, Any]
):
    """
    Allows an authenticated user to change their password.
    This endpoint would typically be called after a user is logged in
    and provides their new password.
    """
    user_id = current_user.id # Changed from .get('id')
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not identified.")

    logger.info(f"Attempting to change password for user: {user_id}")
    try:
        # Use Firebase Admin SDK to update the password directly
        auth.update_user(user_id, password=data.new_password)

        logger.info(f"Password for user {user_id} changed successfully.")
        await log_event(
            'password_changed',
            {'uid': user_id},
            user_id=user_id,
            success=True,
            log_from_backend=True
        )
        return {"message": "Password changed successfully."}
    except firebase_exceptions.FirebaseError as e:
        logger.error(f"Firebase error changing password for user {user_id}: {e}", exc_info=True)
        await log_event(
            'password_changed',
            {'uid': user_id, 'error': str(e)},
            user_id=user_id,
            success=False,
            error_message=f"Failed to change password: {e.code}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to change password: {e.code}")
    except Exception as e:
        logger.error(f"An unexpected error occurred changing password for user {user_id}: {e}", exc_info=True)
        await log_event(
            'password_changed',
            {'uid': user_id, 'error': str(e)},
            user_id=user_id,
            success=False,
            error_message=str(e),
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")