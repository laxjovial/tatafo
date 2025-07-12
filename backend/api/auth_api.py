# backend/api/auth_api.py

from fastapi import APIRouter, HTTPException, status, Depends
from typing import Annotated, Dict, Any
import logging

# Import Pydantic models from our backend.models
from backend.models.user_models import UserCreate, UserLogin, PasswordResetRequest, PasswordResetConfirm, ChangePassword

# Import middleware for protected routes (e.g., change password)
# Now importing the dependency functions for UserManager and FirestoreManager
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
logger.setLevel(logging.DEBUG) # Set to DEBUG for detailed logging during development

router = APIRouter()

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    user_manager: UserManager = Depends(get_user_manager_dependency) # Inject UserManager
):
    """
    Registers a new user in Firebase Authentication and stores profile in Firestore.
    Assigns default tier and roles (from config_manager) and logs the event.
    """
    logger.debug(f"Attempting registration for email: {user_data.email}, username: {user_data.username}")
    try:
        # 1. Create user in Firebase Authentication
        user_record = auth.create_user(email=user_data.email, password=user_data.password, display_name=user_data.username)
        user_id = user_record.uid

        # 2. Set custom claims for tier and roles
        # Fetch default tier and roles from config_manager for consistency
        default_tier = config_manager.get("default_user_tier", "free")
        default_roles = config_manager.get("default_user_roles", ["user"])
        auth.set_custom_user_claims(user_id, {'tier': default_tier, 'roles': default_roles})
        logger.debug(f"Firebase user created with UID: {user_id}, assigned tier: {default_tier}, roles: {default_roles}")
        
        # 3. Store user profile in Firestore
        # Use injected user_manager to create the profile
        await user_manager.create_user_profile(
            user_id=user_id,
            email=user_data.email,
            username=user_data.username,
            initial_tier=default_tier,
            initial_roles=default_roles
        )
        
        logger.info(f"User '{user_data.username}' ({user_data.email}) registered and profile created with UID: {user_id}")
        await log_event(
            'user_registered',
            {'email': user_data.email, 'username': user_data.username, 'tier': default_tier, 'roles': default_roles},
            user_id=user_id,
            success=True,
            log_from_backend=True
        )
        return {"message": "User registered successfully", "uid": user_id, "success": True}
    except firebase_exceptions.FirebaseError as e:
        logger.error(f"Firebase registration error for {user_data.email}: {e.code} - {e.cause}", exc_info=True)
        error_message = e.code
        display_message = f"Registration failed: {error_message}"
        
        if hasattr(e, 'message') and e.message:
            display_message = e.message
        elif error_message == 'auth/email-already-exists':
            display_message = "Email already in use. Please use a different email or log in."
        elif error_message == 'auth/weak-password':
            display_message = "Password is too weak. Please choose a stronger password (at least 6 characters)."
        
        await log_event(
            'user_registered',
            {'email': user_data.email, 'username': user_data.username, 'error': str(e), 'firebase_code': error_message},
            user_id="unauthenticated", # User ID might not be available if creation failed
            success=False,
            error_message=display_message,
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=display_message)
    except Exception as e:
        logger.error(f"An unexpected error occurred during registration for email {user_data.email}: {e}", exc_info=True)
        await log_event(
            'user_registered',
            {'email': user_data.email, 'username': user_data.username, 'error': str(e)},
            user_id="unauthenticated",
            success=False,
            error_message=str(e),
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/login")
async def login_user(
    credentials: UserLogin,
    user_manager: UserManager = Depends(get_user_manager_dependency) # Inject UserManager
):
    """
    Authenticates a user by verifying a Firebase ID Token provided by the client.
    This is the secure way to handle login when using Firebase Auth client-side.
    The client sends the ID token obtained from Firebase JS SDK after successful sign-in.
    """
    logger.debug(f"Attempting login with Firebase ID Token.")
    id_token = credentials.id_token # Assuming UserLogin now includes id_token

    if not id_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Firebase ID token is required for login."
        )

    try:
        # Verify the ID token using Firebase Admin SDK
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        
        # Optionally, check if the user is active/not disabled
        user_data = await user_manager.get_user(uid)
        if not user_data or user_data.get('status') == 'disabled':
            await log_event(
                'user_login_attempt',
                {'uid': uid, 'reason': 'Account disabled or not found'},
                user_id=uid,
                success=False,
                error_message="Account is disabled or not found. Please contact support.",
                log_from_backend=True
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Your account is disabled. Please contact support."
            )

        # Update last login time and potentially other user data via UserManager
        await user_manager.update_last_login(uid)

        logger.info(f"User {uid} authenticated successfully via Firebase ID Token.")
        await log_event(
            'user_logged_in',
            {'uid': uid},
            user_id=uid,
            success=True,
            log_from_backend=True
        )
        return {"message": "Login successful", "uid": uid, "success": True}
    except firebase_exceptions.AuthError as e:
        logger.error(f"Firebase ID Token verification failed: {e}", exc_info=True)
        await log_event(
            'user_logged_in',
            {'error': str(e), 'id_token_provided': bool(id_token)},
            user_id="unauthenticated",
            success=False,
            error_message=f"Authentication failed: {e.code}. Please ensure your token is valid and not expired.",
            log_from_backend=True
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {e.code}. Please log in again."
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during login: {e}", exc_info=True)
        await log_event(
            'user_logged_in',
            {'error': str(e)},
            user_id="unauthenticated",
            success=False,
            error_message=f"An unexpected error occurred: {str(e)}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")


@router.post("/request-password-reset")
async def request_password_reset(request_data: PasswordResetRequest):
    """
    Requests a password reset link for the given email using Firebase Auth.
    Logs the event.
    """
    logger.debug(f"Requesting password reset for email: {request_data.email}")
    try:
        # Firebase Admin SDK generates the link
        reset_link = auth.generate_password_reset_link(request_data.email)
        # In a real application, you would send this link via email.
        logger.info(f"Generated password reset link for {request_data.email}.")
        await log_event(
            'password_reset_requested',
            {'email': request_data.email},
            user_id="unauthenticated", # User not authenticated yet
            success=True,
            log_from_backend=True
        )
        # For security, always return a generic success message even if email not found
        return {"message": "If the email is registered, a password reset link has been sent to your inbox."}
    except firebase_exceptions.UserNotFoundError:
        logger.warning(f"Attempted password reset for non-existent email: {request_data.email}. Returning generic success.")
        await log_event(
            'password_reset_requested',
            {'email': request_data.email, 'error': 'User not found (handled gracefully)'},
            user_id="unauthenticated",
            success=True, # Still considered success from user's perspective for security
            log_from_backend=True
        )
        return {"message": "If the email is registered, a password reset link has been sent to your inbox."}
    except Exception as e:
        logger.error(f"Error requesting password reset for {request_data.email}: {e}", exc_info=True)
        await log_event(
            'password_reset_requested',
            {'email': request_data.email, 'error': str(e)},
            user_id="unauthenticated",
            success=False,
            error_message=str(e),
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to request password reset: {e}")

@router.post("/reset-password")
async def reset_password(confirm_data: PasswordResetConfirm):
    """
    Resets user's password using a valid token (oobCode from Firebase).
    Logs the event.
    """
    logger.debug(f"Attempting password reset with token.")
    try:
        # Verify the password reset code and then confirm the reset
        auth.confirm_password_reset(confirm_data.token, confirm_data.new_password)
        logger.info(f"Password successfully reset using token (oobCode).")
        await log_event(
            'password_reset_confirmed',
            {'token_provided': bool(confirm_data.token)},
            user_id="unauthenticated", # User not authenticated yet
            success=True,
            log_from_backend=True
        )
        return {"message": "Password reset successfully."}
    except Exception as e:
        logger.error(f"Error confirming password reset with token: {e}", exc_info=True)
        await log_event(
            'password_reset_confirmed',
            {'token_provided': bool(confirm_data.token), 'error': str(e)},
            user_id="unauthenticated",
            success=False,
            error_message=f"Invalid or expired token: {e}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid or expired token: {e}")

@router.post("/change-password")
async def change_password(
    data: ChangePassword,
    current_user: Annotated[Dict[str, Any], Depends(get_current_user)] # Use get_current_user
):
    """
    Allows a logged-in user to change their password using Firebase Auth.
    This endpoint expects the client to have re-authenticated the user recently
    or to provide a fresh ID token, as Firebase Admin SDK does not directly
    support changing password with an old password for security reasons.
    Logs the event.
    """
    user_id = current_user["uid"] # Get user_id from the authenticated token
    logger.debug(f"User {user_id} attempting to change password.")

    try:
        # The client-side Firebase SDK would typically handle re-authentication
        # before calling this endpoint, ensuring the user is verified.
        # This backend endpoint then uses Firebase Admin SDK to update the password.
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

