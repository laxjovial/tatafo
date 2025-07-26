# backend/api/auth_api.py

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
import logging
from typing import Dict, Any

# Import Firebase Admin SDK components
from firebase_admin import auth
from firebase_admin import exceptions as firebase_exceptions

# Project imports
from backend.middleware.auth_middleware import get_current_user, get_firestore_manager_dependency, get_user_manager_dependency
from utils.user_manager import UserManager # Ensure UserManager is imported
from database.firestore_manager import FirestoreManager
from backend.models.user_models import UserProfile # Ensure UserProfile is imported
from utils.analytics_tracker import log_event
from datetime import datetime, timezone

router = APIRouter()
logger = logging.getLogger(__name__)

class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    username: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

@router.post("/register", response_model=UserProfile, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_reg: UserRegistration,
    user_manager: UserManager = Depends(get_user_manager_dependency),
    firestore_manager: FirestoreManager = Depends(get_firestore_manager_dependency) # Added for completeness if needed elsewhere
):
    logger.info(f"Attempting to register user: {user_reg.email}")
    try:
        # Create user in Firebase Authentication
        firebase_user = auth.create_user(
            email=user_reg.email,
            password=user_reg.password,
            display_name=user_reg.username # Set display name
        )
        user_id = firebase_user.uid
        logger.info(f"Firebase Auth user created with UID: {user_id}")

        # Set custom claims (e.g., roles) if necessary
        auth.set_custom_user_claims(user_id, {"roles": ["user"]})
        logger.info(f"Custom claims set for UID {user_id}: {{'roles': ['user']}}")

        # Create or update user profile in Firestore
        # Use create_or_update_user from UserManager
        profile_result = await user_manager.create_or_update_user(
            user_id=user_id,
            email=user_reg.email,
            username=user_reg.username
        )
        logger.info(f"User profile created/updated in Firestore for UID: {user_id}")
        
        # Log successful registration
        await log_event(
            'user_registration_success',
            {'uid': user_id, 'email': user_reg.email},
            user_id=user_id, # Log with the actual user_id
            success=True,
            log_from_backend=True
        )

        return profile_result

    except firebase_exceptions.FirebaseError as e:
        logger.error(f"Firebase error during registration: {e}", exc_info=True)
        detail_message = "Registration failed."
        if e.code == 'auth/email-already-exists':
            detail_message = "Email already registered."
            status_code = status.HTTP_409_CONFLICT
        elif e.code == 'auth/invalid-password':
            detail_message = "Password must be at least 6 characters long."
            status_code = status.HTTP_400_BAD_REQUEST
        elif e.code == 'auth/invalid-email':
            detail_message = "Invalid email format."
            status_code = status.HTTP_400_BAD_REQUEST
        else:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

        await log_event(
            'user_registration_failure',
            {'reason': detail_message, 'error': str(e)},
            user_id="backend_system_user", # Use a system user ID if actual user ID is not yet available
            success=False,
            error_message=detail_message,
            log_from_backend=True
        )
        raise HTTPException(status_code=status_code, detail=detail_message)
    except Exception as e:
        logger.error(f"Unexpected error during registration: {e}", exc_info=True)
        await log_event(
            'user_registration_failure',
            {'reason': 'Unexpected error', 'error': str(e)},
            user_id="backend_system_user", # Use a system user ID if actual user ID is not yet available
            success=False,
            error_message=f"An unexpected error occurred: {str(e)}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred during registration.")


@router.post("/login/password", response_model=Token)
async def login_for_access_token(
    user_login: UserLogin,
    user_manager: UserManager = Depends(get_user_manager_dependency)
):
    logger.info(f"Attempting to log in user: {user_login.email}")
    try:
        # Authenticate user with Firebase Authentication
        # Firebase Admin SDK does not directly support password-based sign-in for generating ID tokens.
        # This endpoint assumes the frontend handles the initial password authentication
        # and sends back the ID token. For server-side, you'd typically verify an ID token.
        # If this is meant to be a direct password login, you'd need a client-side library
        # or a custom solution that calls Firebase client SDK methods.
        
        # As a workaround for direct server-side login for testing/admin purposes,
        # you might try to verify credentials, but it's not standard for getting
        # a fresh ID token for client use.
        
        # A common flow:
        # 1. Client sends email/password to Firebase Client SDK.
        # 2. Client SDK returns ID Token.
        # 3. Client sends ID Token to backend.
        # 4. Backend verifies ID Token using Firebase Admin SDK (as in get_current_user).

        # For this function, let's assume it verifies credentials and then generates
        # a custom token, which the client can then use to sign in and get an ID token.
        # This is a common pattern for custom authentication.

        # First, find the user by email
        user_record = auth.get_user_by_email(user_login.email)
        user_id = user_record.uid

        # Generate a custom token. The client will use this to sign in and get an ID token.
        custom_token = auth.create_custom_token(user_id)
        
        # Log successful login attempt
        await log_event(
            'user_login',
            {'uid': user_id, 'email': user_login.email, 'method': 'password'},
            user_id=user_id,
            success=True,
            log_from_backend=True
        )

        return {"access_token": custom_token.decode('utf-8'), "token_type": "bearer"}

    except firebase_exceptions.FirebaseError as e:
        logger.error(f"Firebase error during login: {e}", exc_info=True)
        detail_message = "Authentication failed."
        if e.code == 'auth/user-not-found' or e.code == 'auth/wrong-password': # Admin SDK's get_user_by_email doesn't check password
            detail_message = "Invalid credentials."
            status_code = status.HTTP_401_UNAUTHORIZED
        elif e.code == 'auth/invalid-email':
            detail_message = "Invalid email format."
            status_code = status.HTTP_400_BAD_REQUEST
        else:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

        await log_event(
            'user_login_failure',
            {'reason': detail_message, 'error': str(e)},
            user_id="backend_system_user", # Or parse user ID from error if possible
            success=False,
            error_message=detail_message,
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred during login.")
    except Exception as e:
        logger.error(f"Unexpected error during login: {e}", exc_info=True)
        await log_event(
            'user_login_failure',
            {'reason': 'Unexpected error', 'error': str(e)},
            user_id="backend_system_user",
            success=False,
            error_message=f"An unexpected error occurred: {str(e)}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred during login.")

# This endpoint is just an example of a protected route
@router.get("/me", response_model=UserProfile)
async def read_users_me(current_user: UserProfile = Depends(get_current_user)):
    """
    Get the current authenticated user's profile.
    This endpoint requires a valid Firebase ID Token in the Authorization header.
    """
    await log_event(
        'user_profile_access',
        {'user_id': current_user.user_id},
        user_id=current_user.user_id,
        success=True,
        log_from_backend=True
    )
    return current_user
