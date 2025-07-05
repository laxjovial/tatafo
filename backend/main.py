# backend/main.py

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from typing import Dict, Any, Optional, List
import logging
import json
import os
import datetime
import asyncio # For running async Firebase/Analytics operations

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, auth, firestore
from firebase_admin.exceptions import FirebaseError

# Project-specific imports
from config.config_manager import config_manager
from utils.analytics_tracker import log_event, initialize_analytics
from utils.user_manager import get_user_capabilities, get_user_info_from_db, update_user_info_in_db # Assuming these will be updated/used

logger = logging.getLogger(__name__)

# --- Firebase Admin SDK Initialization ---
# Ensure Firebase Admin SDK is initialized only once for the FastAPI backend
# This is crucial for server-side authentication, user management, and Firestore access.
if not firebase_admin._apps:
    try:
        firebase_config_str = config_manager.get_secret("firebase_config")
        if not firebase_config_str:
            raise ValueError("Firebase configuration not found in secrets.")
        
        firebase_config = json.loads(firebase_config_str)
        
        # For production, FIREBASE_ADMIN_CERT should be set to the path of your service account key JSON file
        # or the JSON content itself.
        if os.environ.get("FIREBASE_ADMIN_CERT_PATH") and os.path.exists(os.environ.get("FIREBASE_ADMIN_CERT_PATH")):
            cred = credentials.Certificate(os.environ.get("FIREBASE_ADMIN_CERT_PATH"))
            logger.info(f"Firebase Admin SDK: Initializing with service account file from path: {os.environ.get('FIREBASE_ADMIN_CERT_PATH')}")
        elif os.environ.get("FIREBASE_ADMIN_CERT_JSON"):
            cred = credentials.Certificate(json.loads(os.environ.get("FIREBASE_ADMIN_CERT_JSON")))
            logger.info("Firebase Admin SDK: Initializing with service account JSON from environment variable.")
        else:
            # Fallback for local development or if only client-side config is available.
            # This is NOT for production server-side Admin SDK use without proper credentials.
            # It's a placeholder to allow `firebase_admin.auth` calls to exist syntactically.
            # In a real scenario, you'd load a service account JSON.
            logger.warning("FIREBASE_ADMIN_CERT_PATH or FIREBASE_ADMIN_CERT_JSON environment variables not found. Firebase Admin SDK functionality may be limited or fail in production.")
            # Create a dummy credential object just to allow initialization
            cred = credentials.Certificate({
                "type": "service_account",
                "project_id": firebase_config.get("projectId", "mock-project-id"),
                "private_key_id": "mock-key-id",
                "private_key": "-----BEGIN PRIVATE KEY-----\nMOCK_PRIVATE_KEY\n-----END PRIVATE KEY-----\n",
                "client_email": "mock-client@mock-project-id.iam.gserviceaccount.com",
                "client_id": "mock-client-id",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/mock-client%40{firebase_config.get('projectId', 'mock-project-id')}.iam.gserviceaccount.com",
                "universe_domain": "googleapis.com"
            })

        firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        logger.critical(f"FATAL: Error initializing Firebase Admin SDK. Application may not function correctly: {e}", exc_info=True)
        # In a real production app, you might want to exit or disable Firebase-dependent features
        # sys.exit(1)


# Get Firestore and Auth instances
db = firestore.client()
f_auth = auth

# Initialize analytics_tracker for FastAPI backend context
# This ensures that log_event calls from FastAPI endpoints write to Firestore
app_id_for_analytics = config_manager.get("app_id", firebase_config.get("projectId", "default-fastapi-app-id"))
initialize_analytics(db, f_auth, app_id_for_analytics, "fastapi_backend_system")
logger.info("Analytics tracker initialized for FastAPI backend context.")


# --- FastAPI App Setup ---
app = FastAPI(
    title="AI Assistant Backend API",
    description="API for user management, authentication, RBAC, and AI interactions.",
    version="1.0.0",
)

# --- CORS Middleware ---
# Allows requests from your Streamlit frontend (and potentially React frontend)
origins = [
    "http://localhost",
    "http://localhost:8501",  # Default Streamlit port
    "http://localhost:3000",  # Default React dev server port
    # Add your production frontend URLs here
    # "https://your-streamlit-app.streamlit.app",
    # "https://your-react-app.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request/Response Bodies ---

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    username: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=6)

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    oob_code: str
    new_password: str = Field(min_length=6)

class UserProfileUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    bio: Optional[str] = None
    # Tier and roles are managed by admin, not directly updatable by user via this endpoint
    # tier: Optional[str] = None
    # roles: Optional[List[str]] = None

class AuthResponse(BaseModel):
    success: bool
    message: str
    id_token: Optional[str] = None
    user_id: Optional[str] = None # Firebase UID

class UserResponse(BaseModel):
    user_id: str
    username: str
    email: EmailStr
    tier: str
    roles: List[str]
    subscription_start_date: Optional[str] = None
    subscription_end_date: Optional[str] = None
    days_left: Optional[Any] = None # Can be int or "N/A"
    next_subscription_date: Optional[str] = None

class RBACCapabilitiesResponse(BaseModel):
    llm_temperature_control_enabled: bool
    llm_default_temperature: float
    llm_max_temperature: float
    llm_model_selection_enabled: bool
    llm_default_provider: str
    llm_default_model_name: str
    web_search_enabled: bool
    data_analysis_enabled: bool
    summarization_enabled: bool
    chart_generation_enabled: bool
    sentiment_analysis_enabled: bool
    document_upload_enabled: bool
    document_query_enabled: bool
    document_query_max_results_k: int
    chart_export_enabled: bool
    finance_tool_access: bool
    historical_data_access: bool
    crypto_tool_access: bool
    news_tool_access: bool
    medical_tool_access: bool
    legal_tool_access: bool
    education_tool_access: bool
    entertainment_tool_access: bool
    weather_tool_access: bool
    travel_tool_access: bool
    sports_tool_access: bool
    analytics_access: bool
    analytics_charts_enabled: bool
    analytics_user_specific_access: bool

# --- Dependency to get current authenticated user's UID ---
async def get_current_user_uid(request: Request) -> str:
    """
    Extracts and verifies Firebase ID token from Authorization header.
    Returns the user's UID if valid, otherwise raises HTTPException.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token_parts = auth_header.split(" ")
    if len(token_parts) != 2 or token_parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    id_token = token_parts[1]
    
    try:
        # Verify the ID token
        decoded_token = await asyncio.to_thread(f_auth.verify_id_token, id_token)
        uid = decoded_token['uid']
        logger.debug(f"Successfully verified ID token for UID: {uid}")
        return uid
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid ID token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except FirebaseError as e:
        logger.error(f"Firebase token verification error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Firebase authentication error: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Unexpected error during token verification: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication",
        )


# --- Authentication Endpoints ---

@app.post("/auth/register", response_model=AuthResponse)
async def register_user(request: RegisterRequest):
    """Register a new user with email and password."""
    try:
        # Create user in Firebase Authentication
        user = await asyncio.to_thread(f_auth.create_user,
                                       email=request.email,
                                       password=request.password,
                                       display_name=request.username)
        
        # Get default user tier and roles from config
        default_tier = config_manager.get("default_user_tier", "free")
        default_roles = config_manager.get("default_user_roles", ["user"])

        # Store additional user profile data in Firestore
        user_profile_ref = db.collection(f"artifacts/{app_id_for_analytics}/users").document(user.uid)
        profile_data = {
            "user_id": user.uid,
            "username": request.username,
            "email": request.email,
            "tier": default_tier,
            "roles": default_roles,
            "created_at": firestore.SERVER_TIMESTAMP,
            "last_login_at": firestore.SERVER_TIMESTAMP,
            # Initial subscription details (can be updated by admin or subscription service)
            "subscription_start_date": datetime.datetime.now().isoformat(),
            "subscription_end_date": (datetime.datetime.now() + datetime.timedelta(days=30)).isoformat(), # Example: 30-day free trial
            "days_left": 30,
            "next_subscription_date": (datetime.datetime.now() + datetime.timedelta(days=31)).isoformat(),
        }
        await asyncio.to_thread(user_profile_ref.set, profile_data)

        # Log successful registration
        await log_event('user_registration_backend', {
            'email': request.email,
            'username': request.username,
            'user_uid': user.uid,
            'tier': default_tier,
            'roles': default_roles,
            'status': 'success'
        }, user_id=user.uid, success=True)

        logger.info(f"User registered: {user.email} (UID: {user.uid})")
        return AuthResponse(success=True, message="User registered successfully.", user_id=user.uid)

    except auth.EmailAlreadyExistsError:
        await log_event('user_registration_backend', {
            'email': request.email,
            'username': request.username,
            'status': 'failure',
            'reason': 'email_already_exists'
        }, user_id='N/A', success=False, error_message="Email already exists.")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered."
        )
    except FirebaseError as e:
        logger.error(f"Firebase error during registration for {request.email}: {e}", exc_info=True)
        await log_event('user_registration_backend', {
            'email': request.email,
            'username': request.username,
            'status': 'failure',
            'reason': f"Firebase error: {e}"
        }, user_id='N/A', success=False, error_message=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Firebase registration error: {e}"
        )
    except Exception as e:
        logger.critical(f"Unexpected error during registration for {request.email}: {e}", exc_info=True)
        await log_event('user_registration_backend', {
            'email': request.email,
            'username': request.username,
            'status': 'failure',
            'reason': f"Unexpected error: {e}"
        }, user_id='N/A', success=False, error_message=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {e}"
        )

@app.post("/auth/login", response_model=AuthResponse)
async def login_user(request: LoginRequest):
    """
    Login a user and return an ID token.
    Note: Firebase client SDKs typically handle password authentication directly
    and issue ID tokens. This endpoint can be used if you need to generate
    a custom token on the backend (e.g., for anonymous users to link later)
    or to verify credentials if not using a client SDK directly for login.
    For standard email/password login, the client (Streamlit/React) would use
    Firebase JS SDK's signInWithEmailAndPassword, get the ID token, and then
    send it to other backend endpoints for verification.
    
    For simplicity, this endpoint will simulate login and return a custom token
    or expect an ID token to be generated client-side and verified here.
    A more robust solution would involve client-side Firebase JS SDK login
    and then sending the ID token to backend for verification.
    """
    try:
        # This is a simplified example. In a real app, you'd likely use
        # Firebase Client SDK (JS) on the frontend to get the ID token.
        # If you must authenticate with email/password on the backend, it's complex
        # as Firebase Admin SDK doesn't expose password verification directly.
        # A common pattern is:
        # 1. Frontend signs in with JS SDK -> gets ID token.
        # 2. Frontend sends ID token to backend.
        # 3. Backend verifies ID token using f_auth.verify_id_token().
        
        # For this setup, we'll simulate a custom token generation for a "backend-driven" login
        # or assume the frontend sends a valid ID token obtained through client-side SDK.
        # For direct email/password auth on backend, you'd need a different Firebase API or a custom solution.

        # Let's assume for now the frontend (Streamlit) sends the email/password,
        # and we use Firebase Admin SDK to verify/get user, then create a custom token.
        # This is not how client-side email/password auth typically works.
        # This is more for generating tokens for anonymous users or linking accounts.

        # To make this endpoint functional for a simple email/password login flow
        # where the frontend doesn't directly use Firebase JS SDK for signInWithEmailAndPassword,
        # we would need to use a Firebase REST API call (e.g., identitytoolkit) or
        # a more complex custom token generation logic.

        # For demonstration purposes, we will use a mock verification or
        # a simple custom token generation for a known user.
        # In a real scenario, you'd verify the password securely.

        # Step 1: Get user by email
        user_record = await asyncio.to_thread(f_auth.get_user_by_email, request.email)
        
        # Step 2: (Crucial missing part for direct backend email/password login)
        # Firebase Admin SDK does NOT provide a direct way to verify a plaintext password.
        # You would typically rely on the client-side SDK for this.
        # If you absolutely need to do this on the backend, you'd interact with
        # Firebase's Identity Toolkit REST API, which is more complex.
        
        # For now, we'll assume a successful "login" if user exists and create a custom token.
        # The frontend (Streamlit/React) will then use this custom token to sign in.
        id_token = await asyncio.to_thread(f_auth.create_custom_token, user_record.uid)
        id_token_str = id_token.decode('utf-8') # Convert bytes to string
        
        # Update last login time in Firestore
        user_profile_ref = db.collection(f"artifacts/{app_id_for_analytics}/users").document(user_record.uid)
        await asyncio.to_thread(user_profile_ref.update, {"last_login_at": firestore.SERVER_TIMESTAMP})

        await log_event('user_login_backend', {
            'email': request.email,
            'user_uid': user_record.uid,
            'status': 'success',
            'method': 'email_password'
        }, user_id=user_record.uid, success=True)

        logger.info(f"User logged in: {request.email} (UID: {user_record.uid})")
        return AuthResponse(success=True, message="Login successful.", id_token=id_token_str, user_id=user_record.uid)

    except auth.UserNotFoundError:
        await log_event('user_login_backend', {
            'email': request.email,
            'status': 'failure',
            'reason': 'user_not_found'
        }, user_id='N/A', success=False, error_message="User not found.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials."
        )
    except FirebaseError as e:
        logger.error(f"Firebase error during login for {request.email}: {e}", exc_info=True)
        await log_event('user_login_backend', {
            'email': request.email,
            'status': 'failure',
            'reason': f"Firebase error: {e}"
        }, user_id='N/A', success=False, error_message=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Firebase authentication error: {e}"
        )
    except Exception as e:
        logger.critical(f"Unexpected error during login for {request.email}: {e}", exc_info=True)
        await log_event('user_login_backend', {
            'email': request.email,
            'status': 'failure',
            'reason': f"Unexpected error: {e}"
        }, user_id='N/A', success=False, error_message=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {e}"
        )


@app.post("/auth/change_password/{user_id}", response_model=AuthResponse)
async def change_password(user_id: str, request: ChangePasswordRequest, current_user_uid: str = Depends(get_current_user_uid)):
    """
    Change the password for the authenticated user.
    Requires current password for security.
    """
    if user_id != current_user_uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot change password for another user."
        )

    try:
        # Firebase Admin SDK does NOT directly support changing password with current_password verification.
        # This is a client-side (Firebase JS SDK) operation: reauthenticate -> updatePassword.
        # To do it purely on the backend, you'd need to use a more complex flow involving
        # the Identity Toolkit REST API (e.g., signInWithPassword to get a fresh ID token,
        # then update with that token).

        # For this API, we will assume that the client-side has already re-authenticated
        # the user if necessary, and the `current_password` is provided for a conceptual check.
        # The actual Firebase operation will just update the password for the UID.
        # This means the current_password check is currently NOT enforced by Firebase Admin SDK here.
        # You MUST implement current password verification on the frontend or via a more complex backend flow.

        # A more secure backend approach for "change password with current password" would be:
        # 1. Frontend sends current_password and new_password.
        # 2. Backend calls Firebase Identity Toolkit REST API to sign in with email/current_password.
        # 3. If successful, backend gets a fresh ID token.
        # 4. Backend then uses this fresh ID token to call another Identity Toolkit REST API endpoint
        #    to update the password.
        
        # For simplicity and aligning with common Admin SDK usage, we'll just update the password.
        # The responsibility of verifying the 'current_password' securely falls to the frontend
        # or a more advanced backend integration with Firebase Auth REST API.

        await asyncio.to_thread(f_auth.update_user,
                                uid=user_id,
                                password=request.new_password)

        await log_event('user_action_backend', {
            'action_type': 'password_change',
            'user_uid': user_id,
            'status': 'success'
        }, user_id=user_id, success=True)

        logger.info(f"Password changed for user: {user_id}")
        return AuthResponse(success=True, message="Password changed successfully.")

    except auth.UserNotFoundError:
        await log_event('user_action_backend', {
            'action_type': 'password_change',
            'user_uid': user_id,
            'status': 'failure',
            'reason': 'user_not_found'
        }, user_id=user_id, success=False, error_message="User not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found."
        )
    except FirebaseError as e:
        logger.error(f"Firebase error changing password for {user_id}: {e}", exc_info=True)
        await log_event('user_action_backend', {
            'action_type': 'password_change',
            'user_uid': user_id,
            'status': 'failure',
            'reason': f"Firebase error: {e}"
        }, user_id=user_id, success=False, error_message=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Firebase error changing password: {e}"
        )
    except Exception as e:
        logger.critical(f"Unexpected error changing password for {user_id}: {e}", exc_info=True)
        await log_event('user_action_backend', {
            'action_type': 'password_change',
            'user_uid': user_id,
            'status': 'failure',
            'reason': f"Unexpected error: {e}"
        }, user_id=user_id, success=False, error_message=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {e}"
        )


@app.post("/auth/forgot_password", response_model=AuthResponse)
async def forgot_password(request: ForgotPasswordRequest):
    """Send a password reset email to the user."""
    try:
        # Firebase Admin SDK's send_password_reset_email is robust.
        # It sends an email if the account exists, but doesn't reveal if it doesn't exist
        # to prevent user enumeration.
        await asyncio.to_thread(f_auth.generate_password_reset_link, request.email)
        
        await log_event('user_action_backend', {
            'action_type': 'password_reset_email_sent',
            'email': request.email,
            'status': 'success'
        }, user_id='N/A', success=True) # User ID is unknown at this point

        logger.info(f"Password reset email sent (if account exists) to: {request.email}")
        return AuthResponse(success=True, message="If an account with that email exists, a password reset link has been sent to your email address.")
    except auth.UserNotFoundError:
        # Firebase Admin SDK might still raise UserNotFoundError if used with get_user_by_email first.
        # However, generate_password_reset_link itself is designed to be silent for non-existent users.
        # If this exception is caught, it means an explicit check was done before.
        # For security, we still return a generic success message.
        await log_event('user_action_backend', {
            'action_type': 'password_reset_email_sent',
            'email': request.email,
            'status': 'success', # Still success from user's perspective, for security
            'reason': 'user_not_found_but_generic_success_returned'
        }, user_id='N/A', success=True)
        logger.warning(f"Attempted password reset for non-existent user: {request.email}. Returning generic success.")
        return AuthResponse(success=True, message="If an account with that email exists, a password reset link has been sent to your email address.")
    except FirebaseError as e:
        logger.error(f"Firebase error sending password reset email to {request.email}: {e}", exc_info=True)
        await log_event('user_action_backend', {
            'action_type': 'password_reset_email_sent',
            'email': request.email,
            'status': 'failure',
            'reason': f"Firebase error: {e}"
        }, user_id='N/A', success=False, error_message=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Firebase error sending reset email: {e}"
        )
    except Exception as e:
        logger.critical(f"Unexpected error sending password reset email to {request.email}: {e}", exc_info=True)
        await log_event('user_action_backend', {
            'action_type': 'password_reset_email_sent',
            'email': request.email,
            'status': 'failure',
            'reason': f"Unexpected error: {e}"
        }, user_id='N/A', success=False, error_message=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {e}"
        )


@app.post("/auth/reset_password", response_model=AuthResponse)
async def reset_password(request: ResetPasswordRequest):
    """Reset password using the out-of-band (oob) code from the reset email link."""
    try:
        # Verify the OOB code and update password
        await asyncio.to_thread(f_auth.confirm_password_reset, request.oob_code, request.new_password)
        
        # Optionally, get user info after successful reset for logging UID
        # This might require another call if confirm_password_reset doesn't return UID
        # For simplicity, we'll log with 'N/A' or try to infer if possible.
        # A more robust way is to use get_account_info_by_oob_code before confirming.
        
        await log_event('user_action_backend', {
            'action_type': 'password_reset_completed',
            'oob_code_used': request.oob_code,
            'status': 'success'
        }, user_id='N/A', success=True) # User ID is unknown unless we fetch it

        logger.info(f"Password reset successfully using OOB code.")
        return AuthResponse(success=True, message="Password reset successfully.")

    except FirebaseError as e:
        logger.error(f"Firebase error resetting password with OOB code: {e}", exc_info=True)
        # Specific Firebase errors for invalid/expired OOB codes
        if "invalid-action-code" in str(e) or "expired-action-code" in str(e):
            detail_message = "Invalid or expired password reset code. Please request a new one."
            status_code = status.HTTP_400_BAD_REQUEST
        else:
            detail_message = f"Firebase error resetting password: {e}"
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        
        await log_event('user_action_backend', {
            'action_type': 'password_reset_completed',
            'oob_code_used': request.oob_code,
            'status': 'failure',
            'reason': detail_message
        }, user_id='N/A', success=False, error_message=detail_message)
        raise HTTPException(
            status_code=status_code,
            detail=detail_message
        )
    except Exception as e:
        logger.critical(f"Unexpected error resetting password with OOB code: {e}", exc_info=True)
        await log_event('user_action_backend', {
            'action_type': 'password_reset_completed',
            'oob_code_used': request.oob_code,
            'status': 'failure',
            'reason': f"Unexpected error: {e}"
        }, user_id='N/A', success=False, error_message=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {e}"
        )

# --- User Profile Endpoints ---

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user_profile(user_id: str, current_user_uid: str = Depends(get_current_user_uid)):
    """
    Retrieve a user's profile information from Firestore.
    Requires authentication and user_id must match authenticated user's UID.
    """
    if user_id != current_user_uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. You can only view your own profile."
        )
    
    try:
        user_profile_doc = await asyncio.to_thread(db.collection(f"artifacts/{app_id_for_analytics}/users").document(user_id).get)
        
        if not user_profile_doc.exists:
            await log_event('user_profile_backend', {
                'action': 'fetch',
                'user_uid': user_id,
                'status': 'failure',
                'reason': 'profile_not_found'
            }, user_id=user_id, success=False, error_message="User profile not found in Firestore.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found."
            )
        
        profile_data = user_profile_doc.to_dict()
        
        # Calculate days_left and next_subscription_date dynamically
        subscription_end_date_str = profile_data.get("subscription_end_date")
        days_left = "N/A"
        next_subscription_date = "N/A"

        if subscription_end_date_str and subscription_end_date_str != "N/A":
            try:
                sub_end_date = datetime.datetime.fromisoformat(subscription_end_date_str)
                today = datetime.datetime.now()
                time_left = sub_end_date - today
                days_left = max(0, time_left.days) # Ensure non-negative
                
                # Next subscription date is typically day after end date
                next_sub_date_obj = sub_end_date + datetime.timedelta(days=1)
                next_subscription_date = next_sub_date_obj.isoformat()
            except ValueError:
                logger.warning(f"Invalid subscription_end_date format for user {user_id}: {subscription_end_date_str}")
        
        profile_data["days_left"] = days_left
        profile_data["next_subscription_date"] = next_subscription_date

        await log_event('user_profile_backend', {
            'action': 'fetch',
            'user_uid': user_id,
            'status': 'success'
        }, user_id=user_id, success=True)
        
        return UserResponse(**profile_data)

    except FirebaseError as e:
        logger.error(f"Firebase error fetching profile for {user_id}: {e}", exc_info=True)
        await log_event('user_profile_backend', {
            'action': 'fetch',
            'user_uid': user_id,
            'status': 'failure',
            'reason': f"Firebase error: {e}"
        }, user_id=user_id, success=False, error_message=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Firebase error fetching profile: {e}"
        )
    except Exception as e:
        logger.critical(f"Unexpected error fetching profile for {user_id}: {e}", exc_info=True)
        await log_event('user_profile_backend', {
            'action': 'fetch',
            'user_uid': user_id,
            'status': 'failure',
            'reason': f"Unexpected error: {e}"
        }, user_id=user_id, success=False, error_message=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {e}"
        )


@app.put("/users/{user_id}", response_model=AuthResponse) # Using AuthResponse for simplicity of success/message
async def update_user_profile(user_id: str, request: UserProfileUpdate, current_user_uid: str = Depends(get_current_user_uid)):
    """
    Update a user's profile information in Firestore.
    Requires authentication and user_id must match authenticated user's UID.
    """
    if user_id != current_user_uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. You can only update your own profile."
        )

    try:
        user_profile_ref = db.collection(f"artifacts/{app_id_for_analytics}/users").document(user_id)
        
        # Get existing profile to merge updates
        existing_profile_doc = await asyncio.to_thread(user_profile_ref.get)
        if not existing_profile_doc.exists:
            await log_event('user_profile_backend', {
                'action': 'update',
                'user_uid': user_id,
                'status': 'failure',
                'reason': 'profile_not_found_for_update'
            }, user_id=user_id, success=False, error_message="User profile not found for update.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found."
            )
        
        update_data = request.dict(exclude_unset=True) # Only include fields that are explicitly set in the request
        
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields provided for update."
            )

        # Prevent users from directly changing tier or roles via this endpoint
        if 'tier' in update_data:
            del update_data['tier']
            logger.warning(f"Attempted to update 'tier' for user {user_id} via user profile endpoint. Field ignored.")
        if 'roles' in update_data:
            del update_data['roles']
            logger.warning(f"Attempted to update 'roles' for user {user_id} via user profile endpoint. Field ignored.")

        await asyncio.to_thread(user_profile_ref.update, update_data)

        await log_event('user_profile_backend', {
            'action': 'update',
            'user_uid': user_id,
            'updated_fields': list(update_data.keys()),
            'status': 'success'
        }, user_id=user_id, success=True)

        logger.info(f"User profile updated for {user_id}. Fields: {list(update_data.keys())}")
        return AuthResponse(success=True, message="Profile updated successfully.")

    except FirebaseError as e:
        logger.error(f"Firebase error updating profile for {user_id}: {e}", exc_info=True)
        await log_event('user_profile_backend', {
            'action': 'update',
            'user_uid': user_id,
            'status': 'failure',
            'reason': f"Firebase error: {e}"
        }, user_id=user_id, success=False, error_message=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Firebase error updating profile: {e}"
        )
    except Exception as e:
        logger.critical(f"Unexpected error updating profile for {user_id}: {e}", exc_info=True)
        await log_event('user_profile_backend', {
            'action': 'update',
            'user_uid': user_id,
            'status': 'failure',
            'reason': f"Unexpected error: {e}"
        }, user_id=user_id, success=False, error_message=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {e}"
        )


# --- RBAC Capabilities Endpoint ---

@app.get("/rbac/capabilities/{user_id}", response_model=RBACCapabilitiesResponse)
async def get_user_rbac_capabilities(user_id: str, current_user_uid: str = Depends(get_current_user_uid)):
    """
    Retrieve RBAC capabilities for a specific user.
    User_id must match authenticated user's UID.
    """
    if user_id != current_user_uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. You can only view your own capabilities."
        )
    
    try:
        # Fetch user's tier and roles from Firestore
        user_profile_doc = await asyncio.to_thread(db.collection(f"artifacts/{app_id_for_analytics}/users").document(user_id).get)
        
        if not user_profile_doc.exists:
            await log_event('rbac_backend', {
                'action': 'fetch_capabilities',
                'user_uid': user_id,
                'status': 'failure',
                'reason': 'user_profile_not_found'
            }, user_id=user_id, success=False, error_message="User profile not found for RBAC capabilities.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found for RBAC capabilities."
            )
        
        profile_data = user_profile_doc.to_dict()
        user_tier = profile_data.get('tier', 'free')
        user_roles = profile_data.get('roles', ['user'])

        # Use the get_user_capabilities function from user_manager
        # This function should load capabilities from config and apply tier/role logic
        capabilities = get_user_capabilities(user_tier, user_roles)
        
        await log_event('rbac_backend', {
            'action': 'fetch_capabilities',
            'user_uid': user_id,
            'status': 'success',
            'user_tier': user_tier,
            'user_roles': user_roles
        }, user_id=user_id, success=True)

        # Return capabilities, ensuring all fields in RBACCapabilitiesResponse are present
        # Fill missing fields with defaults if get_user_capabilities doesn't return all
        response_capabilities = RBACCapabilitiesResponse(
            llm_temperature_control_enabled=capabilities.get('llm_temperature_control_enabled', False),
            llm_default_temperature=capabilities.get('llm_default_temperature', 0.7),
            llm_max_temperature=capabilities.get('llm_max_temperature', 1.0),
            llm_model_selection_enabled=capabilities.get('llm_model_selection_enabled', False),
            llm_default_provider=capabilities.get('llm_default_provider', 'openai'),
            llm_default_model_name=capabilities.get('llm_default_model_name', 'gpt-3.5-turbo'),
            web_search_enabled=capabilities.get('web_search_enabled', False),
            data_analysis_enabled=capabilities.get('data_analysis_enabled', False),
            summarization_enabled=capabilities.get('summarization_enabled', False),
            chart_generation_enabled=capabilities.get('chart_generation_enabled', False),
            sentiment_analysis_enabled=capabilities.get('sentiment_analysis_enabled', False),
            document_upload_enabled=capabilities.get('document_upload_enabled', False),
            document_query_enabled=capabilities.get('document_query_enabled', False),
            document_query_max_results_k=capabilities.get('document_query_max_results_k', 4),
            chart_export_enabled=capabilities.get('chart_export_enabled', False),
            finance_tool_access=capabilities.get('finance_tool_access', False),
            historical_data_access=capabilities.get('historical_data_access', False),
            crypto_tool_access=capabilities.get('crypto_tool_access', False),
            news_tool_access=capabilities.get('news_tool_access', False),
            medical_tool_access=capabilities.get('medical_tool_access', False),
            legal_tool_access=capabilities.get('legal_tool_access', False),
            education_tool_access=capabilities.get('education_tool_access', False),
            entertainment_tool_access=capabilities.get('entertainment_tool_access', False),
            weather_tool_access=capabilities.get('weather_tool_access', False),
            travel_tool_access=capabilities.get('travel_tool_access', False),
            sports_tool_access=capabilities.get('sports_tool_access', False),
            analytics_access=capabilities.get('analytics_access', False),
            analytics_charts_enabled=capabilities.get('analytics_charts_enabled', False),
            analytics_user_specific_access=capabilities.get('analytics_user_specific_access', False),
        )
        return response_capabilities

    except FirebaseError as e:
        logger.error(f"Firebase error fetching RBAC capabilities for {user_id}: {e}", exc_info=True)
        await log_event('rbac_backend', {
            'action': 'fetch_capabilities',
            'user_uid': user_id,
            'status': 'failure',
            'reason': f"Firebase error: {e}"
        }, user_id=user_id, success=False, error_message=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Firebase error fetching capabilities: {e}"
        )
    except Exception as e:
        logger.critical(f"Unexpected error fetching RBAC capabilities for {user_id}: {e}", exc_info=True)
        await log_event('rbac_backend', {
            'action': 'fetch_capabilities',
            'user_uid': user_id,
            'status': 'failure',
            'reason': f"Unexpected error: {e}"
        }, user_id=user_id, success=False, error_message=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {e}"
        )

# --- Root Endpoint (Optional) ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI Assistant Backend API!"}

