# backend/main.py

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import logging
import json
import os
import asyncio
import firebase_admin
from firebase_admin import credentials, auth, firestore
from firebase_admin import exceptions as firebase_exceptions
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime, timezone

# Project imports
from config.config_manager import config_manager
from utils.analytics_tracker import initialize_analytics, log_event
from database.firestore_manager import FirestoreManager
import shared_tools.cloud_storage_utils as cloud_storage_utils_module
import shared_tools.vector_utils as vector_utils_module
from utils.date_parser import parse_date_to_yyyymmdd
from utils.user_manager import UserManager, get_user_tier_capability # Import get_user_tier_capability

# Import domain tools
from domain_tools.finance_tools import FinanceTools
from domain_tools.crypto_tools import CryptoTools
from domain_tools.medical_tools import MedicalTools
from domain_tools.news_tools import NewsTools
from domain_tools.legal_tools import LegalTools
from domain_tools.education_tools import EducationTools
from domain_tools.entertainment_tools import EntertainmentTools
from domain_tools.weather_tools import WeatherTools
from domain_tools.travel_tools import TravelTools
from domain_tools.sports_tools import SportsTools
from domain_tools.document_tools.document_tool import DocumentTools # Ensure this import is present

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# --- Firebase Admin SDK Initialization ---
if not firebase_admin._apps:
    try:
        firebase_credentials_json = os.environ.get("FIREBASE_CREDENTIALS")
        if not firebase_credentials_json:
            logger.error("FIREBASE_CREDENTIALS environment variable not set. Using dummy credentials for local testing.")
            firebase_config_str = config_manager.get_secret("firebase_config")
            firebase_config = json.loads(firebase_config_str) if firebase_config_str else {}
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
        else:
            cred = credentials.Certificate(json.loads(firebase_credentials_json))
        
        firebase_app = firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        logger.error(f"FATAL: Error initializing Firebase Admin SDK: {e}", exc_info=True)
        raise

# Initialize Firestore Manager
db_client = firestore.client(firebase_app)
firestore_manager = FirestoreManager()
logger.info("FirestoreManager initialized.")

# Initialize Cloud Storage Utils Wrapper
cloud_storage_utils = cloud_storage_utils_module.CloudStorageUtilsWrapper(config_manager)
logger.info("CloudStorageUtilsWrapper initialized.")

# Initialize UserManager with FirestoreManager and CloudStorageUtilsWrapper
user_manager = UserManager(firestore_manager, cloud_storage_utils)
logger.info("UserManager initialized.")

# Initialize Analytics Tracker (after UserManager as it might use user_manager for user_id)
app_id_for_analytics = config_manager.get("app_id", "default-backend-app-id")
# Use a service user ID for backend-initiated logs, actual user_id for user-specific logs
backend_service_user_id = "backend-service-user" 
initialize_analytics(db_client, auth, app_id_for_analytics, backend_service_user_id)
logger.info("Analytics Tracker initialized.")


# Initialize Vector Utils Wrapper
vector_utils = vector_utils_module.VectorUtilsWrapper(
    firestore_manager=firestore_manager,
    cloud_storage_utils=cloud_storage_utils,
    config_manager=config_manager
)
logger.info("VectorUtilsWrapper initialized.")


app = FastAPI(
    title="Intelli-Agent Backend",
    description="Backend for the Intelli-Agent, providing various domain-specific tools and user management.",
    version="0.1.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2PasswordBearer for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Authenticates the user using Firebase ID token and retrieves their profile.
    """
    try:
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token['uid']
        user_record = await user_manager.get_user(uid) # Use user_manager to get profile and update last_login_at
        if not user_record:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User profile not found.")
        
        user_record['uid'] = uid # Ensure uid is part of the returned dict
        return user_record
    except Exception as e:
        logger.error(f"Authentication failed: {e}", exc_info=True)
        # Log authentication failure
        await log_event(
            'authentication_failure',
            {'error_details': str(e)},
            user_id="unauthenticated", # No user_id available yet
            success=False,
            error_message=str(e),
            log_from_backend=True
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Dependency to check if the current user is an admin.
    """
    if "admin" not in current_user.get("roles", []):
        # Log authorization failure
        await log_event(
            'authorization_failure',
            {'required_role': 'admin', 'user_roles': current_user.get("roles", [])},
            user_id=current_user.get('uid', 'N/A'),
            success=False,
            error_message="Admin access required",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized. Admin access required.")
    return current_user

# Initialize DocumentTools first, as it's a dependency for other domain tools
document_tools_instance = DocumentTools(vector_utils, firestore_manager, cloud_storage_utils, config_manager, log_event)

# Initialize domain tool instances, passing necessary dependencies
domain_tool_instances = {
    "finance_tools": FinanceTools(
        config_manager,
        firestore_manager,
        log_event,
        document_tools_instance
    ),
    "crypto_tools": CryptoTools(
        config_manager,
        log_event,
        document_tools_instance
    ),
    "medical_tools": MedicalTools(
        config_manager,
        log_event,
        document_tools_instance
    ),
    "news_tools": NewsTools(
        config_manager,
        log_event,
        document_tools_instance
    ),
    "legal_tools": LegalTools(
        config_manager,
        log_event,
        document_tools_instance
    ),
    "education_tools": EducationTools(
        config_manager,
        log_event,
        document_tools_instance
    ),
    "entertainment_tools": EntertainmentTools(
        config_manager,
        log_event,
        document_tools_instance
    ),
    "weather_tools": WeatherTools(
        config_manager,
        log_event,
        document_tools_instance
    ),
    "travel_tools": TravelTools(
        config_manager,
        log_event,
        document_tools_instance
    ),
    "sports_tools": SportsTools(
        config_manager,
        log_event,
        document_tools_instance
    ),
    "document_tools": document_tools_instance,
}

# --- Pydantic Models for Request Bodies ---
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    username: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class FrontendAnalyticsEvent(BaseModel):
    event_type: str
    details: Dict[str, Any]
    user_id: str # Frontend will send the user_id
    success: bool
    error_message: Optional[str] = None

# --- API Endpoints ---

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Intelli-Agent Backend!"}

@app.post("/register")
async def register_user_endpoint(request_data: RegisterRequest, request: Request):
    """Registers a new user with Firebase Authentication and creates a user profile in Firestore."""
    logger.debug(f"Attempting registration for email: {request_data.email}, username: {request_data.username}")
    try:
        user = auth.create_user(email=request_data.email, password=request_data.password)
        logger.debug(f"Firebase user created: {user.uid}")
        # Create user profile in Firestore
        await user_manager.create_user_profile(user.uid, request_data.email, request_data.username)
        
        logger.info(f"User registered and profile created: {user.uid} with email {request_data.email}")
        await log_event(
            'user_registered',
            {'email': request_data.email, 'username': request_data.username},
            user_id=user.uid,
            success=True,
            log_from_backend=True
        )
        return {"message": "User registered successfully", "uid": user.uid, "success": True} # Added success: True
    except firebase_exceptions.FirebaseError as e:
        logger.error(f"Firebase registration error for {request_data.email}: {e.code} - {e.cause}", exc_info=True)
        error_message = e.code
        display_message = f"Firebase error: {error_message}"
        
        if hasattr(e, 'message') and e.message:
            display_message = e.message
        elif error_message == 'auth/email-already-exists':
            display_message = "Email already in use. Please use a different email or log in."
        elif error_message == 'auth/weak-password':
            display_message = "Password is too weak. Please choose a stronger password (at least 6 characters)."
        
        await log_event(
            'user_registered',
            {'email': request_data.email, 'username': request_data.username, 'error': str(e), 'firebase_code': error_message},
            user_id=None, # User ID might not be available if creation failed
            success=False,
            error_message=display_message,
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=display_message) # Raise HTTPException
    except Exception as e:
        logger.error(f"An unexpected error occurred during registration for email {request_data.email}: {e}", exc_info=True)
        await log_event(
            'user_registered',
            {'email': request_data.email, 'username': request_data.username, 'error': str(e)},
            user_id=None,
            success=False,
            error_message=str(e),
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")


@app.post("/login")
async def login_user_endpoint(request_data: LoginRequest, request: Request):
    """
    Authenticates a user and generates a custom Firebase token.
    """
    logger.debug(f"Attempting login for email: {request_data.email}")
    try:
        # Authenticate with Firebase Authentication (this typically happens client-side or via a more secure method)
        # For this example, we assume we can get the user record by email
        user_record = auth.get_user_by_email(request_data.email)
        
        # Create a custom token for the authenticated user
        custom_token = auth.create_custom_token(user_record.uid).decode('utf-8')
        logger.info(f"Custom token generated for user: {user_record.uid}")

        # Log successful login
        await log_event(
            'user_logged_in',
            {'email': request_data.email},
            user_id=user_record.uid,
            success=True,
            log_from_backend=True
        )
        return {"message": "Login successful", "custom_token": custom_token, "uid": user_record.uid, "success": True}
    except firebase_exceptions.FirebaseError as e:
        logger.error(f"Firebase login error for {request_data.email}: {e.code} - {e.cause}", exc_info=True)
        error_message = e.code
        display_message = f"Firebase error: {error_message}"
        
        if hasattr(e, 'message') and e.message:
            display_message = e.message
        elif error_message == 'auth/user-not-found':
            display_message = "No account found with that email. Please register or check your email."
        elif error_message == 'auth/invalid-password':
            display_message = "Invalid password. Please try again."
        
        await log_event(
            'user_logged_in',
            {'email': request_data.email, 'error': str(e), 'firebase_code': error_message},
            user_id=None, # User ID might not be available if login failed
            success=False,
            error_message=display_message,
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=display_message) # Raise HTTPException
    except Exception as e:
        logger.error(f"An unexpected error occurred during login for email {request_data.email}: {e}", exc_info=True)
        await log_event(
            'user_logged_in',
            {'email': request_data.email, 'error': str(e)},
            user_id=None,
            success=False,
            error_message=str(e),
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/log-frontend-analytics")
async def log_frontend_analytics_endpoint(event_data: FrontendAnalyticsEvent):
    """
    Receives analytics events from the frontend (unauthenticated).
    """
    logger.debug(f"Received frontend analytics event: {event_data.event_type} for user {event_data.user_id}")
    try:
        await log_event(
            event_data.event_type,
            event_data.details,
            user_id=event_data.user_id, # Use the user_id provided by the frontend
            success=event_data.success,
            error_message=event_data.error_message,
            log_from_backend=False # This event originates from the frontend
        )
        return {"message": "Analytics event logged successfully"}
    except Exception as e:
        logger.error(f"Failed to log frontend analytics event: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to log analytics event: {e}")


@app.get("/user/profile")
async def get_user_profile_endpoint(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Retrieves the profile of the current authenticated user."""
    logger.info(f"User {current_user['uid']} requesting profile.")
    # current_user already contains the full user profile fetched by get_current_user
    return {"user": current_user, "success": True}

class UserProfileUpdate(BaseModel):
    username: Optional[str] = None # Changed from display_name to username
    phone: Optional[str] = None # Changed from phone_number to phone
    address: Optional[str] = None # Added address
    bio: Optional[str] = None # Added bio
    # Add other updatable fields as needed

@app.put("/user/profile")
async def update_user_profile_endpoint(
    update_data: UserProfileUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Updates the profile of the current authenticated user."""
    uid = current_user['uid']
    logger.info(f"User {uid} updating profile.")
    update_dict = update_data.model_dump(exclude_unset=True)
    
    if not update_dict:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update.")

    try:
        # Delegate to user_manager to update the profile in Firestore
        result = await user_manager.update_user_profile(uid, {"profile_data": update_dict}) # Update profile_data sub-document
        if result["success"]:
            logger.info(f"User {uid} profile updated successfully.")
            await log_event(
                'user_profile_updated',
                {'fields': list(update_dict.keys())},
                user_id=uid,
                success=True,
                log_from_backend=True
            )
            return {"message": "Profile updated successfully", "success": True}
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    except Exception as e:
        logger.error(f"Failed to update profile for user {uid}: {e}", exc_info=True)
        await log_event(
            'user_profile_updated',
            {'error': str(e), 'fields': list(update_dict.keys())},
            user_id=uid,
            success=False,
            error_message=str(e),
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/upload-document")
async def upload_document_endpoint(
    file_name: str,
    file_content_base64: str, # Base64 encoded content of the file
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Uploads a document for the current user, processes it, and stores its vectors.
    """
    user_id = current_user['uid']
    logger.info(f"User {user_id} attempting to upload document: {file_name}")

    # Use the get_user_tier_capability function from user_manager
    if not get_user_tier_capability(user_id, 'document_upload_enabled', False):
        await log_event(
            'document_upload_attempt',
            {'file_name': file_name, 'reason': 'Capability not enabled'},
            user_id=user_id,
            success=False,
            error_message="Document upload is not enabled for your current tier.",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Document upload is not enabled for your current tier.")

    try:
        result = await vector_utils_module.process_uploaded_document(
            user_id=user_id,
            file_name=file_name,
            file_content_base64=file_content_base64,
            firestore_manager=firestore_manager,
            cloud_storage_utils=cloud_storage_utils,
            config_manager=config_manager,
            log_event_func=log_event # Pass the log_event function
        )
        if result["success"]:
            await log_event(
                'document_upload_success',
                {'file_name': file_name, 'document_id': result.get("document_id")},
                user_id=user_id,
                success=True,
                log_from_backend=True
            )
            return {"message": result["message"], "document_id": result.get("document_id"), "success": True}
        else:
            await log_event(
                'document_upload_failure',
                {'file_name': file_name, 'reason': result["message"]},
                user_id=user_id,
                success=False,
                error_message=result["message"],
                log_from_backend=True
            )
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    except Exception as e:
        logger.error(f"Error uploading document for user {user_id}: {e}", exc_info=True)
        await log_event(
            'document_upload_failure',
            {'file_name': file_name, 'reason': str(e)},
            user_id=user_id,
            success=False,
            error_message=str(e),
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to upload document: {e}")


@app.post("/agent/chat")
async def chat_with_agent_endpoint(
    message: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Allows authenticated users to chat with the Intelli-Agent, leveraging available tools.
    """
    user_id = current_user['uid']
    user_tier = current_user.get('tier', 'free')
    user_roles = current_user.get('roles', [])
    logger.info(f"Chat request from user {user_id} (Tier: {user_tier}, Roles: {user_roles}): {message}")

    await log_event(
        'chat_request',
        {'message': message},
        user_id=user_id,
        success=True,
        log_from_backend=True
    )

    response_message = f"Hello {current_user.get('username', 'user')}! I received your message: '{message}'. " \
                       "I'm currently under development, but I can tell you about some tools I have."

    if "stock price" in message.lower():
        try:
            # Pass user_id as user_token to the tool for RBAC checks and logging
            stock_price = await domain_tool_instances["finance_tools"].finance_get_stock_price(symbol="GOOG", user_token=user_id)
            response_message += f"\n\n(Example Tool Call: Google Stock Price: {stock_price})"
        except Exception as e:
            response_message += f"\n\n(Example Tool Call Error: Could not get stock price: {e})"
    
    elif "my documents" in message.lower() or "uploaded files" in message.lower():
        try:
            document_tools = domain_tool_instances["document_tools"]
            # Pass user_id as user_token to the tool for RBAC checks and logging
            doc_query_result = await document_tools.document_query_uploaded_docs(
                query_text="summarize key points from my latest report",
                user_token=user_id
            )
            response_message += f"\n\n(Example Tool Call: Document Query Result: {doc_query_result})"
        except Exception as e:
            response_message += f"\n\n(Example Tool Call Error: Could not query documents: {e})"

    await log_event(
        'chat_response',
        {'response': response_message},
        user_id=user_id,
        success=True,
        log_from_backend=True
    )
    
    return {"response": response_message}

class UserRoleUpdate(BaseModel):
    new_tier: Optional[str] = None
    roles: Optional[List[str]] = None

@app.put("/admin/users/{user_id}/roles-and-tier")
async def update_user_roles_and_tier_endpoint(
    user_id: str, # This is the target user_id to update
    update_data: UserRoleUpdate,
    current_user: Dict[str, Any] = Depends(get_current_admin_user) # This is the admin user
):
    """Updates a user's tier and roles (admin only)."""
    admin_uid = current_user['uid']
    logger.info(f"Admin user {admin_uid} updating roles and tier for user {user_id}.")
    
    result = await user_manager.update_user_roles_and_tier(user_id, update_data.new_tier, update_data.roles)
    
    if result["success"]:
        # The user_manager.update_user_roles_and_tier already logs an event.
        # This log is for the admin action itself.
        await log_event(
            'admin_action_update_user_roles_tier',
            {'target_user_uid': user_id, 'new_tier': update_data.new_tier, 'new_roles': update_data.roles},
            user_id=admin_uid,
            success=True,
            log_from_backend=True
        )
        return {"success": True, "message": "User roles and tier updated successfully."}
    else:
        await log_event(
            'admin_action_update_user_roles_tier',
            {'target_user_uid': user_id, 'new_tier': update_data.new_tier, 'new_roles': update_data.roles, 'reason': result["message"]},
            user_id=admin_uid,
            success=False,
            error_message=result["message"],
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])

@app.get("/admin/analytics/events")
async def get_analytics_events_endpoint(
    event_type: Optional[str] = None,
    user_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_admin_user)
):
    """Retrieves analytics events with optional filters (admin only)."""
    admin_uid = current_user['uid']
    logger.info(f"Admin user {admin_uid} requesting analytics events.")
    
    parsed_start_date = parse_date_to_yyyymmdd(start_date) if start_date else None
    parsed_end_date = parse_date_to_yyyymmdd(end_date) if end_date else None

    try:
        events = await firestore_manager.get_analytics_events(
            event_type=event_type,
            user_id=user_id, # This is the filter for events, not the admin's user_id
            start_date=parsed_start_date,
            end_date=parsed_end_date
        )
        await log_event(
            'admin_action_get_analytics',
            {'filters': {'event_type': event_type, 'target_user_id': user_id, 'start_date': start_date, 'end_date': end_date}},
            user_id=admin_uid,
            success=True,
            log_from_backend=True
        )
        return {"success": True, "events": events}
    except Exception as e:
        logger.error(f"Error retrieving analytics events for admin {admin_uid}: {e}", exc_info=True)
        await log_event(
            'admin_action_get_analytics',
            {'filters': {'event_type': event_type, 'target_user_id': user_id, 'start_date': start_date, 'end_date': end_date}, 'reason': str(e)},
            user_id=admin_uid,
            success=False,
            error_message=str(e),
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

