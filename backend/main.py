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
from pydantic import BaseModel, EmailStr, Field # Import Field for validation
from datetime import datetime, timezone

# Project imports
from config.config_manager import config_manager
from utils.analytics_tracker import initialize_analytics, log_event
from database.firestore_manager import FirestoreManager
import shared_tools.cloud_storage_utils as cloud_storage_utils_module # Import module
import shared_tools.vector_utils as vector_utils_module # Import module
from utils.date_parser import parse_date_to_yyyymmdd # Corrected import: changed from parse_date_string
from utils.user_manager import UserManager

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
# NEW: Import DocumentTools
from domain_tools.document_tools import DocumentTools

logger = logging.getLogger(__name__)

# --- Firebase Admin SDK Initialization ---
# Ensure Firebase Admin SDK is initialized only once
if not firebase_admin._apps:
    try:
        # Load Firebase credentials from environment variable
        firebase_credentials_json = os.environ.get("FIREBASE_CREDENTIALS")
        if not firebase_credentials_json:
            raise ValueError("FIREBASE_CREDENTIALS environment variable not set.")
        
        cred = credentials.Certificate(json.loads(firebase_credentials_json))
        firebase_app = firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Firebase Admin SDK: {e}")
        # Depending on criticality, you might want to exit or raise the exception
        raise

# Initialize Firestore Manager
db_client = firestore.client(firebase_app) # Get the Firestore client instance
auth_client = auth.Client.from_app(firebase_app) # Get the Auth client instance

firestore_manager = FirestoreManager()
logger.info("FirestoreManager initialized.")

# Initialize Cloud Storage Utils
cloud_storage_utils = cloud_storage_utils_module.CloudStorageUtilsWrapper(config_manager)
logger.info("CloudStorageUtilsWrapper initialized.")

# Initialize Vector Utils
vector_utils = vector_utils_module.VectorUtilsWrapper(
    firestore_manager=firestore_manager,
    cloud_storage_utils=cloud_storage_utils,
    config_manager=config_manager
)
logger.info("VectorUtilsWrapper initialized.")

# Initialize UserManager with FirestoreManager and CloudStorageUtilsWrapper
user_manager = UserManager(firestore_manager, cloud_storage_utils)
logger.info("UserManager initialized.")

app = FastAPI(
    title="Intelli-Agent Backend",
    description="Backend for the Intelli-Agent, providing various domain-specific tools and user management.",
    version="0.1.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analytics tracker for backend context
# Pass the actual Firestore client and Auth client
initialize_analytics(db_client, auth_client, config_manager.get("app_id", "default-backend-app-id"), "backend_server")
logger.info("Analytics tracker initialized for backend.")

# --- RBAC and Authentication Dependencies ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Retrieves the current authenticated user based on the Firebase ID token."""
    try:
        # Verify the Firebase ID token
        decoded_token = auth.verify_id_token(token)
        # The decoded_token contains the user's UID and other claims
        uid = decoded_token['uid']
        # Optionally, fetch more user details from Firestore if needed
        user_profile = await user_manager.get_user_profile(uid)
        if not user_profile:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User profile not found.")
        
        # Add capabilities to the user object
        user_capabilities = await user_manager.get_user_capabilities(uid)
        user_profile['capabilities'] = user_capabilities
        user_profile['uid'] = uid # Ensure uid is in the profile for consistency
        return user_profile
    except ValueError as e:
        logger.error(f"Invalid token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Authentication error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Ensures the current user has 'admin' role."""
    if "admin" not in current_user.get("roles", []):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized. Admin access required.")
    return current_user

# --- Tool Initialization ---
# Initialize domain-specific tool classes with necessary dependencies
finance_tools = FinanceTools(config_manager)
crypto_tools = CryptoTools(config_manager)
medical_tools = MedicalTools(config_manager)
news_tools = NewsTools(config_manager)
legal_tools = LegalTools(config_manager)
education_tools = EducationTools(config_manager)
entertainment_tools = EntertainmentTools(config_manager)
weather_tools = WeatherTools(config_manager)
travel_tools = TravelTools(config_manager)
sports_tools = SportsTools(config_manager)
# CORRECTED: Initialize DocumentTools with the vector_utils instance and other managers
document_tools = DocumentTools(
    vector_utils_wrapper=vector_utils,
    config_manager=config_manager,
    firestore_manager=firestore_manager, # Pass firestore_manager
    cloud_storage_utils=cloud_storage_utils, # Pass cloud_storage_utils
    log_event_func=log_event # Pass log_event function
)
logger.info("Domain tools initialized.")

# --- Pydantic Models for Request Bodies ---
class TokenRequest(BaseModel):
    email: EmailStr
    password: str

class UserRegistration(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    username: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class UploadDocumentRequest(BaseModel):
    file_name: str
    file_content_base64: str # Base64 encoded content of the file

class UpdateUserRolesAndTierRequest(BaseModel):
    new_tier: str
    roles: List[str]

# --- Health Check Endpoint ---
@app.get("/")
async def read_root():
    return {"message": "Intelli-Agent Backend is running!"}

# --- Authentication Endpoints ---
@app.post("/auth/register")
async def register(user_data: UserRegistration):
    """Registers a new user."""
    result = await user_manager.register_user(user_data.email, user_data.password, user_data.username)
    if result["success"]:
        return {"message": "User registered successfully", "user_id": result["user_id"]}
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])

@app.post("/auth/token")
async def login(request: Request, user_request: TokenRequest):
    """Generates a custom Firebase token for a user."""
    token_info = await user_manager.login_user(user_request.email, user_request.password)
    if token_info["success"]:
        return {"access_token": token_info["id_token"], "token_type": "bearer"}
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=token_info["message"])

@app.post("/auth/forgot_password")
async def forgot_password(request: Request, forgot_password_data: ForgotPasswordRequest):
    """Handles forgot password requests."""
    result = await user_manager.forgot_password(forgot_password_data.email)
    if result["success"]:
        return {"message": result["message"]}
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])

# --- User Profile Endpoints ---
@app.get("/users/me")
async def read_users_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Retrieves the profile of the currently authenticated user."""
    return current_user

@app.get("/users/{user_id}/profile")
async def get_user_profile_endpoint(user_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Retrieves a specific user's profile (requires admin or self-access)."""
    if current_user["uid"] != user_id and "admin" not in current_user.get("roles", []):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this user's profile.")
    
    profile = await user_manager.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return profile

@app.put("/users/{user_id}/profile")
async def update_user_profile_endpoint(user_id: str, profile_data: Dict[str, Any], current_user: Dict[str, Any] = Depends(get_current_user)):
    """Updates a user's profile (requires admin or self-access)."""
    if current_user["uid"] != user_id and "admin" not in current_user.get("roles", []):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to update this user's profile.")
    
    # Prevent updating sensitive fields like email, password, roles, tier directly here
    # These should have dedicated admin endpoints or Firebase Auth methods
    for key in ["email", "password", "roles", "tier", "uid"]:
        if key in profile_data:
            del profile_data[key] # Ensure sensitive fields are not updated via this endpoint

    result = await user_manager.update_user_profile(user_id, profile_data)
    if result["success"]:
        return {"message": "Profile updated successfully."}
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])

# --- RBAC Endpoints ---
@app.get("/rbac/capabilities/me")
async def get_my_capabilities(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Retrieves the capabilities for the currently authenticated user."""
    return current_user.get('capabilities', {})

@app.get("/rbac/capabilities/{user_id}")
async def get_user_capabilities_endpoint(user_id: str, current_user: Dict[str, Any] = Depends(get_current_admin_user)):
    """Retrieves capabilities for a specific user (admin only)."""
    capabilities = await user_manager.get_user_capabilities(user_id)
    if not capabilities:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found or no capabilities defined.")
    return capabilities

# --- Admin Endpoints ---
@app.put("/admin/users/{user_id}/roles_and_tier")
async def update_user_roles_and_tier_endpoint(
    user_id: str,
    update_data: UpdateUserRolesAndTierRequest,
    current_user: Dict[str, Any] = Depends(get_current_admin_user)
):
    """Updates a user's roles and tier (admin only)."""
    logger.info(f"Admin user {current_user['uid']} updating roles and tier for user {user_id}.")
    result = await user_manager.update_user_roles_and_tier(user_id, update_data.new_tier, update_data.roles)
    if result["success"]:
        return {"success": True, "message": "User roles and tier updated successfully."}
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
    logger.info(f"Admin user {current_user['uid']} requesting analytics events.")
    
    # Parse dates if provided
    # Corrected usage: Renamed parse_date_string to parse_date_to_yyyymmdd
    parsed_start_date = parse_date_to_yyyymmdd(start_date) if start_date else None
    parsed_end_date = parse_date_to_yyyymmdd(end_date) if end_date else None

    try:
        events = await firestore_manager.get_analytics_events(
            event_type=event_type,
            user_id=user_id,
            start_date=parsed_start_date,
            end_date=parsed_end_date
        )
        await log_event('admin_action', {
            'action': 'view_analytics_events',
            'filters': {'event_type': event_type, 'user_id': user_id, 'start_date': start_date, 'end_date': end_date},
            'num_results': len(events)
        }, user_id=current_user['uid'], success=True)
        return events
    except Exception as e:
        logger.error(f"Error retrieving analytics events: {e}", exc_info=True)
        await log_event('admin_action', {
            'action': 'view_analytics_events',
            'filters': {'event_type': event_type, 'user_id': user_id, 'start_date': start_date, 'end_date': end_date},
            'status': 'failed',
            'error': str(e)
        }, user_id=current_user['uid'], success=False, error_message=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve analytics events: {e}")

# --- Document Upload Endpoint ---
class DocumentUploadRequest(BaseModel):
    file_name: str
    file_content_base64: str

@app.post("/documents/upload")
async def upload_document_endpoint(
    upload_request: DocumentUploadRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Endpoint to upload a document to cloud storage and index it for RAG.
    Requires 'document_upload_enabled' capability.
    """
    user_id = current_user['uid']
    file_name = upload_request.file_name
    file_content_base64 = upload_request.file_content_base64

    logger.info(f"Received document upload request for user: {user_id}, file: {file_name}")

    # RBAC check for document upload capability
    if not current_user.get('capabilities', {}).get('document_upload_enabled', False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to document upload is not enabled for your current tier."
        )

    # Call the process_uploaded_document method from vector_utils_module
    # Note: process_uploaded_document now expects the managers as arguments
    result = await vector_utils.process_uploaded_document(
        user_id=user_id,
        file_name=file_name,
        file_content_base64=file_content_base64,
        firestore_manager=firestore_manager,
        cloud_storage_utils=cloud_storage_utils,
        config_manager=config_manager,
        log_event_func=log_event # Pass the log_event function
    )

    if result["success"]:
        return {"message": result["message"], "document_id": result.get("document_id")}
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])

