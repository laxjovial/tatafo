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
from utils.date_parser import parse_date_string
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
        firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Firebase Admin SDK: {e}")
        # Depending on criticality, you might want to exit or raise the exception
        raise

# Initialize Firestore Manager
firestore_manager = FirestoreManager()
logger.info("FirestoreManager initialized.")

# Initialize Analytics Tracker
initialize_analytics(firestore_manager)
logger.info("Analytics Tracker initialized.")

# Initialize Cloud Storage Utils Wrapper
cloud_storage_utils = cloud_storage_utils_module.CloudStorageUtilsWrapper(config_manager)
logger.info("CloudStorageUtilsWrapper initialized.")

# Initialize Vector Utils Wrapper
# Pass the necessary managers to the VectorUtilsWrapper
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

# OAuth2PasswordBearer for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Authenticates the user using Firebase ID token.
    """
    try:
        # Verify the Firebase ID token
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token['uid']
        user_record = await user_manager.get_user(uid) # Fetch user details including roles
        if not user_record:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")
        
        # Add UID to the user_record for easier access
        user_record['uid'] = uid
        return user_record
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
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
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized. Admin access required.")
    return current_user

# Initialize domain tool instances, passing necessary dependencies
# Each tool class receives the managers it needs.
domain_tool_instances = {
    "finance_tools": FinanceTools(firestore_manager, config_manager, log_event),
    "crypto_tools": CryptoTools(firestore_manager, config_manager, log_event),
    "medical_tools": MedicalTools(firestore_manager, config_manager, log_event),
    "news_tools": NewsTools(firestore_manager, config_manager, log_event),
    "legal_tools": LegalTools(firestore_manager, config_manager, log_event),
    "education_tools": EducationTools(firestore_manager, config_manager, log_event),
    "entertainment_tools": EntertainmentTools(firestore_manager, config_manager, log_event),
    "weather_tools": WeatherTools(firestore_manager, config_manager, log_event),
    "travel_tools": TravelTools(firestore_manager, config_manager, log_event),
    "sports_tools": SportsTools(firestore_manager, config_manager, log_event),
    # NEW: Initialize DocumentTools and pass all required dependencies
    "document_tools": DocumentTools(vector_utils, firestore_manager, cloud_storage_utils, config_manager, log_event),
}

# --- API Endpoints ---

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Intelli-Agent Backend!"}

@app.post("/register")
async def register_user_endpoint(email: EmailStr, password: str, request: Request):
    """Registers a new user with Firebase Authentication and creates a user profile in Firestore."""
    try:
        user = auth.create_user(email=email, password=password)
        await user_manager.create_user_profile(user.uid, email)
        logger.info(f"User registered: {user.uid} with email {email}")
        await log_event('user_registered', {'email': email}, user_id=user.uid, success=True)
        return {"message": "User registered successfully", "uid": user.uid}
    except Exception as e:
        logger.error(f"Registration failed for email {email}: {e}")
        await log_event('user_registered', {'email': email, 'error': str(e)}, success=False)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@app.post("/login")
async def login_user_endpoint(email: EmailStr, password: str, request: Request):
    """
    Generates a custom Firebase token for a user.
    Note: In a real application, you'd typically use Firebase Client SDK for login
    and receive an ID token, which you'd then verify on the backend.
    This endpoint is for demonstration or specific backend-initiated authentication flows.
    """
    try:
        # Authenticate user (e.g., using Firebase Admin SDK's ability to get user by email
        # and then create a custom token, or verify credentials against a different system).
        # For simplicity, this example assumes you'd have a way to verify password
        # or that this is for generating a custom token for an already existing Firebase user.
        user_record = auth.get_user_by_email(email)
        custom_token = auth.create_custom_token(user_record.uid).decode('utf-8')
        logger.info(f"Custom token generated for user: {user_record.uid}")
        await log_event('user_logged_in', {'email': email}, user_id=user_record.uid, success=True)
        return {"message": "Login successful", "custom_token": custom_token, "uid": user_record.uid}
    except Exception as e:
        logger.error(f"Login failed for email {email}: {e}")
        await log_event('user_logged_in', {'email': email, 'error': str(e)}, success=False)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

@app.get("/user/profile")
async def get_user_profile_endpoint(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Retrieves the profile of the current authenticated user."""
    logger.info(f"User {current_user['uid']} requesting profile.")
    return {"user": current_user}

class UserProfileUpdate(BaseModel):
    display_name: Optional[str] = None
    phone_number: Optional[str] = None
    photo_url: Optional[str] = None
    # Add other updatable fields as needed

@app.put("/user/profile")
async def update_user_profile_endpoint(
    update_data: UserProfileUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Updates the profile of the current authenticated user."""
    uid = current_user['uid']
    logger.info(f"User {uid} updating profile.")
    update_dict = update_data.model_dump(exclude_unset=True) # Use model_dump to get only provided fields
    
    if not update_dict:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update.")

    try:
        await user_manager.update_user_profile(uid, update_dict)
        logger.info(f"User {uid} profile updated successfully.")
        await log_event('user_profile_updated', {'fields': list(update_dict.keys())}, user_id=uid, success=True)
        return {"message": "Profile updated successfully"}
    except Exception as e:
        logger.error(f"Failed to update profile for user {uid}: {e}")
        await log_event('user_profile_updated', {'error': str(e)}, user_id=uid, success=False)
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

    if not get_user_tier_capability(user_id, 'document_upload_enabled', False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Document upload is not enabled for your current tier.")

    try:
        # Call the process_uploaded_document from vector_utils_module directly
        # It now expects the managers as arguments
        result = await vector_utils_module.process_uploaded_document(
            user_id=user_id,
            file_name=file_name,
            file_content_base64=file_content_base64,
            firestore_manager=firestore_manager,
            cloud_storage_utils=cloud_storage_utils,
            config_manager=config_manager,
            log_event_func=log_event
        )
        if result["success"]:
            return {"message": result["message"], "document_id": result.get("document_id")}
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    except Exception as e:
        logger.error(f"Error uploading document for user {user_id}: {e}", exc_info=True)
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

    # Log the incoming chat request
    await log_event('chat_request', {'message': message}, user_id=user_id, success=True)

    # Placeholder for agent response logic
    # In a real scenario, you would integrate your LLM agent here,
    # passing the message, user_id, user_tier, and the domain_tool_instances.
    # The agent would then decide which tool to use based on the message.

    # Example of how to call a tool (for demonstration)
    # This part would be replaced by your actual agent's tool calling logic
    response_message = f"Hello {current_user.get('display_name', 'user')}! I received your message: '{message}'. " \
                       "I'm currently under development, but I can tell you about some tools I have."

    # Example: If the message contains "finance", use a finance tool
    if "stock price" in message.lower():
        try:
            # Dynamically call the tool method
            # This is a simplified example; your agent's logic would be more sophisticated
            finance_tools = domain_tool_instances["finance_tools"]
            # Assuming get_stock_price takes a symbol and user_token
            # This part needs to be carefully designed based on your agent's output
            # For a real agent, the agent would parse the message to extract 'symbol'
            stock_price = await finance_tools.get_stock_price(symbol="GOOG", user_token=user_id)
            response_message += f"\n\n(Example Tool Call: Google Stock Price: {stock_price})"
        except Exception as e:
            response_message += f"\n\n(Example Tool Call Error: Could not get stock price: {e})"
    
    # NEW: Example of calling the document query tool
    elif "my documents" in message.lower() or "uploaded files" in message.lower():
        try:
            document_tools = domain_tool_instances["document_tools"]
            # This is a simplified example; your agent would extract the actual query
            doc_query_result = await document_tools.query_uploaded_docs(
                query_text="summarize key points from my latest report", # Example query
                user_token=user_id
            )
            response_message += f"\n\n(Example Tool Call: Document Query Result: {doc_query_result})"
        except Exception as e:
            response_message += f"\n\n(Example Tool Call Error: Could not query documents: {e})"


    # Log the outgoing chat response
    await log_event('chat_response', {'response': response_message}, user_id=user_id, success=True)
    
    return {"response": response_message}

class UserRoleUpdate(BaseModel):
    new_tier: Optional[str] = None
    roles: Optional[List[str]] = None

@app.put("/admin/users/{user_id}/roles-and-tier")
async def update_user_roles_and_tier_endpoint(
    user_id: str,
    update_data: UserRoleUpdate,
    current_user: Dict[str, Any] = Depends(get_current_admin_user)
):
    """Updates a user's tier and roles (admin only)."""
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
    parsed_start_date = parse_date_string(start_date) if start_date else None
    parsed_end_date = parse_date_string(end_date) if end_date else None

    try:
        events = await firestore_manager.get_analytics_events(
            event_type=event_type,
            user_id=user_id,
            start_date=parsed_start_date,
            end_date=parsed_end_date
        )
        return {"success": True, "events": events}
    except Exception as e:
        logger.error(f"Error retrieving analytics events: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

