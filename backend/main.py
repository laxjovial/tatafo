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
from pydantic import BaseModel, EmailStr, Field # Keep BaseModel for FrontendAnalyticsEvent if not moved yet
from datetime import datetime, timezone

# Project imports
from config.config_manager import config_manager
from utils.analytics_tracker import initialize_analytics, log_event
from database.firestore_manager import FirestoreManager
import shared_tools.cloud_storage_utils as cloud_storage_utils_module
import shared_tools.vector_utils as vector_utils_module
from utils.date_parser import parse_date_to_yyyymmdd
from utils.user_manager import UserManager, get_user_tier_capability # Keep get_user_tier_capability for direct use

# Import Pydantic models from backend.models
from backend.models.user_models import UserProfile, UserUpdate # Import UserProfile and UserUpdate

# Import API routers
from backend.api.auth_api import router as auth_router
# from backend.api.user_api import router as user_router # Assuming user_api.py will be created/updated
# from backend.api.admin_api import router as admin_router # Assuming admin_api.py will be created/updated
# from backend.api.tool_api import router as tool_router # Assuming tool_api.py will be created/updated
# from backend.api.integrations_api import router as integrations_router # For user/global API management

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
from domain_tools.document_tools.document_tool import DocumentTools

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# --- Firebase Admin SDK Initialization ---
# Ensure Firebase is initialized only once
if not firebase_admin._apps:
    try:
        firebase_credentials_json = os.environ.get("FIREBASE_CREDENTIALS")
        if not firebase_credentials_json:
            logger.warning("FIREBASE_CREDENTIALS environment variable not set. Using dummy credentials for local testing.")
            # Attempt to load from config_manager if available, otherwise use hardcoded mock
            firebase_config_str = config_manager.get_secret("firebase_config")
            firebase_config = json.loads(firebase_config_str) if firebase_config_str else {}
            
            # Ensure all required fields for credentials.Certificate are present, even if mocked
            cred_dict = {
                "type": "service_account",
                "project_id": firebase_config.get("projectId", "mock-project-id"),
                "private_key_id": os.environ.get("FIREBASE_PRIVATE_KEY_ID", "mock-key-id"),
                "private_key": os.environ.get("FIREBASE_PRIVATE_KEY", "-----BEGIN PRIVATE KEY-----\nMOCK_PRIVATE_KEY\n-----END PRIVATE KEY-----\n").replace('\\n', '\n'), # Handle potential escaped newlines
                "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL", "mock-client@mock-project-id.iam.gserviceaccount.com"),
                "client_id": os.environ.get("FIREBASE_CLIENT_ID", "mock-client-id"),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": os.environ.get("FIREBASE_CLIENT_X509_CERT_URL", f"https://www.googleapis.com/robot/v1/metadata/x509/mock-client%40{firebase_config.get('projectId', 'mock-project-id')}.iam.gserviceaccount.com"),
                "universe_domain": "googleapis.com"
            }
            cred = credentials.Certificate(cred_dict)
        else:
            cred = credentials.Certificate(json.loads(firebase_credentials_json))
        
        firebase_app = firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        logger.error(f"FATAL: Error initializing Firebase Admin SDK: {e}", exc_info=True)
        raise

# Initialize Firestore Manager
db_client = firestore.client(firebase_app)
firestore_manager = FirestoreManager(db_client) # Pass db_client to FirestoreManager
logger.info("FirestoreManager initialized.")

# Initialize Cloud Storage Utils Wrapper
cloud_storage_utils = cloud_storage_utils_module.CloudStorageUtilsWrapper(config_manager)
logger.info("CloudStorageUtilsWrapper initialized.")

# Initialize UserManager with FirestoreManager and CloudStorageUtilsWrapper
user_manager = UserManager(firestore_manager, cloud_storage_utils)
logger.info("UserManager initialized.")

# Initialize Analytics Tracker (after UserManager as it might use user_manager for user_id)
app_id_for_analytics = config_manager.get("app_id", "default-backend-app-id")
backend_service_user_id = "backend-service-user" # Use a service user ID for backend-initiated logs
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
    allow_origins=["*"], # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2PasswordBearer for token authentication
# This scheme is used by FastAPI's Depends to extract the token from the Authorization header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # tokenUrl is for docs, actual token verification happens below

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserProfile:
    """
    Authenticates the user using Firebase ID token (JWT) provided in the Authorization header.
    Retrieves and returns their UserProfile, ensuring the account is active.
    """
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

        # Return the user profile as a Pydantic model for type consistency
        # Ensure 'uid' from Firebase is mapped to 'user_id' in UserProfile
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

async def get_current_admin_user(current_user: UserProfile = Depends(get_current_user)):
    """
    Dependency to check if the current user is an admin.
    Returns UserProfile if admin, otherwise raises 403.
    """
    # Check for 'admin' role or creator bypass
    # Assuming 'creator' role implies full admin access
    if "admin" not in current_user.roles and "creator" not in current_user.roles:
        # Log authorization failure
        await log_event(
            'authorization_failure',
            {'required_role': 'admin', 'user_roles': current_user.roles},
            user_id=current_user.user_id,
            success=False,
            error_message="Admin access required",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized. Admin access required.")
    return current_user

# Initialize DocumentTools first, as it's a dependency for other domain tools
document_tools_instance = DocumentTools(vector_utils, firestore_manager, cloud_storage_utils, config_manager, log_event)

# Initialize domain tool instances, passing necessary dependencies
# These instances will be passed to the LLM service for tool calling
domain_tool_instances = {
    "finance_tools": FinanceTools(
        config_manager,
        firestore_manager,
        log_event,
        document_tools_instance # Pass document_tools_instance if finance tools can leverage it
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
    # Add shared tools here if they are directly callable by the agent
    "python_interpreter_tool": shared_tools.python_interpreter_tool.PythonInterpreterTool(log_event),
    "chart_generation_tool": shared_tools.chart_generation_tool.ChartGenerationTool(log_event),
    "doc_summarizer_tool": shared_tools.doc_summarizer.DocumentSummarizerTool(log_event),
    "scrapper_tool": shared_tools.scrapper_tool.ScrapperTool(log_event),
    "sentiment_analysis_tool": shared_tools.sentiment_analysis_tool.SentimentAnalysisTool(log_event),
    "query_uploaded_docs_tool": shared_tools.query_uploaded_docs_tool.QueryUploadedDocsTool(vector_utils, firestore_manager, log_event)
}


# --- Pydantic Models for Request Bodies (Moved to models/user_models.py or other API-specific models) ---
# FrontendAnalyticsEvent will be defined in a separate analytics_models.py or kept here for now
class FrontendAnalyticsEvent(BaseModel):
    event_type: str
    details: Dict[str, Any]
    user_id: str # Frontend will send the user_id
    success: bool
    error_message: Optional[str] = None

# --- API Endpoints ---

# Include routers from other API files
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
# app.include_router(user_router, prefix="/user", tags=["User Management"])
# app.include_router(admin_router, prefix="/admin", tags=["Admin Operations"])
# app.include_router(tool_router, prefix="/tools", tags=["Tool Operations"])
# app.include_router(integrations_router, prefix="/integrations", tags=["API Integrations"])


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Intelli-Agent Backend! Visit /docs for API documentation."}

# Removed duplicate /register and /login endpoints, now handled by auth_api.router

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


# Moved /profile/{user_id} and /profile/update/{user_id} to user_api.py (will be created/updated next)
# The Pydantic model UserProfileUpdate is now imported from user_models.py

@app.post("/upload-document")
async def upload_document_endpoint(
    file_name: str,
    file_content_base64: str, # Base64 encoded content of the file
    current_user: UserProfile = Depends(get_current_user) # Use UserProfile type hint
):
    """
    Uploads a document for the current user, processes it, and stores its vectors.
    Checks user tier capability for document upload.
    """
    user_id = current_user.user_id # Access user_id from UserProfile
    logger.info(f"User {user_id} attempting to upload document: {file_name}")

    # Use the get_user_tier_capability function from user_manager
    if not get_user_tier_capability(current_user.tier, 'document_upload_enabled', False):
        await log_event(
            'document_upload_attempt',
            {'file_name': file_name, 'reason': 'Capability not enabled', 'tier': current_user.tier},
            user_id=user_id,
            success=False,
            error_message="Document upload is not enabled for your current tier.",
            log_from_backend=True
        )
        # Return 403 with a specific detail for frontend to redirect to upgrade page
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail={"message": "Document upload is not enabled for your current tier.", "code": "UPGRADE_REQUIRED"})

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
    except HTTPException:
        raise # Re-raise HTTPExceptions
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
    current_user: UserProfile = Depends(get_current_user) # Use UserProfile type hint
):
    """
    Allows authenticated users to chat with the Intelli-Agent, leveraging available tools.
    This endpoint will eventually integrate LangGraph and dynamic tool selection.
    """
    user_id = current_user.user_id # Access user_id from UserProfile
    user_tier = current_user.tier # Access user_tier from UserProfile
    user_roles = current_user.roles # Access user_roles from UserProfile
    logger.info(f"Chat request from user {user_id} (Tier: {user_tier}, Roles: {user_roles}): {message}")

    # Log the chat request
    await log_event(
        'chat_request',
        {'message': message, 'tier': user_tier},
        user_id=user_id,
        success=True,
        log_from_backend=True
    )

    response_message = f"Hello {current_user.username}! I received your message: '{message}'. " \
                       "I'm currently under development, but I can tell you about some tools I have."

    # Placeholder for future LangGraph integration and dynamic tool calling
    # For now, keep simple conditional logic for demonstration
    if "stock price" in message.lower():
        try:
            # Pass UserProfile object to the tool if it needs tier/roles for internal checks
            stock_price = await domain_tool_instances["finance_tools"].finance_get_stock_price(symbol="GOOG", user_context=current_user)
            response_message += f"\n\n(Example Tool Call: Google Stock Price: {stock_price})"
            await log_event(
                'tool_usage_example',
                {'tool': 'finance_get_stock_price', 'symbol': 'GOOG'},
                user_id=user_id,
                success=True,
                log_from_backend=True
            )
        except HTTPException as e:
            response_message += f"\n\n(Tool Call Error: {e.detail['message'] if isinstance(e.detail, dict) else e.detail})"
            await log_event(
                'tool_usage_example',
                {'tool': 'finance_get_stock_price', 'symbol': 'GOOG', 'error': str(e.detail)},
                user_id=user_id,
                success=False,
                error_message=str(e.detail),
                log_from_backend=True
            )
        except Exception as e:
            response_message += f"\n\n(Example Tool Call Error: Could not get stock price: {e})"
            await log_event(
                'tool_usage_example',
                {'tool': 'finance_get_stock_price', 'symbol': 'GOOG', 'error': str(e)},
                user_id=user_id,
                success=False,
                error_message=str(e),
                log_from_backend=True
            )
    
    elif "my documents" in message.lower() or "uploaded files" in message.lower():
        try:
            document_tools = domain_tool_instances["document_tools"]
            # Pass UserProfile object to the tool for RBAC checks and logging
            doc_query_result = await document_tools.document_query_uploaded_docs(
                query_text="summarize key points from my latest report",
                user_context=current_user
            )
            response_message += f"\n\n(Example Tool Call: Document Query Result: {doc_query_result})"
            await log_event(
                'tool_usage_example',
                {'tool': 'document_query_uploaded_docs'},
                user_id=user_id,
                success=True,
                log_from_backend=True
            )
        except HTTPException as e:
            response_message += f"\n\n(Tool Call Error: {e.detail['message'] if isinstance(e.detail, dict) else e.detail})"
            await log_event(
                'tool_usage_example',
                {'tool': 'document_query_uploaded_docs', 'error': str(e.detail)},
                user_id=user_id,
                success=False,
                error_message=str(e.detail),
                log_from_backend=True
            )
        except Exception as e:
            response_message += f"\n\n(Example Tool Call Error: Could not query documents: {e})"
            await log_event(
                'tool_usage_example',
                {'tool': 'document_query_uploaded_docs', 'error': str(e)},
                user_id=user_id,
                success=False,
                error_message=str(e),
                log_from_backend=True
            )

    # Log the chat response
    await log_event(
        'chat_response',
        {'response': response_message},
        user_id=user_id,
        success=True,
        log_from_backend=True
    )
    
    return {"response": response_message}

# Moved /admin/users/{user_id}/roles-and-tier to admin_api.py (will be created/updated next)
# Moved /admin/analytics/events to admin_api.py (will be created/updated next)

