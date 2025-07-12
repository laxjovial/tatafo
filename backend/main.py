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
from pydantic import BaseModel # Keep BaseModel for FrontendAnalyticsEvent if not moved yet
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
from backend.models.user_models import UserProfile # Only UserProfile is directly used here

# Import API routers
from backend.api.auth_api import router as auth_router
from backend.api.user_api import router as user_router
from backend.api.admin_api import router as admin_router # Now including admin_api
# from backend.api.tool_api import router as tool_router # Assuming tool_api.py will be created/updated
# from backend.api.integrations_api import router as integrations_router # For user/global API management

# Import middleware dependencies (for dependency overrides)
from backend.middleware.auth_middleware import (
    get_current_user, get_current_admin_user,
    get_firestore_manager_dependency, get_user_manager_dependency,
    get_api_usage_service_dependency # NEW: Import for ApiUsageService
)

# Import Services (for initialization and injection)
from backend.services.admin_service import AdminService # Now importing AdminService
from backend.services.api_usage_service import ApiUsageService # NEW: Import ApiUsageService
from backend.services.llm_service import LLMService # NEW: Import LLMService

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

# Import shared tools that might be callable by the agent
import shared_tools.python_interpreter_tool
import shared_tools.chart_generation_tool
import shared_tools.doc_summarizer
import shared_tools.scrapper_tool
import shared_tools.sentiment_analysis_tool
import shared_tools.query_uploaded_docs_tool


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

# Initialize Core Managers and Services (singletons for the app)
_db_client_instance = firestore.client(firebase_app)
_firestore_manager_instance = FirestoreManager(_db_client_instance)
logger.info("FirestoreManager initialized.")

_cloud_storage_utils_instance = cloud_storage_utils_module.CloudStorageUtilsWrapper(config_manager)
logger.info("CloudStorageUtilsWrapper initialized.")

_user_manager_instance = UserManager(_firestore_manager_instance, _cloud_storage_utils_instance)
logger.info("UserManager initialized.")

_vector_utils_instance = vector_utils_module.VectorUtilsWrapper(
    firestore_manager=_firestore_manager_instance,
    cloud_storage_utils=_cloud_storage_utils_instance,
    config_manager=config_manager
)
logger.info("VectorUtilsWrapper initialized.")

_api_usage_service_instance = ApiUsageService(_firestore_manager_instance, config_manager, _user_manager_instance) # NEW: Pass _user_manager_instance to ApiUsageService
logger.info("ApiUsageService initialized.")

_admin_service_instance = AdminService( # NEW: Initialize AdminService with its dependencies
    firestore_manager=_firestore_manager_instance,
    user_manager=_user_manager_instance,
    cloud_storage_utils=_cloud_storage_utils_instance,
    api_usage_service=_api_usage_service_instance
)
logger.info("AdminService initialized.")

_llm_service_instance = LLMService( # NEW: Initialize LLMService with its dependencies
    user_manager=_user_manager_instance,
    api_usage_service=_api_usage_service_instance
)
logger.info("LLMService initialized.")


# Initialize Analytics Tracker (after UserManager as it might use user_manager for user_id)
app_id_for_analytics = config_manager.get("app_id", "default-backend-app-id")
backend_service_user_id = "backend-service-user" # Use a service user ID for backend-initiated logs
initialize_analytics(_db_client_instance, auth, app_id_for_analytics, backend_service_user_id)
logger.info("Analytics Tracker initialized.")


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

# --- Dependency Overrides ---
# Override the dependency functions defined in auth_middleware.py
# to provide the actual initialized instances.
app.dependency_overrides[get_firestore_manager_dependency] = lambda: _firestore_manager_instance
app.dependency_overrides[get_user_manager_dependency] = lambda: _user_manager_instance
app.dependency_overrides[get_api_usage_service_dependency] = lambda: _api_usage_service_instance # NEW: Override for ApiUsageService

# Override AdminService dependency (for admin_api.py)
from backend.api.admin_api import get_admin_service_dependency # Import the dependency function from admin_api
app.dependency_overrides[get_admin_service_dependency] = lambda: _admin_service_instance

# Override LLMService dependency (for chat endpoint)
# Assuming a get_llm_service_dependency will be created in an API file or common place
async def get_llm_service_dependency() -> LLMService:
    """Dependency to get the LLMService instance."""
    raise NotImplementedError("LLMService dependency must be provided by main.py")

app.dependency_overrides[get_llm_service_dependency] = lambda: _llm_service_instance


# Initialize DocumentTools first, as it's a dependency for other domain tools
document_tools_instance = DocumentTools(_vector_utils_instance, _firestore_manager_instance, _cloud_storage_utils_instance, config_manager, log_event)

# Initialize domain tool instances, passing necessary dependencies
# These instances will be passed to the LLM service for tool calling
# Note: Tools themselves will use the LLMService for API checks, but their direct
# instantiation here is for consistency if they have other internal dependencies.
domain_tool_instances = {
    "finance_tools": FinanceTools(
        config_manager,
        _firestore_manager_instance,
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
    # Add shared tools here if they are directly callable by the agent
    "python_interpreter_tool": shared_tools.python_interpreter_tool.PythonInterpreterTool(log_event),
    "chart_generation_tool": shared_tools.chart_generation_tool.ChartGenerationTool(log_event),
    "doc_summarizer_tool": shared_tools.doc_summarizer.DocumentSummarizerTool(log_event),
    "scrapper_tool": shared_tools.scrapper_tool.ScrapperTool(log_event),
    "sentiment_analysis_tool": shared_tools.sentiment_analysis_tool.SentimentAnalysisTool(log_event),
    "query_uploaded_docs_tool": shared_tools.query_uploaded_docs_tool.QueryUploadedDocsTool(_vector_utils_instance, _firestore_manager_instance, log_event)
}


# --- Pydantic Models for Request Bodies (FrontendAnalyticsEvent remains here for now) ---
class FrontendAnalyticsEvent(BaseModel):
    event_type: str
    details: Dict[str, Any]
    user_id: str # Frontend will send the user_id
    success: bool
    error_message: Optional[str] = None

# --- API Endpoints ---

# Include routers from other API files
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(user_router, prefix="/user", tags=["User Management"])
app.include_router(admin_router, prefix="/admin", tags=["Admin Operations"]) # Now including admin_api
# app.include_router(tool_router, prefix="/tools", tags=["Tool Operations"])
# app.include_router(integrations_router, prefix="/integrations", tags=["API Integrations"])


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Intelli-Agent Backend! Visit /docs for API documentation."}

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


@app.post("/upload-document")
async def upload_document_endpoint(
    file_name: str,
    file_content_base64: str, # Base64 encoded content of the file
    current_user: UserProfile = Depends(get_current_user), # Use UserProfile type hint
    user_manager: UserManager = Depends(get_user_manager_dependency) # Inject UserManager
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
            firestore_manager=_firestore_manager_instance, # Use the globally initialized instance
            cloud_storage_utils=_cloud_storage_utils_instance, # Use the globally initialized instance
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
    current_user: UserProfile = Depends(get_current_user), # Use UserProfile type hint
    llm_service: LLMService = Depends(get_llm_service_dependency) # NEW: Inject LLMService
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

    # Use the injected LLMService to handle the chat with agent
    try:
        # Pass the full UserProfile object to the LLMService
        response_message = await llm_service.chat_with_agent(
            prompt=message,
            chat_history=[], # For now, chat history is not fully managed here, but passed for agent context
            user_profile=current_user, # Pass the UserProfile object
            user_provided_temperature=None, # Example: can be passed from frontend
            user_provided_llm_provider=None, # Example: can be passed from frontend
            user_provided_model_name=None # Example: can be passed from frontend
        )
    except HTTPException as e:
        logger.error(f"HTTPException during agent chat for user {user_id}: {e.detail}", exc_info=True)
        await log_event(
            'chat_response_failure',
            {'message': message, 'error': e.detail},
            user_id=user_id,
            success=False,
            error_message=e.detail,
            log_from_backend=True
        )
        raise # Re-raise the HTTPException
    except Exception as e:
        logger.error(f"Unexpected error during agent chat for user {user_id}: {e}", exc_info=True)
        response_message = f"An unexpected error occurred while processing your request: {e}"
        await log_event(
            'chat_response_failure',
            {'message': message, 'error': str(e)},
            user_id=user_id,
            success=False,
            error_message=str(e),
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")

    # Log the chat response
    await log_event(
        'chat_response',
        {'response': response_message},
        user_id=user_id,
        success=True,
        log_from_backend=True
    )
    
    return {"response": response_message}

