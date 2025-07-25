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
from pydantic import BaseModel
from datetime import datetime, timezone

# Project imports
from config.config_manager import config_manager
from utils.analytics_tracker import initialize_analytics, log_event
from database.firestore_manager import FirestoreManager
import shared_tools.cloud_storage_utils as cloud_storage_utils_module
import shared_tools.vector_utils as vector_utils_module
from utils.date_parser import parse_date_to_yyyymmdd
from utils.user_manager import UserManager, get_user_tier_capability

# Import Pydantic models from backend.models
from backend.models.user_models import UserProfile

# Import API routers
from backend.api.auth_api import router as auth_router
from backend.api.user_api import router as user_router
from backend.api.admin_api import router as admin_router
from backend.api.tool_api import router as tool_router
from backend.api.integrations_api import router as integrations_router
from backend.api.docs_api import router as docs_router # For document handling
from backend.api.api_config_api import router as api_config_router # For global API configurations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Firebase Admin SDK Initialization ---
# This section ensures Firebase Admin SDK is initialized correctly
# by prioritizing environment variable over secrets.toml for production readiness.
try:
    # Attempt to load credentials from the FIREBASE_CREDENTIALS environment variable first
    firebase_credentials_json = os.getenv("FIREBASE_CREDENTIALS")
    
    if firebase_credentials_json:
        try:
            cred_dict = json.loads(firebase_credentials_json)
            cred = credentials.Certificate(cred_dict)
            firebase_app = firebase_admin.initialize_app(cred, name="default-app")
            logger.info("Firebase Admin SDK initialized successfully using FIREBASE_CREDENTIALS environment variable.")
        except json.JSONDecodeError as e:
            logger.error(f"FATAL: FIREBASE_CREDENTIALS environment variable contains invalid JSON: {e}", exc_info=True)
            raise ValueError("Invalid JSON in FIREBASE_CREDENTIALS environment variable.") from e
        except Exception as e:
            logger.error(f"FATAL: Error initializing Firebase Admin SDK from environment variable: {e}", exc_info=True)
            # Re-raise to stop the application if Firebase initialization fails
            raise
    else:
        # If environment variable is not set, try loading from secrets.toml via config_manager
        logger.warning("FIREBASE_CREDENTIALS environment variable not set. Attempting to load from secrets.toml.")
        firebase_admin_cert_from_secrets = config_manager.get_secret("firebase_admin_cert")
        
        if firebase_admin_cert_from_secrets:
            try:
                # config_manager.get_secret should already return a dict if secrets.toml is parsed correctly
                # If it's a string, try to parse it as JSON
                if isinstance(firebase_admin_cert_from_secrets, str):
                    cred_dict = json.loads(firebase_admin_cert_from_secrets)
                else:
                    cred_dict = firebase_admin_cert_from_secrets # Assume it's already a dict

                cred = credentials.Certificate(cred_dict)
                firebase_app = firebase_admin.initialize_app(cred, name="default-app")
                logger.info("Firebase Admin SDK initialized successfully using secrets.toml.")
            except json.JSONDecodeError as e:
                logger.error(f"FATAL: 'firebase_admin_cert' in secrets.toml contains invalid JSON: {e}", exc_info=True)
                raise ValueError("Invalid JSON in 'firebase_admin_cert' in secrets.toml.") from e
            except Exception as e:
                logger.error(f"FATAL: Error initializing Firebase Admin SDK from secrets.toml: {e}", exc_info=True)
                raise
        else:
            logger.error("FATAL: Neither FIREBASE_CREDENTIALS environment variable nor 'firebase_admin_cert' in secrets.toml is set. Firebase Admin SDK cannot be initialized.")
            raise ValueError("Firebase Admin SDK credentials not configured. Please set FIREBASE_CREDENTIALS environment variable or configure 'firebase_admin_cert' in secrets.toml.")

except Exception as e:
    logger.critical(f"Application will not start due to Firebase initialization failure: {e}")
    # Exit the application if Firebase cannot be initialized
    exit(1)

# Initialize Analytics Tracker AFTER Firebase Admin SDK is initialized
try:
    firestore_manager = FirestoreManager() # Initialize FirestoreManager after Firebase app is ready
    initialize_analytics(firestore_manager)
except Exception as e:
    logger.critical(f"Failed to initialize analytics tracker: {e}")
    exit(1)


app = FastAPI(
    title="Intelli-Agent Backend",
    description="Backend services for the Intelli-Agent application, providing LLM, tool execution, and user management.",
    version="0.1.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(user_router, prefix="/user", tags=["User Management"])
app.include_router(admin_router, prefix="/admin", tags=["Admin Operations"])
app.include_router(tool_router, prefix="/tool", tags=["Tool Execution"])
app.include_router(integrations_router, prefix="/integrations", tags=["Integrations"])
app.include_router(docs_router, prefix="/docs", tags=["Document Management"])
app.include_router(api_config_router, prefix="/api-config", tags=["API Configuration"])

# OAuth2PasswordBearer for token extraction (used by Depends in routers)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

@app.get("/")
async def read_root():
    return {"message": "Intelli-Agent Backend is running!"}

# You can keep the existing `chat` endpoint or modify it as needed
# Ensure the `chat` endpoint from your original file is integrated here.
# For example, if your original chat endpoint was defined like this:
#
# @app.post("/chat")
# async def chat_endpoint(
#     message: Dict[str, Any],
#     request: Request,
#     current_user: UserProfile = Depends(get_current_user),
#     llm_service: LLMService = Depends(get_llm_service_dependency)
# ):
#     # ... (your existing chat logic) ...
#     pass
#
# You should ensure it's still present in the updated file.
# I'm providing a placeholder if it was removed in the snippet.

# Placeholder for the chat endpoint logic that was likely present in the original main.py
"""
from backend.middleware.auth_middleware import get_current_user
from backend.services.llm_service import LLMService, get_llm_service_dependency

@app.post("/chat")
async def chat_endpoint(
    message: Dict[str, Any],
    request: Request,
    current_user: UserProfile = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service_dependency)
):
    user_id = current_user.user_id # Assuming current_user is a UserProfile object

    try:
        user_query = message.get("query")
        if not user_query:
            raise HTTPException(status_code=400, detail="Query message is required.")

        logger.info(f"Received chat query from user {user_id}: {user_query}")

        response_message = await llm_service.process_user_query_with_agent(
            user_id=user_id,
            user_query=user_query,
            user_profile=current_user,
            chat_history=message.get("chat_history", []),
            llm_temperature=None, # Example: can be passed from frontend
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
"""
