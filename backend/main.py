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
from backend.services.api_usage_service import ApiUsageService # Import ApiUsageService

# Import Pydantic models from backend.models
from backend.models.user_models import UserProfile

# Import API routers
from backend.api.auth_api import router as auth_router
from backend.api.user_api import router as user_router
from backend.api.admin_api import router as admin_router
from backend.api.tool_api import router as tool_router
from backend.api.integrations_api import router as integrations_router
from backend.api.docs_api import router as docs_router

# Import dependency functions directly from auth_middleware
from backend.middleware.auth_middleware import (
    get_firestore_manager_dependency,
    get_user_manager_dependency,
    get_api_usage_service_dependency
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Firebase Admin SDK Initialization ---
try:
    firebase_app = None
    firebase_credentials_json = os.getenv("FIREBASE_CREDENTIALS")

    if firebase_credentials_json:
        try:
            cred_dict = json.loads(firebase_credentials_json)
            cred = credentials.Certificate(cred_dict)
            firebase_app = firebase_admin.initialize_app(cred)  # Default app
            logger.info("Firebase Admin SDK initialized using FIREBASE_CREDENTIALS env var.")
        except json.JSONDecodeError as e:
            logger.error(f"FATAL: FIREBASE_CREDENTIALS env var contains invalid JSON: {e}", exc_info=True)
            raise ValueError("Invalid JSON in FIREBASE_CREDENTIALS environment variable.") from e
        except Exception as e:
            logger.error(f"FATAL: Error initializing Firebase Admin SDK from env var: {e}", exc_info=True)
            raise
    else:
        logger.warning("FIREBASE_CREDENTIALS env var not set. Attempting to load from secrets.toml.")
        firebase_admin_cert_from_secrets = config_manager.get_secret("firebase_admin_cert")

        if firebase_admin_cert_from_secrets:
            logger.info("Initializing Firebase from 'firebase_admin_cert' in secrets.toml.")
            try:
                if isinstance(firebase_admin_cert_from_secrets, str):
                    cred_dict = json.loads(firebase_admin_cert_from_secrets)
                    logger.info("Parsed 'firebase_admin_cert' from string.")
                else:
                    cred_dict = firebase_admin_cert_from_secrets
                    logger.info("'firebase_admin_cert' is already a dict.")

                cred = credentials.Certificate(cred_dict)
                firebase_app = firebase_admin.initialize_app(cred)  # Default app
                logger.info("Firebase Admin SDK initialized using secrets.toml.")
            except json.JSONDecodeError as e:
                logger.error(f"FATAL: 'firebase_admin_cert' in secrets.toml contains invalid JSON: {e}", exc_info=True)
                raise ValueError("Invalid JSON in 'firebase_admin_cert' in secrets.toml.") from e
            except Exception as e:
                logger.error(f"FATAL: Error initializing Firebase from secrets.toml: {e}", exc_info=True)
                raise
        else:
            logger.error("FATAL: Neither FIREBASE_CREDENTIALS nor 'firebase_admin_cert' found.")
            raise ValueError("Firebase Admin SDK credentials not configured.")

except Exception as e:
    logger.critical(f"Application will not start due to Firebase initialization failure: {e}")
    exit(1)

# Initialize FirestoreManager and other services AFTER Firebase Admin SDK is initialized
try:
    if firebase_app:
        firestore_db_client = firestore.client()
        firestore_manager = FirestoreManager(db_instance=firestore_db_client, auth_instance=auth) # Pass the actual Firestore client and auth instance
        user_manager = UserManager(firestore_manager=firestore_manager) # Instantiate UserManager
        api_usage_service = ApiUsageService(firestore_manager=firestore_manager) # Instantiate ApiUsageService

        # Pass the Firestore client and Firebase Auth instance for analytics
        current_app_id = firebase_app.name # Or a more specific app ID if you have one
        current_user_id_for_analytics = "backend_system_user" # A default user ID for backend-initiated logs

        initialize_analytics(firestore_manager._db, auth, current_app_id, current_user_id_for_analytics)
        logger.info("Analytics tracker initialized successfully.")
    else:
        logger.critical("Firebase app not initialized. Skipping analytics tracker.")
        exit(1)
except Exception as e:
    logger.critical(f"Failed to initialize analytics tracker: {e}")
    exit(1)


app = FastAPI(
    title="Intelli-Agent Backend",
    description="Backend services for the Intelli-Agent application, providing LLM, tool execution, and user management.",
    version="0.1.0",
)

# --- Dependency Overrides ---
# Override the dependency functions defined in auth_middleware.py
app.dependency_overrides[get_firestore_manager_dependency] = lambda: firestore_manager
app.dependency_overrides[get_user_manager_dependency] = lambda: user_manager
app.dependency_overrides[get_api_usage_service_dependency] = lambda: api_usage_service


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(user_router, prefix="/user", tags=["User Management"])
app.include_router(admin_router, prefix="/admin", tags=["Admin Operations"])
app.include_router(tool_router, prefix="/tool", tags=["Tool Execution"])
app.include_router(integrations_router, prefix="/integrations", tags=["Integrations"])
app.include_router(docs_router, prefix="/docs", tags=["Document Management"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

@app.get("/")
async def read_root():
    return {"message": "Intelli-Agent Backend is running!"}

# Optional placeholder for the /chat endpoint
# from backend.middleware.auth_middleware import get_current_user
# from backend.services.llm_service import LLMService, get_llm_service_dependency

# @app.post("/chat")
# async def chat_endpoint(
#     message: Dict[str, Any],
#     request: Request,
#     current_user: UserProfile = Depends(get_current_user),
#     llm_service: LLMService = Depends(get_llm_service_dependency)
# ):
#     user_id = current_user.user_id

#     try:
#         user_query = message.get("query")
#         if not user_query:
#             raise HTTPException(status_code=400, detail="Query message is required.")

#         logger.info(f"Received chat query from user {user_id}: {user_query}")

#         response_message = await llm_service.process_user_query_with_agent(
#             user_id=user_id,
#             user_query=user_query,
#             user_profile=current_user,
#             chat_history=message.get("chat_history", []),
#             llm_temperature=None,
#             user_provided_llm_provider=None,
#             user_provided_model_name=None
#         )
#     except HTTPException as e:
#         logger.error(f"HTTPException during chat: {e.detail}", exc_info=True)
#         await log_event('chat_response_failure', {'message': message, 'error': e.detail},
#                         user_id=user_id, success=False, error_message=e.detail, log_from_backend=True)
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error during chat: {e}", exc_info=True)
#         response_message = f"An error occurred: {e}"
#         await log_event('chat_response_failure', {'message': message, 'error': str(e)},
#                         user_id=user_id, success=False, error_message=str(e), log_from_backend=True)
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

#     await log_event('chat_response', {'response': response_message},
#                     user_id=user_id, success=True, log_from_backend=True)
#     return {"response": response_message}
