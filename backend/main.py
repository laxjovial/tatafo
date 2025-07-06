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
from utils.date_parser import parse_date_to_yyyymmdd # CORRECTED: Changed from parse_date_string
from utils.user_manager import UserManager

# Import all domain-specific tool classes
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

logger = logging.getLogger(__name__)

# --- Firebase Admin SDK Initialization (Centralized here) ---
# This block is the SOLE place where Firebase Admin SDK is initialized.
# It prioritizes loading credentials from environment variable (FIREBASE_ADMIN_CERT).
# For local development without env var, it falls back to secrets.toml (firebase_admin_cert_json).
db = None
auth_sdk = None
try:
    if not firebase_admin._apps: # Only initialize if not already initialized
        # Attempt to load Firebase Admin SDK credentials from environment variable first
        firebase_admin_cert_env_var = os.environ.get("FIREBASE_ADMIN_CERT")
        
        cred = None
        if firebase_admin_cert_env_var:
            try:
                # The environment variable should contain the JSON string of the service account key
                cred = credentials.Certificate(json.loads(firebase_admin_cert_env_var))
                logger.info("Firebase Admin SDK credentials loaded from FIREBASE_ADMIN_CERT environment variable.")
            except json.JSONDecodeError as e:
                logger.error(f"FIREBASE_ADMIN_CERT environment variable is not a valid JSON string: {e}. Falling back to secrets.toml.")
            except Exception as e:
                logger.error(f"Error loading Firebase Admin SDK credentials from environment variable: {e}. Falling back to secrets.toml.")
        else:
            logger.warning("FIREBASE_ADMIN_CERT environment variable not found. Attempting to load from secrets.toml.")

        # If not loaded from environment variable, try loading from secrets.toml
        if cred is None:
            # Note: config_manager.get_secret will now read directly from .streamlit/secrets.toml
            # due to the changes in config_manager.py
            # The key 'firebase_admin_cert_json' should contain the service account JSON string
            firebase_admin_cert_json_str = config_manager.get_secret('firebase_admin_cert_json')
            if firebase_admin_cert_json_str:
                try:
                    firebase_admin_cert = json.loads(firebase_admin_cert_json_str)
                    cred = credentials.Certificate(firebase_admin_cert)
                    logger.info("Firebase Admin SDK credentials loaded from secrets.toml (firebase_admin_cert_json).")
                except json.JSONDecodeError as e:
                    logger.critical(f"FATAL: Failed to parse firebase_admin_cert_json from secrets.toml: {e}. Ensure it's a valid JSON string with escaped newlines.")
                    raise ValueError(f"Failed to parse firebase_admin_cert_json from secrets.toml: {e}")
                except Exception as e:
                    logger.critical(f"FATAL: Error loading Firebase Admin SDK credentials from secrets.toml: {e}")
                    raise ValueError(f"Error loading Firebase Admin SDK credentials from secrets.toml: {e}")
            else:
                logger.critical("FATAL: Firebase Admin SDK service account key not found in FIREBASE_ADMIN_CERT environment variable or secrets.toml (firebase_admin_cert_json).")
                raise ValueError("Firebase Admin SDK service account key not found.")

        # Initialize the Firebase app
        firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully.")
    else:
        logger.info("Firebase Admin SDK already initialized.")

    db = firestore.client()
    auth_sdk = auth
    logger.info("Firestore client and Auth SDK instances obtained.")

except Exception as e:
    logger.critical(f"FATAL: Failed to initialize Firebase Admin SDK. Application cannot start: {e}", exc_info=True)
    raise # Re-raise to prevent app from starting without Firebase

# Initialize analytics tracker with live Firebase instances
# This uses the projectId from the initialized Firebase app, which comes from the service account key
app_id_for_analytics = firebase_admin.get_app().project_id
initialize_analytics(db, auth_sdk, app_id_for_analytics, "backend_system")
logger.info("Analytics tracker initialized for FastAPI backend.")

# --- Wrapper for cloud_storage_utils module functions ---
class CloudStorageUtilsWrapper:
    def __init__(self, config_manager_instance):
        self.config_manager = config_manager_instance

    async def upload_file_to_gcs(self, *args, **kwargs):
        return await cloud_storage_utils_module.upload_file_to_gcs(*args, **kwargs)
    
    async def download_file_from_gcs(self, *args, **kwargs):
        return await cloud_storage_utils_module.download_file_from_gcs(*args, **kwargs)
    
    async def delete_file_from_gcs(self, *args, **kwargs):
        return await cloud_storage_utils_module.delete_file_from_gcs(*args, **kwargs)
    
    async def read_file_from_gcs_to_bytes(self, *args, **kwargs):
        return await cloud_storage_utils_module.read_file_from_gcs_to_bytes(*args, **kwargs)
    
    def get_gcs_bucket(self):
        return cloud_storage_utils_module.get_gcs_bucket()

    def get_gcs_client(self):
        return cloud_storage_utils_module.get_gcs_client()

# --- Wrapper for vector_utils module functions ---
class VectorUtilsWrapper:
    def __init__(self, firestore_manager_instance, cloud_storage_utils_instance, config_manager_instance):
        self.firestore_manager = firestore_manager_instance
        self.cloud_storage_utils = cloud_storage_utils_instance
        self.config_manager = config_manager_instance

    # Expose the functions from the module through this wrapper
    async def process_uploaded_document(self, *args, **kwargs):
        return await vector_utils_module.process_uploaded_document(*args, **kwargs)
    
    async def query_documents(self, *args, **kwargs):
        return await vector_utils_module.query_documents(*args, **kwargs)
    
    async def delete_vector_store_collection(self, *args, **kwargs):
        return await vector_utils_module.delete_vector_store_collection(*args, **kwargs)
    
    # You might want to expose other functions if they are called directly, e.g.:
    # def get_embedding_model(self, *args, **kwargs):
    #     return vector_utils_module.get_embedding_model(*args, **kwargs)
    # async def get_vector_store(self, *args, **kwargs):
    #     return await vector_utils_module.get_vector_store(*args, **kwargs)


# Initialize managers and tools - NOW PASS db and auth_sdk instances
firestore_manager = FirestoreManager(db_instance=db, auth_instance=auth_sdk)

# Instantiate the wrappers
cloud_storage_utils = CloudStorageUtilsWrapper(config_manager)
vector_utils = VectorUtilsWrapper(firestore_manager, cloud_storage_utils, config_manager) # Pass the wrapper instance

# Instantiate UserManager with all required dependencies
user_manager = UserManager(db=db, auth_sdk=auth_sdk, firestore_manager=firestore_manager, config_manager=config_manager, log_event=log_event)

# Instantiate all domain-specific tool classes
finance_tools = FinanceTools(config_manager, log_event)
crypto_tools = CryptoTools(config_manager, log_event)
medical_tools = MedicalTools(config_manager, log_event)
news_tools = NewsTools(config_manager, log_event)
legal_tools = LegalTools(config_manager, log_event)
education_tools = EducationTools(config_manager, log_event)
entertainment_tools = EntertainmentTools(config_manager, log_event)
weather_tools = WeatherTools(config_manager, log_event)
travel_tools = TravelTools(config_manager, log_event)
sports_tools = SportsTools(config_manager, log_event)


app = FastAPI(
    title="Intelli-Agent Backend",
    description="Backend for the AI assistant, managing user authentication, document storage, and tool execution.",
    version="1.0.0"
)

# Configure CORS middleware
origins = [
    "http://localhost",
    "http://localhost:8501",   # Streamlit default port
    "http://localhost:3000",   # React default port
    config_manager.get("frontend_url", "http://localhost:8501") # Allow configurable frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2PasswordBearer for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Pydantic Models for Request/Response Bodies ---

class UserData(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    username: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class UserProfileUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    # Add other fields that can be updated by user
    # Do NOT include tier or roles here, as they are admin-only

class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(min_length=6)

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    oobCode: str # Out-of-band code from Firebase
    newPassword: str = Field(min_length=6)

class DocumentUploadRequest(BaseModel):
    file_name: str
    file_content_base64: str # Base64 encoded file content
    content_type: str # e.g., "application/pdf", "text/plain"
    user_id: str # The user ID who is uploading

class DocumentQueryRequest(BaseModel):
    query_text: str
    user_id: str
    collection_name: Optional[str] = None # Added for explicit collection querying
    k: int = 5 # Number of top results to return

class LLMGenerateRequest(BaseModel):
    prompt: str
    user_id: str

class UserRoleTierUpdate(BaseModel):
    new_tier: str
    roles: List[str]

class ToolInvocationRequest(BaseModel):
    tool_name: str # e.g., "finance_tools.get_stock_price"
    tool_args: Dict[str, Any]
    user_id: str # User making the request


# Dependency to get the current user based on Firebase ID token
async def get_current_user(request: Request, token: str = Depends(oauth2_scheme)):
    try:
        # Verify the ID token using Firebase Admin SDK
        decoded_token = auth_sdk.verify_id_token(token)
        user_id = decoded_token['uid']
        
        # Fetch user profile and capabilities from Firestore
        # This call now directly uses the user_manager instance
        user_profile_data = await user_manager.get_user_profile_from_firestore(user_id)
        if not user_profile_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User profile not found.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Attach user capabilities to the request state or return it
        # For simplicity, we'll return a dict that includes uid and capabilities
        return {
            "uid": user_id,
            "email": decoded_token.get('email'),
            "username": user_profile_data.get('username'),
            "capabilities": user_profile_data.get('capabilities', {}), # Ensure capabilities are included
            "tier": user_profile_data.get('tier', 'free'),
            "roles": user_profile_data.get('roles', ['user'])
        }
    except auth_sdk.InvalidIdTokenError as e:
        logger.warning(f"Invalid ID token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Error in get_current_user: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication.",
        )

# Dependency to check for admin role
async def get_current_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)):
    user_roles = current_user.get('roles', [])
    if "admin" not in user_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions. Admin access required."
        )
    return current_user

@app.get("/")
async def read_root():
    """Root endpoint for the API."""
    return {"message": "Welcome to Intelli-Agent Backend!"}

@app.post("/register")
async def register_user_endpoint(user_data: UserData):
    """Registers a new user."""
    logger.info(f"Attempting to register user: {user_data.email}")
    result = await user_manager.register_user(user_data.email, user_data.password, user_data.username)
    if result["success"]:
        return {"success": True, "message": "User registered successfully. Please log in."}
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])

@app.post("/login")
async def login_user_endpoint(user_data: UserData):
    """Logs in a user and returns an ID token."""
    logger.info(f"Attempting to log in user: {user_data.email}")
    result = await user_manager.login_user(user_data.email, user_data.password)
    if result["success"]:
        return {"success": True, "id_token": result["id_token"], "user_id": result["user_id"], "message": "Login successful."}
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=result["message"])

# @app.post("/refresh_token") # This endpoint is typically handled client-side by Firebase JS SDK
# async def refresh_id_token(request: Request):
#     """Refreshes the ID token using a refresh token."""
#     refresh_token = request.headers.get("X-Refresh-Token")
#     if not refresh_token:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Refresh token missing.")
    
#     result = await user_manager.refresh_id_token(refresh_token)
#     if result["success"]:
#         return {"success": True, "id_token": result["id_token"], "message": "Token refreshed successfully."}
#     raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=result["message"])

@app.post("/logout")
async def logout_user_endpoint(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Logs out the current user."""
    user_id = current_user['uid']
    result = await user_manager.logout_user(user_id) # This will revoke refresh token
    if result["success"]:
        return {"success": True, "message": "Logout successful."}
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])

@app.get("/user/profile")
async def get_user_profile_endpoint(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Retrieves the profile of the current authenticated user."""
    user_id = current_user['uid']
    logger.info(f"Fetching profile for user: {user_id}")
    profile_data = await user_manager.get_user_profile_from_firestore(user_id)
    if profile_data:
        # The user_manager.get_user_profile_from_firestore already formats it correctly
        return {"success": True, "profile": profile_data}
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found.")

@app.put("/user/profile")
async def update_user_profile_endpoint(profile_update: UserProfileUpdate, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Updates the profile of the current authenticated user."""
    user_id = current_user['uid']
    logger.info(f"Updating profile for user: {user_id}")
    update_data = profile_update.dict(exclude_unset=True)
    result = await user_manager.update_user_profile_in_firestore(user_id, update_data)
    if result["success"]:
        return {"success": True, "message": "Profile updated successfully."}
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])

@app.post("/user/change_password")
async def change_password_endpoint(password_change: PasswordChange, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Changes the password for the current authenticated user."""
    user_id = current_user['uid']
    # The current_password is for client-side re-authentication. Backend doesn't verify it directly.
    logger.info(f"User {user_id} attempting to change password.")
    result = await user_manager.change_password_auth_sdk(user_id, password_change.new_password)
    if result["success"]:
        return {"success": True, "message": "Password changed successfully."}
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])

@app.post("/user/forgot_password")
async def forgot_password_endpoint(request: ForgotPasswordRequest):
    """Sends a password reset email."""
    logger.info(f"Received forgot password request for: {request.email}")
    result = await user_manager.send_password_reset_email(request.email)
    if result["success"]:
        return {"success": True, "message": "Password reset email sent if account exists."}
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])

@app.post("/user/reset_password")
async def reset_password_endpoint(request: ResetPasswordRequest):
    """Resets password using an out-of-band code."""
    logger.info(f"Received password reset request with oobCode.")
    result = await user_manager.reset_password_with_oob_code(request.oobCode, request.newPassword)
    if result["success"]:
        return {"success": True, "message": "Password reset successfully."}
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])

@app.post("/documents/upload")
async def upload_document_endpoint(doc_request: DocumentUploadRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Uploads a document to cloud storage and processes it for vector search."""
    user_id = current_user['uid']
    if user_id != doc_request.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User ID mismatch.")
    
    # Check user capabilities for document upload
    user_capabilities = current_user.get('capabilities', {})
    if not user_capabilities.get('document_upload_enabled', False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Document upload not enabled for your account tier.")

    logger.info(f"User {user_id} attempting to upload document: {doc_request.file_name}")
    try:
        # Call the process_uploaded_document function from the vector_utils_module
        result = await vector_utils.process_uploaded_document( # Call from the wrapper instance
            doc_request.file_name,
            doc_request.file_content_base64,
            doc_request.content_type,
            user_id
        )
        if result["success"]:
            return {"success": True, "message": result["message"]}
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    except Exception as e:
        logger.error(f"Error processing document upload for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process document: {e}")

@app.post("/documents/query")
async def query_documents_endpoint(query_request: DocumentQueryRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Queries uploaded documents for relevant information."""
    user_id = current_user['uid']
    if user_id != query_request.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User ID mismatch.")

    # Check user capabilities for document query
    user_capabilities = current_user.get('capabilities', {})
    if not user_capabilities.get('document_query_enabled', False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Document querying not enabled for your account tier.")
    
    # Use the max_results_k from user capabilities, or the requested k if it's lower
    max_k_allowed = user_capabilities.get('document_query_max_results_k', 3) # Default to 3 if not set
    k_to_use = min(query_request.k, max_k_allowed)

    logger.info(f"User {user_id} querying documents with query: '{query_request.query_text}' (k={k_to_use})")
    try:
        # Call the query_documents function from the vector_utils_module
        results = await vector_utils.query_documents( # Call from the wrapper instance
            query_request.query_text,
            user_id,
            collection_name=query_request.collection_name, # Pass collection_name
            k=k_to_use
        )
        return {"success": True, "results": results}
    except Exception as e:
        logger.error(f"Error querying documents for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to query documents: {e}")

@app.post("/llm/generate")
async def generate_llm_response_endpoint(llm_request: LLMGenerateRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Generates a response from the LLM, potentially using tools."""
    user_id = current_user['uid']
    if user_id != llm_request.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User ID mismatch.")

    # Check user capabilities for LLM access
    user_capabilities = current_user.get('capabilities', {})
    if not user_capabilities.get('llm_access_enabled', False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="LLM access not enabled for your account tier.")

    logger.info(f"User {user_id} requesting LLM generation for prompt: '{llm_request.prompt}'")
    
    # For now, a simple direct call to a mock LLM.
    # In a real scenario, this would involve calling the actual LLM (e.g., Gemini API)
    # and potentially using an agent to decide on tool usage based on the prompt.
    
    # Mock LLM response (replace with actual LLM call)
    mock_llm_response = f"This is a mock AI response to your query: '{llm_request.prompt}'. " \
                        f"In a real scenario, I would use advanced tools and an LLM to answer this."

    # Example of tool usage logic (conceptual, not fully implemented here)
    # This section is for demonstration of how tools *could* be called from the LLM endpoint
    # if an agent were integrated here. For direct tool invocation, use the /tools/invoke endpoint.
    if "stock price" in llm_request.prompt.lower():
        # This would typically be decided by an agent, not a simple keyword match
        try:
            # Example: Parse date from prompt and call finance tool
            date_str = "today" # Or parse from prompt
            symbol = "GOOG" # Or parse from prompt
            stock_data = await finance_tools.get_stock_price(symbol, date_str, user_id)
            mock_llm_response = f"The stock price for {symbol} on {date_str} is: {stock_data.get('price', 'N/A')}. (Powered by Finance Tool)"
            asyncio.create_task(log_event('tool_usage', {'tool': 'finance_tools.get_stock_price', 'symbol': symbol, 'date': date_str, 'status': 'success'}, user_id=user_id, success=True))
        except Exception as e:
            mock_llm_response = f"Sorry, I couldn't get the stock price. Error: {e}"
            asyncio.create_task(log_event('tool_usage', {'tool': 'finance_tools.get_stock_price', 'status': 'failure', 'error': str(e)}, user_id=user_id, success=False, error_message=str(e)))
    elif "weather" in llm_request.prompt.lower():
        try:
            location = "London" # Or parse from prompt
            weather_data = await weather_tools.get_current_weather(location, user_id)
            # Weather tool returns a string, so we use it directly
            mock_llm_response = f"The current weather in {location} is: {weather_data}. (Powered by Weather Tool)"
            asyncio.create_task(log_event('tool_usage', {'tool': 'weather_tools.get_current_weather', 'location': location, 'status': 'success'}, user_id=user_id, success=True))
        except Exception as e:
            mock_llm_response = f"Sorry, I couldn't get the weather. Error: {e}"
            asyncio.create_task(log_event('tool_usage', {'tool': 'weather_tools.get_current_weather', 'status': 'failure', 'error': str(e)}, user_id=user_id, success=False, error_message=str(e)))
    # Add more tool examples here...

    # In a full implementation, you would use an LLM to decide which tool to use,
    # call the tool, get the result, and then feed the result back to the LLM
    # to generate a natural language response.

    return {"success": True, "response": mock_llm_response}

# --- Generic Tool Invocation Endpoint ---
@app.post("/tools/invoke")
async def invoke_tool_endpoint(
    request: ToolInvocationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Invokes a specified tool with given arguments.
    Performs RBAC check based on the tool's domain access.
    """
    user_id = current_user['uid']
    if user_id != request.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User ID mismatch.")

    tool_name_parts = request.tool_name.split('.')
    if len(tool_name_parts) < 2:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid tool name format (e.g., 'finance_tools.get_stock_price').")
    
    domain_name = tool_name_parts[0].replace('_tools', '') # e.g., 'finance' from 'finance_tools'
    tool_function_name = tool_name_parts[1]

    # Map domain names to tool instances
    domain_tool_instances = {
        "finance": finance_tools,
        "crypto": crypto_tools,
        "medical": medical_tools,
        "news": news_tools,
        "legal": legal_tools,
        "education": education_tools,
        "entertainment": entertainment_tools,
        "weather": weather_tools,
        "travel": travel_tools,
        "sports": sports_tools,
    }

    tool_instance = domain_tool_instances.get(domain_name)
    if not tool_instance:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tool domain '{domain_name}' not found.")

    tool_function = getattr(tool_instance, tool_function_name, None)
    # Ensure the function exists and is callable (and an async function)
    if not tool_function or not callable(tool_function) or not asyncio.iscoroutinefunction(tool_function):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tool function '{tool_function_name}' not found or not callable/async in '{domain_name}_tools'.")

    # RBAC Check for tool access
    # Capability key example: 'finance_tool_access'
    capability_key = f"{domain_name}_tool_access"
    user_capabilities = current_user.get('capabilities', {})
    if not user_capabilities.get(capability_key, False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access to {domain_name} tools not enabled for your account tier."
        )

    logger.info(f"User {user_id} invoking tool: {request.tool_name} with args: {request.tool_args}")
    try:
        # Pass user_id to the tool function for internal logging
        tool_result = await tool_function(**request.tool_args, user_id=user_id)
        return {"success": True, "result": tool_result}
    except HTTPException as e: # Re-raise HTTPExceptions from tool functions
        logger.error(f"HTTPException from tool {request.tool_name}: {e.detail}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Error invoking tool {request.tool_name} for user {user_id}: {e}", exc_info=True)
        # The tool functions themselves should be logging analytics events for success/failure
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error executing tool: {e}")


# --- Admin Endpoints (Requires Admin Role) ---

@app.get("/admin/users")
async def get_all_users_endpoint(current_user: Dict[str, Any] = Depends(get_current_admin_user)):
    """Retrieves all user profiles (admin only)."""
    logger.info(f"Admin user {current_user['uid']} requesting all user profiles.")
    try:
        users = await user_manager.get_all_users() # This gets from Firestore and Firebase Auth
        return {"success": True, "users": users}
    except Exception as e:
        logger.error(f"Error getting all users for admin {current_user['uid']}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve users: {e}")

@app.put("/admin/user/{user_id}/roles_and_tier")
async def update_user_roles_and_tier_endpoint(
    user_id: str,
    update_data: UserRoleTierUpdate,
    current_user: Dict[str, Any] = Depends(get_current_admin_user)
):
    """Updates a specific user's tier and roles (admin only)."""
    if user_id == current_user['uid']:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admins cannot change their own roles or tier via this endpoint for security reasons.")

    logger.info(f"Admin user {current_user['uid']} updating user {user_id}'s tier to {update_data.new_tier} and roles to {update_data.roles}.")
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
    parsed_start_date = parse_date_to_yyyymmdd(start_date) if start_date else None # CORRECTED: Changed function call
    parsed_end_date = parse_date_to_yyyymmdd(end_date) if end_date else None      # CORRECTED: Changed function call

    try:
        events = await firestore_manager.get_analytics_events(
            event_type=event_type,
            user_id=user_id,
            start_date=parsed_start_date,
            end_date=parsed_end_date
        )
        return {"success": True, "events": events}
    except Exception as e:
        logger.error(f"Error retrieving analytics events for admin {current_user['uid']}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve analytics events: {e}")







