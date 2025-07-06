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
from domain_tools.document_tools import DocumentTools # Import the DocumentTools class

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app setup
app = FastAPI(
    title="Tatafo Backend API",
    description="Backend API for Tatafo Assistant, providing various domain-specific tools and user management.",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Firebase Admin SDK Initialization ---
# This should be initialized only once when the application starts
firebase_app = None
firestore_manager = None
user_manager = None
cloud_storage_utils = None
vector_utils = None

@app.on_event("startup")
async def startup_event():
    global firebase_app, firestore_manager, user_manager, cloud_storage_utils, vector_utils

    # Load Firebase Admin SDK credentials from environment variable
    firebase_admin_cert_json = os.environ.get("FIREBASE_ADMIN_CERT_JSON")
    if not firebase_admin_cert_json:
        logger.error("FIREBASE_ADMIN_CERT_JSON environment variable not set.")
        raise ValueError("Firebase Admin SDK credentials not found.")

    try:
        cred = credentials.Certificate(json.loads(firebase_admin_cert_json))
        firebase_app = firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully.")

        # Initialize Firestore Manager
        db = firestore.client(firebase_app)
        firestore_manager = FirestoreManager(db)
        logger.info("FirestoreManager initialized.")

        # Initialize Analytics Tracker with the Firestore DB instance
        initialize_analytics(db, auth, config_manager.get("app_id", "default-backend-app-id"))
        logger.info("Analytics Tracker initialized.")

        # Initialize UserManager with FirestoreManager
        user_manager = UserManager(firestore_manager)
        logger.info("UserManager initialized.")

        # Initialize CloudStorageUtilsWrapper
        gcs_bucket_name = config_manager.get_secret("gcs_bucket_name")
        if not gcs_bucket_name or gcs_bucket_name == "your-gcs-bucket-name-here":
            logger.warning("GCS_BUCKET_NAME not set or is default. Cloud storage features may not work.")
        cloud_storage_utils = cloud_storage_utils_module.CloudStorageUtilsWrapper(gcs_bucket_name)
        logger.info("CloudStorageUtilsWrapper initialized.")

        # Initialize VectorUtilsWrapper
        vector_utils = vector_utils_module.VectorUtilsWrapper(
            firestore_manager=firestore_manager,
            cloud_storage_utils=cloud_storage_utils,
            config_manager=config_manager
        )
        logger.info("VectorUtilsWrapper initialized.")

    except Exception as e:
        logger.critical(f"Failed to initialize Firebase Admin SDK or related services: {e}")
        raise

# OAuth2PasswordBearer for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    Authenticates the user based on the Firebase ID token.
    """
    try:
        # Verify the ID token using the Firebase Admin SDK
        # Ensure firebase_app is initialized before calling auth.verify_id_token
        if not firebase_app:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Firebase app not initialized."
            )
        decoded_token = auth.verify_id_token(token, app=firebase_app)
        # Add the user's tier and capabilities to the decoded token
        user_id = decoded_token.get("uid")
        if user_id and user_manager:
            user_data = await user_manager.get_user_data(user_id)
            if user_data:
                decoded_token["tier"] = user_data.get("tier", "free")
                decoded_token["roles"] = user_data.get("roles", ["user"])
                decoded_token["capabilities"] = await user_manager.get_user_capabilities(user_id)
            else:
                # If user data not found, assign default free tier capabilities
                logger.warning(f"User data not found for {user_id}, assigning default free tier capabilities.")
                decoded_token["tier"] = "free"
                decoded_token["roles"] = ["user"]
                decoded_token["capabilities"] = await user_manager.get_user_capabilities(user_id, default_to_free=True)

        return decoded_token
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Dependency to check if the current user has 'admin' role.
    """
    if "admin" not in current_user.get("roles", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operation forbidden: Not an admin user."
        )
    return current_user

# --- Initialize Domain Tools ---
# These instances will be created once and reused.
# Pass the necessary managers and dependencies to their constructors.
finance_tools = FinanceTools(config_manager=config_manager)
crypto_tools = CryptoTools(config_manager=config_manager)
medical_tools = MedicalTools(config_manager=config_manager)
news_tools = NewsTools(config_manager=config_manager)
legal_tools = LegalTools(config_manager=config_manager)
education_tools = EducationTools(config_manager=config_manager)
entertainment_tools = EntertainmentTools(config_manager=config_manager)
weather_tools = WeatherTools(config_manager=config_manager)
travel_tools = TravelTools(config_manager=config_manager)
sports_tools = SportsTools(config_manager=config_manager)

# DocumentTools requires more dependencies
document_tools = None # Initialize as None, set in startup
@app.on_event("startup")
async def init_document_tools():
    global document_tools
    # Ensure all required global variables are initialized
    if not all([vector_utils, config_manager, firestore_manager, cloud_storage_utils, log_event]):
        logger.error("Dependencies for DocumentTools not fully initialized at startup.")
        # Depending on criticality, you might want to raise an exception here
        # or handle it more gracefully if the app can function partially.
        return

    document_tools = DocumentTools(
        vector_utils_wrapper=vector_utils,
        config_manager=config_manager,
        firestore_manager=firestore_manager,
        cloud_storage_utils=cloud_storage_utils,
        log_event_func=log_event
    )
    logger.info("DocumentTools initialized.")


# --- API Endpoints ---

@app.get("/")
async def read_root():
    return {"message": "Welcome to Tatafo Backend API!"}

# --- User Management Endpoints (Admin Only) ---

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    display_name: Optional[str] = None
    roles: Optional[List[str]] = ["user"]
    tier: Optional[str] = "free"

class UserUpdate(BaseModel):
    new_tier: Optional[str] = None
    roles: Optional[List[str]] = None

@app.post("/admin/users", status_code=status.HTTP_201_CREATED)
async def create_user_endpoint(user_data: UserCreate, current_user: Dict[str, Any] = Depends(get_current_admin_user)):
    """Creates a new user (admin only)."""
    logger.info(f"Admin user {current_user['uid']} attempting to create user: {user_data.email}")
    try:
        user_record = await user_manager.create_user(
            email=user_data.email,
            password=user_data.password,
            display_name=user_data.display_name,
            roles=user_data.roles,
            tier=user_data.tier
        )
        return {"message": "User created successfully", "uid": user_record.uid, "email": user_record.email}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating user: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during user creation.")

@app.get("/admin/users/{user_id}")
async def get_user_endpoint(user_id: str, current_user: Dict[str, Any] = Depends(get_current_admin_user)):
    """Retrieves user data by ID (admin only)."""
    logger.info(f"Admin user {current_user['uid']} requesting user data for {user_id}")
    user_data = await user_manager.get_user_data(user_id)
    if user_data:
        return user_data
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

@app.get("/admin/users")
async def list_users_endpoint(current_user: Dict[str, Any] = Depends(get_current_admin_user)):
    """Lists all users (admin only)."""
    logger.info(f"Admin user {current_user['uid']} requesting list of all users.")
    users = await user_manager.list_users()
    return users

@app.put("/admin/users/{user_id}")
async def update_user_endpoint(user_id: str, update_data: UserUpdate, current_user: Dict[str, Any] = Depends(get_current_admin_user)):
    """Updates user roles and/or tier (admin only)."""
    logger.info(f"Admin user {current_user['uid']} updating user {user_id} with data: {update_data.dict()}")
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
        return events
    except Exception as e:
        logger.error(f"Error retrieving analytics events: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve analytics events.")


# --- Tool Endpoints (Accessed by authenticated users) ---

@app.get("/tools")
async def list_available_tools(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Lists all tools available to the current user based on their capabilities."""
    available_tools = []
    user_capabilities = current_user.get("capabilities", {})

    # Helper to check if a capability is enabled
    def is_enabled(capability_key):
        return user_capabilities.get(capability_key, False)

    # Finance Tools
    if is_enabled("finance_tool_access"):
        available_tools.extend([
            {"name": "finance_get_stock_price", "description": "Get real-time stock price."},
            {"name": "finance_get_company_overview", "description": "Get company overview."},
            {"name": "finance_get_income_statement", "description": "Get company income statement."},
            {"name": "finance_get_balance_sheet", "description": "Get company balance sheet."},
            {"name": "finance_get_cash_flow", "description": "Get company cash flow."},
            {"name": "finance_get_exchange_rate", "description": "Get exchange rate between two currencies."},
        ])
        if is_enabled("historical_data_access"):
            available_tools.append({"name": "finance_get_historical_stock_data", "description": "Get historical stock data."})

    # Crypto Tools
    if is_enabled("crypto_tool_access"):
        available_tools.extend([
            {"name": "crypto_get_crypto_price", "description": "Get real-time cryptocurrency price."},
            {"name": "crypto_get_trending_cryptos", "description": "Get a list of trending cryptocurrencies."},
            {"name": "crypto_search_crypto_news", "description": "Search for cryptocurrency news."},
            {"name": "crypto_summarize_document_by_path", "description": "Summarize a crypto-related document by path."},
            {"name": "crypto_search_web", "description": "Search the web for crypto information."}
        ])

    # Medical Tools
    if is_enabled("medical_tool_access"):
        available_tools.extend([
            {"name": "medical_search_condition", "description": "Search for information about a medical condition."},
            {"name": "medical_search_drug", "description": "Search for information about a drug."},
            {"name": "medical_search_symptom", "description": "Search for information about a medical symptom."},
            {"name": "medical_summarize_document_by_path", "description": "Summarize a medical document by path."},
            {"name": "medical_search_web", "description": "Search the web for medical information."}
        ])

    # News Tools
    if is_enabled("news_tool_access"):
        available_tools.extend([
            {"name": "news_get_top_headlines", "description": "Get top news headlines."},
            {"name": "news_search_news", "description": "Search for news articles by keyword."},
            {"name": "news_get_news_by_category", "description": "Get news articles by category."},
            {"name": "news_summarize_document_by_path", "description": "Summarize a news-related document by path."},
            {"name": "news_search_web", "description": "Search the web for general news information."}
        ])

    # Legal Tools
    if is_enabled("legal_tool_access"):
        available_tools.extend([
            {"name": "legal_search_case_law", "description": "Search for case law information."},
            {"name": "legal_search_statute", "description": "Search for statute information."},
            {"name": "legal_summarize_document_by_path", "description": "Summarize a legal document by path."},
            {"name": "legal_search_web", "description": "Search the web for general legal information."}
        ])

    # Education Tools
    if is_enabled("education_tool_access"):
        available_tools.extend([
            {"name": "education_search_course", "description": "Search for educational courses."},
            {"name": "education_search_institution", "description": "Search for educational institutions."},
            {"name": "education_summarize_document_by_path", "description": "Summarize an educational document by path."},
            {"name": "education_search_web", "description": "Search the web for general education information."}
        ])

    # Entertainment Tools
    if is_enabled("entertainment_tool_access"):
        available_tools.extend([
            {"name": "entertainment_search_movie", "description": "Search for movie information."},
            {"name": "entertainment_search_tv_show", "description": "Search for TV show information."},
            {"name": "entertainment_summarize_document_by_path", "description": "Summarize an entertainment-related document by path."},
            {"name": "entertainment_search_web", "description": "Search the web for general entertainment information."}
        ])

    # Weather Tools
    if is_enabled("weather_tool_access"):
        available_tools.extend([
            {"name": "weather_get_current_weather", "description": "Get current weather conditions for a location."},
            {"name": "weather_get_forecast", "description": "Get weather forecast for a location."},
            {"name": "weather_summarize_document_by_path", "description": "Summarize a weather-related document by path."},
            {"name": "weather_search_web", "description": "Search the web for general weather information."}
        ])

    # Travel Tools
    if is_enabled("travel_tool_access"):
        available_tools.extend([
            {"name": "travel_search_flight", "description": "Search for flight information."},
            {"name": "travel_search_hotel", "description": "Search for hotel information."},
            {"name": "travel_summarize_document_by_path", "description": "Summarize a travel-related document by path."},
            {"name": "travel_search_web", "description": "Search the web for general travel information."}
        ])

    # Sports Tools
    if is_enabled("sports_tool_access"):
        available_tools.extend([
            {"name": "sports_get_latest_scores", "description": "Get latest sports scores for a league/sport."},
            {"name": "sports_get_team_info", "description": "Get information about a sports team."},
            {"name": "sports_get_player_stats", "description": "Get statistics for a sports player."},
            {"name": "sports_summarize_document_by_path", "description": "Summarize a sports-related document by path."},
            {"name": "sports_search_web", "description": "Search the web for general sports information."}
        ])

    # Document Tools
    if is_enabled("document_query_enabled"):
        available_tools.append({"name": "document_query_uploaded_docs", "description": "Query previously uploaded and indexed documents."})
    if is_enabled("document_upload_enabled"):
        available_tools.append({"name": "document_process_uploaded_document", "description": "Upload and process a document for indexing."})
    if is_enabled("summarization_enabled"): # This is a general capability, but document_summarize_document_by_path uses it
         # Only add if document_tools is initialized, as it's a method of that class
        if document_tools:
            available_tools.append({"name": "document_summarize_document_by_path", "description": "Summarize a document by its file path."})
    if is_enabled("web_search_enabled"): # General web search, but document_search_web uses it
        if document_tools:
            available_tools.append({"name": "document_search_web", "description": "Search the web for document-related information."})


    return {"available_tools": available_tools}


@app.post("/tools/{tool_name}")
async def use_tool(tool_name: str, request: Request, current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Generic endpoint to use any available tool.
    The tool's arguments are passed in the request body as JSON.
    """
    user_token = current_user.get("uid", "default")
    request_data = await request.json()
    logger.info(f"User {user_token} attempting to use tool: {tool_name} with params: {request_data}")

    # Add user_token to request_data for RBAC checks within tools
    request_data['user_token'] = user_token

    try:
        # Dynamically call the tool based on tool_name
        # This requires careful mapping and security considerations
        tool_function = None
        tool_instance = None

        # Mapping tool_name to actual function and instance
        if tool_name.startswith("finance_"):
            tool_instance = finance_tools
        elif tool_name.startswith("crypto_"):
            tool_instance = crypto_tools
        elif tool_name.startswith("medical_"):
            tool_instance = medical_tools
        elif tool_name.startswith("news_"):
            tool_instance = news_tools
        elif tool_name.startswith("legal_"):
            tool_instance = legal_tools
        elif tool_name.startswith("education_"):
            tool_instance = education_tools
        elif tool_name.startswith("entertainment_"):
            tool_instance = entertainment_tools
        elif tool_name.startswith("weather_"):
            tool_instance = weather_tools
        elif tool_name.startswith("travel_"):
            tool_instance = travel_tools
        elif tool_name.startswith("sports_"):
            tool_instance = sports_tools
        elif tool_name.startswith("document_"):
            tool_instance = document_tools
        
        if tool_instance:
            tool_function = getattr(tool_instance, tool_name, None)

        if tool_function:
            # Check if the user has the capability to use this specific tool
            # This is a redundant check if the list_available_tools is accurate,
            # but adds an extra layer of security.
            # A more robust solution would map tool_name to its required capability.
            # For now, we rely on the in-tool RBAC checks.

            result = await tool_function(**request_data)
            return {"tool_name": tool_name, "result": result}
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tool '{tool_name}' not found or not accessible.")

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
        # Log the error with analytics
        await log_event(
            user_id=user_token,
            event_type="tool_execution_error",
            event_details={"tool_name": tool_name, "error": str(e), "params": request_data},
            success=False
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error executing tool: {e}")

