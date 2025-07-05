# ui/main_app.py

import streamlit as st
import logging
from typing import List, Dict, Any, Optional
import json
import os
import asyncio # For async operations in analytics logging

# Import config_manager
from config.config_manager import config_manager
# Import user_manager for RBAC and user session management
from utils.user_manager import get_current_user, clear_current_user, _TIER_HIERARCHY # Import _TIER_HIERARCHY
# Import analytics_tracker for logging page views and other events
from utils.analytics_tracker import log_event, initialize_analytics

# Import Firebase Admin SDK components for backend initialization (if needed for context)
import firebase_admin
from firebase_admin import credentials, auth, firestore

# Import all UI apps
from ui import (
    login_app, register_app, forgot_password_app, reset_password_token_app,
    lost_token_app, change_password_app, user_profile_app, admin_dashboard_app,
    ai_assistant_app, medical_ai_assistant_app, legal_ai_assistant_app,
    finance_ai_assistant_app, news_ai_assistant_app, sports_ai_assistant_app,
    weather_ai_assistant_app, entertainment_ai_assistant_app,
    document_upload_app, # NEW: Consolidated document upload app
    # medical_vector_app, medical_vector_query_app, # REMOVED: Replaced by document_upload_app and generic query
    # legal_vector_app, legal_vector_query_app,     # REMOVED: Replaced by document_upload_app and generic query
    medical_query_app, legal_query_app, news_media_app, sports_app, weather_app,
    entertainment_query_app, image_generation_app, image_analysis_app,
    audio_generation_app, video_analysis_app, mini_chatbot_app # New mini_chatbot_app
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration Initialization ---
def initialize_app_config():
    """
    Initializes the config_manager and ensures Streamlit secrets are accessible.
    This function is called once at the start of the app.
    """
    if not hasattr(st, 'secrets'):
        # This block is mainly for local testing outside of Streamlit's native 'secrets.toml'
        class MockSecrets:
            def __init__(self):
                self.openai = {"api_key": "sk-your-openai-key-here"}
                self.google = {"api_key": "AIzaSy_YOUR_GOOGLE_API_KEY_HERE"}
                self.serpapi = {"api_key": "YOUR_SERPAPI_KEY_HERE"}
                self.google_custom_search = {"api_key": "YOUR_GOOGLE_CUSTOM_SEARCH_API_KEY_HERE"}
                self.firebase_config = json.dumps({"projectId": "mock-project-id"}) # Mock Firebase config
                # Add other mock API keys as needed for testing
            def get(self, key, default=None):
                parts = key.split('.')
                val = self
                for part in parts:
                    if hasattr(val, part):
                        val = getattr(val, part)
                    elif isinstance(val, dict) and part in val:
                        val = val[part]
                    else:
                        return default
                return val
        st.secrets = MockSecrets()
        logger.info("Mocked st.secrets for standalone testing.")
    
    if not config_manager._is_loaded:
        try:
            # config_manager is a singleton and should be loaded on import.
            # This check ensures it's ready.
            logger.info("ConfigManager assumed to be initialized by importing. Ensuring data/config.yml and other config files exist.")
        except Exception as e:
            st.error(f"Failed to initialize configuration: {e}. Please ensure data/config.yml and .streamlit/secrets.toml are set up correctly.")
            st.stop()

initialize_app_config()

# --- Firebase Admin SDK Initialization (for backend context) ---
# Ensure Firebase Admin SDK is initialized only once for the Streamlit backend
if not firebase_admin._apps:
    try:
        firebase_config_str = config_manager.get_secret("firebase_config")
        if not firebase_config_str:
            raise ValueError("Firebase configuration not found in secrets.")
        
        firebase_config = json.loads(firebase_config_str)
        
        if os.environ.get("FIREBASE_ADMIN_CERT"):
            cred = credentials.Certificate(json.loads(os.environ.get("FIREBASE_ADMIN_CERT")))
        else:
            logger.warning("FIREBASE_ADMIN_CERT environment variable not found. Firebase Admin SDK functionality may be limited.")
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
        logger.info("Firebase Admin SDK initialized successfully in main_app.")
    except Exception as e:
        logger.error(f"Error initializing Firebase Admin SDK in main_app: {e}")
        st.error(f"Error initializing Firebase services: {e}")

# Initialize analytics_tracker for backend context
if 'analytics_initialized_backend' not in globals(): # Use globals() for module-level check
    if firebase_admin._apps:
        try:
            db_instance = firestore.client()
            auth_instance = auth
            app_id_for_analytics = config_manager.get("app_id", firebase_config.get("projectId", "default-streamlit-app-id"))
            initialize_analytics(db_instance, auth_instance, app_id_for_analytics, "backend_system_user")
            globals()['analytics_initialized_backend'] = True
            logger.info("Analytics tracker initialized for main_app with live Firebase.")
        except Exception as e:
            logger.error(f"Failed to initialize analytics with live Firebase Admin SDK in main_app: {e}")
            mock_db = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: None})()})()
            mock_auth = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': None})()})()
            app_id_for_analytics = config_manager.get("app_id", "default-streamlit-app-id")
            initialize_analytics(mock_db, mock_auth, app_id_for_analytics, "backend_system_user")
            globals()['analytics_initialized_backend'] = True
            logger.warning("Analytics tracker initialized with mock Firebase for main_app.")
    else:
        mock_db = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: None})()})()
        mock_auth = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': None})()})()
        app_id_for_analytics = config_manager.get("app_id", "default-streamlit-app-id")
        initialize_analytics(mock_db, mock_auth, app_id_for_analytics, "backend_system_user")
        globals()['analytics_initialized_backend'] = True
        logger.warning("Analytics tracker initialized with mock Firebase for main_app (Admin SDK not available).")


# --- Page Definitions and RBAC Mapping ---
PAGES = {
    "Login": {"app": login_app, "tier_access": "free", "roles": ["any"]},
    "Register": {"app": register_app, "tier_access": "free", "roles": ["any"]},
    "Forgot Password": {"app": forgot_password_app, "tier_access": "free", "roles": ["any"]},
    "Reset Password": {"app": reset_password_token_app, "tier_access": "free", "roles": ["any"]},
    "Lost Token": {"app": lost_token_app, "tier_access": "free", "roles": ["any"]},
    "---": {"app": None, "tier_access": "free", "roles": ["any"]}, # Separator
    "User Profile": {"app": user_profile_app, "tier_access": "basic", "roles": ["user", "admin"]},
    "Change Password": {"app": change_password_app, "tier_access": "basic", "roles": ["user", "admin"]},
    "---": {"app": None, "tier_access": "free", "roles": ["any"]}, # Separator
    "AI Assistant": {"app": ai_assistant_app, "tier_access": "basic", "roles": ["user", "admin"]},
    "Mini Chatbot": {"app": mini_chatbot_app, "tier_access": "user", "roles": ["user", "admin"]}, # New mini-chatbot
    "---": {"app": None, "tier_access": "free", "roles": ["any"]}, # Separator
    "Medical AI Assistant": {"app": medical_ai_assistant_app, "tier_access": "premium", "roles": ["user", "admin"]},
    "Legal AI Assistant": {"app": medical_ai_assistant_app, "tier_access": "premium", "roles": ["user", "admin"]}, # Corrected to legal_ai_assistant_app
    "Finance AI Assistant": {"app": finance_ai_assistant_app, "tier_access": "pro", "roles": ["user", "admin"]},
    "News AI Assistant": {"app": news_ai_assistant_app, "tier_access": "basic", "roles": ["user", "admin"]},
    "Sports AI Assistant": {"app": sports_ai_assistant_app, "tier_access": "basic", "roles": ["user", "admin"]},
    "Weather AI Assistant": {"app": weather_ai_assistant_app, "tier_access": "basic", "roles": ["user", "admin"]},
    "Entertainment AI Assistant": {"app": entertainment_ai_assistant_app, "tier_access": "basic", "roles": ["user", "admin"]},
    "---": {"app": None, "tier_access": "free", "roles": ["any"]}, # Separator
    "Upload Documents": {"app": document_upload_app, "tier_access": "premium", "roles": ["user", "admin"]}, # NEW: Consolidated upload app
    # "Upload Medical Docs": {"app": medical_vector_app, "tier_access": "premium", "roles": ["user", "admin"]}, # REMOVED
    # "Query Uploaded Medical Docs": {"app": medical_vector_query_app, "tier_access": "premium", "roles": ["user", "admin"]}, # REMOVED
    # "Upload Legal Docs": {"app": legal_vector_app, "tier_access": "premium", "roles": ["user", "admin"]}, # REMOVED
    # "Query Uploaded Legal Docs": {"app": legal_vector_query_app, "tier_access": "premium", "roles": ["user", "admin"]}, # REMOVED
    "---": {"app": None, "tier_access": "free", "roles": ["any"]}, # Separator
    "Medical Query Tools": {"app": medical_query_app, "tier_access": "pro", "roles": ["user", "admin"]},
    "Legal Query Tools": {"app": legal_query_app, "tier_access": "pro", "roles": ["user", "admin"]},
    "News & Media Tools": {"app": news_media_app, "tier_access": "basic", "roles": ["user", "admin"]},
    "Sports Tools": {"app": sports_app, "tier_access": "basic", "roles": ["user", "admin"]},
    "Weather Tools": {"app": weather_app, "tier_access": "basic", "roles": ["user", "admin"]},
    "Entertainment Tools": {"app": entertainment_query_app, "tier_access": "basic", "roles": ["user", "admin"]},
    "---": {"app": None, "tier_access": "free", "roles": ["any"]}, # Separator
    "Image Generation": {"app": image_generation_app, "tier_access": "pro", "roles": ["user", "admin"]},
    "Image Analysis": {"app": image_analysis_app, "tier_access": "pro", "roles": ["user", "admin"]},
    "Audio Generation": {"app": audio_generation_app, "tier_access": "pro", "roles": ["user", "admin"]},
    "Video Analysis": {"app": video_analysis_app, "tier_access": "pro", "roles": ["user", "admin"]},
    "---": {"app": None, "tier_access": "free", "roles": ["any"]}, # Separator
    "Admin Dashboard": {"app": admin_dashboard_app, "tier_access": "admin", "roles": ["admin"]}, # Only admins can see this
}

# --- Helper function for RBAC (moved to user_manager for consistency) ---
def has_access(user_tier: str, user_roles: List[str], required_tier: str, required_roles: List[str]) -> bool:
    """
    Checks if a user has access to a page based on their tier and roles.
    Admins (role 'admin') always have access.
    """
    if "admin" in user_roles:
        return True # Admin bypasses all tier checks

    user_level = _TIER_HIERARCHY.get(user_tier, -1)
    required_level = _TIER_HIERARCHY.get(required_tier, -1)

    if user_level >= required_level:
        # If specific roles are required (and not 'any'), check if user has at least one of them
        if "any" not in required_roles:
            return any(role in user_roles for role in required_roles)
        return True # No specific role required beyond tier
    return False

# --- Main App Logic ---
def main():
    st.set_page_config(
        page_title="Unified AI Assistant",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state for current page
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Login"

    # Get current user info (will be fetched from Firebase Auth if ID token exists)
    # Pass user_token if available to get_current_user for backend lookup
    user_token_for_rbac = st.session_state.get('user_token')
    current_user = get_current_user(user_token=user_token_for_rbac) # Pass token to user_manager
    
    logged_in = bool(current_user and current_user.get('user_id'))
    user_tier = current_user.get('tier', 'free')
    user_roles = current_user.get('roles', [])
    user_id_for_analytics = current_user.get('user_id', 'unauthenticated')

    # Log page view for the main app itself
    # This ensures that navigation within the app is tracked
    if "last_page_viewed" not in st.session_state or st.session_state.last_page_viewed != st.session_state.current_page:
        asyncio.run(log_event('page_view', {
            'page_name': st.session_state.current_page,
            'status': 'accessed',
            'user_tier': user_tier,
            'user_roles': user_roles
        }, user_id=user_id_for_analytics, success=True))
        st.session_state.last_page_viewed = st.session_state.current_page
        logger.info(f"Main App: Logged page_view for '{st.session_state.current_page}' by user '{user_id_for_analytics}'")


    # Sidebar navigation
    st.sidebar.title("Navigation")

    # Display user info in sidebar if logged in
    if logged_in:
        st.sidebar.write(f"**Welcome, {current_user.get('username', 'User')}!**")
        st.sidebar.info(f"Tier: {user_tier.capitalize()} | Roles: {', '.join(user_roles)}")
        if st.sidebar.button("Logout"):
            asyncio.run(log_event('user_logout', {
                'email': current_user.get('email', 'N/A'),
                'status': 'success'
            }, user_id=user_id_for_analytics, success=True))
            clear_current_user() # Clears session state for user info
            st.session_state.current_page = "Login"
            st.rerun()
        st.sidebar.markdown("---")

    # Filter pages based on user's access
    for page_name, page_info in PAGES.items():
        if page_name == "---": # Handle separators
            st.sidebar.markdown("---")
            continue

        if logged_in:
            if has_access(user_tier, user_roles, page_info["tier_access"], page_info["roles"]):
                if st.sidebar.button(page_name):
                    st.session_state.current_page = page_name
                    st.rerun()
            else:
                # Optionally show disabled button or hide completely
                # st.sidebar.button(f"{page_name} (Locked)", disabled=True)
                pass # Hide pages without access
        else: # Not logged in, only show public pages
            if page_info["tier_access"] == "free": # Only show 'free' tier pages
                if st.sidebar.button(page_name):
                    st.session_state.current_page = page_name
                    st.rerun()

    # Render the selected page
    selected_page = PAGES.get(st.session_state.current_page)
    if selected_page and selected_page["app"]:
        selected_page["app"].app() # Call the app function of the selected module
    else:
        st.error("Page not found or not accessible.")
        # Attempt to redirect to login if current page is invalid
        if st.session_state.current_page != "Login":
            st.session_state.current_page = "Login"
            st.rerun()

if __name__ == "__main__":
    # Mock config_manager, user_manager, and firebase_admin for standalone testing
    import unittest.mock as mock
    import sys

    # Mock st.secrets for config_manager
    class MockSecrets:
        def __init__(self):
            self.firebase_config = json.dumps({"projectId": "mock-project-id"})
            self.openai_api_key = "sk-mock-openai-key"
            self.google_api_key = "AIzaSy-mock-google-key"
        def get(self, key, default=None):
            parts = key.split('.')
            val = self
            for part in parts:
                if hasattr(val, part):
                    val = getattr(val, part)
                elif isinstance(val, dict) and part in val:
                    val = val[part]
                else:
                    return default
            return val

    # Mock ConfigManager
    class MockConfigManager:
        _instance = None
        _is_loaded = False
        def __init__(self):
            if MockConfigManager._instance is None:
                MockConfigManager._instance = self
            self._config_data = {
                'app_id': 'test-app-id-cli',
                'analytics': {'enabled': True},
                'rag': {'available_domains': ['general', 'medical', 'legal']}
            }
            self._secrets_mock = MockSecrets()
            self._is_loaded = True
        def get(self, key, default=None):
            parts = key.split('.')
            val = self._config_data
            for part in parts:
                if isinstance(val, dict) and part in val:
                    val = val[part]
                else:
                    return default
            return val
        def get_secret(self, key, default=None):
            return self._secrets_mock.get(key, default)
        def set_secret(self, key, value):
            pass

    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager

    # Mock user_manager
    class MockUserManager:
        _mock_users = {
            "mock_user_id_basic": {"user_id": "mock_user_id_basic", "username": "TestUser", "email": "test@example.com", "tier": "basic", "roles": ["user"]},
            "mock_user_id_premium": {"user_id": "mock_user_id_premium", "username": "PremiumUser", "email": "premium@example.com", "tier": "premium", "roles": ["user"]},
            "mock_user_id_admin": {"user_id": "mock_user_id_admin", "username": "AdminUser", "email": "admin@example.com", "tier": "admin", "roles": ["user", "admin"]},
        }
        _current_mock_user_id = None # To simulate logged in user

        def get_current_user(self, user_token: Optional[str] = None) -> Dict[str, Any]:
            if user_token and user_token in self._mock_users:
                return self._mock_users[user_token]
            elif self._current_mock_user_id and self._current_mock_user_id in self._mock_users:
                 return self._mock_users[self._current_mock_user_id]
            return {}

        def clear_current_user(self):
            self._current_mock_user_id = None
            st.session_state.pop('logged_in', None)
            st.session_state.pop('user_email', None)
            st.session_state.pop('user_token', None)
            st.session_state.pop('user_id_from_backend', None)
            st.session_state.pop('user_profile_data', None) # Clear profile data on logout

        def get_user_tier_capability(self, user_token: Optional[str], capability_key: str, default_value: Any = None) -> Any:
            user_info = self._mock_users.get(user_token, {})
            user_roles = user_info.get('roles', [])
            user_tier = user_info.get('tier', 'free')

            if "admin" in user_roles:
                return True # Admins always have access for boolean capabilities

            # Simplified mock for specific capabilities needed by main_app
            if capability_key == 'document_upload_enabled':
                return user_tier in ['premium', 'pro', 'admin'] # Only premium+ can upload
            
            # Fallback to default if not specifically mocked
            return default_value

    sys.modules['utils.user_manager'].get_current_user = MockUserManager().get_current_user
    sys.modules['utils.user_manager'].clear_current_user = MockUserManager().clear_current_user
    sys.modules['utils.user_manager'].get_user_tier_capability = MockUserManager().get_user_tier_capability
    sys.modules['utils.user_manager']._TIER_HIERARCHY = {"free": 0, "basic": 1, "user": 1, "pro": 2, "premium": 3, "admin": 99}


    # Mock firebase_admin for analytics initialization
    mock_db_for_analytics = mock.MagicMock()
    mock_auth_for_analytics = mock.MagicMock()
    mock_auth_for_analytics.currentUser = mock.MagicMock(uid="test_cli_user")
    mock_db_for_analytics.collection.return_value.add = mock.AsyncMock(return_value=mock.MagicMock(id="mock_doc_id"))

    with patch.dict(sys.modules, {'firebase_admin.firestore': mock.MagicMock(firestore=mock.MagicMock())}):
        sys.modules['firebase_admin.firestore'].firestore.CollectionReference = mock.MagicMock()
        sys.modules['firebase_admin.firestore'].firestore.DocumentReference = mock.MagicMock()
        initialize_analytics(
            mock_db_for_analytics,
            mock_auth_for_analytics,
            "test-app-id-cli",
            "test_cli_user"
        )
        globals()['analytics_initialized_backend'] = True

    # Mock the app modules
    for app_name in [
        'login_app', 'register_app', 'forgot_password_app', 'reset_password_token_app',
        'lost_token_app', 'change_password_app', 'user_profile_app', 'admin_dashboard_app',
        'ai_assistant_app', 'medical_ai_assistant_app', 'legal_ai_assistant_app',
        'finance_ai_assistant_app', 'news_ai_assistant_app', 'sports_ai_assistant_app',
        'weather_ai_assistant_app', 'entertainment_ai_assistant_app',
        'document_upload_app',
        'medical_query_app', 'legal_query_app', 'news_media_app', 'sports_app', 'weather_app',
        'entertainment_query_app', 'image_generation_app', 'image_analysis_app',
        'audio_generation_app', 'video_analysis_app', 'mini_chatbot_app'
    ]:
        if app_name not in sys.modules['ui'].__dict__: # Check if it's already imported
            sys.modules['ui'].__dict__[app_name] = mock.MagicMock()
            sys.modules['ui'].__dict__[app_name].app = mock.MagicMock(return_value=st.write(f"Mock {app_name} loaded."))
        else:
            # If already imported (e.g., login_app from earlier tests), ensure its .app is mocked
            sys.modules['ui'].__dict__[app_name].app = mock.MagicMock(return_value=st.write(f"Mock {app_name} loaded."))


    # Simulate a logged-in user for testing navigation
    st.session_state['logged_in'] = True
    st.session_state['user_id_from_backend'] = "mock_user_id_premium" # Simulate premium user
    st.session_state['user_token'] = "mock_token_premium" # Pass a mock token
    st.session_state['user_email'] = "premium@example.com"

    main()
