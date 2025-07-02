# ui/main_app.py

import streamlit as st
import logging
from utils.user_manager import get_current_user, clear_current_user, _TIER_HIERARCHY # Import _TIER_HIERARCHY
from config.config_manager import config_manager # Import config_manager

# Import all UI apps
from ui import (
    login_app, register_app, forgot_password_app, reset_password_token_app,
    lost_token_app, change_password_app, user_profile_app, admin_dashboard_app,
    ai_assistant_app, medical_ai_assistant_app, legal_ai_assistant_app,
    finance_ai_assistant_app, news_ai_assistant_app, sports_ai_assistant_app,
    weather_ai_assistant_app, entertainment_ai_assistant_app,
    medical_vector_app, medical_vector_query_app,
    legal_vector_app, legal_vector_query_app,
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
                # Add other mock API keys as needed for testing
                self.firebase_config = "{}" # Mock empty config for Firebase if not set
        st.secrets = MockSecrets()
        logger.info("Mocked st.secrets for standalone testing.")
    
    if not config_manager._is_loaded:
        try:
            logger.info("ConfigManager assumed to be initialized by importing. Ensure data/config.yml and other config files exist.")
        except Exception as e:
            st.error(f"Failed to initialize configuration: {e}. Please ensure data/config.yml and .streamlit/secrets.toml are set up correctly.")
            st.stop()

initialize_app_config()


# --- Page Definitions and RBAC Mapping ---
# Use the _TIER_HIERARCHY loaded from utils.user_manager
# This ensures consistency with the backend and user_manager's RBAC logic.

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
    "Legal AI Assistant": {"app": legal_ai_assistant_app, "tier_access": "premium", "roles": ["user", "admin"]},
    "Finance AI Assistant": {"app": finance_ai_assistant_app, "tier_access": "pro", "roles": ["user", "admin"]},
    "News AI Assistant": {"app": news_ai_assistant_app, "tier_access": "basic", "roles": ["user", "admin"]},
    "Sports AI Assistant": {"app": sports_ai_assistant_app, "tier_access": "basic", "roles": ["user", "admin"]},
    "Weather AI Assistant": {"app": weather_ai_assistant_app, "tier_access": "basic", "roles": ["user", "admin"]},
    "Entertainment AI Assistant": {"app": entertainment_ai_assistant_app, "tier_access": "basic", "roles": ["user", "admin"]},
    "---": {"app": None, "tier_access": "free", "roles": ["any"]}, # Separator
    "Upload Medical Docs": {"app": medical_vector_app, "tier_access": "premium", "roles": ["user", "admin"]},
    "Query Uploaded Medical Docs": {"app": medical_vector_query_app, "tier_access": "premium", "roles": ["user", "admin"]},
    "Upload Legal Docs": {"app": legal_vector_app, "tier_access": "premium", "roles": ["user", "admin"]},
    "Query Uploaded Legal Docs": {"app": legal_vector_query_app, "tier_access": "premium", "roles": ["user", "admin"]},
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
    current_user = get_current_user()
    logged_in = bool(current_user and current_user.get('user_id'))
    user_tier = current_user.get('tier', 'free')
    user_roles = current_user.get('roles', [])

    # Sidebar navigation
    st.sidebar.title("Navigation")

    # Display user info in sidebar if logged in
    if logged_in:
        st.sidebar.write(f"**Welcome, {current_user.get('username', 'User')}!**")
        st.sidebar.info(f"Tier: {user_tier.capitalize()} | Roles: {', '.join(user_roles)}")
        if st.sidebar.button("Logout"):
            clear_current_user()
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
        st.session_state.current_page = "Login" # Redirect to login on error
        st.rerun()

if __name__ == "__main__":
    main()
