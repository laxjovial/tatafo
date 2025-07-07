# backend/login_app.py

import streamlit as st
import requests
import json
import os
import logging
import asyncio # For async operations in CLI test
import sys # Import sys
from pathlib import Path # Import Path
from typing import Dict, Any, Optional # <-- Ensure Optional is imported here

# --- Add project root to sys.path ---
# This allows imports like 'from config.config_manager import config_manager' to work
# when running Streamlit apps from nested directories.
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[1] # Go up two levels from login_app.py to the 'tatafo' root
sys.path.insert(0, str(project_root))
# --- End sys.path modification ---

# Import config_manager (if needed for frontend config like backend URL, though we'll hardcode for now)
from config.config_manager import config_manager # Keep for potential future use or if other configs are needed

logger = logging.getLogger(__name__)

# --- Configuration for FastAPI Backend ---
# IMPORTANT: Replace this with your actual Codespace URL or deployed backend URL
# Example: "https://friendly-doodle-x5x6qvv74vr6h655x-8000.app.github.dev/"
FASTAPI_BASE_URL = "https://friendly-doodle-x5x6qvv74vr6h655x-8000.app.github.dev" # Use your actual Codespace URL here!

# --- Frontend-specific Analytics Logging (sends to FastAPI backend) ---
async def frontend_log_event(event_type: str, details: dict, user_id: str = "unauthenticated", success: bool = True, error_message: Optional[str] = None):
    """
    Sends analytics event to the FastAPI backend.
    This replaces direct Firebase Admin SDK logging from the frontend.
    """
    try:
        payload = {
            "event_type": event_type,
            "details": details,
            "user_id": user_id,
            "success": success,
            "error_message": error_message
        }
        # UPDATED: Point to the new unauthenticated analytics endpoint
        response = requests.post(f"{FASTAPI_BASE_URL}/log-frontend-analytics", json=payload)
        response.raise_for_status()
        logger.info(f"Frontend Analytics Logged: {event_type} for user {user_id}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send analytics event to backend: {e}")

def login_user_backend(email: str, password: str) -> Dict[str, Any]:
    """Sends login request to the backend and returns the response."""
    try:
        payload = {
            "email": email,
            "password": password
        }
        headers = {"Content-Type": "application/json"}
        # Corrected endpoint: /login as defined in main.py
        response = requests.post(f"{FASTAPI_BASE_URL}/login", json=payload, headers=headers)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error during login: {e.response.status_code} - {e.response.text}", exc_info=True)
        # Parse error message from backend if available
        try:
            error_detail = e.response.json().get("message", e.response.json().get("detail", str(e))) # Check 'message' first
        except json.JSONDecodeError:
            error_detail = e.response.text
        return {"success": False, "message": f"Login failed: {error_detail}"}
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error to backend: {e}", exc_info=True)
        return {"success": False, "message": f"Could not connect to the backend server. Please ensure it is running: {e}"}
    except requests.exceptions.RequestException as e:
        logger.error(f"An unexpected request error occurred during login: {e}", exc_info=True)
        return {"success": False, "message": f"An unexpected error occurred: {e}"}

def app():
    st.title("Login")

    # Log page view for analytics
    asyncio.run(frontend_log_event('page_view', {
        'page_name': 'Login',
        'status': 'accessed'
    }, user_id=st.session_state.get('user_id_from_backend', 'unauthenticated'), success=True))


    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login"):
        if not email or not password:
            st.error("Please enter both email and password.")
            asyncio.run(frontend_log_event('ui_interaction', {
                'component': 'LoginButton',
                'action': 'Click',
                'details': {'email': email, 'status': 'failure', 'reason': 'missing_credentials'},
            }, user_id=email.replace('.', '_') if email else 'N/A', success=False, error_message="Missing credentials"))
            return

        with st.spinner("Logging in..."):
            response = login_user_backend(email, password)

            # --- DEBUGGING LINE ---
            st.write("Backend Login Response:")
            st.write(response)
            # --- END DEBUGGING LINE ---

            if response.get("message") == "Login successful": # Check for specific success message
                st.success("Logged in successfully!")
                st.session_state['user_email'] = email
                st.session_state['logged_in'] = True
                # Backend returns 'custom_token' and 'uid'
                st.session_state['user_token'] = response.get('custom_token') # Store custom token
                st.session_state['user_id_from_backend'] = response.get('uid') # Store Firebase UID
                
                # Log successful login with actual user_id from backend
                asyncio.run(frontend_log_event('user_login', {
                    'email': email,
                    'status': 'success',
                    'method': 'email_password',
                    'user_uid': response.get('uid')
                }, user_id=response.get('uid'), success=True))
                
                # Redirect to a default page after login (e.g., AI Assistant)
                if 'current_page' in st.session_state:
                    st.session_state.current_page = "AI Assistant" # Or "User Profile"
                    st.rerun()
            else:
                error_message = response.get("message", response.get("detail", "An unknown error occurred."))
                st.error(f"Login failed: {error_message}")
                asyncio.run(frontend_log_event('user_login', {
                    'email': email,
                    'status': 'failure',
                    'method': 'email_password',
                    'error_message': error_message
                }, user_id=email.replace('.', '_') if email else 'N/A', success=False, error_message=error_message))

    st.markdown("---")
    # Changed markdown links to Streamlit buttons for internal navigation
    if st.button("Don't have an account? Register here", key="register_button_from_login"):
        st.session_state.current_page = "Register"
        st.rerun()
    if st.button("Forgot your password? Reset here", key="forgot_password_button_from_login"):
        st.session_state.current_page = "ForgotPassword" # Assuming a ForgotPassword page
        st.rerun()


# Example of how to run this app standalone for testing
if __name__ == "__main__":
    # Mock requests.post for backend calls if running standalone without FastAPI
    import unittest.mock as mock
    original_requests_post = requests.post

    def mock_requests_post(url, json, *args, **kwargs):
        if f"{FASTAPI_BASE_URL}/login" in url: # Corrected mock URL
            email = json.get("email")
            password = json.get("password")

            if email == "test@example.com" and password == "password123":
                mock_response = mock.Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "message": "Login successful", # Changed to match backend
                    "custom_token": "mock_custom_token_123", # Changed to match backend
                    "uid": "mock_firebase_uid_test" # Changed to match backend
                }
                mock_response.raise_for_status = lambda: None
                return mock_response
            elif email == "locked@example.com":
                mock_response = mock.Mock()
                mock_response.status_code = 403
                mock_response.json.return_value = {"detail": "Account locked or disabled."} # Changed to match FastAPI error format
                mock_response.raise_for_status = lambda: requests.exceptions.HTTPError("403 Forbidden: Account locked", response=mock_response)
                return mock_response
            else:
                mock_response = mock.Mock()
                mock_response.status_code = 401
                mock_response.json.return_value = {"detail": "Invalid credentials."} # Changed to match FastAPI error format
                mock_response.raise_for_status = lambda: requests.exceptions.HTTPError("401 Unauthorized: Invalid credentials", response=mock_response)
                return mock_response
        return original_requests_post(url, json, *args, **kwargs)

    requests.post = mock_requests_post
    
    # Initialize Streamlit session state for standalone testing
    if 'user_id_from_backend' not in st.session_state:
        st.session_state.user_id_from_backend = 'unauthenticated_test_user'
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Login"

    app()

    # Restore original requests.post after testing
    requests.post = original_requests_post
 
