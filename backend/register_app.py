# backend/register_app.py

import streamlit as st
import requests
import logging
from typing import Dict, Any, Optional
import json
import os
import asyncio # For async operations in CLI test

# Import config_manager (if needed for frontend config like backend URL)
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
        # Assuming an analytics endpoint exists on your backend
        # You might need to add a dedicated endpoint in main.py for frontend analytics
        response = requests.post(f"{FASTAPI_BASE_URL}/admin/analytics/events", json=payload)
        response.raise_for_status()
        logger.info(f"Frontend Analytics Logged: {event_type} for user {user_id}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send analytics event to backend: {e}")

def register_user_backend(email: str, password: str, username: str) -> Dict[str, Any]:
    """Sends registration request to the backend and returns the response."""
    try:
        payload = {
            "email": email,
            "password": password,
            "username": username
        }
        headers = {"Content-Type": "application/json"}
        # Corrected endpoint: /register as defined in main.py (no /auth/)
        response = requests.post(f"{FASTAPI_BASE_URL}/register", json=payload, headers=headers)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error during registration: {e.response.status_code} - {e.response.text}", exc_info=True)
        # Parse error message from backend if available
        try:
            error_detail = e.response.json().get("detail", str(e))
        except json.JSONDecodeError:
            error_detail = e.response.text
        return {"success": False, "message": f"Registration failed: {error_detail}"}
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error to backend: {e}", exc_info=True)
        return {"success": False, "message": f"Could not connect to the backend server. Please ensure it is running: {e}"}
    except requests.exceptions.RequestException as e:
        logger.error(f"An unexpected request error occurred during registration: {e}", exc_info=True)
        return {"success": False, "message": f"An unexpected error occurred: {e}"}

def app():
    st.title("Register New Account")

    # Log page view for analytics
    asyncio.run(frontend_log_event('page_view', {
        'page_name': 'Register',
        'status': 'accessed'
    }, user_id=st.session_state.get('user_id_from_backend', 'unauthenticated'), success=True))

    username = st.text_input("Username", key="register_username")
    email = st.text_input("Email", key="register_email")
    password = st.text_input("Password", type="password", key="register_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")

    if st.button("Register"):
        if not username or not email or not password or not confirm_password:
            st.error("All fields are required.")
            asyncio.run(frontend_log_event('ui_interaction', {
                'component': 'RegisterButton',
                'action': 'Click',
                'details': {'email': email, 'username': username, 'status': 'failure', 'reason': 'missing_fields'},
            }, user_id=email.replace('.', '_') if email else 'N/A', success=False, error_message="Missing fields"))
            return

        if password != confirm_password:
            st.error("Passwords do not match.")
            asyncio.run(frontend_log_event('ui_interaction', {
                'component': 'RegisterButton',
                'action': 'Click',
                'details': {'email': email, 'username': username, 'status': 'failure', 'reason': 'password_mismatch'},
            }, user_id=email.replace('.', '_') if email else 'N/A', success=False, error_message="Password mismatch"))
            return

        if len(password) < 6:
            st.error("Password must be at least 6 characters long.")
            asyncio.run(frontend_log_event('ui_interaction', {
                'component': 'RegisterButton',
                'action': 'Click',
                'details': {'email': email, 'username': username, 'status': 'failure', 'reason': 'password_too_short'},
            }, user_id=email.replace('.', '_') if email else 'N/A', success=False, error_message="Password too short"))
            return

        with st.spinner("Registering..."):
            response = register_user_backend(email, password, username)

            # FastAPI backend returns "User registered successfully", and "uid"
            if response.get("message") == "User registered successfully":
                st.success("Account created successfully! Please log in.")
                asyncio.run(frontend_log_event('user_registration', {
                    'email': email,
                    'username': username,
                    'status': 'success',
                    'user_uid': response.get('uid') # Log the actual Firebase UID if returned
                }, user_id=response.get('uid', email.replace('.', '_')), success=True))
                # Redirect to login page
                if 'current_page' in st.session_state:
                    st.session_state.current_page = "Login"
                    st.rerun()
            else:
                error_message = response.get("message", "An unknown error occurred.")
                st.error(f"Registration failed: {error_message}")
                asyncio.run(frontend_log_event('user_registration', {
                    'email': email,
                    'username': username,
                    'status': 'failure',
                    'reason': error_message
                }, user_id=email.replace('.', '_') if email else 'N/A', success=False, error_message=error_message))

    st.markdown("---")
    st.markdown("Already have an account? [Login here](/login)")

# Example of how to run this app standalone for testing
if __name__ == "__main__":
    # Mock requests.post for backend calls if running standalone without FastAPI
    import unittest.mock as mock
    from typing import Optional # Import Optional for type hinting in mock
    original_requests_post = requests.post

    def mock_requests_post(url, json, *args, **kwargs):
        if f"{FASTAPI_BASE_URL}/register" in url: # Corrected mock URL
            email = json.get("email")
            password = json.get("password")
            username = json.get("username")

            if "@" not in email or len(password) < 6:
                mock_response = mock.Mock()
                mock_response.status_code = 400
                mock_response.json.return_value = {"detail": "Invalid email or password format."} # Changed to match FastAPI error format
                mock_response.raise_for_status = lambda: requests.exceptions.HTTPError("400 Client Error: Bad Request for url", response=mock_response)
                return mock_response
            elif email == "existing@example.com":
                mock_response = mock.Mock()
                mock_response.status_code = 409
                mock_response.json.return_value = {"detail": "User already exists."} # Changed to match FastAPI error format
                mock_response.raise_for_status = lambda: requests.exceptions.HTTPError("409 Conflict: User already exists", response=mock_response)
                return mock_response
            else:
                mock_response = mock.Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"message": "User registered successfully.", "uid": f"mock_uid_{email.split('@')[0]}"} # Changed to match backend
                mock_response.raise_for_status = lambda: None
                return mock_response
        return original_requests_post(url, json, *args, **kwargs)

    requests.post = mock_requests_post
    
    # Initialize Streamlit session state for standalone testing
    if 'user_id_from_backend' not in st.session_state:
        st.session_state.user_id_from_backend = 'unauthenticated_test_user'
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Register"

    app()

    # Restore original requests.post after testing
    requests.post = original_requests_post

