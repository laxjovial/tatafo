# backend/register_app.py

import streamlit as st
import requests
import logging
from typing import Dict, Any

# Import config_manager to access Firebase configuration and other settings
from config.config_manager import config_manager
# Import analytics_tracker for logging events
from utils.analytics_tracker import log_event, initialize_analytics
# Import Firebase Admin SDK components for backend initialization (if needed for context)
import firebase_admin
from firebase_admin import credentials, auth, firestore
import json
import os
import asyncio # For async operations in CLI test

logger = logging.getLogger(__name__)

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
            st.warning("FIREBASE_ADMIN_CERT environment variable not found. Firebase Admin SDK functionality may be limited.")
            # Create a dummy credential object just to allow initialization for local testing
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
        logger.info("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Firebase Admin SDK: {e}")
        st.error(f"Error initializing Firebase services: {e}")

# Initialize analytics_tracker for Streamlit backend context
if 'analytics_initialized_backend' not in st.session_state:
    # Use actual Firebase Admin SDK db/auth instances if available, otherwise mocks
    # For Streamlit, the primary use of analytics_tracker here is for backend-initiated events
    # like user registration success/failure.
    
    # Check if Firebase Admin SDK app is initialized
    if firebase_admin._apps:
        try:
            db_instance = firestore.client()
            auth_instance = auth
            app_id_for_analytics = config_manager.get("app_id", firebase_config.get("projectId", "default-streamlit-app-id"))
            initialize_analytics(db_instance, auth_instance, app_id_for_analytics, "backend_system_user")
            st.session_state.analytics_initialized_backend = True
            logger.info("Analytics tracker initialized for Streamlit backend context with live Firebase.")
        except Exception as e:
            logger.error(f"Failed to initialize analytics with live Firebase Admin SDK: {e}")
            # Fallback to mock if live initialization fails
            mock_db = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: None})()})()
            mock_auth = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': None})()})()
            app_id_for_analytics = config_manager.get("app_id", "default-streamlit-app-id")
            initialize_analytics(mock_db, mock_auth, app_id_for_analytics, "backend_system_user")
            st.session_state.analytics_initialized_backend = True
            logger.warning("Analytics tracker initialized with mock Firebase for Streamlit backend context.")
    else:
        # If Firebase Admin SDK itself failed to initialize, use mocks for analytics
        mock_db = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: None})()})()
        mock_auth = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': None})()})()
        app_id_for_analytics = config_manager.get("app_id", "default-streamlit-app-id")
        initialize_analytics(mock_db, mock_auth, app_id_for_analytics, "backend_system_user")
        st.session_state.analytics_initialized_backend = True
        logger.warning("Analytics tracker initialized with mock Firebase for Streamlit backend context (Admin SDK not available).")


# --- Configuration for FastAPI Backend ---
FASTAPI_BASE_URL = "http://localhost:8000" # Assuming FastAPI runs on port 8000

def register_user_backend(email: str, password: str, username: str) -> Dict[str, Any]:
    """Sends registration request to the backend and returns the response."""
    try:
        payload = {
            "email": email,
            "password": password,
            "username": username
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{FASTAPI_BASE_URL}/auth/register", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error registering user via backend: {e}", exc_info=True)
        return {"success": False, "message": f"Communication error with backend: {e}"}

def app():
    st.title("Register New Account")

    username = st.text_input("Username", key="register_username")
    email = st.text_input("Email", key="register_email")
    password = st.text_input("Password", type="password", key="register_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")

    if st.button("Register"):
        if not username or not email or not password or not confirm_password:
            st.error("All fields are required.")
            asyncio.run(log_event('user_registration', {
                'email': email,
                'username': username,
                'status': 'failure',
                'reason': 'missing_fields'
            }, user_id=email.replace('.', '_') if email else 'N/A', success=False))
            return

        if password != confirm_password:
            st.error("Passwords do not match.")
            asyncio.run(log_event('user_registration', {
                'email': email,
                'username': username,
                'status': 'failure',
                'reason': 'password_mismatch'
            }, user_id=email.replace('.', '_') if email else 'N/A', success=False))
            return

        if len(password) < 6:
            st.error("Password must be at least 6 characters long.")
            asyncio.run(log_event('user_registration', {
                'email': email,
                'username': username,
                'status': 'failure',
                'reason': 'password_too_short'
            }, user_id=email.replace('.', '_') if email else 'N/A', success=False))
            return

        with st.spinner("Registering..."):
            response = register_user_backend(email, password, username)

            if response.get("success"):
                st.success("Account created successfully! Please log in.")
                asyncio.run(log_event('user_registration', {
                    'email': email,
                    'username': username,
                    'status': 'success',
                    'user_uid': response.get('user_id') # Log the actual Firebase UID if returned
                }, user_id=response.get('user_id', email.replace('.', '_')), success=True))
                # Redirect to login page
                if 'current_page' in st.session_state:
                    st.session_state.current_page = "Login"
                    st.rerun()
            else:
                error_message = response.get("message", "An unknown error occurred.")
                st.error(f"Registration failed: {error_message}")
                asyncio.run(log_event('user_registration', {
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
    original_requests_post = requests.post

    def mock_requests_post(url, json, *args, **kwargs):
        if f"{FASTAPI_BASE_URL}/auth/register" in url:
            email = json.get("email")
            password = json.get("password")
            username = json.get("username")

            if "@" not in email or len(password) < 6:
                mock_response = mock.Mock()
                mock_response.status_code = 400
                mock_response.json.return_value = {"success": False, "message": "Invalid email or password format."}
                mock_response.raise_for_status = lambda: requests.exceptions.HTTPError("400 Client Error: Bad Request for url")
                return mock_response
            elif email == "existing@example.com":
                mock_response = mock.Mock()
                mock_response.status_code = 409
                mock_response.json.return_value = {"success": False, "message": "User already exists."}
                mock_response.raise_for_status = lambda: requests.exceptions.HTTPError("409 Conflict: User already exists")
                return mock_response
            else:
                mock_response = mock.Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"success": True, "message": "User created successfully.", "user_id": f"mock_uid_{email.split('@')[0]}"}
                mock_response.raise_for_status = lambda: None
                return mock_response
        return original_requests_post(url, json, *args, **kwargs)

    requests.post = mock_requests_post
    
    # Initialize analytics for the test run if not already done
    if 'analytics_initialized_backend' not in st.session_state:
        # Mock Firebase Admin SDK for analytics initialization in test context
        mock_db_for_analytics = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: asyncio.sleep(0.01)})()})()
        mock_auth_for_analytics = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': "test_cli_user"})()})()
        initialize_analytics(mock_db_for_analytics, mock_auth_for_analytics, "test-app-id-cli", "test_cli_user")
        st.session_state.analytics_initialized_backend = True

    app()

    # Restore original requests.post after testing
    requests.post = original_requests_post
