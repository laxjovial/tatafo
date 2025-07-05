# backend/forgot_password_app.py

import streamlit as st
import requests
import logging
from typing import Dict, Any, Optional
import asyncio # For async operations in CLI test

# Import config_manager to access Firebase configuration and other settings
from config.config_manager import config_manager
# Import analytics_tracker for logging events
from utils.analytics_tracker import log_event, initialize_analytics

# Import Firebase Admin SDK components for backend initialization (if needed for context)
import firebase_admin
from firebase_admin import credentials, auth, firestore
import json
import os

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

def request_password_reset_backend(email: str) -> Dict[str, Any]:
    """Sends password reset email request to the backend and returns the response."""
    try:
        payload = {"email": email}
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{FASTAPI_BASE_URL}/auth/forgot_password", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error requesting password reset for {email} via backend: {e}", exc_info=True)
        return {"success": False, "message": f"Communication error with backend: {e}"}

def app():
    st.title("Forgot Password")
    st.info("Enter your email address below and we'll send you a link to reset your password.")

    email = st.text_input("Email", key="forgot_password_email")

    if st.button("Send Reset Link"):
        if not email:
            st.error("Please enter your email address.")
            asyncio.run(log_event('ui_interaction', {
                'component': 'ForgotPasswordForm',
                'action': 'Submit Reset Request',
                'details': {'email': email, 'reason': 'missing_email'},
                'user_id': email.replace('.', '_') if email else 'N/A',
                'success': False,
                'error_message': 'Email address not provided'
            }))
            return

        with st.spinner("Sending reset link..."):
            response = request_password_reset_backend(email)

            if response.get("success"):
                st.success("If an account with that email exists, a password reset link has been sent to your email address.")
                # Log successful password reset request (even if email doesn't exist, for security reasons Firebase confirms vaguely)
                asyncio.run(log_event('user_action', {
                    'action_type': 'password_reset_request',
                    'email': email,
                    'status': 'success'
                }, user_id=email.replace('.', '_'), success=True))
            else:
                error_message = response.get("message", "An unknown error occurred.")
                st.error(f"Failed to send reset link: {error_message}")
                # Log failed password reset request
                asyncio.run(log_event('user_action', {
                    'action_type': 'password_reset_request',
                    'email': email,
                    'status': 'failure',
                    'error_message': error_message
                }, user_id=email.replace('.', '_'), success=False, error_message=error_message))

    st.markdown("---")
    st.markdown("Remember your password? [Login here](/login)")
    st.markdown("Don't have an account? [Register here](/register)")

# Example of how to run this app standalone for testing
if __name__ == "__main__":
    # Mock requests.post for backend calls if running standalone without FastAPI
    import unittest.mock as mock
    original_requests_post = requests.post

    def mock_requests_post(url, json, headers, *args, **kwargs):
        if f"{FASTAPI_BASE_URL}/auth/forgot_password" in url:
            email = json.get("email")

            if email == "user@example.com":
                mock_response = mock.Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"success": True, "message": "Password reset email sent."}
                mock_response.raise_for_status = lambda: None
                return mock_response
            elif email == "nonexistent@example.com":
                # Firebase often returns success even for non-existent emails to prevent enumeration
                mock_response = mock.Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"success": True, "message": "If an account with that email exists, a password reset link has been sent."}
                mock_response.raise_for_status = lambda: None
                return mock_response
            else:
                mock_response = mock.Mock()
                mock_response.status_code = 500
                mock_response.json.return_value = {"success": False, "message": "Internal server error during reset request."}
                mock_response.raise_for_status = lambda: requests.exceptions.HTTPError("500 Internal Server Error")
                return mock_response
        return original_requests_post(url, json, headers, *args, **kwargs)

    requests.post = mock_requests_post
    
    # Initialize analytics for the test run if not already done
    if 'analytics_initialized_backend' not in st.session_state:
        mock_db_for_analytics = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: asyncio.sleep(0.01)})()})()
        mock_auth_for_analytics = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': "test_cli_user"})()})()
        initialize_analytics(mock_db_for_analytics, mock_auth_for_analytics, "test-app-id-cli", "test_cli_user")
        st.session_state.analytics_initialized_backend = True

    app()

    # Restore original requests.post after testing
    requests.post = original_requests_post
