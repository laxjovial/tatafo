# backend/reset_password_token_app.py

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

def reset_password_backend(oob_code: str, new_password: str) -> Dict[str, Any]:
    """Sends password reset request with OOB code to the backend and returns the response."""
    try:
        payload = {
            "oob_code": oob_code,
            "new_password": new_password
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{FASTAPI_BASE_URL}/auth/reset_password", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error resetting password via backend: {e}", exc_info=True)
        return {"success": False, "message": f"Communication error with backend: {e}"}

def app():
    st.title("Reset Password")
    st.info("Enter your new password below.")

    # Get OOB code from URL parameters (if running as a direct page)
    # In a real deployment, Streamlit might not directly expose URL query params easily
    # for a multi-page app. This assumes the main_app.py or a redirect handles it.
    # For local testing, you might manually set it in session_state or mock it.
    oob_code = st.query_params.get("oobCode") # For direct Streamlit page
    if not oob_code:
        # Fallback if not in query_params, check session state (e.g., if passed from main_app)
        oob_code = st.session_state.get("reset_password_oob_code")
        if not oob_code:
            st.error("Missing password reset code. Please ensure you clicked the link from your email.")
            asyncio.run(log_event('page_view', {
                'page_name': 'ResetPassword',
                'status': 'access_denied',
                'reason': 'missing_oob_code'
            }, user_id='unauthenticated', success=False))
            return

    new_password = st.text_input("New Password", type="password", key="reset_new_password")
    confirm_new_password = st.text_input("Confirm New Password", type="password", key="reset_confirm_new_password")

    if st.button("Reset Password"):
        # Client-side validation
        if not new_password or not confirm_new_password:
            st.error("Both new password fields are required.")
            asyncio.run(log_event('ui_interaction', {
                'component': 'ResetPasswordForm',
                'action': 'Submit Reset Password',
                'details': {'oob_code_present': bool(oob_code), 'reason': 'missing_fields'},
                'user_id': 'unauthenticated', # User is not logged in yet
                'success': False,
                'error_message': 'Missing new password fields'
            }))
            return

        if new_password != confirm_new_password:
            st.error("New passwords do not match.")
            asyncio.run(log_event('ui_interaction', {
                'component': 'ResetPasswordForm',
                'action': 'Submit Reset Password',
                'details': {'oob_code_present': bool(oob_code), 'reason': 'password_mismatch'},
                'user_id': 'unauthenticated',
                'success': False,
                'error_message': 'New passwords do not match'
            }))
            return

        if len(new_password) < 6:
            st.error("New password must be at least 6 characters long.")
            asyncio.run(log_event('ui_interaction', {
                'component': 'ResetPasswordForm',
                'action': 'Submit Reset Password',
                'details': {'oob_code_present': bool(oob_code), 'reason': 'password_too_short'},
                'user_id': 'unauthenticated',
                'success': False,
                'error_message': 'New password too short'
            }))
            return

        with st.spinner("Resetting password..."):
            response = reset_password_backend(oob_code, new_password)

            if response.get("success"):
                st.success("Your password has been reset successfully! You can now log in with your new password.")
                # Log successful password reset
                asyncio.run(log_event('user_action', {
                    'action_type': 'password_reset_complete',
                    'status': 'success',
                    'user_id_from_backend': response.get('user_id') # If backend returns user_id
                }, user_id=response.get('user_id', 'unauthenticated'), success=True))
                # Redirect to login page
                if 'current_page' in st.session_state:
                    st.session_state.current_page = "Login"
                    st.rerun()
            else:
                error_message = response.get("message", "An unknown error occurred.")
                st.error(f"Failed to reset password: {error_message}")
                # Log failed password reset
                asyncio.run(log_event('user_action', {
                    'action_type': 'password_reset_complete',
                    'status': 'failure',
                    'error_message': error_message
                }, user_id='unauthenticated', success=False, error_message=error_message))

    st.markdown("---")
    st.markdown("Remember your password? [Login here](/login)")

# Example of how to run this app standalone for testing
if __name__ == "__main__":
    # For standalone testing, you might set a mock oob_code in query_params or session_state
    # Example: st.query_params["oobCode"] = "mock_oob_code_123"
    if "reset_password_oob_code" not in st.session_state:
        st.session_state.reset_password_oob_code = "mock_valid_oob_code" # Simulate a valid code

    # Mock requests.post for backend calls if running standalone without FastAPI
    import unittest.mock as mock
    original_requests_post = requests.post

    def mock_requests_post(url, json, headers, *args, **kwargs):
        if f"{FASTAPI_BASE_URL}/auth/reset_password" in url:
            oob_code = json.get("oob_code")
            new_password = json.get("new_password")

            if oob_code == "mock_valid_oob_code" and new_password == "newsecurepassword":
                mock_response = mock.Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"success": True, "message": "Password reset successfully.", "user_id": "mock_reset_user_uid"}
                mock_response.raise_for_status = lambda: None
                return mock_response
            elif oob_code == "mock_invalid_oob_code":
                mock_response = mock.Mock()
                mock_response.status_code = 400
                mock_response.json.return_value = {"success": False, "message": "Invalid or expired password reset code."}
                mock_response.raise_for_status = lambda: requests.exceptions.HTTPError("400 Bad Request: Invalid code")
                return mock_response
            else:
                mock_response = mock.Mock()
                mock_response.status_code = 500
                mock_response.json.return_value = {"success": False, "message": "Internal server error during password reset."}
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
