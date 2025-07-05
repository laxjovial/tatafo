# backend/change_password_app.py

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

def change_password_backend(user_id: str, id_token: str, current_password: str, new_password: str) -> Dict[str, Any]:
    """Sends change password request to the backend and returns the response."""
    try:
        payload = {
            "current_password": current_password,
            "new_password": new_password
        }
        headers = {"Authorization": f"Bearer {id_token}", "Content-Type": "application/json"}
        response = requests.post(f"{FASTAPI_BASE_URL}/auth/change_password/{user_id}", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error changing password for {user_id} via backend: {e}", exc_info=True)
        return {"success": False, "message": f"Communication error with backend: {e}"}

def app():
    st.title("Change Password")

    # Ensure user is logged in
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.warning("Please log in to change your password.")
        asyncio.run(log_event('page_view', {
            'page_name': 'ChangePassword',
            'status': 'access_denied',
            'reason': 'not_logged_in'
        }, user_id='unauthenticated', success=False))
        return

    user_id = st.session_state.get('user_id_from_backend')
    id_token = st.session_state.get('user_token')
    user_email = st.session_state.get('user_email')

    if not user_id or not id_token or not user_email:
        st.error("User authentication information missing. Please log in again.")
        asyncio.run(log_event('page_view', {
            'page_name': 'ChangePassword',
            'status': 'access_denied',
            'reason': 'missing_auth_info'
        }, user_id='unknown_user', success=False))
        return

    st.info(f"Changing password for: {user_email}")

    current_password = st.text_input("Current Password", type="password", key="current_password")
    new_password = st.text_input("New Password", type="password", key="new_password")
    confirm_new_password = st.text_input("Confirm New Password", type="password", key="confirm_new_password")

    if st.button("Change Password"):
        # Client-side validation
        if not current_password or not new_password or not confirm_new_password:
            st.error("All fields are required.")
            asyncio.run(log_event('ui_interaction', {
                'component': 'ChangePasswordForm',
                'action': 'Submit Change Password',
                'details': {'user_id': user_id, 'reason': 'missing_fields'},
                'user_id': user_id,
                'success': False,
                'error_message': 'Missing fields'
            }))
            return

        if new_password != confirm_new_password:
            st.error("New passwords do not match.")
            asyncio.run(log_event('ui_interaction', {
                'component': 'ChangePasswordForm',
                'action': 'Submit Change Password',
                'details': {'user_id': user_id, 'reason': 'new_password_mismatch'},
                'user_id': user_id,
                'success': False,
                'error_message': 'New passwords do not match'
            }))
            return

        if len(new_password) < 6:
            st.error("New password must be at least 6 characters long.")
            asyncio.run(log_event('ui_interaction', {
                'component': 'ChangePasswordForm',
                'action': 'Submit Change Password',
                'details': {'user_id': user_id, 'reason': 'new_password_too_short'},
                'user_id': user_id,
                'success': False,
                'error_message': 'New password too short'
            }))
            return

        with st.spinner("Changing password..."):
            response = change_password_backend(user_id, id_token, current_password, new_password)

            if response.get("success"):
                st.success("Password changed successfully! Please log in with your new password.")
                # Log successful password change
                asyncio.run(log_event('user_action', {
                    'action_type': 'password_change',
                    'status': 'success',
                    'user_uid': user_id
                }, user_id=user_id, success=True))
                # Force logout after password change for security
                st.session_state['logged_in'] = False
                st.session_state.pop('user_email', None)
                st.session_state.pop('user_token', None)
                st.session_state.pop('user_id_from_backend', None)
                st.session_state.current_page = "Login"
                st.rerun()
            else:
                error_message = response.get("message", "An unknown error occurred.")
                st.error(f"Failed to change password: {error_message}")
                # Log failed password change
                asyncio.run(log_event('user_action', {
                    'action_type': 'password_change',
                    'status': 'failure',
                    'user_uid': user_id,
                    'error_message': error_message
                }, user_id=user_id, success=False, error_message=error_message))

    st.markdown("---")
    st.markdown("[Back to User Profile](/user_profile)")

# Example of how to run this app standalone for testing
if __name__ == "__main__":
    # Mock session state for standalone testing
    if "user_id_from_backend" not in st.session_state:
        st.session_state.user_id_from_backend = "mock_user_uid_123"
        st.session_state.user_token = "mock_id_token_123"
        st.session_state.user_email = "test@example.com"
        st.session_state.logged_in = True
    
    # Mock requests.post for backend calls if running standalone without FastAPI
    import unittest.mock as mock
    original_requests_post = requests.post

    def mock_requests_post(url, json, headers, *args, **kwargs):
        if f"{FASTAPI_BASE_URL}/auth/change_password/" in url:
            user_id = url.split('/')[-1]
            current_password = json.get("current_password")
            new_password = json.get("new_password")

            if user_id == "mock_user_uid_123" and current_password == "oldpassword" and new_password == "newpassword123":
                mock_response = mock.Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"success": True, "message": "Password changed successfully."}
                mock_response.raise_for_status = lambda: None
                return mock_response
            elif user_id == "mock_user_uid_123" and current_password != "oldpassword":
                mock_response = mock.Mock()
                mock_response.status_code = 401
                mock_response.json.return_value = {"success": False, "message": "Invalid current password."}
                mock_response.raise_for_status = lambda: requests.exceptions.HTTPError("401 Unauthorized: Invalid current password")
                return mock_response
            else:
                mock_response = mock.Mock()
                mock_response.status_code = 500
                mock_response.json.return_value = {"success": False, "message": "Internal server error."}
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
