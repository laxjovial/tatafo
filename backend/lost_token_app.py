# backend/lost_token_app.py

import streamlit as st
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


def app():
    st.title("Lost Password Reset Token?")
    st.info("It seems you've landed on this page because your password reset link might have expired, or you didn't receive it.")

    st.write("""
        If you requested a password reset and did not receive an email, or if the link in the email has expired,
        please try requesting a new password reset link.
    """)

    st.markdown("---")

    if st.button("Request a New Password Reset Link"):
        # Redirect to the Forgot Password page
        if 'current_page' in st.session_state:
            st.session_state.current_page = "Forgot Password"
            asyncio.run(log_event('ui_interaction', {
                'component': 'LostTokenPage',
                'action': 'Request New Reset Link Click',
                'details': {'redirect_to': 'Forgot Password'},
                'user_id': st.session_state.get('user_id_from_backend', 'unauthenticated'),
                'success': True
            }))
            st.rerun()
        else:
            st.warning("Cannot redirect. Please navigate to 'Forgot Password' manually.")
            asyncio.run(log_event('ui_interaction', {
                'component': 'LostTokenPage',
                'action': 'Request New Reset Link Click',
                'details': {'redirect_to': 'Forgot Password', 'status': 'failed_redirect'},
                'user_id': st.session_state.get('user_id_from_backend', 'unauthenticated'),
                'success': False,
                'error_message': 'Session state current_page not found for redirect'
            }))


    st.markdown("---")
    st.markdown("Remember your password? [Login here](/login)")

# Example of how to run this app standalone for testing
if __name__ == "__main__":
    # Initialize analytics for the test run if not already done
    if 'analytics_initialized_backend' not in st.session_state:
        mock_db_for_analytics = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: asyncio.sleep(0.01)})()})()
        mock_auth_for_analytics = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': "test_cli_user"})()})()
        initialize_analytics(mock_db_for_analytics, mock_auth_for_analytics, "test-app-id-cli", "test_cli_user")
        st.session_state.analytics_initialized_backend = True
    
    # Log page view for standalone test
    asyncio.run(log_event('page_view', {
        'page_name': 'LostToken',
        'status': 'accessed',
        'user_id': st.session_state.get('user_id_from_backend', 'unauthenticated')
    }, user_id=st.session_state.get('user_id_from_backend', 'unauthenticated'), success=True))

    app()
