# backend/user_profile_app.py

import streamlit as st
import requests
import logging
from datetime import datetime
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

def get_user_profile_from_backend(user_id: str, id_token: str) -> Dict[str, Any]:
    """Fetches user profile from the backend."""
    try:
        headers = {"Authorization": f"Bearer {id_token}"}
        response = requests.get(f"{FASTAPI_BASE_URL}/users/{user_id}", headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching user profile for {user_id}: {e}", exc_info=True)
        return {"user_id": user_id, "username": "Error", "email": "error@example.com", "tier": "Unknown", "roles": ["error"],
                "subscription_start_date": "N/A", "subscription_end_date": "N/A",
                "days_left": "N/A", "next_subscription_date": "N/A", "message": f"Could not load profile: {e}"} # Fallback

def update_user_profile_backend(user_id: str, id_token: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Sends updated user profile data to the backend."""
    try:
        headers = {"Authorization": f"Bearer {id_token}", "Content-Type": "application/json"}
        response = requests.put(f"{FASTAPI_BASE_URL}/users/{user_id}", json=profile_data, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error updating user profile for {user_id}: {e}", exc_info=True)
        return {"success": False, "message": f"Communication error with backend: {e}"}


def app():
    st.title("👤 User Profile")

    # Ensure user is logged in
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.warning("Please log in to view your profile.")
        # Log page view attempt by unauthenticated user
        asyncio.run(log_event('page_view', {
            'page_name': 'UserProfile',
            'status': 'access_denied',
            'reason': 'not_logged_in'
        }, user_id='unauthenticated', success=False))
        return

    # Get user_id and id_token from session state
    user_id = st.session_state.get('user_id_from_backend')
    id_token = st.session_state.get('user_token')

    if not user_id or not id_token:
        st.error("User authentication information missing. Please log in again.")
        asyncio.run(log_event('page_view', {
            'page_name': 'UserProfile',
            'status': 'access_denied',
            'reason': 'missing_auth_info'
        }, user_id='unknown_user', success=False))
        return

    # Log page view for authenticated access
    asyncio.run(log_event('page_view', {
        'page_name': 'UserProfile',
        'status': 'accessed',
        'user_id': user_id
    }, user_id=user_id, success=True))

    # Fetch user profile data
    # Use a unique key for the profile data in session state to prevent re-fetching on every rerun
    if 'user_profile_data' not in st.session_state or st.session_state.user_profile_data.get('user_id') != user_id:
        with st.spinner("Loading profile..."):
            st.session_state.user_profile_data = get_user_profile_from_backend(user_id, id_token)
            if not st.session_state.user_profile_data.get('success', True): # Check for backend success flag
                st.error(st.session_state.user_profile_data.get('message', 'Failed to load profile.'))
                asyncio.run(log_event('data_fetch', {
                    'entity': 'UserProfile',
                    'action': 'load',
                    'status': 'failure',
                    'error_message': st.session_state.user_profile_data.get('message', 'Unknown error during fetch')
                }, user_id=user_id, success=False))
                return

    user_profile = st.session_state.user_profile_data
    
    # Initialize edit state if not present
    if 'is_editing_profile' not in st.session_state:
        st.session_state.is_editing_profile = False
    if 'edited_profile_data' not in st.session_state:
        st.session_state.edited_profile_data = user_profile.copy() # Initialize with fetched data

    st.header(f"Welcome, {user_profile.get('username', 'User')}!")
    st.markdown("---")

    st.subheader("Account Information")
    
    # Display fields for editing or viewing
    if st.session_state.is_editing_profile:
        new_username = st.text_input("Username", value=st.session_state.edited_profile_data.get('username', ''), key="edit_username")
        new_email = st.text_input("Email", value=st.session_state.edited_profile_data.get('email', ''), key="edit_email")
        new_phone = st.text_input("Phone", value=st.session_state.edited_profile_data.get('phone', ''), key="edit_phone")
        new_address = st.text_area("Address", value=st.session_state.edited_profile_data.get('address', ''), key="edit_address")
        new_bio = st.text_area("Bio", value=st.session_state.edited_profile_data.get('bio', ''), key="edit_bio")

        st.session_state.edited_profile_data['username'] = new_username
        st.session_state.edited_profile_data['email'] = new_email
        st.session_state.edited_profile_data['phone'] = new_phone
        st.session_state.edited_profile_data['address'] = new_address
        st.session_state.edited_profile_data['bio'] = new_bio

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Changes"):
                with st.spinner("Saving profile..."):
                    update_response = update_user_profile_backend(user_id, id_token, st.session_state.edited_profile_data)
                    if update_response.get("success"):
                        st.success("Profile updated successfully!")
                        st.session_state.is_editing_profile = False
                        # Force re-fetch of profile data to get latest from backend
                        st.session_state.user_profile_data = get_user_profile_from_backend(user_id, id_token)
                        asyncio.run(log_event('ui_interaction', {
                            'component': 'UserProfileForm',
                            'action': 'Save Profile',
                            'details': {'changes_made': True, 'user_id': user_id},
                            'user_id': user_id,
                            'success': True
                        }))
                        st.rerun()
                    else:
                        st.error(f"Failed to update profile: {update_response.get('message', 'Unknown error')}")
                        asyncio.run(log_event('ui_interaction', {
                            'component': 'UserProfileForm',
                            'action': 'Save Profile',
                            'details': {'changes_made': True, 'user_id': user_id, 'error': update_response.get('message')},
                            'user_id': user_id,
                            'success': False,
                            'error_message': update_response.get('message')
                        }))
        with col2:
            if st.button("Cancel"):
                st.session_state.is_editing_profile = False
                st.session_state.edited_profile_data = user_profile.copy() # Revert changes
                st.warning("Changes cancelled.")
                asyncio.run(log_event('ui_interaction', {
                    'component': 'UserProfileForm',
                    'action': 'Cancel Edit',
                    'details': {'user_id': user_id},
                    'user_id': user_id,
                    'success': True
                }))
                st.rerun()
    else:
        st.write(f"**User ID:** `{user_profile.get('user_id', 'N/A')}`")
        st.write(f"**Username:** {user_profile.get('username', 'N/A')}")
        st.write(f"**Email:** {user_profile.get('email', 'N/A')}")
        st.write(f"**Phone:** {user_profile.get('phone', 'N/A')}")
        st.write(f"**Address:** {user_profile.get('address', 'N/A')}")
        st.write(f"**Bio:** {user_profile.get('bio', 'N/A')}")
        st.write(f"**Tier:** {user_profile.get('tier', 'N/A').capitalize()}")
        st.write(f"**Roles:** {', '.join(user_profile.get('roles', []))}")

        if st.button("Edit Profile"):
            st.session_state.is_editing_profile = True
            st.session_state.edited_profile_data = user_profile.copy() # Load current data into edit state
            asyncio.run(log_event('ui_interaction', {
                'component': 'UserProfilePage',
                'action': 'Edit Profile Button Click',
                'details': {'user_id': user_id},
                'user_id': user_id,
                'success': True
            }))
            st.rerun()

    st.markdown("---")
    st.subheader("Subscription Details")
    
    sub_start = user_profile.get('subscription_start_date', 'N/A')
    sub_end = user_profile.get('subscription_end_date', 'N/A')
    days_left = user_profile.get('days_left', 'N/A')
    next_sub_date = user_profile.get('next_subscription_date', 'N/A')

    st.write(f"**Subscription Start Date:** {sub_start}")
    st.write(f"**Subscription End Date:** {sub_end}")
    st.write(f"**Days Remaining:** {days_left}")
    st.write(f"**Next Subscription Date:** {next_sub_date}")

    if user_profile.get('tier') == 'free':
        st.info("You are currently on the Free tier. Upgrade to unlock more features and extended subscriptions!")
    elif days_left != "N/A" and isinstance(days_left, int) and days_left <= 7:
        st.warning(f"Your subscription is ending in {days_left} day(s)! Renew now to continue enjoying all features.")
    elif days_left == 0 and user_profile.get('tier') != 'free':
        st.error("Your subscription has expired! Please renew to regain access to premium features.")
            
    st.markdown("---")
    st.subheader("Manage Account")
    # These buttons would navigate to other pages in a multi-page app
    if st.button("Change Password", key="change_password_btn"):
        st.session_state.current_page = "Change Password" # Assuming main_app.py handles navigation
        asyncio.run(log_event('ui_interaction', {
            'component': 'UserProfilePage',
            'action': 'Change Password Button Click',
            'details': {'user_id': user_id},
            'user_id': user_id,
            'success': True
        }))
        st.rerun()
    if st.button("Manage Subscription", key="manage_subscription_btn"):
        st.info("This would navigate to a subscription management page (future feature).")
        asyncio.run(log_event('ui_interaction', {
            'component': 'UserProfilePage',
            'action': 'Manage Subscription Button Click',
            'details': {'user_id': user_id},
            'user_id': user_id,
            'success': True
        }))


# Example of how to run this app standalone for testing
if __name__ == "__main__":
    # Mock session state for standalone testing
    if "user_id_from_backend" not in st.session_state:
        st.session_state.user_id_from_backend = "mock_premium_uid_123" # Must be a UID
        st.session_state.user_token = "mock_id_token_premium" # Must be an ID token
        st.session_state.logged_in = True
    
    # Mock requests.get and requests.put for backend calls if running standalone without FastAPI
    import unittest.mock as mock
    original_requests_get = requests.get
    original_requests_put = requests.put

    mock_profiles = {
        "mock_free_uid_123": {
            "user_id": "mock_free_uid_123", "username": "FreeUser", "email": "free@example.com", "phone": "N/A", "address": "N/A", "bio": "Free tier user.", "tier": "free", "roles": ["user"],
            "subscription_start_date": "N/A", "subscription_end_date": "N/A", "days_left": "N/A", "next_subscription_date": "N/A", "success": True
        },
        "mock_basic_uid_456": {
            "user_id": "mock_basic_uid_456", "username": "BasicUser", "email": "basic@example.com", "phone": "111-222-3333", "address": "456 Oak Ave", "bio": "Basic tier user.", "tier": "basic", "roles": ["user"],
            "subscription_start_date": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
            "subscription_end_date": (datetime.now() + timedelta(days=20)).strftime("%Y-%m-%d"),
            "days_left": 20, "next_subscription_date": (datetime.now() + timedelta(days=21)).strftime("%Y-%m-%d"), "success": True
        },
        "mock_premium_uid_123": { # This is the default for standalone test
            "user_id": "mock_premium_uid_123", "username": "PremiumUser", "email": "premium@example.com", "phone": "987-654-3210", "address": "789 Pine Ln", "bio": "Premium tier user with full access.", "tier": "premium", "roles": ["user"],
            "subscription_start_date": (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d"),
            "subscription_end_date": (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d"),
            "days_left": 45, "next_subscription_date": (datetime.now() + timedelta(days=46)).strftime("%Y-%m-%d"), "success": True
        }
    }

    def mock_requests_get(url, headers, *args, **kwargs):
        if f"{FASTAPI_BASE_URL}/users/" in url:
            user_id = url.split('/')[-1]
            user_data = mock_profiles.get(user_id)
            if user_data:
                mock_response = mock.Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = user_data
                mock_response.raise_for_status = lambda: None
                return mock_response
            else:
                mock_response = mock.Mock()
                mock_response.status_code = 404
                mock_response.json.return_value = {"success": False, "message": "User not found."}
                mock_response.raise_for_status = lambda: requests.exceptions.HTTPError("404 Not Found: User not found")
                return mock_response
        return original_requests_get(url, headers, *args, **kwargs)

    def mock_requests_put(url, json, headers, *args, **kwargs):
        if f"{FASTAPI_BASE_URL}/users/" in url:
            user_id = url.split('/')[-1]
            if user_id in mock_profiles:
                # Simulate update
                mock_profiles[user_id].update(json)
                mock_response = mock.Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"success": True, "message": "Profile updated successfully."}
                mock_response.raise_for_status = lambda: None
                return mock_response
            else:
                mock_response = mock.Mock()
                mock_response.status_code = 404
                mock_response.json.return_value = {"success": False, "message": "User not found for update."}
                mock_response.raise_for_status = lambda: requests.exceptions.HTTPError("404 Not Found: User not found for update")
                return mock_response
        return original_requests_put(url, json, headers, *args, **kwargs)

    requests.get = mock_requests_get
    requests.put = mock_requests_put
    
    # Initialize analytics for the test run if not already done
    if 'analytics_initialized_backend' not in st.session_state:
        # Mock Firebase Admin SDK for analytics initialization in test context
        mock_db_for_analytics = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: asyncio.sleep(0.01)})()})()
        mock_auth_for_analytics = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': "test_cli_user"})()})()
        initialize_analytics(mock_db_for_analytics, mock_auth_for_analytics, "test-app-id-cli", "test_cli_user")
        st.session_state.analytics_initialized_backend = True

    app()

    # Restore original requests.get and requests.put after testing
    requests.get = original_requests_get
    requests.put = original_requests_put
