# backend/admin_dashboard_app.py

import streamlit as st
import logging
from typing import List, Dict, Any, Optional
import asyncio # For async operations
import pandas as pd # For displaying data in a table

# Import config_manager to access configurations
from config.config_manager import config_manager
# Import analytics_tracker for logging events
from utils.analytics_tracker import log_event, initialize_analytics
# Import UserManager for backend interactions
from utils.user_manager import UserManager

# Import Firebase Admin SDK components for backend initialization (if needed for context)
import firebase_admin
from firebase_admin import credentials, auth, firestore
import json
import os

logger = logging.getLogger(__name__)

# --- Firebase Admin SDK Initialization (for backend context) ---
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
        logger.info("Firebase Admin SDK initialized successfully in admin_dashboard_app.")
    except Exception as e:
        logger.error(f"Error initializing Firebase Admin SDK in admin_dashboard_app: {e}")

# Initialize analytics_tracker for backend context
if 'analytics_initialized_backend' not in globals(): # Use globals() for module-level check
    if firebase_admin._apps:
        try:
            db_instance = firestore.client()
            auth_instance = auth
            app_id_for_analytics = config_manager.get("app_id", firebase_config.get("projectId", "default-streamlit-app-id"))
            initialize_analytics(db_instance, auth_instance, app_id_for_analytics, "backend_system_user")
            globals()['analytics_initialized_backend'] = True
            logger.info("Analytics tracker initialized for admin_dashboard_app with live Firebase.")
        except Exception as e:
            logger.error(f"Failed to initialize analytics with live Firebase Admin SDK in admin_dashboard_app: {e}")
            mock_db = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: None})()})()
            mock_auth = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': None})()})()
            app_id_for_analytics = config_manager.get("app_id", "default-streamlit-app-id")
            initialize_analytics(mock_db, mock_auth, app_id_for_analytics, "backend_system_user")
            globals()['analytics_initialized_backend'] = True
            logger.warning("Analytics tracker initialized with mock Firebase for admin_dashboard_app.")
    else:
        mock_db = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: None})()})()
        mock_auth = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': None})()})()
        app_id_for_analytics = config_manager.get("app_id", "default-streamlit-app-id")
        initialize_analytics(mock_db, mock_auth, app_id_for_analytics, "backend_system_user")
        globals()['analytics_initialized_backend'] = True
        logger.warning("Analytics tracker initialized with mock Firebase for admin_dashboard_app (Admin SDK not available).")


# Initialize UserManager
user_manager = UserManager()

async def load_all_users_data():
    """Loads all user data from the backend."""
    response = await user_manager.get_all_users_admin()
    if response.get("success"):
        return response["users"]
    else:
        st.error(f"Failed to load user data: {response.get('message', 'Unknown error')}")
        return []

async def update_user_data(user_id: str, new_tier: str, new_roles: List[str]):
    """Updates a user's tier and roles via the backend."""
    response = await user_manager.update_user_roles_and_tier_admin(user_id, new_tier, new_roles)
    if response.get("success"):
        st.success(f"Successfully updated user {user_id}'s tier to '{new_tier}' and roles to {new_roles}.")
        return True
    else:
        st.error(f"Failed to update user {user_id}: {response.get('message', 'Unknown error')}")
        return False

def app():
    st.title("🛡️ Admin Dashboard")
    st.info("Manage user accounts, tiers, and roles.")

    # Ensure user is logged in
    if not user_manager.st.session_state.is_authenticated:
        st.warning("Please log in to access the Admin Dashboard.")
        asyncio.run(log_event('page_view', {
            'page_name': 'AdminDashboard',
            'status': 'access_denied',
            'reason': 'not_logged_in'
        }, user_id='unauthenticated', success=False))
        return

    user_id = user_manager.st.session_state.user_id
    user_capabilities = user_manager.st.session_state.user_capabilities

    # RBAC check for admin access
    if not user_capabilities.get('analytics_access', False): # Using analytics_access as a proxy for admin access for now
        st.warning("You do not have administrative access to this page.")
        asyncio.run(log_event('page_view', {
            'page_name': 'AdminDashboard',
            'status': 'access_denied',
            'reason': 'rbac_denied'
        }, user_id=user_id, success=False))
        return

    # Log successful page view
    asyncio.run(log_event('page_view', {
        'page_name': 'AdminDashboard',
        'status': 'accessed',
        'user_id': user_id
    }, user_id=user_id, success=True))

    st.subheader("All Users")

    # Load users data
    users_data = asyncio.run(load_all_users_data())

    if users_data:
        df = pd.DataFrame(users_data)
        # Ensure columns are in a desired order, handle missing ones gracefully
        display_cols = ['user_id', 'email', 'username', 'tier', 'roles', 'created_at', 'last_login_at']
        for col in display_cols:
            if col not in df.columns:
                df[col] = None # Add missing columns as None

        st.dataframe(df[display_cols].set_index('user_id'), use_container_width=True)

        st.subheader("Update User Tier and Roles")
        
        # Get list of existing user IDs for selection
        user_ids_list = [user['user_id'] for user in users_data]
        
        selected_user_id = st.selectbox("Select User to Update", options=user_ids_list, key="admin_user_select")

        # Fetch current tier and roles for the selected user to pre-fill
        current_user_profile = next((user for user in users_data if user['user_id'] == selected_user_id), None)
        
        if current_user_profile:
            current_tier = current_user_profile.get('tier', 'free')
            current_roles = current_user_profile.get('roles', ['user'])
            
            st.write(f"**Current Tier:** `{current_tier}`")
            st.write(f"**Current Roles:** `{', '.join(current_roles)}`")

            # Get available tiers and roles from config (or hardcode if not dynamic)
            available_tiers = config_manager.get("available_tiers", ["free", "basic", "pro", "premium", "admin"])
            available_roles = config_manager.get("available_roles", ["user", "admin", "moderator", "developer"]) # Example roles

            new_tier = st.selectbox("New Tier", options=available_tiers, index=available_tiers.index(current_tier) if current_tier in available_tiers else 0, key="admin_new_tier")
            new_roles = st.multiselect("New Roles", options=available_roles, default=current_roles, key="admin_new_roles")

            if st.button(f"Update {selected_user_id}"):
                if selected_user_id == user_id: # Prevent admin from changing their own roles/tier via this UI
                    st.warning("For security, you cannot change your own tier or roles through this interface. Please contact another administrator if needed.")
                    asyncio.run(log_event('admin_action_frontend', {
                        'action_type': 'update_user_roles_and_tier',
                        'target_user_uid': selected_user_id,
                        'new_tier': new_tier,
                        'new_roles': new_roles,
                        'status': 'failure',
                        'reason': 'self_update_attempt'
                    }, user_id=user_id, success=False, error_message="Admin attempted to change own roles/tier."))
                else:
                    with st.spinner(f"Updating user {selected_user_id}..."):
                        success = asyncio.run(update_user_data(selected_user_id, new_tier, new_roles))
                        if success:
                            st.rerun() # Refresh the page to show updated data
        else:
            st.info("Select a user to view and update their details.")
    else:
        st.info("No users found or failed to load user data.")

# Example of how to run this app standalone for testing
if __name__ == "__main__":
    # Mock session state for standalone testing
    if "user_id" not in st.session_state:
        st.session_state.user_id = "mock_admin_uid"
        st.session_state.username = "MockAdmin"
        st.session_state.email = "admin@example.com"
        st.session_state.id_token = "mock_admin_token"
        st.session_state.is_authenticated = True
        st.session_state.user_profile = {"tier": "admin", "roles": ["user", "admin"]}
        st.session_state.user_capabilities = {
            'analytics_access': True, # Ensure admin access for testing
            'document_upload_enabled': True,
            'document_query_enabled': True,
            'document_query_max_results_k': 10,
            # ... other capabilities
        }
    
    # Mock UserManager methods
    import unittest.mock as mock
    
    mock_users_list = [
        {"user_id": "user1_uid", "email": "user1@example.com", "username": "User One", "tier": "free", "roles": ["user"], "created_at": "2023-01-01T10:00:00Z", "last_login_at": "2024-07-01T15:30:00Z"},
        {"user_id": "user2_uid", "email": "user2@example.com", "username": "User Two", "tier": "pro", "roles": ["user"], "created_at": "2023-02-15T11:00:00Z", "last_login_at": "2024-07-02T09:00:00Z"},
        {"user_id": "admin_uid_test", "email": "testadmin@example.com", "username": "Test Admin", "tier": "admin", "roles": ["user", "admin"], "created_at": "2022-12-01T08:00:00Z", "last_login_at": "2024-07-05T13:00:00Z"},
    ]

    mock_user_manager_instance = mock.MagicMock(spec=UserManager)
    mock_user_manager_instance.st = st # Allow mock to access st.session_state
    mock_user_manager_instance.get_all_users_admin = mock.AsyncMock(return_value={"success": True, "users": mock_users_list})
    mock_user_manager_instance.update_user_roles_and_tier_admin = mock.AsyncMock(return_value={"success": True, "message": "Updated"})

    with patch('utils.user_manager.UserManager', return_value=mock_user_manager_instance):
        # Initialize analytics for the test run if not already done
        if 'analytics_initialized_backend' not in st.session_state:
            mock_db_for_analytics = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: asyncio.sleep(0.01)})()})()
            mock_auth_for_analytics = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': "test_cli_user"})()})()
            initialize_analytics(mock_db_for_analytics, mock_auth_for_analytics, "test-app-id-cli", "test_cli_user")
            st.session_state.analytics_initialized_backend = True

        st.write("Running standalone Admin Dashboard App test. You should see a list of mock users.")
        app()
