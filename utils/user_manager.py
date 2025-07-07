# utils/user_manager.py

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import asyncio

# Import analytics_tracker for logging events
from utils.analytics_tracker import log_event

logger = logging.getLogger(__name__)

# --- RBAC Capabilities Configuration (Centralized) ---
# This dictionary defines what capabilities each tier/role has.
# This would typically be loaded from a config file or database in a larger app.
_RBAC_CAPABILITIES_CONFIG = {
    'capabilities': {
        'finance_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'crypto_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'medical_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'news_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'legal_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'education_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'entertainment_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'weather_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'travel_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'sports_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'document_upload_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'document_query_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'web_search_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'web_search_max_results': {'default': 2, 'tiers': {'pro': 7, 'premium': 15}},
        'web_search_limit_chars': {'default': 500, 'tiers': {'pro': 3000, 'premium': 10000}},
        'data_analysis_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'summarization_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'chart_generation_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'sentiment_analysis_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'analytics_access': {'default': False, 'roles': {'admin': True}},
        'analytics_charts_enabled': {'default': False, 'roles': {'admin': True}},
        'analytics_user_specific_access': {'default': False, 'roles': {'admin': True}},
    }
}

class UserManager:
    """
    Manages user profiles in Firestore, including creation, retrieval, and updates.
    Handles user tiers and roles.
    This class is intended for backend use (FastAPI).
    """
    def __init__(self, firestore_manager: Any, cloud_storage_utils: Any):
        self.firestore_manager = firestore_manager
        self.cloud_storage_utils = cloud_storage_utils # For future use, e.g., profile pictures
        logger.info("UserManager instantiated.")

    async def create_user_profile(self, uid: str, email: str, username: str, initial_tier: str = "free", initial_roles: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Creates a new user profile document in Firestore.
        """
        if initial_roles is None:
            initial_roles = ["user"] # Default role for new users

        user_profile_data = {
            "uid": uid,
            "email": email,
            "username": username,
            "tier": initial_tier,
            "roles": initial_roles,
            "created_at": datetime.now(timezone.utc),
            "last_login_at": datetime.now(timezone.utc),
            "profile_data": {} # Placeholder for additional profile fields
        }
        try:
            # Path: /users/{uid}
            await self.firestore_manager.set_doc(f"users/{uid}", user_profile_data, merge=True)
            logger.info(f"User profile created/updated for UID: {uid}")
            # Log analytics event
            await log_event(
                'user_profile_creation',
                {'uid': uid, 'email': email, 'username': username, 'tier': initial_tier},
                user_id=uid,
                success=True,
                log_from_backend=True
            )
            return {"success": True, "message": "User profile created successfully."}
        except Exception as e:
            logger.error(f"Error creating user profile for UID {uid}: {e}", exc_info=True)
            # Log analytics event for failure
            await log_event(
                'user_profile_creation',
                {'uid': uid, 'email': email, 'username': username, 'tier': initial_tier, 'error': str(e)},
                user_id=uid,
                success=False,
                error_message=str(e),
                log_from_backend=True
            )
            return {"success": False, "message": f"Failed to create user profile: {e}"}

    async def get_user(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a user's profile from Firestore.
        """
        try:
            user_data = await self.firestore_manager.get_doc("users", uid) # Pass collection and doc_id separately
            if user_data:
                # Update last_login_at if the user is being retrieved (implying a login or active session)
                await self.update_user_profile(uid, {"last_login_at": datetime.now(timezone.utc)})
            return user_data
        except Exception as e:
            logger.error(f"Error retrieving user profile for UID {uid}: {e}", exc_info=True)
            return None

    async def get_all_users_admin(self) -> Dict[str, Any]:
        """
        Retrieves all user profiles (admin only).
        """
        try:
            users = await self.firestore_manager.get_collection("users") # Assuming get_collection exists
            return {"success": True, "users": users}
        except Exception as e:
            logger.error(f"Error retrieving all user profiles: {e}", exc_info=True)
            return {"success": False, "message": f"Failed to retrieve users: {e}"}

    async def update_user_profile(self, uid: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates specific fields in a user's profile.
        """
        try:
            await self.firestore_manager.update_doc(f"users/{uid}", updates)
            logger.info(f"User profile updated for UID: {uid}. Fields: {list(updates.keys())}")
            # Log analytics event
            await log_event(
                'user_profile_update',
                {'uid': uid, 'updated_fields': list(updates.keys())},
                user_id=uid,
                success=True,
                log_from_backend=True
            )
            return {"success": True, "message": "User profile updated successfully."}
        except Exception as e:
            logger.error(f"Error updating user profile for UID {uid}: {e}", exc_info=True)
            # Log analytics event for failure
            await log_event(
                'user_profile_update',
                {'uid': uid, 'updated_fields': list(updates.keys()), 'error': str(e)},
                user_id=uid,
                success=False,
                error_message=str(e),
                log_from_backend=True
            )
            return {"success": False, "message": f"Failed to update user profile: {e}"}

    async def update_user_roles_and_tier(self, uid: str, new_tier: Optional[str] = None, new_roles: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Updates a user's tier and/or roles.
        """
        updates = {}
        if new_tier:
            updates["tier"] = new_tier
        if new_roles is not None: # Allow setting roles to an empty list
            updates["roles"] = new_roles
        
        if not updates:
            return {"success": False, "message": "No tier or roles provided for update."}

        result = await self.update_user_profile(uid, updates)
        # Log this specific admin action
        await log_event(
            'admin_user_roles_tier_update',
            {'target_uid': uid, 'new_tier': new_tier, 'new_roles': new_roles, 'admin_action_result': result.get('message')},
            user_id=None, # Admin user ID will be logged by the FastAPI endpoint calling this
            success=result['success'],
            error_message=result.get('message') if not result['success'] else None,
            log_from_backend=True
        )
        return result

# --- Re-implemented get_user_tier_capability to fetch from Firestore ---
async def get_user_tier_capability(user_id: str, capability_key: str, default_value: Any = None) -> Any:
    """
    Determines user capabilities based on their actual tier and roles fetched from Firestore.
    This function requires an initialized UserManager and FirestoreManager.
    """
    # Import UserManager and FirestoreManager locally to avoid circular dependencies
    # if this function is called very early in the app startup before full initialization.
    # In a real app, you might pass these as arguments or use a global singleton pattern.
    from database.firestore_manager import FirestoreManager
    from shared_tools.cloud_storage_utils import CloudStorageUtilsWrapper
    from config.config_manager import config_manager

    # Re-instantiate UserManager and FirestoreManager if they aren't globally available
    # (This is a simplified approach for this environment. In a larger app, you'd
    # likely pass the already initialized instances or use FastAPI's dependency injection
    # for this function if it were an endpoint dependency itself.)
    try:
        # Attempt to get the globally initialized instances if available
        global_db_client = firestore.client(firebase_admin.get_app())
        global_firestore_manager = FirestoreManager(global_db_client)
        global_cloud_storage_utils = CloudStorageUtilsWrapper(config_manager)
        manager = UserManager(global_firestore_manager, global_cloud_storage_utils)
    except Exception:
        # Fallback for testing or if Firebase app isn't initialized yet
        logger.warning("Firebase app not fully initialized for get_user_tier_capability. Using mock data.")
        # Fallback to the old mock logic if Firebase/Firestore isn't ready
        return _RBAC_CAPABILITIES_CONFIG['capabilities'].get(capability_key, {}).get('default', default_value)


    user_profile = await manager.get_user(user_id)
    if not user_profile:
        logger.warning(f"User profile not found for UID {user_id}. Returning default capability for {capability_key}.")
        return _RBAC_CAPABILITIES_CONFIG['capabilities'].get(capability_key, {}).get('default', default_value)

    user_tier = user_profile.get('tier', 'free')
    user_roles = user_profile.get('roles', [])

    capability_config = _RBAC_CAPABILITIES_CONFIG.get('capabilities', {}).get(capability_key)
    if not capability_config:
        return default_value

    # Admin role overrides all
    if "admin" in user_roles:
        if isinstance(capability_config.get('default'), bool): return True
        if isinstance(capability_config.get('default'), (int, float)): return float('inf')

    # Check roles first
    for role in user_roles:
        if role in capability_config.get('roles', {}):
            return capability_config['roles'][role]
    
    # Then check tiers
    if user_tier in capability_config.get('tiers', {}):
        return capability_config['tiers'][user_tier]

    return capability_config.get('default', default_value)


# CLI Test (optional) - This part remains the same for testing purposes
if __name__ == "__main__":
    import asyncio
    from unittest.mock import MagicMock, AsyncMock, patch
    import sys
    import os
    import json
    import firebase_admin # Import firebase_admin for the test setup
    from firebase_admin import credentials, firestore, auth as firebase_auth

    logging.basicConfig(level=logging.DEBUG)

    # Mock Firebase Admin SDK initialization for CLI test
    # This part is crucial for making the UserManager testable outside FastAPI
    if not firebase_admin._apps:
        # Create a dummy credential object for local testing
        dummy_cred_path = "/tmp/dummy_firebase_admin_cert_um.json"
        dummy_cred_content = {
            "type": "service_account",
            "project_id": "test-project-um",
            "private_key_id": "dummy_key_id_um",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMOCK_PRIVATE_KEY_UM\n-----END PRIVATE KEY-----\n",
            "client_email": "dummy_um@test-project-um.iam.gserviceaccount.com",
            "client_id": "dummy_client_id_um",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/dummy_um.iam.gserviceaccount.com",
            "universe_domain": "googleapis.com"
        }
        with open(dummy_cred_path, "w") as f:
            json.dump(dummy_cred_content, f)
        os.environ["FIREBASE_ADMIN_CERT"] = json.dumps(dummy_cred_content) # Set env var for initialization

        try:
            firebase_admin.initialize_app(credentials.Certificate(dummy_cred_path), name="test-app-um")
            logger.info("Firebase Admin SDK initialized for UserManager CLI test.")
        except ValueError as e:
            logger.warning(f"Firebase Admin SDK already initialized or error in UserManager test: {e}")
        finally:
            # Clean up dummy cred file
            if os.path.exists(dummy_cred_path):
                os.remove(dummy_cred_path)

    async def run_tests():
        # Mock FirestoreManager and CloudStorageUtilsWrapper for UserManager tests
        mock_firestore_manager = MagicMock()
        mock_cloud_storage_utils = MagicMock()

        # Configure mock_firestore_manager methods to be async mocks
        mock_firestore_manager.set_doc = AsyncMock(return_value=True)
        mock_firestore_manager.get_doc = AsyncMock(return_value=None) # Default: user not found
        mock_firestore_manager.update_doc = AsyncMock(return_value=True)
        mock_firestore_manager.get_collection = AsyncMock(return_value=[])

        # Configure mock log_event to be an async mock
        with patch('utils.analytics_tracker.log_event', new_callable=AsyncMock) as mock_log_event:
            manager = UserManager(mock_firestore_manager, mock_cloud_storage_utils)

            # Test create_user_profile
            print("\n--- Testing create_user_profile ---")
            test_uid = "test_user_123"
            test_email = "test@example.com"
            test_username = "tester"
            await manager.create_user_profile(test_uid, test_email, test_username)
            mock_firestore_manager.set_doc.assert_called_once_with(
                f"users/{test_uid}",
                {
                    "uid": test_uid,
                    "email": test_email,
                    "username": test_username,
                    "tier": "free",
                    "roles": ["user"],
                    "created_at": Any, # Use Any for datetime objects
                    "last_login_at": Any,
                    "profile_data": {}
                },
                merge=True
            )
            mock_log_event.assert_called_once()
            assert mock_log_event.call_args[0][0] == 'user_profile_creation'
            mock_firestore_manager.set_doc.reset_mock()
            mock_log_event.reset_mock()

            # Test get_user (user not found)
            print("\n--- Testing get_user (not found) ---")
            user = await manager.get_user("non_existent_user")
            print(f"User (not found): {user}")
            assert user is None

            # Test get_user (user found)
            print("\n--- Testing get_user (found) ---")
            mock_user_data = {
                "uid": test_uid,
                "email": test_email,
                "username": test_username,
                "tier": "pro",
                "roles": ["user"],
                "created_at": datetime.now(timezone.utc),
                "last_login_at": datetime.now(timezone.utc),
                "profile_data": {"phone": "123-456-7890"}
            }
            mock_firestore_manager.get_doc.return_value = mock_user_data
            user = await manager.get_user(test_uid)
            print(f"User (found): {user}")
            assert user is not None
            assert user['uid'] == test_uid
            mock_firestore_manager.get_doc.reset_mock()
            mock_firestore_manager.update_doc.reset_mock() # update_user_profile is called internally

            # Test update_user_profile
            print("\n--- Testing update_user_profile ---")
            updates = {"username": "new_tester_name", "profile_data.bio": "Updated bio"}
            success = await manager.update_user_profile(test_uid, updates)
            print(f"Update profile successful: {success}")
            assert success
            mock_firestore_manager.update_doc.assert_called_once_with(f"users/{test_uid}", updates)
            mock_log_event.assert_called_once()
            assert mock_log_event.call_args[0][0] == 'user_profile_update'
            mock_firestore_manager.update_doc.reset_mock()
            mock_log_event.reset_mock()

            # Test update_user_roles_and_tier
            print("\n--- Testing update_user_roles_and_tier ---")
            success = await manager.update_user_roles_and_tier(test_uid, new_tier="premium", new_roles=["user", "special"])
            print(f"Update roles/tier successful: {success}")
            assert success
            mock_firestore_manager.update_doc.assert_called_once_with(f"users/{test_uid}", {"tier": "premium", "roles": ["user", "special"]})
            mock_log_event.assert_called_once()
            assert mock_log_event.call_args[0][0] == 'admin_user_roles_tier_update'
            mock_firestore_manager.update_doc.reset_mock()
            mock_log_event.reset_mock()

            # Test get_user_tier_capability (real lookup)
            print("\n--- Testing get_user_tier_capability (real lookup) ---")
            # Mock get_user to return a specific user profile
            mock_firestore_manager.get_doc.return_value = {
                "uid": "user_with_pro_tier",
                "email": "pro@example.com",
                "username": "ProUser",
                "tier": "pro",
                "roles": ["user"],
                "created_at": datetime.now(timezone.utc),
                "last_login_at": datetime.now(timezone.utc),
                "profile_data": {}
            }
            
            # Patch the UserManager and FirestoreManager instances used inside get_user_tier_capability
            with patch('utils.user_manager.UserManager', autospec=True) as MockUserManager:
                MockUserManager.return_value.get_user = AsyncMock(return_value=mock_firestore_manager.get_doc.return_value)
                
                can_upload = await get_user_tier_capability("user_with_pro_tier", "document_upload_enabled")
                print(f"User with pro tier can upload documents: {can_upload}")
                assert can_upload is True

                max_results = await get_user_tier_capability("user_with_pro_tier", "web_search_max_results")
                print(f"User with pro tier max web search results: {max_results}")
                assert max_results == 7

                # Test admin capabilities
                mock_firestore_manager.get_doc.return_value = {
                    "uid": "admin_user_id",
                    "email": "admin@example.com",
                    "username": "Admin",
                    "tier": "admin",
                    "roles": ["user", "admin"],
                    "created_at": datetime.now(timezone.utc),
                    "last_login_at": datetime.now(timezone.utc),
                    "profile_data": {}
                }
                MockUserManager.return_value.get_user = AsyncMock(return_value=mock_firestore_manager.get_doc.return_value)

                can_analytics = await get_user_tier_capability("admin_user_id", "analytics_access")
                print(f"Admin can access analytics: {can_analytics}")
                assert can_analytics is True

                admin_max_results = await get_user_tier_capability("admin_user_id", "web_search_max_results")
                print(f"Admin max web search results: {admin_max_results}")
                assert admin_max_results == float('inf')


            print("\nAll UserManager tests completed.")

    if __name__ == "__main__":
        asyncio.run(run_tests())


