# utils/user_manager.py

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import asyncio
from fastapi import Depends # Ensure 'Depends' is imported for injection

# Import analytics_tracker for logging events
from utils.analytics_tracker import log_event

logger = logging.getLogger(__name__)

# Firebase imports (assuming these are globally available or handled by the main app)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, auth as firebase_auth
    from database.firestore_manager import FirestoreManager # Keep this import
    from shared_tools.cloud_storage_utils import CloudStorageUtilsWrapper
    from config.config_manager import config_manager
    from backend.models.user_models import UserProfile # Also ensure UserProfile is imported for type hinting
except ImportError:
    logger.warning("Firebase Admin SDK or related modules not found. `get_user_tier_capability` will use mock data for CLI tests.")
    firebase_admin = None
    firestore = None
    firebase_auth = None
    FirestoreManager = None
    CloudStorageUtilsWrapper = None
    config_manager = None
    # Define a dummy UserProfile for testing if import fails
    class UserProfile:
        user_id: str = "dummy"
        username: str = "dummy_user"
        email: str = "dummy@example.com"
        tier: str = "free"
        roles: List[str] = []
        created_at: datetime = datetime.now(timezone.utc)
        last_login_at: datetime = datetime.now(timezone.utc)
        profile_data: Dict[str, Any] = {}


# --- RBAC Capabilities Configuration (Centralized) ---
# This dictionary defines what capabilities each tier/role has.
# This would typically be loaded from a config file or database in a larger app.
_RBAC_CAPABILITIES_CONFIG = {
    'capabilities': {
        'finance_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'crypto_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'medical_tool_access': {'default': False, 'roles': {'premium': True, 'admin': True}},
        'document_upload_limit': {'default': 5, 'tiers': {'pro': 50, 'premium': 500, 'admin': float('inf')}}, # Max documents
        'web_search_max_results': {'default': 3, 'tiers': {'pro': 10, 'premium': 20, 'admin': float('inf')}}, # Max web search results
        'analytics_access': {'default': False, 'roles': {'admin': True}},
        'llm_temperature_control_enabled': {'default': False, 'roles': {'premium': True, 'admin': True}},
        'llm_default_model_name': {'default': 'gemini-1.5-flash', 'roles': {'premium': 'gemini-1.5-pro', 'admin': 'gemini-1.5-pro'}}, # LLM Model selection
        'api_key_management_enabled': {'default': False, 'roles': {'premium': True, 'admin': True}},
        'tool_creation_enabled': {'default': False, 'roles': {'admin': True}},
        'code_execution_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'data_visualization_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'sentiment_analysis_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'export_results_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'dynamic_api_limit_adjustment_control': {'default': False, 'roles': {'admin': True}},
        'rag_query_k_limit': {'default': 3, 'tiers': {'pro': 5, 'premium': 10, 'admin': float('inf')}}, # Max documents to retrieve from RAG
    },
    'roles_priority': ['admin', 'premium', 'pro', 'user', 'guest'] # Order of role evaluation
}

class UserManager:
    """
    Manages user profiles and RBAC (Role-Based Access Control).
    """
    # Inject FirestoreManager here
    def __init__(self, firestore_manager: FirestoreManager = Depends(FirestoreManager)):
        self.firestore_manager = firestore_manager
        logger.info("UserManager initialized.")

    async def get_user(self, user_id: str) -> Optional[UserProfile]:
        """
        Retrieves a user's profile from Firestore.
        """
        if self.firestore_manager:
            user_data = await self.firestore_manager.get_doc("users", user_id)
            if user_data:
                return UserProfile(**user_data)
        return None

    async def create_or_update_user(self, user_id: str, email: str, username: Optional[str] = None) -> UserProfile:
        """
        Creates a new user profile or updates an existing one on first login.
        """
        user_profile = await self.get_user(user_id)
        current_time = datetime.now(timezone.utc)

        if user_profile:
            # Update existing user's last login
            update_data = {"last_login_at": current_time}
            if username and user_profile.username == "Guest User": # Update username if it's still default
                update_data["username"] = username
            await self.firestore_manager.update_doc("users", user_id, update_data, merge=True)
            user_profile.last_login_at = current_time
            if username and user_profile.username == "Guest User":
                user_profile.username = username
            log_event(event_type="user_login", user_id=user_id, details={"method": "existing_user_update"})
            return user_profile
        else:
            # Create new user
            new_user_data = {
                "user_id": user_id,
                "email": email,
                "username": username if username else "Guest User",
                "tier": "free", # Default tier
                "roles": ["user"], # Default role
                "created_at": current_time,
                "last_login_at": current_time,
                "profile_data": {} # Empty dictionary for additional profile data
            }
            await self.firestore_manager.add_doc("users", new_user_data, doc_id=user_id)
            log_event(event_type="user_signup", user_id=user_id, details={"method": "new_user_creation"})
            return UserProfile(**new_user_data)

    async def update_user_profile_data(self, user_id: str, profile_data: Dict[str, Any]) -> UserProfile:
        """
        Updates specific profile data for a user.
        """
        await self.firestore_manager.update_doc("users", user_id, {"profile_data": profile_data}, merge=True)
        updated_user_data = await self.get_user(user_id)
        if not updated_user_data:
            raise ValueError("User not found after update.")
        return updated_user_data

    async def update_user_tier_and_roles(self, user_id: str, tier: Optional[str] = None, roles: Optional[List[str]] = None) -> UserProfile:
        """
        Updates a user's tier and/or roles. (Admin function)
        """
        update_data = {}
        if tier:
            update_data["tier"] = tier
        if roles is not None: # Allow setting roles to empty list
            update_data["roles"] = roles

        if not update_data:
            raise ValueError("No tier or roles provided for update.")

        await self.firestore_manager.update_doc("users", user_id, update_data, merge=True)
        updated_user_data = await self.get_user(user_id)
        if not updated_user_data:
            raise ValueError("User not found after tier/role update.")
        return updated_user_data

def get_user_tier_capability(user_id: str, capability_name: str, user_profile: Optional[UserProfile] = None) -> Any:
    """
    Checks if a user's tier or role grants them a specific capability.
    This function should be called with an already loaded UserProfile for efficiency.
    If user_profile is None, it will attempt to fetch it (less efficient).
    """
    # Fallback for testing environments where FirestoreManager might be mocked or not initialized
    if FirestoreManager is None:
        logger.debug("FirestoreManager not available, using mock capabilities.")
        # Provide sensible defaults for CLI testing without Firebase
        if capability_name == "web_search_max_results": return 3
        if capability_name == "llm_default_model_name": return 'gemini-1.5-flash'
        return _RBAC_CAPABILITIES_CONFIG['capabilities'].get(capability_name, {}).get('default', False)

    # In a real FastAPI app, user_profile should be passed via Depends(get_current_user)
    # For standalone use, if user_profile is not provided, fetch it (less efficient)
    if not user_profile:
        # This part should ideally be avoided in hot paths of FastAPI requests.
        # It's here for standalone script compatibility.
        manager = UserManager(firestore_manager=FirestoreManager()) # Ensure FirestoreManager is initialized
        user_profile_data = asyncio.run(manager.get_user(user_id))
        if user_profile_data:
            user_profile = user_profile_data
        else:
            logger.warning(f"User {user_id} not found, using default capabilities.")
            user_profile = UserProfile(user_id=user_id, email="guest@example.com", username="Guest User", tier="free", roles=["user"])

    capabilities = _RBAC_CAPABILITIES_CONFIG['capabilities']
    capability_config = capabilities.get(capability_name)

    if not capability_config:
        logger.warning(f"Capability '{capability_name}' not defined in RBAC config. Returning default (False).")
        return False

    # Check tier-specific values first (e.g., limits like document_upload_limit)
    if 'tiers' in capability_config:
        tier_value = capability_config['tiers'].get(user_profile.tier)
        if tier_value is not None:
            return tier_value
        # If specific tier not found, fall back to default for this capability
        return capability_config.get('default')

    # Then check role-based access
    if 'roles' in capability_config:
        for role in _RBAC_CAPABILITIES_CONFIG['roles_priority']:
            if role in user_profile.roles:
                if role in capability_config['roles']:
                    return capability_config['roles'][role]
        # If no matching role found, fall back to default for this capability
        return capability_config.get('default')

    # If neither 'tiers' nor 'roles' are defined, return the default value
    return capability_config.get('default')


    if __name__ == '__main__':
        # This block is for local testing/demonstration.
        # It won't be executed in the context of the FastAPI application.
        # You'll need to set up Firebase Admin SDK initialization if you want
        # to test this with a real Firestore instance locally.
        print("Running UserManager tests...")
        
        # Mock Firebase Admin SDK for tests if not initialized
        if not firebase_admin._apps:
            print("Initializing dummy Firebase app for tests...")
            from firebase_admin import initialize_app, firestore
            from firebase_admin import credentials
            
            # Using a mock credential or service account key for testing if available
            # For actual tests, you'd point to a test project's service account key
            try:
                # Attempt to use a dummy credential for initialization
                cred = credentials.Certificate({
                    "type": "service_account",
                    "project_id": "test-project-id",
                    "private_key_id": "dummy_key_id",
                    "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
                    "client_email": "test@test-project-id.iam.gserviceaccount.com",
                    "client_id": "dummy_client_id",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/test@test-project-id.iam.gserviceaccount.com"
                })
                initialize_app(cred, name="test-app")
            except ValueError:
                print("Firebase app already initialized.")
            except Exception as e:
                print(f"Could not initialize Firebase for tests (this is normal if running in a non-Firebase environment): {e}")

        
        # Test cases for UserManager
        async def run_tests():
            print("\n--- Testing UserManager ---")
            mock_firestore_instance = type('obj', (object,), {
                'get_doc': AsyncMock(return_value=None),
                'add_doc': AsyncMock(return_value=None),
                'update_doc': AsyncMock(return_value=None),
                'delete_doc': AsyncMock(return_value=True),
            })()
            
            # Create a mock FirestoreManager instance to pass
            mock_firestore_manager = FirestoreManager(db_instance=mock_firestore_instance, auth_instance=None)
            
            # Create UserManager with the mock FirestoreManager
            mock_user_manager_instance = UserManager(firestore_manager=mock_firestore_manager)


            # Test 1: Create a new user
            print("\n--- Test 1: Create a new user ---")
            new_user_id = "test_user_123"
            new_user_email = "test123@example.com"
            new_user_username = "TestUser"
            user_profile = await mock_user_manager_instance.create_or_update_user(new_user_id, new_user_email, new_user_username)
            print(f"Created User: {user_profile.model_dump()}")
            assert user_profile.user_id == new_user_id
            mock_firestore_instance.add_doc.assert_called_once()
            mock_firestore_instance.add_doc.reset_mock() # Reset mock for next test

            # Test 2: Update an existing user (simulated)
            print("\n--- Test 2: Update an existing user ---")
            # Mock get_doc to return the newly created user for update test
            mock_firestore_instance.get_doc.return_value = {
                "user_id": new_user_id,
                "email": new_user_email,
                "username": new_user_username,
                "tier": "free",
                "roles": ["user"],
                "created_at": user_profile.created_at.isoformat(),
                "last_login_at": (user_profile.last_login_at - timedelta(days=1)).isoformat(), # Simulate old last login
                "profile_data": {}
            }
            updated_user_profile = await mock_user_manager_instance.create_or_update_user(new_user_id, new_user_email)
            print(f"Updated User: {updated_user_profile.model_dump()}")
            assert updated_user_profile.user_id == new_user_id
            assert updated_user_profile.last_login_at > user_profile.last_login_at - timedelta(days=1)
            mock_firestore_instance.update_doc.assert_called_once()
            mock_firestore_instance.get_doc.reset_mock() # Reset mock for next test
            mock_firestore_instance.update_doc.reset_mock()

            # Test 3: Update user profile data
            print("\n--- Test 3: Update user profile data ---")
            mock_firestore_instance.get_doc.return_value = {
                "user_id": new_user_id,
                "email": new_user_email,
                "username": new_user_username,
                "tier": "free",
                "roles": ["user"],
                "created_at": user_profile.created_at.isoformat(),
                "last_login_at": user_profile.last_login_at.isoformat(),
                "profile_data": {"fav_color": "blue"}
            }
            updated_profile_data = {"fav_color": "green", "age": 30}
            updated_user = await mock_user_manager_instance.update_user_profile_data(new_user_id, updated_profile_data)
            print(f"Updated User Profile Data: {updated_user.model_dump()}")
            assert updated_user.profile_data["fav_color"] == "green"
            assert updated_user.profile_data["age"] == 30
            mock_firestore_instance.update_doc.assert_called_once_with("users", new_user_id, {"profile_data": updated_profile_data}, merge=True)
            mock_firestore_instance.get_doc.reset_mock()
            mock_firestore_instance.update_doc.reset_mock()

            # Test 4: Update user tier and roles
            print("\n--- Test 4: Update user tier and roles ---")
            mock_firestore_instance.get_doc.return_value = {
                "user_id": new_user_id,
                "email": new_user_email,
                "username": new_user_username,
                "tier": "pro",
                "roles": ["user", "pro"],
                "created_at": user_profile.created_at.isoformat(),
                "last_login_at": user_profile.last_login_at.isoformat(),
                "profile_data": updated_user.profile_data
            }
            updated_user_tier_roles = await mock_user_manager_instance.update_user_tier_and_roles(new_user_id, tier="pro", roles=["user", "pro"])
            print(f"Updated User Tier/Roles: {updated_user_tier_roles.model_dump()}")
            assert updated_user_tier_roles.tier == "pro"
            assert "pro" in updated_user_tier_roles.roles
            mock_firestore_instance.update_doc.assert_called_once_with("users", new_user_id, {"tier": "pro", "roles": ["user", "pro"]}, merge=True)
            mock_firestore_instance.get_doc.reset_mock()
            mock_firestore_instance.update_doc.reset_mock()
            

            # Test RBAC capabilities (using get_user_tier_capability)
            print("\n--- Test RBAC Capabilities ---")
            mock_user_free = UserProfile(user_id="free_user", email="free@example.com", username="Free User", tier="free", roles=["user"])
            mock_user_pro = UserProfile(user_id="pro_user", email="pro@example.com", username="Pro User", tier="pro", roles=["user", "pro"])
            mock_user_admin = UserProfile(user_id="admin_user", email="admin@example.com", username="Admin User", tier="admin", roles=["user", "admin"])

            # Mock get_user to return specific profiles for get_user_tier_capability tests
            mock_firestore_manager_for_rbac_tests = FirestoreManager(db_instance=mock_firestore_instance, auth_instance=None)
            mock_user_manager_instance_for_rbac = UserManager(firestore_manager=mock_firestore_manager_for_rbac_tests)

            # Test 5: Free user access to finance tool
            mock_firestore_instance.get_doc.return_value = mock_user_free.model_dump()
            can_access_finance = get_user_tier_capability("free_user", "finance_tool_access", user_profile=mock_user_free)
            print(f"Free user can access finance tool: {can_access_finance}")
            assert can_access_finance is False

            # Test 6: Pro user access to finance tool
            mock_firestore_instance.get_doc.return_value = mock_user_pro.model_dump()
            can_access_finance_pro = get_user_tier_capability("pro_user", "finance_tool_access", user_profile=mock_user_pro)
            print(f"Pro user can access finance tool: {can_access_finance_pro}")
            assert can_access_finance_pro is True

            # Test 7: Admin user access to analytics
            mock_firestore_instance.get_doc.return_value = mock_user_admin.model_dump()
            can_access_analytics = get_user_tier_capability("admin_user", "analytics_access", user_profile=mock_user_admin)
            print(f"Admin user can access analytics: {can_access_analytics}")
            assert can_access_analytics is True

            # Test 8: Document upload limit for Free user
            mock_firestore_instance.get_doc.return_value = mock_user_free.model_dump()
            doc_limit_free = get_user_tier_capability("free_user", "document_upload_limit", user_profile=mock_user_free)
            print(f"Free user document limit: {doc_limit_free}")
            assert doc_limit_free == 5

            # Test 9: Web search max results for Premium user
            mock_user_premium = UserProfile(user_id="premium_user", email="premium@example.com", username="Premium User", tier="premium", roles=["user", "premium"])
            mock_firestore_instance.get_doc.return_value = mock_user_premium.model_dump()
            web_search_max_results = get_user_tier_capability("premium_user", "web_search_max_results", user_profile=mock_user_premium)
            print(f"Premium user web search max results: {web_search_max_results}")
            assert web_search_max_results == 20

            # Test 10: LLM default model for Premium user
            llm_model_premium = get_user_tier_capability("premium_user", "llm_default_model_name", user_profile=mock_user_premium)
            print(f"Premium user default LLM model: {llm_model_premium}")
            assert llm_model_premium == 'gemini-1.5-pro'
            
            # Test 11: LLM temperature control for Pro user (should be False)
            llm_temp_control_pro = get_user_tier_capability("pro_user", "llm_temperature_control_enabled", user_profile=mock_user_pro)
            print(f"Pro user LLM temperature control enabled: {llm_temp_control_pro}")
            assert llm_temp_control_pro is False

            # Test 12: LLM temperature control for Premium user (should be True)
            llm_temp_control_premium = get_user_tier_capability("premium_user", "llm_temperature_control_enabled", user_profile=mock_user_premium)
            print(f"Premium user LLM temperature control enabled: {llm_temp_control_premium}")
            assert llm_temp_control_premium is True

            # Test 13: RAG query k limit for Pro user
            rag_k_pro = get_user_tier_capability("pro_user", "rag_query_k_limit", user_profile=mock_user_pro)
            print(f"Pro user RAG query k limit: {rag_k_pro}")
            assert rag_k_pro == 5

            # Test case for admin with get_user_tier_capability directly
            mock_firestore_manager_for_analytics = FirestoreManager(db_instance=mock_firestore_instance, auth_instance=None)
            mock_user_pro_profile = UserProfile(user_id="admin_user_id", username="Admin", tier="admin", roles=["user", "admin"])
            mock_firestore_instance.get_doc.return_value = {
                    "user_id": "admin_user_id",
                    "email": "admin@example.com",
                    "username": "Admin",
                    "tier": "admin",
                    "roles": ["user", "admin"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "last_login_at": datetime.now(timezone.utc).isoformat(),
                    "profile_data": {}
                }
            # Note: For get_user_tier_capability without an explicit user_profile passed,
            # it internally fetches the user. This test is confirming that path.
            # In a real FastAPI app, it's better to pass `user_profile` directly.

            can_analytics = get_user_tier_capability("admin_user_id", "analytics_access")
            print(f"Admin can access analytics (direct call): {can_analytics}")
            assert can_analytics is True

            admin_max_results = get_user_tier_capability("admin_user_id", "web_search_max_results")
            print(f"Admin max web search results (direct call): {admin_max_results}")
            assert admin_max_results == float('inf')

            admin_can_control_temp = get_user_tier_capability("admin_user_id", "llm_temperature_control_enabled")
            print(f"Admin can control LLM temperature (direct call): {admin_can_control_temp}")
            assert admin_can_control_temp is True

            admin_llm_model = get_user_tier_capability("admin_user_id", "llm_default_model_name")
            print(f"Admin default LLM model (direct call): {admin_llm_model}")
            assert admin_llm_model == 'gemini-1.5-pro' # Should be 'gemini-1.5-pro' from config for admin


        print("\nAll UserManager tests completed.")

if __name__ == "__main__":
    # This ensures that `Depends(FirestoreManager)` works correctly if
    # you are running these files directly for testing purposes.
    # For a full FastAPI application, FastAPI's dependency injection system
    # handles the instantiation automatically.
    
    # We need AsyncMock for testing async functions
    from unittest.mock import AsyncMock
    import sys
    
    # Set up a mock for firestore.Increment if it's not available in test context
    class MockIncrement:
        def __init__(self, value):
            self.value = value
        def __repr__(self):
            return f"MockIncrement({self.value})"

    if 'firestore' not in sys.modules or not hasattr(sys.modules['firestore'], 'Increment'):
        # Only patch if firestore.Increment is not genuinely available
        sys.modules['firestore'] = type('obj', (object,), {'Increment': MockIncrement, 'client': lambda: None})()


    asyncio.run(run_tests())
