import pytest
from unittest.mock import AsyncMock, patch
from backend.services.admin_service import AdminService
from backend.models.user_models import UserProfile

@pytest.fixture
def admin_service():
    with patch('backend.services.admin_service.FirestoreManager') as mock_firestore_manager, \
         patch('backend.services.admin_service.UserManager') as mock_user_manager, \
         patch('backend.services.admin_service.CloudStorageUtilsWrapper') as mock_cloud_storage_utils, \
         patch('backend.services.admin_service.ApiUsageService') as mock_api_usage_service:
        service = AdminService(mock_firestore_manager, mock_user_manager, mock_cloud_storage_utils, mock_api_usage_service)
        return service

@pytest.mark.asyncio
async def test_get_all_user_profiles(admin_service):
    admin_service.user_manager.get_all_user_profiles.return_value = {"success": True, "users": [{"uid": "1", "username": "testuser"}]}
    result = await admin_service.get_all_user_profiles()
    assert result == [{"uid": "1", "username": "testuser"}]

@pytest.mark.asyncio
async def test_update_user_profile_admin(admin_service):
    admin_service.user_manager.get_user.return_value = {"uid": "1", "username": "testuser", "roles": ["user"], "tier": "free"}
    admin_service.user_manager.update_user_profile.return_value = {"success": True}
    with patch('backend.services.admin_service.auth') as mock_auth:
        mock_auth.get_user.return_value = AsyncMock(custom_claims={})
        result = await admin_service.update_user_profile_admin("1", {"username": "newuser"}, UserProfile(user_id="admin", roles=["creator"]))
        assert result["username"] == "newuser"

@pytest.mark.asyncio
async def test_update_user_status_admin(admin_service):
    admin_service.user_manager.get_user.return_value = {"uid": "1", "username": "testuser", "roles": ["user"], "tier": "free"}
    admin_service.user_manager.update_user_profile.return_value = {"success": True}
    with patch('backend.services.admin_service.auth') as mock_auth:
        result = await admin_service.update_user_status_admin("1", "disabled", UserProfile(user_id="admin", roles=["creator"]))
        assert result["status"] == "disabled"

@pytest.mark.asyncio
async def test_purge_user_sessions(admin_service):
    with patch('backend.services.admin_service.auth') as mock_auth:
        await admin_service.purge_user_sessions("1", UserProfile(user_id="admin", roles=["creator"]))
        mock_auth.revoke_refresh_tokens.assert_called_once_with("1")

@pytest.mark.asyncio
async def test_grant_admin_access(admin_service):
    with patch('backend.services.admin_service.auth') as mock_auth:
        mock_auth.get_user.return_value = AsyncMock(custom_claims={})
        await admin_service.grant_admin_access("1", {"can_manage_users": True}, False, UserProfile(user_id="admin", roles=["creator"]))
        mock_auth.set_custom_user_claims.assert_called_once_with("1", {"can_manage_users": True, "roles": ["user", "admin"]})

@pytest.mark.asyncio
async def test_get_rbac_capabilities(admin_service):
    admin_service.firestore_manager.get_global_config.return_value = {"capabilities": {"can_do_stuff": True}}
    result = await admin_service.get_rbac_capabilities()
    assert result == {"can_do_stuff": True}

@pytest.mark.asyncio
async def test_update_rbac_capabilities(admin_service):
    admin_service.firestore_manager.get_global_config.return_value = {"capabilities": {}}
    await admin_service.update_rbac_capabilities({"capability_key": "can_do_stuff", "default_value": True}, UserProfile(user_id="admin", roles=["creator"]))
    admin_service.firestore_manager.set_global_config.assert_called_once_with("rbac_capabilities", {"capabilities": {"can_do_stuff": {"default": True, "roles": {}}}})

@pytest.mark.asyncio
async def test_get_tier_hierarchy(admin_service):
    admin_service.firestore_manager.get_global_config.return_value = {"tiers": {"free": 0}}
    result = await admin_service.get_tier_hierarchy()
    assert result == {"free": 0}

@pytest.mark.asyncio
async def test_update_tier_hierarchy(admin_service):
    admin_service.firestore_manager.get_global_config.return_value = {"tiers": {}}
    await admin_service.update_tier_hierarchy({"tier_name": "free", "level": 0}, UserProfile(user_id="admin", roles=["creator"]))
    admin_service.firestore_manager.set_global_config.assert_called_once_with("tiers", {"tiers": {"free": {"level": 0, "description": ""}}})

@pytest.mark.asyncio
async def test_create_global_api_config(admin_service):
    await admin_service.create_global_api_config({"name": "test_api"}, UserProfile(user_id="admin", roles=["creator"]))
    admin_service.firestore_manager.set_global_config_document.assert_called_once()

@pytest.mark.asyncio
async def test_get_global_api_configs(admin_service):
    admin_service.firestore_manager.get_all_global_config_documents.return_value = [{"name": "test_api"}]
    result = await admin_service.get_global_api_configs()
    assert result == [{"name": "test_api"}]

@pytest.mark.asyncio
async def test_update_global_api_config(admin_service):
    admin_service.firestore_manager.get_global_config_document.return_value = {"name": "new_test_api"}
    result = await admin_service.update_global_api_config("1", {"name": "new_test_api"}, UserProfile(user_id="admin", roles=["creator"]))
    admin_service.firestore_manager.update_global_config_document.assert_called_once()
    assert result["name"] == "new_test_api"

@pytest.mark.asyncio
async def test_delete_global_api_config(admin_service):
    await admin_service.delete_global_api_config("1", UserProfile(user_id="admin", roles=["creator"]))
    admin_service.firestore_manager.delete_global_config_document.assert_called_once_with("global_api_configs", "1")

@pytest.mark.asyncio
async def test_update_api_limits(admin_service):
    admin_service.firestore_manager.get_global_config.return_value = {"limits": {}}
    await admin_service.update_api_limits({"tier": "free", "limits": {"monthly_calls": 100}}, UserProfile(user_id="admin", roles=["creator"]))
    admin_service.firestore_manager.set_global_config.assert_called_once_with("api_limits", {"limits": {"free": {"monthly_calls": 100}}})

@pytest.mark.asyncio
async def test_get_unanswered_queries_analytics(admin_service):
    admin_service.firestore_manager.get_all_global_config_documents.return_value = [{"query": "unanswered"}]
    result = await admin_service.get_unanswered_queries_analytics(UserProfile(user_id="admin", roles=["creator"]))
    assert result == [{"query": "unanswered"}]
