import pytest
from fastapi.testclient import TestClient
from backend.main import app
from unittest.mock import patch, AsyncMock

@pytest.fixture
def client():
    return TestClient(app)

@patch('backend.middleware.auth_middleware.auth.verify_id_token', new_callable=AsyncMock)
@patch('backend.api.user_api.UserManager.get_user', new_callable=AsyncMock)
@patch('utils.analytics_tracker.log_event', new_callable=AsyncMock)
def test_get_user_profile(mock_log_event, mock_get_user, mock_verify_id_token, client):
    mock_verify_id_token.return_value = {"uid": "test_uid"}
    mock_get_user.return_value = {"uid": "test_uid", "username": "testuser", "roles": ["user"], "tier": "free"}
    response = client.get(
        "/user/test_uid",
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"

@patch('backend.middleware.auth_middleware.auth.verify_id_token', new_callable=AsyncMock)
@patch('backend.api.user_api.UserManager.get_user', new_callable=AsyncMock)
@patch('backend.api.user_api.UserManager.update_user_profile', new_callable=AsyncMock)
@patch('backend.api.user_api.auth.get_user', new_callable=AsyncMock)
@patch('backend.api.user_api.auth.set_custom_user_claims', new_callable=AsyncMock)
@patch('backend.api.user_api.auth.revoke_refresh_tokens', new_callable=AsyncMock)
@patch('utils.analytics_tracker.log_event', new_callable=AsyncMock)
def test_update_user_profile(mock_log_event, mock_revoke_refresh_tokens, mock_set_custom_user_claims, mock_auth_get_user, mock_update_user_profile, mock_get_user, mock_verify_id_token, client):
    mock_verify_id_token.return_value = {"uid": "test_uid", "roles": []}
    mock_get_user.return_value = {"uid": "test_uid", "username": "testuser", "roles": ["user"], "tier": "free"}
    mock_auth_get_user.return_value = AsyncMock(custom_claims={})
    response = client.put(
        "/user/test_uid",
        json={"username": "newuser"},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json()["username"] == "newuser"

@patch('backend.middleware.auth_middleware.auth.verify_id_token', new_callable=AsyncMock)
@patch('backend.api.user_api.UserManager.get_all_users_admin', new_callable=AsyncMock)
def test_get_all_users_api(mock_get_all_users_admin, mock_verify_id_token, client):
    mock_verify_id_token.return_value = {"uid": "test_uid", "roles": ["admin"]}
    mock_get_all_users_admin.return_value = {"success": True, "users": [{"uid": "1", "username": "testuser"}]}
    response = client.get(
        "/user/",
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json()[0]["username"] == "testuser"

@patch('backend.middleware.auth_middleware.auth.verify_id_token', new_callable=AsyncMock)
@patch('backend.api.user_api.UserManager.get_user', new_callable=AsyncMock)
@patch('utils.user_manager._RBAC_CAPABILITIES_CONFIG', new_callable=dict)
def test_get_user_capabilities_route(mock_rbac_config, mock_get_user, mock_verify_id_token, client):
    mock_verify_id_token.return_value = {"uid": "test_uid"}
    mock_get_user.return_value = {"uid": "test_uid", "tier": "free", "roles": ["user"]}
    mock_rbac_config.get.return_value.get.return_value = {}
    response = client.get(
        "/user/capabilities/test_uid",
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json() == {}
