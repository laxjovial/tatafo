import pytest
from fastapi.testclient import TestClient
from backend.main import app
from unittest.mock import patch, AsyncMock

@pytest.fixture
def client():
    return TestClient(app)

@patch('backend.api.auth_api.auth.create_user', new_callable=AsyncMock)
@patch('backend.api.auth_api.auth.set_custom_user_claims', new_callable=AsyncMock)
@patch('backend.api.auth_api.UserManager.create_user_profile', new_callable=AsyncMock)
@patch('utils.analytics_tracker.log_event', new_callable=AsyncMock)
def test_register_user(mock_log_event, mock_create_user_profile, mock_set_custom_user_claims, mock_create_user, client):
    mock_create_user.return_value = AsyncMock(uid="test_uid")
    response = client.post(
        "/auth/register",
        json={"email": "test@example.com", "password": "password", "username": "testuser"}
    )
    assert response.status_code == 201
    assert response.json()["uid"] == "test_uid"

@patch('backend.api.auth_api.auth.generate_password_reset_link', new_callable=AsyncMock)
@patch('utils.analytics_tracker.log_event', new_callable=AsyncMock)
def test_lost_token(mock_log_event, mock_generate_password_reset_link, client):
    mock_generate_password_reset_link.return_value = "reset_link"
    response = client.post(
        "/auth/lost-token",
        json={"email": "test@example.com"}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "If the email is registered, a password reset link has been sent to your inbox."

@patch('backend.api.auth_api.auth.verify_id_token', new_callable=AsyncMock)
@patch('backend.api.auth_api.UserManager.get_user', new_callable=AsyncMock)
@patch('backend.api.auth_api.UserManager.update_last_login', new_callable=AsyncMock)
@patch('utils.analytics_tracker.log_event', new_callable=AsyncMock)
def test_login_user(mock_log_event, mock_update_last_login, mock_get_user, mock_verify_id_token, client):
    mock_verify_id_token.return_value = {"uid": "test_uid"}
    mock_get_user.return_value = {"uid": "test_uid", "status": "active"}
    response = client.post(
        "/auth/login",
        json={"id_token": "test_token"}
    )
    assert response.status_code == 200
    assert response.json()["uid"] == "test_uid"

@patch('backend.api.auth_api.auth.generate_password_reset_link', new_callable=AsyncMock)
@patch('utils.analytics_tracker.log_event', new_callable=AsyncMock)
def test_request_password_reset(mock_log_event, mock_generate_password_reset_link, client):
    mock_generate_password_reset_link.return_value = "reset_link"
    response = client.post(
        "/auth/request-password-reset",
        json={"email": "test@example.com"}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "If the email is registered, a password reset link has been sent to your inbox."

@patch('backend.api.auth_api.auth.confirm_password_reset', new_callable=AsyncMock)
@patch('utils.analytics_tracker.log_event', new_callable=AsyncMock)
def test_reset_password(mock_log_event, mock_confirm_password_reset, client):
    response = client.post(
        "/auth/reset-password",
        json={"token": "test_token", "new_password": "new_password"}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Password reset successfully."

@patch('backend.api.auth_api.auth.update_user', new_callable=AsyncMock)
@patch('backend.middleware.auth_middleware.auth.verify_id_token', new_callable=AsyncMock)
@patch('utils.analytics_tracker.log_event', new_callable=AsyncMock)
def test_change_password(mock_log_event, mock_verify_id_token, mock_update_user, client):
    mock_verify_id_token.return_value = {"uid": "test_uid"}
    response = client.post(
        "/auth/change-password",
        json={"new_password": "new_password"},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Password changed successfully."
