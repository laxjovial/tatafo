import pytest
from unittest.mock import AsyncMock, patch
from backend.services.api_usage_service import ApiUsageService
from backend.models.user_models import UserProfile

@pytest.fixture
def api_usage_service():
    with patch('backend.services.api_usage_service.FirestoreManager') as mock_firestore_manager, \
         patch('backend.services.api_usage_service.ConfigManager') as mock_config_manager, \
         patch('backend.services.api_usage_service.UserManager') as mock_user_manager:
        service = ApiUsageService(mock_firestore_manager, mock_config_manager, mock_user_manager)
        return service

@pytest.mark.asyncio
async def test_check_api_limit(api_usage_service):
    api_usage_service.get_user_api_usage_document.return_value = {"monthly_usage": 0, "daily_usage": 0}
    api_usage_service.get_api_limits_config.return_value = {"free": {"monthly_calls": 100, "daily_calls": 10}}
    result = await api_usage_service.check_api_limit(UserProfile(user_id="1", tier="free"), "test_api")
    assert result is True

@pytest.mark.asyncio
async def test_increment_api_usage(api_usage_service):
    api_usage_service.get_user_api_usage_document.return_value = {"monthly_usage": 0, "daily_usage": 0, "last_reset_month": "2023-01", "last_reset_day": "2023-01-01"}
    await api_usage_service.increment_api_usage("1", "test_api")
    api_usage_service.firestore_manager.set_user_data_document.assert_called_once()

@pytest.mark.asyncio
async def test_get_user_api_usage(api_usage_service):
    api_usage_service.firestore_manager.get_user_data_document.return_value = {"monthly_usage": 10, "daily_usage": 1}
    result = await api_usage_service.get_user_api_usage("1")
    assert result == {"monthly_usage": 10, "daily_usage": 1}

@pytest.mark.asyncio
async def test_get_api_limits_for_tier(api_usage_service):
    api_usage_service.get_api_limits_config.return_value = {"free": {"monthly_calls": 100, "daily_calls": 10}}
    result = await api_usage_service.get_api_limits_for_tier("free")
    assert result == {"monthly_calls": 100, "daily_calls": 10}

@pytest.mark.asyncio
async def test_update_api_limits(api_usage_service):
    api_usage_service.get_api_limits_config.return_value = {}
    await api_usage_service.update_api_limits({"tier": "free", "limits": {"monthly_calls": 100}})
    api_usage_service.firestore_manager.set_global_config.assert_called_once_with("api_limits", {"limits": {"free": {"monthly_calls": 100}}})
