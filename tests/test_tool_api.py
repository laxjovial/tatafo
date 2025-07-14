import pytest
from fastapi.testclient import TestClient
from backend.main import app
from unittest.mock import patch, AsyncMock

@pytest.fixture
def client():
    return TestClient(app)

@patch('backend.services.llm_service.LLMService.run_tool_by_name', new_callable=AsyncMock)
def test_run_tool_endpoint_success(mock_run_tool, client):
    mock_run_tool.return_value = "Tool executed successfully"
    response = client.post(
        "/tools/run-tool",
        json={"tool_name": "test_tool", "tool_args": {"arg1": "value1"}},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json() == {"result": "Tool executed successfully", "success": True}

@patch('backend.services.llm_service.LLMService.run_tool_by_name', new_callable=AsyncMock)
def test_run_tool_endpoint_not_found(mock_run_tool, client):
    mock_run_tool.side_effect = ValueError("Tool not found")
    response = client.post(
        "/tools/run-tool",
        json={"tool_name": "non_existent_tool", "tool_args": {}},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Tool not found"}

@patch('backend.services.llm_service.LLMService.run_tool_by_name', new_callable=AsyncMock)
def test_run_tool_endpoint_permission_denied(mock_run_tool, client):
    mock_run_tool.side_effect = PermissionError("Permission denied")
    response = client.post(
        "/tools/run-tool",
        json={"tool_name": "admin_tool", "tool_args": {}},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 403
    assert response.json() == {"detail": "Permission denied"}

@patch('backend.services.llm_service.LLMService.run_tool_by_name', new_callable=AsyncMock)
def test_run_tool_endpoint_internal_error(mock_run_tool, client):
    mock_run_tool.side_effect = Exception("Internal server error")
    response = client.post(
        "/tools/run-tool",
        json={"tool_name": "test_tool", "tool_args": {}},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "An unexpected error occurred: Internal server error"}
