# utils/user_manager.py

import httpx
import logging
from typing import Dict, Any, Optional, List
import json # For parsing JSON responses
import datetime # To handle date calculations if needed client-side

from config.config_manager import config_manager

logger = logging.getLogger(__name__)

# Base URL for the FastAPI backend
# This should be configured in your config.ini or environment variables
FASTAPI_BASE_URL = config_manager.get("fastapi_base_url", "http://localhost:8000")

# --- Helper function for making authenticated requests ---
async def _make_authenticated_request(
    method: str,
    endpoint: str,
    id_token: str,
    json_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Helper to make authenticated HTTP requests to the FastAPI backend.
    """
    headers = {
        "Authorization": f"Bearer {id_token}",
        "Content-Type": "application/json"
    }
    url = f"{FASTAPI_BASE_URL}{endpoint}"
    
    try:
        async with httpx.AsyncClient() as client:
            if method == "GET":
                response = await client.get(url, headers=headers)
            elif method == "POST":
                response = await client.post(url, headers=headers, json=json_data)
            elif method == "PUT":
                response = await client.put(url, headers=headers, json=json_data)
            elif method == "DELETE":
                response = await client.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status() # Raise an exception for 4xx or 5xx responses
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error for {method} {url}: {e.response.status_code} - {e.response.text}", exc_info=True)
        # Attempt to parse error detail from backend
        try:
            error_detail = e.response.json().get("detail", "An unknown error occurred.")
        except json.JSONDecodeError:
            error_detail = e.response.text or "An unknown error occurred."
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except httpx.RequestError as e:
        logger.error(f"Network error for {method} {url}: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Network error: Could not connect to backend. {e}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during request to {url}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


# --- Custom HTTPException for consistent error handling ---
# Re-define HTTPException if it's not imported from FastAPI (as this is a utility file)
# This allows us to raise consistent errors that Streamlit can catch.
class HTTPException(Exception):
    def __init__(self, status_code: int, detail: Any = None):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP Error {status_code}: {detail}")


# --- Authentication Functions (interacting with backend) ---

async def register_user_backend(email: str, password: str, username: str) -> Dict[str, Any]:
    """
    Registers a new user via the FastAPI backend.
    Returns the backend's AuthResponse on success.
    """
    endpoint = "/auth/register"
    data = {"email": email, "password": password, "username": username}
    logger.info(f"Attempting to register user: {email}")
    try:
        response = await httpx.AsyncClient().post(f"{FASTAPI_BASE_URL}{endpoint}", json=data)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Registration failed for {email}: {e.response.status_code} - {e.response.text}", exc_info=True)
        try:
            error_detail = e.response.json().get("detail", "Registration failed.")
        except json.JSONDecodeError:
            error_detail = e.response.text or "Registration failed."
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except httpx.RequestError as e:
        logger.error(f"Network error during registration for {email}: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Network error: Could not connect to backend during registration. {e}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during registration for {email}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during registration: {e}")


async def login_user_backend(email: str, password: str) -> Dict[str, Any]:
    """
    Logs in a user via the FastAPI backend.
    Returns the backend's AuthResponse (containing id_token and user_id) on success.
    """
    endpoint = "/auth/login"
    data = {"email": email, "password": password}
    logger.info(f"Attempting to log in user: {email}")
    try:
        response = await httpx.AsyncClient().post(f"{FASTAPI_BASE_URL}{endpoint}", json=data)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Login failed for {email}: {e.response.status_code} - {e.response.text}", exc_info=True)
        try:
            error_detail = e.response.json().get("detail", "Login failed.")
        except json.JSONDecodeError:
            error_detail = e.response.text or "Login failed."
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except httpx.RequestError as e:
        logger.error(f"Network error during login for {email}: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Network error: Could not connect to backend during login. {e}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during login for {email}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during login: {e}")


async def change_password_backend(user_id: str, id_token: str, current_password: str, new_password: str) -> Dict[str, Any]:
    """
    Changes a user's password via the FastAPI backend.
    """
    endpoint = f"/auth/change_password/{user_id}"
    data = {"current_password": current_password, "new_password": new_password}
    logger.info(f"Attempting to change password for user: {user_id}")
    return await _make_authenticated_request("POST", endpoint, id_token, json_data=data)

async def forgot_password_backend(email: str) -> Dict[str, Any]:
    """
    Requests a password reset email via the FastAPI backend.
    """
    endpoint = "/auth/forgot_password"
    data = {"email": email}
    logger.info(f"Requesting password reset for email: {email}")
    try:
        response = await httpx.AsyncClient().post(f"{FASTAPI_BASE_URL}{endpoint}", json=data)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Forgot password failed for {email}: {e.response.status_code} - {e.response.text}", exc_info=True)
        try:
            error_detail = e.response.json().get("detail", "Failed to send password reset email.")
        except json.JSONDecodeError:
            error_detail = e.response.text or "Failed to send password reset email."
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except httpx.RequestError as e:
        logger.error(f"Network error during forgot password for {email}: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Network error: Could not connect to backend for password reset. {e}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during forgot password for {email}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during password reset: {e}")


async def reset_password_backend(oob_code: str, new_password: str) -> Dict[str, Any]:
    """
    Resets password using an OOB code via the FastAPI backend.
    """
    endpoint = "/auth/reset_password"
    data = {"oob_code": oob_code, "new_password": new_password}
    logger.info(f"Attempting to reset password with OOB code.")
    try:
        response = await httpx.AsyncClient().post(f"{FASTAPI_BASE_URL}{endpoint}", json=data)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Reset password failed with OOB code: {e.response.status_code} - {e.response.text}", exc_info=True)
        try:
            error_detail = e.response.json().get("detail", "Password reset failed.")
        except json.JSONDecodeError:
            error_detail = e.response.text or "Password reset failed."
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except httpx.RequestError as e:
        logger.error(f"Network error during reset password with OOB code: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Network error: Could not connect to backend for password reset. {e}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during reset password with OOB code: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during password reset: {e}")


# --- User Profile Functions (interacting with backend) ---

async def get_user_info_from_db(user_id: str, id_token: str) -> Dict[str, Any]:
    """
    Fetches user profile information from the FastAPI backend.
    """
    endpoint = f"/users/{user_id}"
    logger.info(f"Fetching user info for {user_id}")
    return await _make_authenticated_request("GET", endpoint, id_token)

async def update_user_info_in_db(user_id: str, id_token: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Updates user profile information via the FastAPI backend.
    """
    endpoint = f"/users/{user_id}"
    logger.info(f"Updating user info for {user_id} with data: {update_data.keys()}")
    return await _make_authenticated_request("PUT", endpoint, id_token, json_data=update_data)


# --- RBAC Capabilities Function (interacting with backend) ---

async def get_user_capabilities(user_id: str, id_token: str) -> Dict[str, Any]:
    """
    Fetches RBAC capabilities for a user from the FastAPI backend.
    """
    endpoint = f"/rbac/capabilities/{user_id}"
    logger.info(f"Fetching RBAC capabilities for {user_id}")
    return await _make_authenticated_request("GET", endpoint, id_token)


# --- Mock/Default Capabilities (Fallback if backend not accessible or for testing) ---
# This dictionary should ideally mirror the structure of RBACCapabilitiesResponse
# and be used only if the backend call fails or for initial setup before backend is fully live.
# In a production setup, the backend is the source of truth for capabilities.
DEFAULT_CAPABILITIES = {
    "llm_temperature_control_enabled": False,
    "llm_default_temperature": 0.7,
    "llm_max_temperature": 1.0,
    "llm_model_selection_enabled": False,
    "llm_default_provider": "openai",
    "llm_default_model_name": "gpt-3.5-turbo",
    "web_search_enabled": False,
    "data_analysis_enabled": False,
    "summarization_enabled": False,
    "chart_generation_enabled": False,
    "sentiment_analysis_enabled": False,
    "document_upload_enabled": False,
    "document_query_enabled": False,
    "document_query_max_results_k": 4,
    "chart_export_enabled": False,
    "finance_tool_access": False,
    "historical_data_access": False,
    "crypto_tool_access": False,
    "news_tool_access": False,
    "medical_tool_access": False,
    "legal_tool_access": False,
    "education_tool_access": False,
    "entertainment_tool_access": False,
    "weather_tool_access": False,
    "travel_tool_access": False,
    "sports_tool_access": False,
    "analytics_access": False,
    "analytics_charts_enabled": False,
    "analytics_user_specific_access": False,
}

