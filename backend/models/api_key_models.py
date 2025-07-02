# backend/models/api_key_models.py

from pydantic import BaseModel, Field
from typing import Optional, Dict

class ApiKeyCreate(BaseModel):
    """
    Pydantic model for creating a new API key entry.
    """
    service_name: str = Field(..., description="Name of the service this API key belongs to (e.g., 'OpenAI', 'SerpAPI').")
    key_value: str = Field(..., description="The actual API key value (should be securely handled/encrypted).")
    description: Optional[str] = Field(None, description="Optional description for the API key.")
    is_active: bool = Field(True, description="Whether the API key is currently active.")
    # Consider adding:
    # created_by: str = Field(..., description="User ID of the admin who created this key.")
    # expires_at: Optional[str] = Field(None, description="Expiration date for the key (YYYY-MM-DD).")

class ApiKeyUpdate(BaseModel):
    """
    Pydantic model for updating an existing API key entry.
    All fields are optional for partial updates.
    """
    key_value: Optional[str] = Field(None, description="New API key value.")
    description: Optional[str] = Field(None, description="Updated description for the API key.")
    is_active: Optional[bool] = Field(None, description="New active status for the API key.")

class ApiKeyResponse(BaseModel):
    """
    Pydantic model for returning API key information (excluding sensitive value if not authorized).
    """
    key_id: str = Field(..., description="Unique identifier for the API key entry.")
    service_name: str = Field(..., description="Name of the service this API key belongs to.")
    description: Optional[str] = Field(None, description="Description for the API key.")
    is_active: bool = Field(..., description="Whether the API key is currently active.")
    # For security, the actual 'key_value' should only be returned to highly privileged users
    # or not at all via API, only for direct input.
    # key_value: Optional[str] = Field(None, description="The actual API key value (only for privileged access).")

    class Config:
        from_attributes = True
