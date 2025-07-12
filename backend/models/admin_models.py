# backend/models/admin_models.py

from pydantic import BaseModel, Field, conint
from typing import List, Dict, Any, Optional

# --- User Management Models ---
class UserUpdateAdmin(BaseModel):
    """
    Model for updating user profiles by an administrator.
    Allows changing username, tier, roles, and status.
    """
    username: Optional[str] = Field(None, description="New username for the user.")
    # Updated description to reflect all defined tiers
    tier: Optional[str] = Field(None, description="New subscription tier for the user (e.g., 'free', 'basic', 'pro', 'elite', 'premium').")
    roles: Optional[List[str]] = Field(None, description="New list of roles for the user (e.g., ['user', 'dev', 'admin']).")
    status: Optional[str] = Field(None, description="New account status (e.g., 'active', 'disabled', 'suspended').")

class UserStatusUpdate(BaseModel):
    """
    Model for an administrator to update a user's account status.
    """
    status: str = Field(..., description="New account status (e.g., 'active', 'disabled', 'suspended').")

class PurgeSessionsRequest(BaseModel):
    """
    Model for requesting to purge sessions for a specific user or all users.
    """
    user_id: Optional[str] = Field(None, description="Optional: The ID of the specific user whose sessions to purge. If not provided, all sessions will be purged (creator-only).")
    purge_all: Optional[bool] = Field(False, description="Set to true to purge all active sessions for all users. Requires creator privileges.")

class GrantAdminAccessRequest(BaseModel):
    """
    Model for the creator to grant specific administrative permissions to another admin.
    This updates the target user's custom claims.
    """
    target_user_id: str = Field(..., description="The ID of the user to whom to grant admin permissions.")
    # Example: {'can_manage_basic_tier': True, 'can_view_all_analytics': True}
    permissions: Dict[str, Any] = Field(..., description="A dictionary of specific administrative permissions to grant or revoke (e.g., {'can_manage_tier_pro': True, 'can_disable_users': False}).")
    # Optional: If you want to replace all existing admin permissions
    replace_all_permissions: Optional[bool] = Field(False, description="If true, replaces all existing admin permissions with the new ones. If false, merges them.")

# --- RBAC Capabilities Management Models ---
class CapabilityUpdate(BaseModel):
    """
    Model for updating a specific RBAC capability or the entire capabilities document.
    If capability_key is provided, it updates that specific capability.
    If not, it expects full_capabilities to replace the entire document.
    """
    capability_key: Optional[str] = Field(None, description="The key of the capability to update (e.g., 'data_analysis_enabled').")
    default_value: Optional[Any] = Field(None, description="The new default value for the capability.")
    roles: Optional[Dict[str, Any]] = Field(None, description="A dictionary of roles and their specific values for this capability.")
    full_capabilities: Optional[Dict[str, Any]] = Field(None, description="Optional: The full capabilities dictionary to replace the existing one. Used if capability_key is not provided.")

# --- Tier Hierarchy Management Models ---
class TierUpdate(BaseModel):
    """
    Model for updating a specific tier or the entire tier hierarchy document.
    If tier_name is provided, it updates that specific tier.
    If not, it expects full_tiers to replace the entire document.
    """
    tier_name: Optional[str] = Field(None, description="The name of the tier to update (e.g., 'pro', 'new_tier').")
    level: Optional[conint(ge=0)] = Field(None, description="The new numerical level for the tier (higher means higher priority).")
    description: Optional[str] = Field(None, description="A new description for the tier.")
    full_tiers: Optional[Dict[str, Any]] = Field(None, description="Optional: The full tiers dictionary to replace the existing one. Used if tier_name is not provided.")

# --- Global/Default API Management Models ---
class GlobalApiConfig(BaseModel):
    """
    Model for defining a single global/default API configuration.
    """
    api_id: Optional[str] = Field(None, description="Unique ID for the API (generated if not provided).")
    name: str = Field(..., description="Display name for the API (e.g., 'Default Stock Data API').")
    base_url: str = Field(..., description="Base URL of the external API.")
    auth_type: str = Field(..., description="Authentication type (e.g., 'api_key_header', 'bearer_token', 'none').")
    api_key_env_var: Optional[str] = Field(None, description="Name of the environment variable holding the API key (for server-side use).")
    # You might add fields for rate limits, usage tracking, etc. for this specific API
    description: Optional[str] = Field(None, description="A brief description of the API's functionality.")
    # Tiers that this API is available to by default
    available_to_tiers: List[str] = Field([], description="List of tiers this default API is available to.")

class GlobalApiConfigCreate(GlobalApiConfig):
    """Model for creating a new global/default API configuration."""
    # api_id should not be provided on creation
    pass

class GlobalApiConfigUpdate(GlobalApiConfig):
    """Model for updating an existing global/default API configuration."""
    # All fields are optional for partial updates
    name: Optional[str] = None
    base_url: Optional[str] = None
    auth_type: Optional[str] = None
    api_key_env_var: Optional[str] = None
    description: Optional[str] = None
    available_to_tiers: Optional[List[str]] = None

class ApiCallLimitUpdate(BaseModel):
    """
    Model for updating default API call limits per tier.
    """
    tier: str = Field(..., description="The tier for which to update API call limits.")
    # Example: {'default_api_calls_per_month': 1000, 'default_api_calls_per_day': 50}
    limits: Dict[str, int] = Field(..., description="A dictionary of API call limits (e.g., {'default_api_calls_per_month': 1000, 'default_api_calls_per_day': 50}).")
    # Optional: If you want to replace all existing limits for a tier
    replace_all_limits: Optional[bool] = Field(False, description="If true, replaces all existing limits for the tier with the new ones. If false, merges them.")

