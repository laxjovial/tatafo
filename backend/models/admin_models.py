# backend/models/admin_models.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- User Management Models ---
class UserUpdateAdmin(BaseModel):
    """
    Model for updating user profiles by an administrator.
    Allows changing username, tier, and roles.
    """
    username: Optional[str] = Field(None, description="New username for the user.")
    tier: Optional[str] = Field(None, description="New tier for the user (e.g., 'free', 'pro', 'premium', 'admin').")
    roles: Optional[List[str]] = Field(None, description="New list of roles for the user (e.g., ['user', 'dev']).")

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
    level: Optional[int] = Field(None, description="The new numerical level for the tier (higher means higher priority).")
    description: Optional[str] = Field(None, description="A new description for the tier.")
    full_tiers: Optional[Dict[str, Any]] = Field(None, description="Optional: The full tiers dictionary to replace the existing one. Used if tier_name is not provided.")

