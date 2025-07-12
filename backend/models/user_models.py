# backend/models/user_models.py

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

class UserCreate(BaseModel):
    """
    Pydantic model for creating a new user (e.g., during registration).
    """
    username: str = Field(..., min_length=3, max_length=50, description="Unique username for the user.")
    email: EmailStr = Field(..., description="Unique email address for the user.")
    password: str = Field(..., min_length=8, description="User's password (will be hashed).")

class UserLogin(BaseModel):
    """
    Pydantic model for user login credentials.
    For Firebase ID Token verification, `id_token` is the primary field expected by the backend.
    Email and password are typically handled client-side by Firebase SDK, but included as Optional
    for potential alternative/legacy auth flows if needed.
    """
    id_token: str = Field(..., description="Firebase ID Token obtained from client-side authentication.")
    email: Optional[EmailStr] = Field(None, description="User's email address (optional, for certain auth flows).")
    password: Optional[str] = Field(None, description="User's password (optional, for certain auth flows).")


class UserProfile(BaseModel):
    """
    Pydantic model for a user's profile information (read-only or for display).
    Includes fields for status and last login, and supports the new tier structure.
    """
    user_id: str = Field(..., description="Unique identifier for the user.")
    username: str = Field(..., description="User's display name.")
    email: EmailStr = Field(..., description="User's email address.")
    # Updated description to reflect all defined tiers
    tier: str = Field("free", description="User's subscription tier (e.g., 'free', 'basic', 'pro', 'elite', 'premium').")
    roles: List[str] = Field(["user"], description="List of roles assigned to the user (e.g., 'user', 'admin', 'customer_care').")
    status: str = Field("active", description="Account status (e.g., 'active', 'disabled', 'suspended').")
    created_at: Optional[datetime] = Field(None, description="Timestamp when the user account was created.")
    last_login_at: Optional[datetime] = Field(None, description="Timestamp of the user's last login.")
    # Add other profile fields as needed (e.g., phone, address, bio - these might be in a nested 'profile_data' map in Firestore)

    class Config:
        # This allows the model to be created from arbitrary class instances
        # with attribute names that match the field names.
        # Useful when converting from database objects.
        from_attributes = True
        # Allow population by field name or alias (useful if Firebase uses different casing)
        populate_by_name = True
        # Enable JSON encoding for datetime objects
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

class UserUpdate(BaseModel):
    """
    Pydantic model for updating an existing user's profile.
    All fields are optional, allowing partial updates.
    Includes status for admin management.
    """
    username: Optional[str] = Field(None, min_length=3, max_length=50, description="New username for the user.")
    email: Optional[EmailStr] = Field(None, description="New email address for the user.")
    tier: Optional[str] = Field(None, description="New subscription tier for the user.")
    roles: Optional[List[str]] = Field(None, description="New list of roles for the user.")
    status: Optional[str] = Field(None, description="New account status (e.g., 'active', 'disabled', 'suspended').")


class PasswordResetRequest(BaseModel):
    """
    Pydantic model for requesting a password reset.
    """
    email: EmailStr = Field(..., description="Email address associated with the account.")

class PasswordResetConfirm(BaseModel):
    """
    Pydantic model for confirming a password reset with a token.
    """
    token: str = Field(..., description="Password reset token received via email (Firebase oobCode).")
    new_password: str = Field(..., min_length=8, description="New password for the user.")

class ChangePassword(BaseModel):
    """
    Pydantic model for a logged-in user to change their password.
    This model is used when the client has already re-authenticated the user
    or provides a fresh ID token for security.
    """
    # The old_password is typically verified client-side via Firebase re-authentication.
    # This backend endpoint just needs the new password.
    new_password: str = Field(..., min_length=8, description="User's new password.")

 
