# backend/models/user_models.py

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List

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
    """
    email: EmailStr = Field(..., description="User's email address.")
    password: str = Field(..., description="User's password.")

class UserProfile(BaseModel):
    """
    Pydantic model for a user's profile information (read-only or for display).
    """
    user_id: str = Field(..., description="Unique identifier for the user.")
    username: str = Field(..., description="User's display name.")
    email: EmailStr = Field(..., description="User's email address.")
    tier: str = Field("free", description="User's subscription tier (e.g., 'free', 'basic', 'pro', 'premium').")
    roles: List[str] = Field(["user"], description="List of roles assigned to the user (e.g., 'user', 'admin', 'customer_care').")
    # Add other profile fields as needed (e.g., created_at, last_login)

    class Config:
        # This allows the model to be created from arbitrary class instances
        # with attribute names that match the field names.
        # Useful when converting from database objects.
        from_attributes = True 

class UserUpdate(BaseModel):
    """
    Pydantic model for updating an existing user's profile.
    All fields are optional, allowing partial updates.
    """
    username: Optional[str] = Field(None, min_length=3, max_length=50, description="New username for the user.")
    email: Optional[EmailStr] = Field(None, description="New email address for the user.")
    tier: Optional[str] = Field(None, description="New subscription tier for the user.")
    roles: Optional[List[str]] = Field(None, description="New list of roles for the user.")

class PasswordResetRequest(BaseModel):
    """
    Pydantic model for requesting a password reset.
    """
    email: EmailStr = Field(..., description="Email address associated with the account.")

class PasswordResetConfirm(BaseModel):
    """
    Pydantic model for confirming a password reset with a token.
    """
    token: str = Field(..., description="Password reset token received via email.")
    new_password: str = Field(..., min_length=8, description="New password for the user.")

class ChangePassword(BaseModel):
    """
    Pydantic model for a logged-in user to change their password.
    """
    old_password: str = Field(..., description="User's current password.")
    new_password: str = Field(..., min_length=8, description="User's new password.")

