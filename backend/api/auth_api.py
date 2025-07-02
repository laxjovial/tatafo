# backend/api/auth_api.py

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr
from typing import Optional

# Import Firestore manager (will be created/modified later)
# from database.firestore_manager import FirestoreManager
# from utils.user_manager import create_user, authenticate_user, generate_password_reset_token, reset_password_with_token

router = APIRouter()

# Pydantic models for request/response bodies
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

class ChangePassword(BaseModel):
    user_id: str # Or current_user_id from auth
    old_password: str
    new_password: str

# Mock user data for initial testing (will be replaced by Firestore)
_mock_users_db = {} # In a real app, this would be Firestore

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate):
    """
    Registers a new user.
    """
    # This logic will eventually call a service that interacts with Firestore
    if user_data.email in _mock_users_db:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User with this email already exists")
    
    # In a real scenario, hash the password before storing
    _mock_users_db[user_data.email] = {
        "username": user_data.username,
        "email": user_data.email,
        "password_hash": user_data.password, # Placeholder: should be hashed
        "tier": "free",
        "roles": ["user"]
    }
    return {"message": "User registered successfully"}

@router.post("/login")
async def login_user(credentials: UserLogin):
    """
    Authenticates a user and returns a token.
    """
    # This logic will eventually call a service that interacts with Firestore
    user = _mock_users_db.get(credentials.email)
    if not user or user["password_hash"] != credentials.password: # Placeholder: should compare hashed passwords
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
    # In a real app, generate a JWT token here
    return {"message": "Login successful", "access_token": "mock_jwt_token", "token_type": "bearer", "user_id": credentials.email}

@router.post("/request-password-reset")
async def request_password_reset(request: PasswordResetRequest):
    """
    Requests a password reset token for the given email.
    """
    # This logic will eventually call a service that interacts with Firestore and sends email
    if request.email not in _mock_users_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    # In a real app, generate and store a token, then send it via email
    print(f"Mock: Password reset token sent to {request.email}")
    return {"message": "If the email is registered, a password reset link has been sent."}

@router.post("/reset-password")
async def reset_password(confirm: PasswordResetConfirm):
    """
    Resets user's password using a valid token.
    """
    # This logic will eventually call a service that interacts with Firestore
    # In a real app, validate token and update password
    if confirm.token != "valid_mock_token": # Placeholder
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token")
    
    # Find user by token (mock logic)
    for email, user_data in _mock_users_db.items():
        if email == "mock_user_for_token_reset@example.com": # Placeholder for finding user by token
            user_data["password_hash"] = confirm.new_password # Placeholder: should be hashed
            return {"message": "Password reset successfully."}
    
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User associated with token not found.")

@router.post("/change-password")
async def change_password(data: ChangePassword):
    """
    Allows a logged-in user to change their password.
    """
    # This logic will eventually call a service that interacts with Firestore
    # In a real app, verify old password and update new password
    user = _mock_users_db.get(data.user_id) # Assuming user_id is email for mock
    if not user or user["password_hash"] != data.old_password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid old password or user not found.")
    
    user["password_hash"] = data.new_password # Placeholder: should be hashed
    return {"message": "Password changed successfully."}
