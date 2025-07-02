# backend/api/auth_api.py

from fastapi import APIRouter, HTTPException, status, Depends
from typing import Annotated # For FastAPI Depends type hinting

# Import Pydantic models from our backend.models
from backend.models.user_models import UserCreate, UserLogin, PasswordResetRequest, PasswordResetConfirm, ChangePassword

# Import middleware for protected routes (e.g., change password)
from backend.middleware.auth_middleware import get_current_active_user

router = APIRouter()

# Mock user data for initial testing (will be replaced by Firestore)
# IMPORTANT: In a real application, passwords MUST be hashed (e.g., using bcrypt)
_mock_users_db = {} # Stores user_id as key, user_info as value

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate):
    """
    Registers a new user.
    """
    # In a real scenario, hash the password before storing
    # For mock, we'll use email as user_id
    user_id = user_data.email
    if user_id in _mock_users_db:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User with this email already exists")
    
    _mock_users_db[user_id] = {
        "user_id": user_id,
        "username": user_data.username,
        "email": user_data.email,
        "password_hash": user_data.password, # Placeholder: should be hashed
        "tier": "free",
        "roles": ["user"]
    }
    return {"message": "User registered successfully", "user_id": user_id}

@router.post("/login")
async def login_user(credentials: UserLogin):
    """
    Authenticates a user and returns a token.
    """
    user_id = credentials.email # Using email as user_id for mock
    user = _mock_users_db.get(user_id)
    if not user or user["password_hash"] != credentials.password: # Placeholder: should compare hashed passwords
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
    # In a real app, generate a JWT token here based on user_id and roles/tier
    # For mock, we will return a hardcoded mock token associated with the user_id
    mock_token = "mock_jwt_token" if user_id == "alice@example.com" else "mock_admin_token" if user_id == "bob@example.com" else "mock_pro_token" if user_id == "charlie@example.com" else "mock_jwt_token" # Fallback
    
    # Populate the mock_valid_tokens in auth_middleware for testing purposes
    from backend.middleware.auth_middleware import _mock_valid_tokens
    _mock_valid_tokens[mock_token] = {
        "user_id": user_id,
        "username": user.get("username"),
        "email": user.get("email"),
        "tier": user.get("tier"),
        "roles": user.get("roles")
    }

    return {"message": "Login successful", "access_token": mock_token, "token_type": "bearer", "user_id": user_id}

@router.post("/request-password-reset")
async def request_password_reset(request: PasswordResetRequest):
    """
    Requests a password reset token for the given email.
    """
    if request.email not in _mock_users_db:
        # For security, always return a generic success message even if email not found
        print(f"Mock: Attempted password reset for non-existent email: {request.email}")
        return {"message": "If the email is registered, a password reset link has been sent."}
    
    # In a real app, generate and store a token, then send it via email
    print(f"Mock: Password reset token sent to {request.email}")
    return {"message": "If the email is registered, a password reset link has been sent."}

@router.post("/reset-password")
async def reset_password(confirm: PasswordResetConfirm):
    """
    Resets user's password using a valid token.
    """
    # This mock logic is very simplified. In a real app:
    # 1. Validate the token against stored tokens (e.g., in Firestore).
    # 2. Ensure token is not expired and has not been used.
    # 3. Retrieve user_id associated with the token.
    # 4. Hash and update the user's password.
    # 5. Invalidate the token.

    if confirm.token == "valid_mock_token_for_reset": # Placeholder token
        # Assume we found a user associated with this token
        user_id_for_reset = "mock_user_for_token_reset@example.com" # Example user ID
        if user_id_for_reset in _mock_users_db:
            _mock_users_db[user_id_for_reset]["password_hash"] = confirm.new_password # Placeholder: hash password
            return {"message": "Password reset successfully."}
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User associated with token not found.")
    
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token")

@router.post("/change-password")
async def change_password(data: ChangePassword, current_user: Annotated[dict, Depends(get_current_active_user)]):
    """
    Allows a logged-in user to change their password.
    """
    user_id = current_user["user_id"] # Get user_id from the authenticated token
    user = _mock_users_db.get(user_id)

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    
    # Verify old password
    if user["password_hash"] != data.old_password: # Placeholder: compare hashed passwords
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid old password.")
    
    # Update with new password
    user["password_hash"] = data.new_password # Placeholder: hash new password
    return {"message": "Password changed successfully."}

