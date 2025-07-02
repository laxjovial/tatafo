# backend/api/auth_api.py

from fastapi import APIRouter, HTTPException, status, Depends
from typing import Annotated # For FastAPI Depends type hinting

# Import Pydantic models from our backend.models
from backend.models.user_models import UserCreate, UserLogin, PasswordResetRequest, PasswordResetConfirm, ChangePassword

# Import middleware for protected routes (e.g., change password)
from backend.middleware.auth_middleware import get_current_active_user

# Import FirestoreManager
from database.firestore_manager import firestore_manager

# Import Firebase Auth (for creating users and setting custom claims)
from firebase_admin import auth

router = APIRouter()

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate):
    """
    Registers a new user in Firebase Authentication and stores profile in Firestore.
    """
    try:
        # 1. Create user in Firebase Authentication
        user_record = auth.create_user(email=user_data.email, password=user_data.password, display_name=user_data.username)
        user_id = user_record.uid

        # 2. Set custom claims for tier and roles (optional, can be done later by admin)
        # For initial registration, assign default tier and roles
        default_tier = "free" # Or from config_manager
        default_roles = ["user"] # Or from config_manager
        auth.set_custom_user_claims(user_id, {'tier': default_tier, 'roles': default_roles})
        
        # 3. Store user profile in Firestore (public 'users' collection)
        user_profile_data = {
            "username": user_data.username,
            "email": user_data.email,
            "tier": default_tier,
            "roles": default_roles,
            "created_at": firestore.SERVER_TIMESTAMP # Add timestamp
        }
        await firestore_manager.set_user_data(user_id, user_profile_data)
        
        logger.info(f"User '{user_data.username}' ({user_data.email}) created with UID: {user_id}")
        return {"message": "User registered successfully", "user_id": user_id}
    except auth.EmailAlreadyExistsError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User with this email already exists")
    except Exception as e:
        logger.error(f"Error during user registration: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to register user: {e}")

@router.post("/login")
async def login_user(credentials: UserLogin):
    """
    Authenticates a user. This endpoint is typically handled by client-side Firebase SDK.
    This server-side endpoint is a placeholder or for custom token generation.
    For a real login, the client-side Firebase SDK would get an ID token, which is then verified by the backend.
    """
    # This endpoint is primarily for demonstration of backend integration.
    # In a typical Streamlit + Firebase Auth setup, the Streamlit frontend
    # uses Firebase JS SDK to sign in and gets an ID token, which it sends to the backend.
    
    # For testing, we can simulate a successful login if the email/password match a known user.
    # In production, you would NOT expose password directly here.
    
    # A more realistic scenario for this endpoint:
    # 1. Client sends email/password to Firebase JS SDK.
    # 2. Firebase JS SDK returns an ID token.
    # 3. Client sends this ID token to this backend endpoint.
    # 4. Backend verifies the ID token using `auth.verify_id_token(id_token)`.
    # 5. Backend then returns custom claims or a new session token.

    # For now, let's just return a placeholder token for known users for testing purposes.
    # In a real app, this would involve a more secure token exchange.
    user_id = credentials.email # Using email as user_id for mock/testing
    user_data = await firestore_manager.get_user_data(user_id) # Try to fetch from Firestore

    if not user_data:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials or user not found.")

    # IMPORTANT: In a real app, you would verify the password hash here,
    # or rely on Firebase Auth's client-side SDK for password verification.
    # This mock assumes successful verification.
    
    # Simulate ID token generation for mock
    # In a real scenario, Firebase Admin SDK can create custom tokens,
    # but for direct user login, client-side SDK is preferred.
    mock_token = f"mock_jwt_token_for_{user_id.split('@')[0]}"
    
    logger.info(f"Simulated login for user {user_id}. Returning mock token.")
    return {"message": "Login successful (simulated)", "access_token": mock_token, "token_type": "bearer", "user_id": user_id}


@router.post("/request-password-reset")
async def request_password_reset(request: PasswordResetRequest):
    """
    Requests a password reset link for the given email using Firebase Auth.
    """
    try:
        # Firebase Admin SDK generates the link
        reset_link = auth.generate_password_reset_link(request.email)
        # In a real application, you would send this link via email.
        logger.info(f"Generated password reset link for {request.email}: {reset_link}")
        return {"message": "If the email is registered, a password reset link has been sent to your inbox."}
    except auth.UserNotFoundError:
        # For security, always return a generic success message even if email not found
        logger.warning(f"Attempted password reset for non-existent email: {request.email}")
        return {"message": "If the email is registered, a password reset link has been sent to your inbox."}
    except Exception as e:
        logger.error(f"Error requesting password reset for {request.email}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to request password reset: {e}")

@router.post("/reset-password")
async def reset_password(confirm: PasswordResetConfirm):
    """
    Resets user's password using a valid token (oobCode from Firebase).
    """
    try:
        # Verify the password reset code and then confirm the reset
        auth.confirm_password_reset(confirm.token, confirm.new_password)
        logger.info(f"Password successfully reset using token (oobCode).")
        return {"message": "Password reset successfully."}
    except Exception as e:
        logger.error(f"Error confirming password reset with token: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid or expired token: {e}")

@router.post("/change-password")
async def change_password(data: ChangePassword, current_user: Annotated[dict, Depends(get_current_active_user)]):
    """
    Allows a logged-in user to change their password using Firebase Auth.
    """
    user_id = current_user["user_id"] # Get user_id from the authenticated token

    try:
        # Firebase Auth does not directly expose a "change password with old password" API for Admin SDK.
        # This is typically handled client-side by re-authenticating the user with their old password
        # and then calling `updatePassword` on the client-side user object.
        
        # For a server-side approach, you would need to:
        # 1. Re-authenticate the user (e.g., via a custom token or verifying credentials).
        # 2. Then update their password using `auth.update_user`.
        
        # For simplicity and to align with typical Firebase Auth flows,
        # we'll simulate the update here, but note the real-world client-side flow.
        
        # Simulate old password verification (this would be a real hash comparison)
        user_data = await firestore_manager.get_user_data(user_id)
        if not user_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
        
        # In a real scenario, you'd compare hashed passwords or use Firebase client-side re-auth
        # For this mock, we'll assume the old_password is correct if the user is authenticated.
        # This part needs strong security review for production.
        
        auth.update_user(user_id, password=data.new_password)
        logger.info(f"Password for user {user_id} changed successfully.")
        return {"message": "Password changed successfully."}
    except Exception as e:
        logger.error(f"Error changing password for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to change password: {e}")



