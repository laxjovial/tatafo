# backend/api/user_api.py

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any

# Import Firestore manager (will be created/modified later)
# from database.firestore_manager import FirestoreManager
# from utils.user_manager import get_user_by_id, get_all_users, update_user_profile

router = APIRouter()

# Pydantic models for request/response bodies
class UserProfile(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    tier: Optional[str] = None
    roles: Optional[List[str]] = None

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None # Allow changing email, but requires verification
    tier: Optional[str] = None
    roles: Optional[List[str]] = None

# Mock user data for initial testing (will be replaced by Firestore)
# This mock should ideally be shared or fetched from auth_api's mock
_mock_users_db_user_api = {
    "user123": {"username": "Alice", "email": "alice@example.com", "tier": "free", "roles": ["user"]},
    "admin456": {"username": "Bob", "email": "bob@example.com", "tier": "premium", "roles": ["user", "admin"]},
}

@router.get("/{user_id}", response_model=UserProfile)
async def get_user_profile(user_id: str):
    """
    Retrieves a user's profile by ID.
    Requires authentication and authorization (e.g., user can only view their own profile, admin can view any).
    """
    user_data = _mock_users_db_user_api.get(user_id)
    if not user_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    # In a real app, add authorization check here
    return UserProfile(**user_data)

@router.put("/{user_id}", response_model=UserProfile)
async def update_user_profile(user_id: str, user_update: UserUpdate):
    """
    Updates a user's profile.
    Requires authentication and authorization (e.g., user can update their own profile, admin can update any).
    """
    user_data = _mock_users_db_user_api.get(user_id)
    if not user_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    # In a real app, add authorization check here (e.g., only admin can change tier/roles)
    update_data = user_update.model_dump(exclude_unset=True)
    user_data.update(update_data)
    
    # Ensure roles are stored as list if coming from Pydantic
    if 'roles' in user_data and isinstance(user_data['roles'], str):
        user_data['roles'] = user_data['roles'].split(',')

    return UserProfile(**user_data)

@router.get("/", response_model=List[UserProfile])
async def get_all_users_api():
    """
    Retrieves a list of all users.
    Requires admin authorization.
    """
    # In a real app, add admin authorization check here
    users_list = []
    for user_id, user_info in _mock_users_db_user_api.items():
        user_dict = {"user_id": user_id}
        user_dict.update(user_info)
        users_list.append(UserProfile(**user_dict))
    return users_list
