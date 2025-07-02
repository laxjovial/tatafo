# backend/api/user_api.py

from fastapi import APIRouter, HTTPException, status, Depends
from typing import Annotated, List, Dict, Any

# Import Pydantic models from our backend.models
from backend.models.user_models import UserProfile, UserUpdate

# Import middleware for authentication and authorization
from backend.middleware.auth_middleware import get_current_active_user, get_current_admin_user

router = APIRouter()

# Mock user data for initial testing (will be replaced by Firestore)
# This mock should be consistent with _mock_users_db in auth_api.py for testing
_mock_users_db_user_api = {
    "user123": {"user_id": "user123", "username": "Alice", "email": "alice@example.com", "tier": "basic", "roles": ["user"]},
    "admin456": {"user_id": "admin456", "username": "Bob", "email": "bob@example.com", "tier": "premium", "roles": ["user", "admin"]},
    "pro789": {"user_id": "pro789", "username": "Charlie", "email": "charlie@example.com", "tier": "pro", "roles": ["user"]},
    "cc101": {"user_id": "cc101", "username": "Diana", "email": "diana@example.com", "tier": "basic", "roles": ["user", "customer_care"]},
    "an202": {"user_id": "an202", "username": "Eve", "email": "eve@example.com", "tier": "basic", "roles": ["user", "analytics"]},
    "dev303": {"user_id": "dev303", "username": "Frank", "email": "frank@example.com", "tier": "basic", "roles": ["user", "dev"]},
    "api404": {"user_id": "api404", "username": "Grace", "email": "grace@example.com", "tier": "basic", "roles": ["user", "api_manager"]},
    "mgmt505": {"user_id": "mgmt505", "username": "Heidi", "email": "heidi@example.com", "tier": "basic", "roles": ["user", "management"]},
}


@router.get("/{user_id}", response_model=UserProfile)
async def get_user_profile(user_id: str, current_user: Annotated[dict, Depends(get_current_active_user)]):
    """
    Retrieves a user's profile by ID.
    Requires authentication. User can view their own profile; admin can view any.
    """
    # Authorization check: A user can only view their own profile unless they are an admin
    if current_user["user_id"] != user_id and "admin" not in current_user.get("roles", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this user's profile"
        )

    user_data = _mock_users_db_user_api.get(user_id)
    if not user_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    return UserProfile(**user_data)

@router.put("/{user_id}", response_model=UserProfile)
async def update_user_profile(user_id: str, user_update: UserUpdate, current_user: Annotated[dict, Depends(get_current_active_user)]):
    """
    Updates a user's profile.
    Requires authentication. User can update their own profile (limited fields); admin can update any.
    """
    # Authorization check: A user can only update their own profile unless they are an admin
    is_admin = "admin" in current_user.get("roles", [])
    if current_user["user_id"] != user_id and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this user's profile"
        )

    user_data = _mock_users_db_user_api.get(user_id)
    if not user_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    update_data = user_update.model_dump(exclude_unset=True)

    # Restrict non-admin users from changing tier or roles
    if not is_admin:
        if 'tier' in update_data:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators can change user tiers.")
        if 'roles' in update_data:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators can change user roles.")
        # Optionally, restrict email changes for non-admins or require verification
        if 'email' in update_data and update_data['email'] != user_data['email']:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Email change requires specific verification process or admin privileges.")

    user_data.update(update_data)
    
    # Ensure roles are stored as list
    if 'roles' in user_data and isinstance(user_data['roles'], str):
        user_data['roles'] = user_data['roles'].split(',')

    return UserProfile(**user_data)

@router.get("/", response_model=List[UserProfile])
async def get_all_users_api(current_user: Annotated[dict, Depends(get_current_admin_user)]):
    """
    Retrieves a list of all users.
    Requires admin authorization.
    """
    # The `get_current_admin_user` dependency already ensures only admins can access this.
    users_list = []
    for user_id, user_info in _mock_users_db_user_api.items():
        user_dict = {"user_id": user_id}
        user_dict.update(user_info)
        users_list.append(UserProfile(**user_dict))
    return users_list

