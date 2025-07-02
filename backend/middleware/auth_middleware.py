# backend/middleware/auth_middleware.py

from fastapi import Header, HTTPException, status
from typing import Optional, Dict, Any

# In a real application, you would import your user manager or JWT utility here
# from utils.user_manager import get_user_by_id_from_token # Example

# Mock for demonstration purposes. In a real app, this would involve JWT decoding and validation.
_mock_valid_tokens = {
    "mock_jwt_token": {"user_id": "user123", "username": "Alice", "email": "alice@example.com", "tier": "basic", "roles": ["user"]},
    "mock_admin_token": {"user_id": "admin456", "username": "Bob", "email": "bob@example.com", "tier": "premium", "roles": ["user", "admin"]},
    "mock_pro_token": {"user_id": "pro789", "username": "Charlie", "email": "charlie@example.com", "tier": "pro", "roles": ["user"]},
    "mock_customer_care_token": {"user_id": "cc101", "username": "Diana", "email": "diana@example.com", "tier": "basic", "roles": ["user", "customer_care"]},
    "mock_analytics_token": {"user_id": "an202", "username": "Eve", "email": "eve@example.com", "tier": "basic", "roles": ["user", "analytics"]},
    "mock_dev_token": {"user_id": "dev303", "username": "Frank", "email": "frank@example.com", "tier": "basic", "roles": ["user", "dev"]},
    "mock_api_manager_token": {"user_id": "api404", "username": "Grace", "email": "grace@example.com", "tier": "basic", "roles": ["user", "api_manager"]},
    "mock_management_token": {"user_id": "mgmt505", "username": "Heidi", "email": "heidi@example.com", "tier": "basic", "roles": ["user", "management"]},
}

async def verify_token(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    FastAPI dependency to verify an authentication token.
    Extracts the token from the Authorization header (Bearer token).
    Returns the user's information if the token is valid, otherwise raises HTTPException.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, token = authorization.split()
    if scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # In a real application, you would decode and validate the JWT here
    # For now, we check against a mock dictionary of valid tokens
    user_info = _mock_valid_tokens.get(token)
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user_info

async def get_current_active_user(current_user: Dict[str, Any] = Depends(verify_token)) -> Dict[str, Any]:
    """
    FastAPI dependency to get the currently authenticated and active user.
    Can be extended to check for user activity status in a database.
    """
    # For now, simply returns the verified user info.
    # In a real app, you might fetch the full user object from Firestore here
    # and perform additional checks (e.g., is_active flag).
    return current_user

async def get_current_admin_user(current_user: Dict[str, Any] = Depends(verify_token)) -> Dict[str, Any]:
    """
    FastAPI dependency to get the currently authenticated user with 'admin' role.
    """
    if "admin" not in current_user.get("roles", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized: Admin access required"
        )
    return current_user

# You can define similar dependencies for other roles:
async def get_current_customer_care_user(current_user: Dict[str, Any] = Depends(verify_token)) -> Dict[str, Any]:
    if "customer_care" not in current_user.get("roles", []):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized: Customer Care access required")
    return current_user

async def get_current_api_manager_user(current_user: Dict[str, Any] = Depends(verify_token)) -> Dict[str, Any]:
    if "api_manager" not in current_user.get("roles", []):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized: API Manager access required")
    return current_user

# Add more role-specific dependencies as needed for analytics, dev, management, etc.

