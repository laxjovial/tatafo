# backend/api/admin_api.py

import logging
from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Dict, Any, Annotated

# Import the AdminService and its dependencies
from backend.services.admin_service import AdminService # For type hinting in Depends
from backend.middleware.auth_middleware import get_current_admin_user, get_firestore_manager_dependency, get_user_manager_dependency
from database.firestore_manager import FirestoreManager # For type hinting
from utils.user_manager import UserManager # For type hinting
from shared_tools.cloud_storage_utils import CloudStorageUtilsWrapper # For type hinting
from backend.services.api_usage_service import ApiUsageService # For type hinting

# Import Pydantic models for request/response validation
from backend.models.user_models import UserProfile # For returning user profiles
from backend.models.admin_models import (
    UserUpdateAdmin, CapabilityUpdate, TierUpdate, UserStatusUpdate,
    PurgeSessionsRequest, GrantAdminAccessRequest, GlobalApiConfigCreate,
    GlobalApiConfigUpdate, ApiCallLimitUpdate, GlobalApiConfig
)

# Project imports for analytics
from utils.analytics_tracker import log_event

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

router = APIRouter()

# Dependency to provide AdminService instance
async def get_admin_service_dependency(
    firestore_manager: FirestoreManager = Depends(get_firestore_manager_dependency),
    user_manager: UserManager = Depends(get_user_manager_dependency),
    cloud_storage_utils: CloudStorageUtilsWrapper = Depends(lambda: None), # Placeholder, will be provided by main.py
    api_usage_service: ApiUsageService = Depends(lambda: None) # Placeholder, will be provided by main.py
) -> AdminService:
    """Dependency to get the AdminService instance."""
    # This will be overridden by main.py's dependency override.
    # It's here to provide a type hint and a placeholder.
    # The actual instance will be created in main.py and injected.
    raise NotImplementedError("AdminService dependency must be provided by main.py")


# All endpoints in this router will automatically require admin privileges
# by depending on get_current_admin_user
@router.get("/users", response_model=List[UserProfile])
async def get_all_users_admin(
    current_admin: Annotated[UserProfile, Depends(get_current_admin_user)],
    admin_service: AdminService = Depends(get_admin_service_dependency)
):
    """
    Retrieves a list of all user profiles. Requires admin privileges.
    """
    logger.info(f"Admin user {current_admin.user_id} requesting all user profiles.")
    try:
        all_users = await admin_service.get_all_user_profiles()
        # Ensure roles are a list for the Pydantic model
        users_list = []
        for user_data in all_users:
            if isinstance(user_data.get('roles'), str):
                user_data['roles'] = user_data['roles'].split(',')
            user_data['user_id'] = user_data.get('uid') # Ensure uid is mapped to user_id
            users_list.append(UserProfile(**user_data))

        await log_event(
            'admin_action_get_all_users',
            {},
            user_id=current_admin.user_id,
            success=True,
            log_from_backend=True
        )
        return users_list
    except Exception as e:
        logger.error(f"Error fetching all user profiles for admin {current_admin.user_id}: {e}", exc_info=True)
        await log_event(
            'admin_action_get_all_users',
            {'error': str(e)},
            user_id=current_admin.user_id,
            success=False,
            error_message=f"Failed to fetch all user profiles: {e}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch all user profiles: {e}")

@router.put("/users/{user_id}", response_model=UserProfile)
async def update_user_profile_by_admin(
    user_id: str,
    user_update: UserUpdateAdmin,
    current_admin: Annotated[UserProfile, Depends(get_current_admin_user)],
    admin_service: AdminService = Depends(get_admin_service_dependency)
):
    """
    Updates a specific user's profile (including username, tier, roles, and status) by an administrator.
    Requires admin privileges. Admin's own permissions (e.g., can_manage_tier_pro) will be checked.
    """
    logger.info(f"Admin user {current_admin.user_id} updating profile for user: {user_id}")
    try:
        updated_user = await admin_service.update_user_profile_admin(user_id, user_update, current_admin)
        # Ensure roles are a list for the Pydantic model
        if isinstance(updated_user.get('roles'), str):
            updated_user['roles'] = updated_user['roles'].split(',')
        updated_user['user_id'] = user_id # Ensure user_id is set for the response model
        
        await log_event(
            'admin_action_update_user_profile',
            {'target_user_id': user_id, 'updates': user_update.model_dump(exclude_unset=True)},
            user_id=current_admin.user_id,
            success=True,
            log_from_backend=True
        )
        return UserProfile(**updated_user)
    except HTTPException:
        raise # Re-raise HTTPExceptions
    except Exception as e:
        logger.error(f"Error updating user profile for user {user_id} by admin {current_admin.user_id}: {e}", exc_info=True)
        await log_event(
            'admin_action_update_user_profile',
            {'target_user_id': user_id, 'updates': user_update.model_dump(exclude_unset=True), 'error': str(e)},
            user_id=current_admin.user_id,
            success=False,
            error_message=f"Failed to update user profile: {e}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update user profile: {e}")

@router.put("/users/{user_id}/status", response_model=UserProfile)
async def update_user_status_by_admin(
    user_id: str,
    status_update: UserStatusUpdate,
    current_admin: Annotated[UserProfile, Depends(get_current_admin_user)],
    admin_service: AdminService = Depends(get_admin_service_dependency)
):
    """
    Updates a specific user's account status (active, disabled, suspended) by an administrator.
    Requires admin privileges.
    """
    logger.info(f"Admin user {current_admin.user_id} updating status for user: {user_id} to {status_update.status}")
    try:
        updated_user = await admin_service.update_user_status_admin(user_id, status_update.status, current_admin)
        if isinstance(updated_user.get('roles'), str):
            updated_user['roles'] = updated_user['roles'].split(',')
        updated_user['user_id'] = user_id
        
        await log_event(
            'admin_action_update_user_status',
            {'target_user_id': user_id, 'new_status': status_update.status},
            user_id=current_admin.user_id,
            success=True,
            log_from_backend=True
        )
        return UserProfile(**updated_user)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user status for user {user_id} by admin {current_admin.user_id}: {e}", exc_info=True)
        await log_event(
            'admin_action_update_user_status',
            {'target_user_id': user_id, 'new_status': status_update.status, 'error': str(e)},
            user_id=current_admin.user_id,
            success=False,
            error_message=f"Failed to update user status: {e}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update user status: {e}")

@router.post("/sessions/purge", status_code=status.HTTP_200_OK)
async def purge_sessions_admin(
    request: PurgeSessionsRequest,
    current_admin: Annotated[UserProfile, Depends(get_current_admin_user)],
    admin_service: AdminService = Depends(get_admin_service_dependency)
):
    """
    Purges active sessions for a specific user or all users.
    Requires admin privileges; purging all sessions requires 'creator' role.
    """
    logger.info(f"Admin user {current_admin.user_id} requesting session purge: {request.model_dump()}")
    try:
        if request.purge_all:
            if "creator" not in current_admin.roles:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the creator can purge all sessions.")
            await admin_service.purge_all_sessions(current_admin)
            message = "All user sessions purged successfully."
            log_details = {'action': 'purge_all_sessions'}
        elif request.user_id:
            await admin_service.purge_user_sessions(request.user_id, current_admin)
            message = f"Sessions for user {request.user_id} purged successfully."
            log_details = {'action': 'purge_user_sessions', 'target_user_id': request.user_id}
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Either 'user_id' or 'purge_all' must be provided.")
        
        await log_event(
            'admin_action_purge_sessions',
            log_details,
            user_id=current_admin.user_id,
            success=True,
            log_from_backend=True
        )
        return {"message": message, "success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error purging sessions by admin {current_admin.user_id}: {e}", exc_info=True)
        await log_event(
            'admin_action_purge_sessions',
            {'request_data': request.model_dump(), 'error': str(e)},
            user_id=current_admin.user_id,
            success=False,
            error_message=f"Failed to purge sessions: {e}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to purge sessions: {e}")

@router.post("/admin-access/grant", status_code=status.HTTP_200_OK)
async def grant_admin_access(
    request: GrantAdminAccessRequest,
    current_admin: Annotated[UserProfile, Depends(get_current_admin_user)],
    admin_service: AdminService = Depends(get_admin_service_dependency)
):
    """
    Grants specific administrative permissions (custom claims) to another user.
    Requires 'creator' role or specific permissions to grant roles.
    """
    logger.info(f"Admin user {current_admin.user_id} attempting to grant admin access to {request.target_user_id}.")
    try:
        # Only creator can grant/revoke arbitrary admin permissions
        if "creator" not in current_admin.roles:
            # Implement more granular checks here if other admins can grant limited permissions
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the creator can grant/modify admin permissions.")
        
        await admin_service.grant_admin_access(
            target_user_id=request.target_user_id,
            permissions=request.permissions,
            replace_all=request.replace_all_permissions,
            current_admin=current_admin
        )
        
        await log_event(
            'admin_action_grant_admin_access',
            {'target_user_id': request.target_user_id, 'permissions': request.permissions, 'replace_all': request.replace_all_permissions},
            user_id=current_admin.user_id,
            success=True,
            log_from_backend=True
        )
        return {"message": f"Admin access permissions for user {request.target_user_id} updated successfully.", "success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error granting admin access by admin {current_admin.user_id}: {e}", exc_info=True)
        await log_event(
            'admin_action_grant_admin_access',
            {'target_user_id': request.target_user_id, 'permissions': request.permissions, 'error': str(e)},
            user_id=current_admin.user_id,
            success=False,
            error_message=f"Failed to grant admin access: {e}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to grant admin access: {e}")


@router.get("/config/capabilities", response_model=Dict[str, Any])
async def get_rbac_capabilities_admin(
    current_admin: Annotated[UserProfile, Depends(get_current_admin_user)],
    admin_service: AdminService = Depends(get_admin_service_dependency)
):
    """
    Retrieves the current RBAC capabilities configuration. Requires admin privileges.
    """
    logger.info(f"Admin user {current_admin.user_id} requesting RBAC capabilities.")
    try:
        capabilities = await admin_service.get_rbac_capabilities()
        await log_event(
            'admin_action_get_rbac_capabilities',
            {},
            user_id=current_admin.user_id,
            success=True,
            log_from_backend=True
        )
        return capabilities
    except Exception as e:
        logger.error(f"Error fetching RBAC capabilities for admin {current_admin.user_id}: {e}", exc_info=True)
        await log_event(
            'admin_action_get_rbac_capabilities',
            {'error': str(e)},
            user_id=current_admin.user_id,
            success=False,
            error_message=f"Failed to retrieve RBAC capabilities: {e}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve RBAC capabilities: {e}")

@router.put("/config/capabilities", response_model=Dict[str, Any])
async def update_rbac_capabilities_admin(
    capability_update: CapabilityUpdate,
    current_admin: Annotated[UserProfile, Depends(get_current_admin_user)],
    admin_service: AdminService = Depends(get_admin_service_dependency)
):
    """
    Updates the RBAC capabilities configuration. Requires admin privileges.
    Can update a specific capability or replace the entire document.
    """
    logger.info(f"Admin user {current_admin.user_id} updating RBAC capabilities.")
    try:
        updated_capabilities = await admin_service.update_rbac_capabilities(capability_update, current_admin)
        await log_event(
            'admin_action_update_rbac_capabilities',
            {'update_data': capability_update.model_dump(exclude_unset=True)},
            user_id=current_admin.user_id,
            success=True,
            log_from_backend=True
        )
        return updated_capabilities
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating RBAC capabilities for admin {current_admin.user_id}: {e}", exc_info=True)
        await log_event(
            'admin_action_update_rbac_capabilities',
            {'update_data': capability_update.model_dump(exclude_unset=True), 'error': str(e)},
            user_id=current_admin.user_id,
            success=False,
            error_message=f"Failed to update RBAC capabilities: {e}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update RBAC capabilities: {e}")

@router.get("/config/tiers", response_model=Dict[str, Any])
async def get_tier_hierarchy_admin(
    current_admin: Annotated[UserProfile, Depends(get_current_admin_user)],
    admin_service: AdminService = Depends(get_admin_service_dependency)
):
    """
    Retrieves the current tier hierarchy configuration. Requires admin privileges.
    """
    logger.info(f"Admin user {current_admin.user_id} requesting tier hierarchy.")
    try:
        tiers = await admin_service.get_tier_hierarchy()
        await log_event(
            'admin_action_get_tier_hierarchy',
            {},
            user_id=current_admin.user_id,
            success=True,
            log_from_backend=True
        )
        return tiers
    except Exception as e:
        logger.error(f"Error fetching tier hierarchy for admin {current_admin.user_id}: {e}", exc_info=True)
        await log_event(
            'admin_action_get_tier_hierarchy',
            {'error': str(e)},
            user_id=current_admin.user_id,
            success=False,
            error_message=f"Failed to retrieve tier hierarchy: {e}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve tier hierarchy: {e}")

@router.put("/config/tiers", response_model=Dict[str, Any])
async def update_tier_hierarchy_admin(
    tier_update: TierUpdate,
    current_admin: Annotated[UserProfile, Depends(get_current_admin_user)],
    admin_service: AdminService = Depends(get_admin_service_dependency)
):
    """
    Updates the tier hierarchy configuration. Requires admin privileges.
    Can update a specific tier or replace the entire document.
    """
    logger.info(f"Admin user {current_admin.user_id} updating tier hierarchy.")
    try:
        updated_tiers = await admin_service.update_tier_hierarchy(tier_update, current_admin)
        await log_event(
            'admin_action_update_tier_hierarchy',
            {'update_data': tier_update.model_dump(exclude_unset=True)},
            user_id=current_admin.user_id,
            success=True,
            log_from_backend=True
        )
        return updated_tiers
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating tier hierarchy for admin {current_admin.user_id}: {e}", exc_info=True)
        await log_event(
            'admin_action_update_tier_hierarchy',
            {'update_data': tier_update.model_dump(exclude_unset=True), 'error': str(e)},
            user_id=current_admin.user_id,
            success=False,
            error_message=f"Failed to update tier hierarchy: {e}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update tier hierarchy: {e}")

# --- Global API Management Endpoints (Creator/Admin with specific permissions) ---

@router.post("/global_apis", response_model=GlobalApiConfig, status_code=status.HTTP_201_CREATED)
async def create_global_api_config(
    api_config: GlobalApiConfigCreate,
    current_admin: Annotated[UserProfile, Depends(get_current_admin_user)],
    admin_service: AdminService = Depends(get_admin_service_dependency)
):
    """
    Creates a new global/default API configuration. Requires admin privileges.
    Only creator or admins with 'can_manage_global_apis' permission can perform this.
    """
    logger.info(f"Admin user {current_admin.user_id} creating global API config: {api_config.name}")
    try:
        # Check for creator role or specific permission
        if "creator" not in current_admin.roles and not current_admin.get('can_manage_global_apis', False):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to create global APIs.")

        created_api = await admin_service.create_global_api_config(api_config, current_admin)
        await log_event(
            'admin_action_create_global_api',
            {'api_name': api_config.name, 'api_id': created_api.get('api_id')},
            user_id=current_admin.user_id,
            success=True,
            log_from_backend=True
        )
        return GlobalApiConfig(**created_api)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating global API config by admin {current_admin.user_id}: {e}", exc_info=True)
        await log_event(
            'admin_action_create_global_api',
            {'api_name': api_config.name, 'error': str(e)},
            user_id=current_admin.user_id,
            success=False,
            error_message=f"Failed to create global API config: {e}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create global API config: {e}")

@router.get("/global_apis", response_model=List[GlobalApiConfig])
async def get_global_api_configs(
    current_admin: Annotated[UserProfile, Depends(get_current_admin_user)],
    admin_service: AdminService = Depends(get_admin_service_dependency)
):
    """
    Retrieves all global/default API configurations. Requires admin privileges.
    """
    logger.info(f"Admin user {current_admin.user_id} requesting global API configs.")
    try:
        configs = await admin_service.get_global_api_configs()
        # Ensure api_id is present for Pydantic model
        return [GlobalApiConfig(api_id=config.get('id'), **config) for config in configs]
    except Exception as e:
        logger.error(f"Error fetching global API configs by admin {current_admin.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve global API configs: {e}")

@router.put("/global_apis/{api_id}", response_model=GlobalApiConfig)
async def update_global_api_config(
    api_id: str,
    api_config_update: GlobalApiConfigUpdate,
    current_admin: Annotated[UserProfile, Depends(get_current_admin_user)],
    admin_service: AdminService = Depends(get_admin_service_dependency)
):
    """
    Updates an existing global/default API configuration. Requires admin privileges.
    Only creator or admins with 'can_manage_global_apis' permission can perform this.
    """
    logger.info(f"Admin user {current_admin.user_id} updating global API config: {api_id}")
    try:
        # Check for creator role or specific permission
        if "creator" not in current_admin.roles and not current_admin.get('can_manage_global_apis', False):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to update global APIs.")

        updated_api = await admin_service.update_global_api_config(api_id, api_config_update, current_admin)
        await log_event(
            'admin_action_update_global_api',
            {'api_id': api_id, 'updates': api_config_update.model_dump(exclude_unset=True)},
            user_id=current_admin.user_id,
            success=True,
            log_from_backend=True
        )
        return GlobalApiConfig(api_id=api_id, **updated_api)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating global API config {api_id} by admin {current_admin.user_id}: {e}", exc_info=True)
        await log_event(
            'admin_action_update_global_api',
            {'api_id': api_id, 'updates': api_config_update.model_dump(exclude_unset=True), 'error': str(e)},
            user_id=current_admin.user_id,
            success=False,
            error_message=f"Failed to update global API config: {e}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update global API config: {e}")

@router.delete("/global_apis/{api_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_global_api_config(
    api_id: str,
    current_admin: Annotated[UserProfile, Depends(get_current_admin_user)],
    admin_service: AdminService = Depends(get_admin_service_dependency)
):
    """
    Deletes a global/default API configuration. Requires admin privileges.
    Only creator or admins with 'can_manage_global_apis' permission can perform this.
    """
    logger.info(f"Admin user {current_admin.user_id} deleting global API config: {api_id}")
    try:
        # Check for creator role or specific permission
        if "creator" not in current_admin.roles and not current_admin.get('can_manage_global_apis', False):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to delete global APIs.")

        await admin_service.delete_global_api_config(api_id, current_admin)
        await log_event(
            'admin_action_delete_global_api',
            {'api_id': api_id},
            user_id=current_admin.user_id,
            success=True,
            log_from_backend=True
        )
        return {"message": "Global API config deleted successfully."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting global API config {api_id} by admin {current_admin.user_id}: {e}", exc_info=True)
        await log_event(
            'admin_action_delete_global_api',
            {'api_id': api_id, 'error': str(e)},
            user_id=current_admin.user_id,
            success=False,
            error_message=f"Failed to delete global API config: {e}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete global API config: {e}")

@router.put("/api_limits", response_model=Dict[str, Any])
async def update_api_limits_admin(
    limit_update: ApiCallLimitUpdate,
    current_admin: Annotated[UserProfile, Depends(get_current_admin_user)],
    admin_service: AdminService = Depends(get_admin_service_dependency)
):
    """
    Updates default API call limits for a specific tier. Requires admin privileges.
    Only creator or admins with 'can_manage_api_limits' permission can perform this.
    """
    logger.info(f"Admin user {current_admin.user_id} updating API limits for tier: {limit_update.tier}")
    try:
        # Check for creator role or specific permission
        if "creator" not in current_admin.roles and not current_admin.get('can_manage_api_limits', False):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to manage API limits.")

        updated_limits = await admin_service.update_api_limits(limit_update, current_admin)
        await log_event(
            'admin_action_update_api_limits',
            {'tier': limit_update.tier, 'updates': limit_update.model_dump(exclude_unset=True)},
            user_id=current_admin.user_id,
            success=True,
            log_from_backend=True
        )
        return updated_limits
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating API limits for tier {limit_update.tier} by admin {current_admin.user_id}: {e}", exc_info=True)
        await log_event(
            'admin_action_update_api_limits',
            {'tier': limit_update.tier, 'updates': limit_update.model_dump(exclude_unset=True), 'error': str(e)},
            user_id=current_admin.user_id,
            success=False,
            error_message=f"Failed to update API limits: {e}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update API limits: {e}")

@router.get("/analytics/unanswered_queries", response_model=List[Dict[str, Any]])
async def get_unanswered_queries_analytics(
    current_admin: Annotated[UserProfile, Depends(get_current_admin_user)],
    admin_service: AdminService = Depends(get_admin_service_dependency)
):
    """
    Retrieves analytics data on unanswered queries and AI-generated tool suggestions.
    Requires admin privileges. Admins with 'can_view_analytics' permission.
    """
    logger.info(f"Admin user {current_admin.user_id} requesting unanswered queries analytics.")
    try:
        # Check for creator role or specific permission
        if "creator" not in current_admin.roles and not current_admin.get('can_view_analytics', False):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view analytics.")

        analytics_data = await admin_service.get_unanswered_queries_analytics(current_admin)
        await log_event(
            'admin_action_get_unanswered_queries_analytics',
            {},
            user_id=current_admin.user_id,
            success=True,
            log_from_backend=True
        )
        return analytics_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching unanswered queries analytics for admin {current_admin.user_id}: {e}", exc_info=True)
        await log_event(
            'admin_action_get_unanswered_queries_analytics',
            {'error': str(e)},
            user_id=current_admin.user_id,
            success=False,
            error_message=f"Failed to retrieve unanswered queries analytics: {e}",
            log_from_backend=True
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve unanswered queries analytics: {e}")

