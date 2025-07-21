# backend/services/admin_service.py

import logging
from typing import List, Dict, Any, Optional
from fastapi import HTTPException, status
from firebase_admin import auth, exceptions as firebase_exceptions # Import Firebase Auth for session management

# Import Managers and Services for dependency injection
from database.firestore_manager import FirestoreManager
from utils.user_manager import UserManager
from shared_tools.cloud_storage_utils import CloudStorageUtilsWrapper # For type hinting
from backend.services.api_usage_service import ApiUsageService # NEW: Import ApiUsageService

# Import Pydantic models for request/response validation
from backend.models.user_models import UserProfile # For current_admin and return types
from backend.models.admin_models import (
    UserUpdateAdmin, CapabilityUpdate, TierUpdate, UserStatusUpdate,
    PurgeSessionsRequest, GrantAdminAccessRequest, GlobalApiConfigCreate,
    GlobalApiConfigUpdate, ApiCallLimitUpdate, GlobalApiConfig # Import new models
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class AdminService:
    """
    Provides business logic for administrative operations,
    including user management, dynamic configuration of RBAC capabilities and tiers,
    session management, and global API configuration.
    """
    def __init__(
        self,
        firestore_manager: FirestoreManager,
        user_manager: UserManager,
        cloud_storage_utils: CloudStorageUtilsWrapper, # Added for completeness
        api_usage_service: ApiUsageService # NEW: Inject ApiUsageService
    ):
        self.firestore_manager = firestore_manager
        self.user_manager = user_manager
        self.cloud_storage_utils = cloud_storage_utils # Store the injected instance
        self.api_usage_service = api_usage_service # Store the injected instance
        logger.info("AdminService initialized.")

    # --- User Management Operations ---

    async def get_all_users(self, current_admin: UserProfile) -> List[UserProfile]:
        """Retrieves all user profiles."""
        logger.debug(f"Admin {current_admin.user_id} requesting all user profiles.")
        if "creator" not in current_admin.roles and not current_admin.get('can_view_users', False):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view all users.")
        
        users = await self.user_manager.get_all_users()
        return [UserProfile(**user) for user in users]

    async def update_user_profile(self, user_id: str, update_data: UserUpdateAdmin, current_admin: UserProfile) -> Dict[str, Any]:
        """Updates a user's profile, tier, and roles."""
        logger.debug(f"Admin {current_admin.user_id} updating profile for user_id: {user_id}")
        
        if "creator" not in current_admin.roles and not current_admin.get('can_manage_users', False):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to manage users.")

        # Prepare update data for Firestore
        firestore_update_data = update_data.model_dump(exclude_unset=True, exclude={'roles', 'tier', 'status'})
        
        # Handle roles and tier updates separately via UserManager methods
        if update_data.roles is not None:
            await self.user_manager.update_user_roles(user_id, update_data.roles)
        if update_data.tier is not None:
            await self.user_manager.update_user_tier(user_id, update_data.tier)
        
        # Handle user status update
        if update_data.status is not None:
            await self.user_manager.update_user_status(user_id, update_data.status)

        # Update other profile data in Firestore
        if firestore_update_data:
            await self.firestore_manager.update_doc(f"users/{user_id}", firestore_update_data)

        logger.info(f"User profile for {user_id} updated by admin {current_admin.user_id}.")
        return {"success": True, "message": f"User {user_id} profile updated."}

    async def grant_admin_access(self, request: GrantAdminAccessRequest, current_admin: UserProfile) -> Dict[str, Any]:
        """Grants or revokes admin access to a user."""
        logger.debug(f"Admin {current_admin.user_id} modifying admin access for user {request.user_id}.")
        
        # Only 'creator' role can grant/revoke admin access
        if "creator" not in current_admin.roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only 'creator' can grant/revoke admin access.")
        
        user_profile = await self.user_manager.get_user(request.user_id)
        if not user_profile:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target user not found.")

        current_roles = set(user_profile.get("roles", []))
        
        if request.grant:
            current_roles.add("admin")
            await self.user_manager.update_user_roles(request.user_id, list(current_roles))
            message = f"Admin access granted to user {request.user_id}."
            logger.info(message)
        else:
            if "admin" in current_roles:
                current_roles.remove("admin")
                await self.user_manager.update_user_roles(request.user_id, list(current_roles))
                message = f"Admin access revoked from user {request.user_id}."
                logger.info(message)
            else:
                message = f"User {request.user_id} does not have admin access to revoke."
                logger.warning(message)

        return {"success": True, "message": message}

    async def delete_user(self, user_id: str, current_admin: UserProfile) -> Dict[str, Any]:
        """Deletes a user from Firebase Auth and Firestore."""
        logger.debug(f"Admin {current_admin.user_id} deleting user: {user_id}")

        if "creator" not in current_admin.roles and not current_admin.get('can_delete_users', False):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to delete users.")

        try:
            # Delete from Firebase Authentication
            auth.delete_user(user_id)
            # Delete user profile from Firestore
            await self.firestore_manager.delete_doc(f"users/{user_id}")
            # Optionally, delete user-related data from cloud storage (e.g., vector stores)
            await self.cloud_storage_utils.delete_user_data_folder(user_id)

            await log_event(
                'admin_action_delete_user',
                {'target_user_id': user_id},
                user_id=current_admin.user_id,
                success=True,
                log_from_backend=True
            )
            logger.info(f"User {user_id} deleted successfully by admin {current_admin.user_id}.")
            return {"success": True, "message": f"User {user_id} and associated data deleted."}
        except firebase_exceptions.FirebaseError as e:
            logger.error(f"Firebase error deleting user {user_id}: {e}", exc_info=True)
            await log_event(
                'admin_action_delete_user',
                {'target_user_id': user_id, 'error': str(e)},
                user_id=current_admin.user_id,
                success=False,
                error_message=f"Failed to delete user: {e.code}",
                log_from_backend=True
            )
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to delete user: {e.code}")
        except Exception as e:
            logger.error(f"Unexpected error deleting user {user_id}: {e}", exc_info=True)
            await log_event(
                'admin_action_delete_user',
                {'target_user_id': user_id, 'error': str(e)},
                user_id=current_admin.user_id,
                success=False,
                error_message=f"Unexpected error: {e}",
                log_from_backend=True
            )
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete user: {e}")

    # --- RBAC and Tier Management ---

    async def update_capability(self, capability_update: CapabilityUpdate, current_admin: UserProfile) -> Dict[str, Any]:
        """Dynamically updates a specific RBAC capability."""
        logger.debug(f"Admin {current_admin.user_id} updating capability: {capability_update.capability_key}.")

        if "creator" not in current_admin.roles and not current_admin.get('can_manage_capabilities', False):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to manage capabilities.")
        
        result = await self.user_manager.update_capability_config(
            capability_update.capability_key, 
            capability_update.new_default_value, 
            capability_update.tier_overrides
        )
        await log_event(
            'admin_action_update_capability',
            {'capability_key': capability_update.capability_key},
            user_id=current_admin.user_id,
            success=True,
            log_from_backend=True
        )
        return result

    async def update_tier_config(self, tier_update: TierUpdate, current_admin: UserProfile) -> Dict[str, Any]:
        """Dynamically updates a tier's configuration (e.g., capabilities)."""
        logger.debug(f"Admin {current_admin.user_id} updating tier config for tier: {tier_update.tier_name}.")

        if "creator" not in current_admin.roles and not current_admin.get('can_manage_tiers', False):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to manage tiers.")

        result = await self.user_manager.update_tier_config(
            tier_update.tier_name, 
            tier_update.capabilities
        )
        await log_event(
            'admin_action_update_tier_config',
            {'tier_name': tier_update.tier_name},
            user_id=current_admin.user_id,
            success=True,
            log_from_backend=True
        )
        return result

    async def get_all_capabilities(self, current_admin: UserProfile) -> Dict[str, Any]:
        """Retrieves the current RBAC capabilities configuration."""
        logger.debug(f"Admin {current_admin.user_id} requesting all capabilities.")
        if "creator" not in current_admin.roles and not current_admin.get('can_view_capabilities', False):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view capabilities.")
        
        return await self.user_manager.get_all_capabilities_config()

    async def get_all_tiers_config(self, current_admin: UserProfile) -> Dict[str, Any]:
        """Retrieves the current tiers configuration."""
        logger.debug(f"Admin {current_admin.user_id} requesting all tiers config.")
        if "creator" not in current_admin.roles and not current_admin.get('can_view_tiers', False):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view tiers.")
        
        return await self.user_manager.get_all_tiers_config()

    # --- Session Management ---

    async def purge_user_sessions(self, request: PurgeSessionsRequest, current_admin: UserProfile) -> Dict[str, Any]:
        """Revokes all refresh tokens for a given user, effectively logging them out from all devices."""
        logger.debug(f"Admin {current_admin.user_id} purging sessions for user: {request.user_id}.")

        if "creator" not in current_admin.roles and not current_admin.get('can_manage_sessions', False):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to purge user sessions.")

        try:
            auth.revoke_refresh_tokens(request.user_id)
            await log_event(
                'admin_action_purge_sessions',
                {'target_user_id': request.user_id},
                user_id=current_admin.user_id,
                success=True,
                log_from_backend=True
            )
            logger.info(f"Revoked all refresh tokens for user {request.user_id} by admin {current_admin.user_id}.")
            return {"success": True, "message": f"All sessions for user {request.user_id} purged."}
        except firebase_exceptions.FirebaseError as e:
            logger.error(f"Firebase error purging sessions for user {request.user_id}: {e}", exc_info=True)
            await log_event(
                'admin_action_purge_sessions',
                {'target_user_id': request.user_id, 'error': str(e)},
                user_id=current_admin.user_id,
                success=False,
                error_message=f"Failed to purge sessions: {e.code}",
                log_from_backend=True
            )
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to purge sessions: {e.code}")
        except Exception as e:
            logger.error(f"Unexpected error purging sessions for user {request.user_id}: {e}", exc_info=True)
            await log_event(
                'admin_action_purge_sessions',
                {'target_user_id': request.user_id, 'error': str(e)},
                user_id=current_admin.user_id,
                success=False,
                error_message=f"Unexpected error: {e}",
                log_from_backend=True
            )
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to purge sessions: {e}")

    # --- Global API Configuration Management ---

    async def create_global_api_config(self, config_data: GlobalApiConfigCreate, current_admin: UserProfile) -> Dict[str, Any]:
        """Creates a new global API configuration."""
        logger.debug(f"Admin {current_admin.user_id} creating global API config for {config_data.api_id}.")
        if "creator" not in current_admin.roles and not current_admin.get('can_manage_api_configs', False):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to manage global API configurations.")
        
        return await self.api_usage_service.create_global_api_config(config_data)

    async def get_global_api_configs(self, current_admin: UserProfile) -> List[Dict[str, Any]]:
        """Retrieves all global API configurations."""
        logger.debug(f"Admin {current_admin.user_id} requesting all global API configs.")
        if "creator" not in current_admin.roles and not current_admin.get('can_view_api_configs', False):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view global API configurations.")
        
        return await self.api_usage_service.get_global_api_configs()

    async def update_global_api_config(self, api_id: str, config_data: GlobalApiConfigUpdate, current_admin: UserProfile) -> Dict[str, Any]:
        """Updates an existing global API configuration."""
        logger.debug(f"Admin {current_admin.user_id} updating global API config for {api_id}.")
        if "creator" not in current_admin.roles and not current_admin.get('can_manage_api_configs', False):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to manage global API configurations.")
        
        return await self.api_usage_service.update_global_api_config(api_id, config_data)

    async def delete_global_api_config(self, api_id: str, current_admin: UserProfile) -> Dict[str, Any]:
        """Deletes a global API configuration."""
        logger.debug(f"Admin {current_admin.user_id} deleting global API config for {api_id}.")
        if "creator" not in current_admin.roles and not current_admin.get('can_manage_api_configs', False):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to manage global API configurations.")
        
        return await self.api_usage_service.delete_global_api_config(api_id)

    async def update_api_limits(self, limit_update: ApiCallLimitUpdate, current_admin: UserProfile) -> Dict[str, Any]:
        """Updates default API call limits for a specific API across different tiers."""
        logger.debug(f"Admin {current_admin.user_id} calling ApiUsageService to update API limits for API {limit_update.api_id}.")
        if "creator" not in current_admin.roles and not current_admin.get('can_manage_api_limits', False):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to manage API limits.")
            
        return await self.api_usage_service.update_api_call_limits(limit_update.api_id, limit_update)

    async def get_all_api_limits(self, current_admin: UserProfile) -> Dict[str, Any]:
        """Retrieves all global API limits configured."""
        logger.debug(f"Admin {current_admin.user_id} requesting all API limits.")
        if "creator" not in current_admin.roles and not current_admin.get('can_view_api_limits', False):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view API limits.")
            
        return await self.api_usage_service.get_all_api_limits()

    # --- Analytics for Unanswered Queries ---
    async def get_unanswered_queries_analytics(self, current_admin: UserProfile) -> List[Dict[str, Any]]:
        """
        Retrieves analytics data on unanswered queries and AI-generated tool suggestions.
        """
        logger.debug(f"Admin {current_admin.user_id} fetching unanswered queries analytics.")
        try:
            # Check for creator role or specific permission
            if "creator" not in current_admin.roles and not current_admin.get('can_view_analytics', False):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view analytics.")

            # This will query a specific collection in Firestore, e.g., 'analytics/unanswered_queries'
            # Assuming 'analytics_tracker' or a dedicated 'AnalyticsService' will log these.
            # For now, we'll just fetch from a predefined collection.
            unanswered_docs = await self.firestore_manager.get_all_global_config_documents(collection_name="unanswered_queries_analytics")
            return unanswered_docs
        except Exception as e:
            logger.error(f"Error fetching unanswered queries analytics for admin {current_admin.user_id}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve unanswered queries analytics: {e}")