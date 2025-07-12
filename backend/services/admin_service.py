# backend/services/admin_service.py

import logging
from typing import List, Dict, Any, Optional
from fastapi import HTTPException, status
from firebase_admin import auth # Import Firebase Auth for session management

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
        cloud_storage_utils: CloudStorageUtilsWrapper, # Added for completeness, though not directly used in all methods here
        api_usage_service: ApiUsageService # NEW: Inject ApiUsageService
    ):
        self.firestore_manager = firestore_manager
        self.user_manager = user_manager
        self.cloud_storage_utils = cloud_storage_utils
        self.api_usage_service = api_usage_service # Store the injected service
        logger.info("AdminService initialized with dependencies.")

    async def get_all_user_profiles(self) -> List[Dict[str, Any]]:
        """
        Retrieves all user profiles.
        """
        try:
            users = await self.user_manager.get_all_user_profiles() # Use injected UserManager
            return list(users.values())
        except Exception as e:
            logger.error(f"Error retrieving all user profiles: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve user profiles: {e}")

    async def update_user_profile_admin(self, user_id: str, user_update: UserUpdateAdmin, current_admin: UserProfile) -> Dict[str, Any]:
        """
        Updates a user's profile (including username, tier, roles, and status) as an administrator.
        Performs granular permission checks based on current_admin's roles/claims.
        """
        logger.debug(f"Admin {current_admin.user_id} attempting to update user {user_id} with data: {user_update.model_dump(exclude_unset=True)}")
        try:
            target_user_profile = await self.user_manager.get_user(user_id)
            if not target_user_profile:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

            update_data = user_update.model_dump(exclude_unset=True)
            
            # --- Granular Admin Permission Checks ---
            # Creator has full override. Otherwise, check specific permissions.
            is_creator = "creator" in current_admin.roles

            if 'tier' in update_data and not is_creator:
                new_tier = update_data['tier']
                # Check if admin has permission to manage this specific tier
                if not current_admin.get(f'can_manage_tier_{new_tier}', False):
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Not authorized to set user tier to '{new_tier}'.")
            
            if 'roles' in update_data and not is_creator:
                # Check if admin has permission to assign all requested roles
                for role in update_data['roles']:
                    if not current_admin.get(f'can_assign_role_{role}', False):
                        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Not authorized to assign role '{role}'.")
            
            if 'status' in update_data and not is_creator:
                # Check if admin has permission to change user status
                if not current_admin.get('can_change_user_status', False):
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to change user account status.")

            # Update Firestore document and Firebase Auth custom claims via UserManager
            result = await self.user_manager.update_user_profile(user_id, update_data)
            
            if not result["success"]:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])

            # Fetch and return the updated user profile
            updated_user_info = await self.user_manager.get_user(user_id)
            if not updated_user_info: # Should not happen if update was successful
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve updated user profile.")

            return updated_user_info
        except HTTPException:
            raise # Re-raise HTTPExceptions
        except Exception as e:
            logger.error(f"Error updating user profile {user_id} by admin {current_admin.user_id}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update user profile: {e}")

    async def update_user_status_admin(self, user_id: str, new_status: str, current_admin: UserProfile) -> Dict[str, Any]:
        """
        Updates a user's account status (active, disabled, suspended) as an administrator.
        Requires 'can_change_user_status' permission or 'creator' role.
        """
        logger.debug(f"Admin {current_admin.user_id} attempting to update status for user {user_id} to {new_status}")
        try:
            if "creator" not in current_admin.roles and not current_admin.get('can_change_user_status', False):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to change user account status.")

            # Use UserManager to update status
            result = await self.user_manager.update_user_profile(user_id, {"status": new_status})
            if not result["success"]:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])

            # If status is disabled/suspended, revoke Firebase tokens to force logout
            if new_status in ["disabled", "suspended"]:
                try:
                    auth.revoke_refresh_tokens(user_id)
                    logger.info(f"Firebase refresh tokens revoked for user {user_id} due to status change to {new_status}.")
                except firebase_exceptions.FirebaseError as e:
                    logger.warning(f"Failed to revoke Firebase tokens for {user_id}: {e}")

            updated_user_info = await self.user_manager.get_user(user_id)
            if not updated_user_info:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve updated user profile after status change.")
            return updated_user_info
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating user status for user {user_id} by admin {current_admin.user_id}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update user status: {e}")

    async def purge_user_sessions(self, user_id: str, current_admin: UserProfile):
        """
        Purges active sessions for a specific user by revoking Firebase refresh tokens.
        Requires admin privileges.
        """
        logger.debug(f"Admin {current_admin.user_id} attempting to purge sessions for user {user_id}")
        try:
            # Admins can purge individual user sessions. Creator can purge all.
            # If a non-creator admin wants to purge, they must have 'can_purge_user_sessions'
            if "creator" not in current_admin.roles and not current_admin.get('can_purge_user_sessions', False):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to purge user sessions.")

            auth.revoke_refresh_tokens(user_id)
            logger.info(f"Firebase refresh tokens revoked for user {user_id}.")
        except firebase_exceptions.UserNotFoundError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
        except Exception as e:
            logger.error(f"Error revoking tokens for user {user_id} by admin {current_admin.user_id}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to purge user sessions: {e}")

    async def purge_all_sessions(self, current_admin: UserProfile):
        """
        Purges all active sessions for all users by revoking all Firebase refresh tokens.
        Requires 'creator' role.
        """
        logger.debug(f"Creator {current_admin.user_id} attempting to purge all sessions.")
        if "creator" not in current_admin.roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the creator can purge all sessions.")
        
        # Firebase Admin SDK does not have a direct 'revoke_all_tokens' function.
        # This would typically involve iterating through all users and revoking tokens individually,
        # or a more complex solution if millions of users. For now, we'll simulate/log.
        # In a real large-scale app, this might be a background task or specific Firebase feature.
        logger.warning(f"Simulating purge of ALL user sessions. This is a placeholder for a large-scale operation.")
        # Example: Fetch all UIDs and call auth.revoke_refresh_tokens(uid) for each.
        # For a truly massive scale, this needs careful consideration of Firebase limits.
        
        # For now, we'll just log and return success.
        # A more robust implementation would fetch users in batches and revoke.
        # users = await self.user_manager.get_all_user_profiles()
        # for user in users:
        #     try:
        #         auth.revoke_refresh_tokens(user['uid'])
        #     except Exception as e:
        #         logger.warning(f"Failed to revoke token for user {user['uid']} during all-purge: {e}")
        logger.info(f"All user sessions conceptually purged by creator {current_admin.user_id}.")


    async def grant_admin_access(self, target_user_id: str, permissions: Dict[str, Any], replace_all: bool, current_admin: UserProfile):
        """
        Grants specific administrative permissions (custom claims) to another user.
        Requires 'creator' role.
        """
        logger.debug(f"Creator {current_admin.user_id} attempting to grant admin access to {target_user_id} with permissions: {permissions}, replace_all: {replace_all}")
        if "creator" not in current_admin.roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the creator can grant/modify admin permissions.")
        
        try:
            user = auth.get_user(target_user_id)
            current_claims = user.custom_claims if user.custom_claims else {}

            if replace_all:
                new_claims = permissions
            else:
                new_claims = {**current_claims, **permissions}
            
            # Ensure 'admin' role is added if any admin permission is granted and not already present
            if any(key.startswith('can_') for key in new_claims.keys()) and 'admin' not in new_claims.get('roles', []):
                roles = set(new_claims.get('roles', []))
                roles.add('admin')
                new_claims['roles'] = list(roles)

            auth.set_custom_user_claims(target_user_id, new_claims)
            logger.info(f"Custom claims updated for user {target_user_id}: {new_claims}")
            # Invalidate token to ensure new claims take effect
            auth.revoke_refresh_tokens(target_user_id)
            logger.info(f"Tokens revoked for {target_user_id} after claims update.")

            # Also update Firestore profile to reflect roles (tier is handled by user_manager update)
            # Ensure roles are consistent between custom claims and Firestore profile
            await self.user_manager.update_user_profile(target_user_id, {"roles": new_claims.get('roles', ['user'])})

        except firebase_exceptions.UserNotFoundError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target user not found.")
        except Exception as e:
            logger.error(f"Error granting admin access to {target_user_id} by creator {current_admin.user_id}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to grant admin access: {e}")


    # --- Global Configuration Management (RBAC Capabilities & Tiers) ---
    async def get_rbac_capabilities(self) -> Dict[str, Any]:
        """
        Retrieves the current RBAC capabilities configuration from Firestore.
        """
        try:
            capabilities_doc = await self.firestore_manager.get_global_config("rbac_capabilities")
            if capabilities_doc and capabilities_doc.get('capabilities'):
                return capabilities_doc['capabilities']
            return {} # Return empty dict if not found
        except Exception as e:
            logger.error(f"Error retrieving RBAC capabilities from Firestore: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve RBAC capabilities: {e}")

    async def update_rbac_capabilities(self, capability_update: CapabilityUpdate, current_admin: UserProfile) -> Dict[str, Any]:
        """
        Updates a specific RBAC capability or the entire capabilities document in Firestore.
        Requires 'creator' role.
        """
        logger.debug(f"Admin {current_admin.user_id} attempting to update RBAC capabilities.")
        if "creator" not in current_admin.roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the creator can update RBAC capabilities.")
        
        try:
            current_capabilities_doc = await self.firestore_manager.get_global_config("rbac_capabilities")
            current_capabilities = current_capabilities_doc.get('capabilities', {}) if current_capabilities_doc else {}

            updated_capabilities = current_capabilities
            if capability_update.capability_key:
                if capability_update.capability_key not in updated_capabilities:
                    updated_capabilities[capability_update.capability_key] = {"default": False, "roles": {}}
                
                if capability_update.default_value is not None:
                    updated_capabilities[capability_update.capability_key]['default'] = capability_update.default_value
                
                if capability_update.roles is not None:
                    updated_capabilities[capability_update.capability_key]['roles'] = capability_update.roles
            else:
                updated_capabilities = capability_update.full_capabilities or {}

            await self.firestore_manager.set_global_config("rbac_capabilities", {"capabilities": updated_capabilities})
            logger.info("RBAC capabilities updated in Firestore.")
            return updated_capabilities
        except Exception as e:
            logger.error(f"Error updating RBAC capabilities in Firestore by admin {current_admin.user_id}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update RBAC capabilities: {e}")

    async def get_tier_hierarchy(self) -> Dict[str, Any]:
        """
        Retrieves the current tier hierarchy configuration from Firestore.
        """
        try:
            tiers_doc = await self.firestore_manager.get_global_config("tiers")
            if tiers_doc and tiers_doc.get('tiers'):
                return tiers_doc['tiers']
            return {}
        except Exception as e:
            logger.error(f"Error retrieving tier hierarchy from Firestore: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve tier hierarchy: {e}")

    async def update_tier_hierarchy(self, tier_update: TierUpdate, current_admin: UserProfile) -> Dict[str, Any]:
        """
        Updates a specific tier or the entire tier hierarchy document in Firestore.
        Requires 'creator' role.
        """
        logger.debug(f"Admin {current_admin.user_id} attempting to update tier hierarchy.")
        if "creator" not in current_admin.roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the creator can update tier hierarchy.")

        try:
            current_tiers_doc = await self.firestore_manager.get_global_config("tiers")
            current_tiers = current_tiers_doc.get('tiers', {}) if current_tiers_doc else {}

            updated_tiers = current_tiers
            if tier_update.tier_name:
                if tier_update.tier_name not in updated_tiers:
                    updated_tiers[tier_update.tier_name] = {"level": 0, "description": ""}
                
                if tier_update.level is not None:
                    updated_tiers[tier_update.tier_name]['level'] = tier_update.level
                
                if tier_update.description is not None:
                    updated_tiers[tier_update.tier_name]['description'] = tier_update.description
            else:
                updated_tiers = tier_update.full_tiers or {}

            await self.firestore_manager.set_global_config("tiers", {"tiers": updated_tiers})
            logger.info("Tier hierarchy updated in Firestore.")
            return updated_tiers
        except Exception as e:
            logger.error(f"Error updating tier hierarchy in Firestore by admin {current_admin.user_id}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update tier hierarchy: {e}")

    # --- Global API Management Methods (Delegated to ApiUsageService) ---
    async def create_global_api_config(self, api_config: GlobalApiConfigCreate, current_admin: UserProfile) -> Dict[str, Any]:
        """Creates a new global/default API configuration."""
        # Permission check already in API endpoint, but can be duplicated here for robustness
        logger.debug(f"Admin {current_admin.user_id} calling ApiUsageService to create global API config.")
        return await self.api_usage_service.create_global_api_config(api_config)

    async def get_global_api_configs(self) -> List[Dict[str, Any]]:
        """Retrieves all global/default API configurations."""
        return await self.api_usage_service.get_global_api_configs()

    async def update_global_api_config(self, api_id: str, api_config_update: GlobalApiConfigUpdate, current_admin: UserProfile) -> Dict[str, Any]:
        """Updates an existing global/default API configuration."""
        logger.debug(f"Admin {current_admin.user_id} calling ApiUsageService to update global API config {api_id}.")
        return await self.api_usage_service.update_global_api_config(api_id, api_config_update)

    async def delete_global_api_config(self, api_id: str, current_admin: UserProfile):
        """Deletes a global/default API configuration."""
        logger.debug(f"Admin {current_admin.user_id} calling ApiUsageService to delete global API config {api_id}.")
        await self.api_usage_service.delete_global_api_config(api_id)

    async def update_api_limits(self, limit_update: ApiCallLimitUpdate, current_admin: UserProfile) -> Dict[str, Any]:
        """Updates default API call limits for a specific tier."""
        logger.debug(f"Admin {current_admin.user_id} calling ApiUsageService to update API limits for tier {limit_update.tier}.")
        return await self.api_usage_service.update_api_limits(limit_update)

    # --- Analytics for Unanswered Queries ---
    async def get_unanswered_queries_analytics(self, current_admin: UserProfile) -> List[Dict[str, Any]]:
        """
        Retrieves analytics data on unanswered queries and AI-generated tool suggestions.
        """
        logger.debug(f"Admin {current_admin.user_id} fetching unanswered queries analytics.")
        try:
            # This will query a specific collection in Firestore, e.g., 'analytics/unanswered_queries'
            # Assuming 'analytics_tracker' or a dedicated 'AnalyticsService' will log these.
            # For now, we'll just fetch from a predefined collection.
            unanswered_docs = await self.firestore_manager.get_all_global_config_documents(collection_name="unanswered_queries_analytics")
            return unanswered_docs
        except Exception as e:
            logger.error(f"Error fetching unanswered queries analytics for admin {current_admin.user_id}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve unanswered queries analytics: {e}")

