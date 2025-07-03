# backend/services/admin_service.py

import logging
from typing import List, Dict, Any, Optional
from fastapi import HTTPException, status

# Import FirestoreManager for direct database interaction
from database.firestore_manager import firestore_manager

# Import UserManager for user-related Firebase Auth operations
from utils.user_manager import get_all_users, update_user_tier_and_roles, get_user_by_id

# Import Pydantic models for request/response validation
from backend.models.admin_models import UserUpdateAdmin, CapabilityUpdate, TierUpdate

logger = logging.getLogger(__name__)

class AdminService:
    """
    Provides business logic for administrative operations,
    including user management and dynamic configuration of RBAC capabilities and tiers.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AdminService, cls).__new__(cls)
            logger.info("AdminService initialized.")
        return cls._instance

    async def get_all_user_profiles(self) -> List[Dict[str, Any]]:
        """
        Retrieves all user profiles.
        """
        try:
            # This uses the get_all_users from user_manager which interacts with Firebase Auth
            users = await get_all_users()
            return list(users.values())
        except Exception as e:
            logger.error(f"Error retrieving all user profiles: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve user profiles: {e}")

    async def update_user_profile_admin(self, user_id: str, user_update: UserUpdateAdmin) -> Dict[str, Any]:
        """
        Updates a user's profile (including tier and roles) as an administrator.
        """
        try:
            # Fetch current user info to ensure user exists
            current_user_info = await get_user_by_id(user_id)
            if not current_user_info:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

            update_data = user_update.model_dump(exclude_unset=True)
            
            # Update Firebase Auth custom claims if tier or roles are provided
            if 'tier' in update_data or 'roles' in update_data:
                new_tier = update_data.get('tier', current_user_info['tier'])
                new_roles = update_data.get('roles', current_user_info['roles'])
                
                await update_user_tier_and_roles(user_id, new_tier, new_roles)
                logger.info(f"Admin updated user {user_id} custom claims: tier={new_tier}, roles={new_roles}")

            # Update other profile data in Firestore if provided (e.g., username, email, if allowed)
            # Note: Email changes in Firebase Auth are separate and more complex.
            # For simplicity, we'll assume only username is directly updatable via Firestore for now.
            firestore_update_data = {}
            if 'username' in update_data:
                firestore_update_data['username'] = update_data['username']
            # If you want to allow email change via admin, it's `auth.update_user(user_id, email=new_email)`
            # but then you'd also need to update the Firestore record.

            if firestore_update_data:
                await firestore_manager.update_user_data(user_id, firestore_update_data)
                logger.info(f"Admin updated user {user_id} Firestore data: {firestore_update_data}")

            # Fetch and return the updated user profile
            updated_user_info = await get_user_by_id(user_id)
            if not updated_user_info: # Should not happen if update was successful
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve updated user profile.")

            return updated_user_info
        except HTTPException:
            raise # Re-raise HTTPExceptions
        except Exception as e:
            logger.error(f"Error updating user profile {user_id} by admin: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update user profile: {e}")

    # --- Global Configuration Management (RBAC Capabilities & Tiers) ---
    async def get_rbac_capabilities(self) -> Dict[str, Any]:
        """
        Retrieves the current RBAC capabilities configuration from Firestore.
        """
        try:
            capabilities_doc = await firestore_manager.get_global_config("rbac_capabilities")
            if capabilities_doc and capabilities_doc.get('capabilities'):
                return capabilities_doc['capabilities']
            return {} # Return empty dict if not found
        except Exception as e:
            logger.error(f"Error retrieving RBAC capabilities from Firestore: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve RBAC capabilities: {e}")

    async def update_rbac_capabilities(self, capability_update: CapabilityUpdate) -> Dict[str, Any]:
        """
        Updates a specific RBAC capability or the entire capabilities document in Firestore.
        """
        try:
            # Fetch current capabilities to merge updates
            current_capabilities_doc = await firestore_manager.get_global_config("rbac_capabilities")
            current_capabilities = current_capabilities_doc.get('capabilities', {}) if current_capabilities_doc else {}

            # Apply updates
            updated_capabilities = current_capabilities
            if capability_update.capability_key: # Update a specific capability
                if capability_update.capability_key not in updated_capabilities:
                    updated_capabilities[capability_update.capability_key] = {"default": False, "roles": {}} # Initialize if new
                
                # Update default value if provided
                if capability_update.default_value is not None:
                    updated_capabilities[capability_update.capability_key]['default'] = capability_update.default_value
                
                # Update roles if provided
                if capability_update.roles is not None:
                    # Merge or overwrite roles based on logic. Here, we overwrite for simplicity.
                    updated_capabilities[capability_update.capability_key]['roles'] = capability_update.roles
            else: # If no specific key, assume full overwrite (or deep merge if more complex)
                # This path is for replacing the entire 'capabilities' map
                updated_capabilities = capability_update.full_capabilities or {}

            await firestore_manager.set_global_config("rbac_capabilities", {"capabilities": updated_capabilities})
            logger.info("RBAC capabilities updated in Firestore.")
            return updated_capabilities
        except Exception as e:
            logger.error(f"Error updating RBAC capabilities in Firestore: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update RBAC capabilities: {e}")

    async def get_tier_hierarchy(self) -> Dict[str, Any]:
        """
        Retrieves the current tier hierarchy configuration from Firestore.
        """
        try:
            tiers_doc = await firestore_manager.get_global_config("tiers")
            if tiers_doc and tiers_doc.get('tiers'):
                return tiers_doc['tiers']
            return {} # Return empty dict if not found
        except Exception as e:
            logger.error(f"Error retrieving tier hierarchy from Firestore: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve tier hierarchy: {e}")

    async def update_tier_hierarchy(self, tier_update: TierUpdate) -> Dict[str, Any]:
        """
        Updates a specific tier or the entire tier hierarchy document in Firestore.
        """
        try:
            # Fetch current tiers to merge updates
            current_tiers_doc = await firestore_manager.get_global_config("tiers")
            current_tiers = current_tiers_doc.get('tiers', {}) if current_tiers_doc else {}

            updated_tiers = current_tiers
            if tier_update.tier_name: # Update a specific tier
                if tier_update.tier_name not in updated_tiers:
                    updated_tiers[tier_update.tier_name] = {"level": 0, "description": ""} # Initialize if new
                
                # Update level if provided
                if tier_update.level is not None:
                    updated_tiers[tier_update.tier_name]['level'] = tier_update.level
                
                # Update description if provided
                if tier_update.description is not None:
                    updated_tiers[tier_update.tier_name]['description'] = tier_update.description
            else: # If no specific tier_name, assume full overwrite (or deep merge)
                updated_tiers = tier_update.full_tiers or {}

            await firestore_manager.set_global_config("tiers", {"tiers": updated_tiers})
            logger.info("Tier hierarchy updated in Firestore.")
            return updated_tiers
        except Exception as e:
            logger.error(f"Error updating tier hierarchy in Firestore: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update tier hierarchy: {e}")

# Instantiate the AdminService as a singleton
admin_service = AdminService()
