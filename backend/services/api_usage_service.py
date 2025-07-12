# backend/services/api_usage_service.py

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
import uuid

# Import FirestoreManager and ConfigManager
from database.firestore_manager import FirestoreManager
from config.config_manager import config_manager

# Import Pydantic models for global API config
from backend.models.admin_models import GlobalApiConfigCreate, GlobalApiConfigUpdate, ApiCallLimitUpdate

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ApiUsageService:
    """
    Manages API call limits, usage tracking, and dynamic distribution for default APIs.
    Also handles user-defined API configurations and overrides.
    """
    def __init__(self, firestore_manager: FirestoreManager, config_manager_instance):
        self.firestore_manager = firestore_manager
        self.config_manager = config_manager_instance
        logger.info("ApiUsageService initialized.")

    # --- Global/Default API Configuration Management ---
    async def create_global_api_config(self, api_config: GlobalApiConfigCreate) -> Dict[str, Any]:
        """Creates a new global/default API configuration in Firestore."""
        api_id = str(uuid.uuid4())
        api_data = api_config.model_dump()
        api_data['id'] = api_id # Store id within the document
        api_data['created_at'] = datetime.now(timezone.utc)
        api_data['last_updated_at'] = datetime.now(timezone.utc)

        # Store in a global config collection, e.g., 'global_configs/api_configs'
        await self.firestore_manager.set_global_config_document(
            collection_name="global_api_configs",
            document_id=api_id,
            data=api_data
        )
        logger.info(f"Created global API config: {api_id} - {api_config.name}")
        return api_data

    async def get_global_api_configs(self) -> List[Dict[str, Any]]:
        """Retrieves all global/default API configurations from Firestore."""
        configs = await self.firestore_manager.get_all_global_config_documents(collection_name="global_api_configs")
        return configs

    async def update_global_api_config(self, api_id: str, api_config_update: GlobalApiConfigUpdate) -> Dict[str, Any]:
        """Updates an existing global/default API configuration in Firestore."""
        update_data = api_config_update.model_dump(exclude_unset=True)
        update_data['last_updated_at'] = datetime.now(timezone.utc)
        
        await self.firestore_manager.update_global_config_document(
            collection_name="global_api_configs",
            document_id=api_id,
            data=update_data
        )
        logger.info(f"Updated global API config: {api_id}")
        # Fetch and return the updated document
        updated_config = await self.firestore_manager.get_global_config_document(
            collection_name="global_api_configs",
            document_id=api_id
        )
        if not updated_config:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Global API config not found after update.")
        return updated_config

    async def delete_global_api_config(self, api_id: str):
        """Deletes a global/default API configuration from Firestore."""
        await self.firestore_manager.delete_global_config_document(
            collection_name="global_api_configs",
            document_id=api_id
        )
        logger.info(f"Deleted global API config: {api_id}")

    # --- API Call Limits Management (Tier-based and Dynamic) ---
    async def get_api_limits_config(self) -> Dict[str, Any]:
        """Retrieves the current API call limits configuration from Firestore."""
        limits_doc = await self.firestore_manager.get_global_config("api_limits")
        return limits_doc.get('limits', {}) if limits_doc else {}

    async def update_api_limits(self, limit_update: ApiCallLimitUpdate) -> Dict[str, Any]:
        """Updates API call limits for a specific tier or replaces the entire limits document."""
        current_limits_doc = await self.firestore_manager.get_global_config("api_limits")
        current_limits = current_limits_doc.get('limits', {}) if current_limits_doc else {}

        updated_limits = current_limits
        if limit_update.tier: # Update limits for a specific tier
            if limit_update.replace_all_limits:
                updated_limits[limit_update.tier] = limit_update.limits
            else:
                updated_limits[limit_update.tier] = {
                    **updated_limits.get(limit_update.tier, {}),
                    **limit_update.limits
                }
        else: # If no specific tier, assume full overwrite (should not happen with ApiCallLimitUpdate model)
            # This case is primarily for initial setup or full replacement of the 'limits' map
            # The ApiCallLimitUpdate model expects a 'tier'
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tier must be specified for API limit update.")

        await self.firestore_manager.set_global_config("api_limits", {"limits": updated_limits})
        logger.info(f"API limits updated in Firestore for tier: {limit_update.tier}")
        return updated_limits

    async def get_user_api_usage(self, user_id: str, api_id: str, period: str = "month") -> int:
        """Retrieves a user's API usage for a specific API and period."""
        # This will need to query user-specific usage logs in Firestore
        # Placeholder for now
        return 0

    async def increment_api_usage(self, user_id: str, api_id: str, is_default_api: bool = True):
        """Increments API usage for a user for a specific API."""
        # This will update usage counters in Firestore
        # Placeholder for now
        logger.debug(f"Incrementing usage for user {user_id}, API {api_id}, default: {is_default_api}")
        pass

    async def check_api_limit(self, user_id: str, api_id: str, user_tier: str) -> bool:
        """
        Checks if a user has remaining calls for a given API, considering all override rules.
        This is a core method that LLMService/Tools will call.
        """
        # This is where the complex logic for:
        # 1. Creator override (unlimited)
        # 2. User-defined API override
        # 3. Tier-based default API limits (with dynamic adjustment)
        # 4. Individual user overrides
        # will be implemented.
        
        # Placeholder for now: Always allow
        logger.debug(f"Checking API limit for user {user_id}, API {api_id}, tier {user_tier}. (Placeholder: Always True)")
        return True # Placeholder: Always allow for now

    # --- User-Defined API Management ---
    async def create_user_api_config(self, user_id: str, api_config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Creates a new user-defined API configuration in Firestore."""
        # This will store in artifacts/{appId}/users/{userId}/user_api_configs/{api_id}
        api_id = str(uuid.uuid4())
        api_config_data['id'] = api_id
        api_config_data['created_at'] = datetime.now(timezone.utc)
        api_config_data['last_updated_at'] = datetime.now(timezone.utc)
        
        await self.firestore_manager.set_user_data_document(
            user_id=user_id,
            collection_name="user_api_configs",
            document_id=api_id,
            data=api_config_data
        )
        logger.info(f"User {user_id} created personal API config: {api_id}")
        return api_config_data

    async def get_user_api_configs(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieves all user-defined API configurations for a user from Firestore."""
        configs = await self.firestore_manager.get_all_user_data_documents(
            user_id=user_id,
            collection_name="user_api_configs"
        )
        return configs

    async def update_user_api_config(self, user_id: str, api_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Updates an existing user-defined API configuration in Firestore."""
        update_data['last_updated_at'] = datetime.now(timezone.utc)
        await self.firestore_manager.update_user_data_document(
            user_id=user_id,
            collection_name="user_api_configs",
            document_id=api_id,
            data=update_data
        )
        logger.info(f"User {user_id} updated personal API config: {api_id}")
        updated_config = await self.firestore_manager.get_user_data_document(
            user_id=user_id,
            collection_name="user_api_configs",
            document_id=api_id
        )
        if not updated_config:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User API config not found after update.")
        return updated_config

    async def delete_user_api_config(self, user_id: str, api_id: str):
        """Deletes a user-defined API configuration from Firestore."""
        await self.firestore_manager.delete_user_data_document(
            user_id=user_id,
            collection_name="user_api_configs",
            document_id=api_id
        )
        logger.info(f"User {user_id} deleted personal API config: {api_id}")

    # --- Dynamic API Call Distribution Logic (Placeholder for future expansion) ---
    async def _adjust_tier_limits_dynamically(self, global_usage_percentage: float):
        """
        Internal method to dynamically adjust tier limits based on global usage.
        This would be triggered by a background task or a usage threshold.
        """
        logger.info(f"Dynamic limit adjustment triggered. Global usage: {global_usage_percentage}%")
        # Example: if global_usage_percentage > 80, reduce Free tier limits by 50%
        # This will involve reading current limits, calculating new limits, and writing back to Firestore.
        pass

    async def _get_global_api_usage(self, api_id: str, period: str = "month") -> int:
        """Internal method to get global usage of a default API."""
        # This would query aggregated usage data
        return 0 # Placeholder

    async def _monitor_global_api_usage(self):
        """
        Method to be run as a background task to monitor global API usage
        and trigger dynamic adjustments.
        """
        # This would periodically call _get_global_api_usage and _adjust_tier_limits_dynamically
        pass
