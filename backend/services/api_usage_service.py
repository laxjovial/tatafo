# backend/services/api_usage_service.py

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
import uuid
from fastapi import HTTPException, status # Import HTTPException and status for error handling

# Import FirestoreManager and ConfigManager
from database.firestore_manager import FirestoreManager
from config.config_manager import config_manager
from utils.user_manager import UserManager # Will need UserManager to get user profile and check creator status/overrides

# Import Pydantic models for global API config
from backend.models.admin_models import GlobalApiConfigCreate, GlobalApiConfigUpdate, ApiCallLimitUpdate
from backend.models.user_models import UserProfile # To receive UserProfile in check_api_limit

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ApiUsageService:
    """
    Manages API call limits, usage tracking, and dynamic distribution for default APIs.
    Also handles user-defined API configurations and overrides.

    Firestore Data Structures (Conceptual):
    - Global API Configs: 'global_configs/global_api_configs/{api_id}'
    - API Limits (Global Configuration): 'global_configs/api_limits'
        - { "limits": { "tier_name": { "monthly_calls": int, "daily_calls": int, "dynamic_monthly_adjustment": int, "dynamic_daily_adjustment": int } } }
    - User-Defined API Configs: 'artifacts/{appId}/users/{userId}/user_api_configs/{api_id}'
        - { "api_id": ..., "name": ..., "base_url": ..., "user_defined_limit_monthly": int, "user_defined_limit_daily": int, "creator_override_monthly": int, "creator_override_daily": int, "creator_override_unlimited": bool }
    - User API Usage Tracking: 'artifacts/{appId}/users/{userId}/api_usage/{api_id}'
        - { "monthly_usage": int, "daily_usage": int, "last_reset_month": "YYYY-MM", "last_reset_day": "YYYY-MM-DD" }
    """
    def __init__(self, firestore_manager: FirestoreManager, config_manager_instance, user_manager: UserManager):
        self.firestore_manager = firestore_manager
        self.config_manager = config_manager_instance
        self.user_manager = user_manager # Store UserManager for fetching user profiles/claims
        self.app_id = config_manager_instance.get("app_id", "default-app-id")
        logger.info("ApiUsageService initialized.")

    # --- Helper Methods for Date/Time ---
    def _get_current_month_str(self) -> str:
        """Returns current month in YYYY-MM format."""
        return datetime.now(timezone.utc).strftime("%Y-%m")

    def _get_current_day_str(self) -> str:
        """Returns current day in YYYY-MM-DD format."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # --- Global/Default API Configuration Management (already in skeletal, kept for completeness) ---
    async def create_global_api_config(self, api_config: GlobalApiConfigCreate) -> Dict[str, Any]:
        """Creates a new global/default API configuration in Firestore."""
        api_id = str(uuid.uuid4())
        api_data = api_config.model_dump()
        api_data['id'] = api_id # Store id within the document
        api_data['created_at'] = datetime.now(timezone.utc)
        api_data['last_updated_at'] = datetime.now(timezone.utc)

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
        else: # This case should ideally not be hit with the current ApiCallLimitUpdate model
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tier must be specified for API limit update.")

        await self.firestore_manager.set_global_config("api_limits", {"limits": updated_limits})
        logger.info(f"API limits updated in Firestore for tier: {limit_update.tier}")
        return updated_limits

    async def get_user_api_usage_document(self, user_id: str, api_id: str) -> Dict[str, Any]:
        """Retrieves a user's API usage document for a specific API."""
        return await self.firestore_manager.get_user_data_document(
            user_id=user_id,
            collection_name="api_usage",
            document_id=api_id
        )

    async def _reset_usage_if_needed(self, usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resets daily/monthly usage counters if the period has passed."""
        current_month = self._get_current_month_str()
        current_day = self._get_current_day_str()

        if usage_data.get('last_reset_month') != current_month:
            usage_data['monthly_usage'] = 0
            usage_data['last_reset_month'] = current_month
            logger.debug(f"Monthly usage reset for API {usage_data.get('id')}.")

        if usage_data.get('last_reset_day') != current_day:
            usage_data['daily_usage'] = 0
            usage_data['last_reset_day'] = current_day
            logger.debug(f"Daily usage reset for API {usage_data.get('id')}.")
        
        return usage_data

    async def increment_api_usage(self, user_id: str, api_id: str):
        """
        Increments API usage for a user for a specific API.
        Handles daily/monthly resets automatically.
        """
        try:
            # Get current usage data or initialize if not exists
            usage_data = await self.get_user_api_usage_document(user_id, api_id)
            if not usage_data:
                usage_data = {
                    "id": api_id,
                    "monthly_usage": 0,
                    "daily_usage": 0,
                    "last_reset_month": self._get_current_month_str(),
                    "last_reset_day": self._get_current_day_str(),
                    "created_at": datetime.now(timezone.utc)
                }
            
            # Reset usage if new period
            usage_data = self._reset_usage_if_needed(usage_data)

            usage_data['monthly_usage'] += 1
            usage_data['daily_usage'] += 1
            usage_data['last_updated_at'] = datetime.now(timezone.utc)

            await self.firestore_manager.set_user_data_document(
                user_id=user_id,
                collection_name="api_usage",
                document_id=api_id,
                data=usage_data
            )
            logger.debug(f"Incremented usage for user {user_id}, API {api_id}. Monthly: {usage_data['monthly_usage']}, Daily: {usage_data['daily_usage']}")
        except Exception as e:
            logger.error(f"Error incrementing API usage for user {user_id}, API {api_id}: {e}", exc_info=True)
            # Do not raise HTTPException here, as this is a background operation that shouldn't block the main request

    async def check_api_limit(self, user_profile: UserProfile, api_id: str) -> bool:
        """
        Checks if a user has remaining calls for a given API, considering all override rules.
        This is a core method that LLMService/Tools will call.
        Priority: Creator Override > Individual User Override (by Creator) > User-Defined API Limit > Tier-Based Default API Limit.
        """
        user_id = user_profile.user_id
        user_tier = user_profile.tier
        is_creator = "creator" in user_profile.roles

        logger.debug(f"Checking API limit for user {user_id} (Tier: {user_tier}) for API: {api_id}")

        # 1. Creator's Global Unlimited Override
        # This can be a custom claim or a specific flag in the user's profile set by the creator.
        if is_creator and user_profile.get('unlimited_api_access', False):
            logger.debug(f"Creator {user_id} has unlimited API access.")
            return True

        # Fetch current usage for this user and API
        user_api_usage = await self.get_user_api_usage_document(user_id, api_id)
        if not user_api_usage:
            user_api_usage = {
                "monthly_usage": 0,
                "daily_usage": 0,
                "last_reset_month": self._get_current_month_str(),
                "last_reset_day": self._get_current_day_str(),
            }
        else:
            user_api_usage = self._reset_usage_if_needed(user_api_usage) # Ensure usage is current

        current_monthly_usage = user_api_usage.get('monthly_usage', 0)
        current_daily_usage = user_api_usage.get('daily_usage', 0)

        # 2. Individual User Override (set by Creator for this specific user/API)
        # These overrides can be stored directly on the user's api_usage document or a dedicated 'overrides' collection.
        # Let's assume they are stored on the user's api_usage document for this API.
        user_api_config = await self.get_user_api_config_document(user_id, api_id) # Try to get user's specific API config

        creator_override_unlimited = user_api_config.get('creator_override_unlimited', False) if user_api_config else False
        creator_override_monthly = user_api_config.get('creator_override_monthly', None) if user_api_config else None
        creator_override_daily = user_api_config.get('creator_override_daily', None) if user_api_config else None

        if creator_override_unlimited:
            logger.debug(f"User {user_id} has creator-granted unlimited access for API {api_id}.")
            return True
        
        if creator_override_monthly is not None or creator_override_daily is not None:
            # Apply creator's specific numeric override
            if creator_override_monthly is not None and current_monthly_usage >= creator_override_monthly:
                logger.info(f"User {user_id} exceeded creator's monthly override limit for API {api_id}.")
                return False
            if creator_override_daily is not None and current_daily_usage >= creator_override_daily:
                logger.info(f"User {user_id} exceeded creator's daily override limit for API {api_id}.")
                return False
            logger.debug(f"User {user_id} within creator's override limits for API {api_id}.")
            return True # If creator override exists and limits not hit, allow.

        # 3. User-Defined API Limit (if this is a user's custom API)
        # This applies if the user has provided their own API key and set a limit on its usage.
        user_defined_limit_monthly = user_api_config.get('user_defined_limit_monthly', None) if user_api_config else None
        user_defined_limit_daily = user_api_config.get('user_defined_limit_daily', None) if user_api_config else None

        if user_defined_limit_monthly is not None or user_defined_limit_daily is not None:
            if user_defined_limit_monthly is not None and current_monthly_usage >= user_defined_limit_monthly:
                logger.info(f"User {user_id} exceeded their own monthly limit for custom API {api_id}.")
                return False
            if user_defined_limit_daily is not None and current_daily_usage >= user_defined_limit_daily:
                logger.info(f"User {user_id} exceeded their own daily limit for custom API {api_id}.")
                return False
            logger.debug(f"User {user_id} within their own defined limits for API {api_id}.")
            return True # If user-defined limit exists and not hit, allow.

        # 4. Tier-Based Default API Limit (with dynamic adjustments)
        global_api_limits_config = await self.get_api_limits_config()
        tier_limits = global_api_limits_config.get(user_tier, {})

        default_monthly_limit = tier_limits.get('monthly_calls', -1) # -1 means no default limit
        default_daily_limit = tier_limits.get('daily_calls', -1)

        # Apply dynamic adjustments from global config
        dynamic_monthly_adjustment = tier_limits.get('dynamic_monthly_adjustment', 0)
        dynamic_daily_adjustment = tier_limits.get('dynamic_daily_adjustment', 0)

        adjusted_monthly_limit = default_monthly_limit + dynamic_monthly_adjustment
        adjusted_daily_limit = default_daily_limit + dynamic_daily_adjustment

        # Ensure limits don't go below zero due to aggressive adjustments
        adjusted_monthly_limit = max(0, adjusted_monthly_limit) if default_monthly_limit != -1 else -1
        adjusted_daily_limit = max(0, adjusted_daily_limit) if default_daily_limit != -1 else -1

        if adjusted_monthly_limit != -1 and current_monthly_usage >= adjusted_monthly_limit:
            logger.info(f"User {user_id} (Tier: {user_tier}) exceeded adjusted monthly limit for default API {api_id}.")
            return False
        
        if adjusted_daily_limit != -1 and current_daily_usage >= adjusted_daily_limit:
            logger.info(f"User {user_id} (Tier: {user_tier}) exceeded adjusted daily limit for default API {api_id}.")
            return False
        
        logger.debug(f"User {user_id} within tier-based default limits for API {api_id}.")
        return True # If no limits hit, allow.

    # --- User-Defined API Management (already in skeletal, kept for completeness) ---
    async def create_user_api_config(self, user_id: str, api_config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Creates a new user-defined API configuration in Firestore."""
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

    async def get_user_api_config_document(self, user_id: str, api_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single user-defined API configuration document."""
        return await self.firestore_manager.get_user_data_document(
            user_id=user_id,
            collection_name="user_api_configs",
            document_id=api_id
        )

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

    # --- Dynamic API Call Distribution Logic (Fuller Implementation) ---
    async def _get_total_default_api_usage(self, api_id: str, period: str = "month") -> int:
        """
        Aggregates total usage for a specific default API across all users for a given period.
        This can be a heavy operation and might need to be optimized with Firestore aggregations
        or run as a periodic background task.
        """
        total_usage = 0
        current_period_str = self._get_current_month_str() if period == "month" else self._get_current_day_str()
        
        # Query all user_api_usage documents for this API
        # This can be inefficient for many users/APIs. Consider Firestore aggregation queries or
        # a separate daily/monthly aggregation document for each global API.
        # For now, a simple iteration:
        users_collection_path = f"artifacts/{self.app_id}/users"
        users_docs = await self.firestore_manager.get_all_documents_in_collection(users_collection_path)

        for user_doc in users_docs:
            user_id = user_doc.id
            usage_doc = await self.firestore_manager.get_user_data_document(
                user_id=user_id,
                collection_name="api_usage",
                document_id=api_id
            )
            if usage_doc:
                if period == "month" and usage_doc.get('last_reset_month') == current_period_str:
                    total_usage += usage_doc.get('monthly_usage', 0)
                elif period == "day" and usage_doc.get('last_reset_day') == current_period_str:
                    total_usage += usage_doc.get('daily_usage', 0)
        
        logger.debug(f"Total {period} usage for default API {api_id}: {total_usage}")
        return total_usage

    async def _adjust_tier_limits_dynamically(self, api_id: str):
        """
        Dynamically adjusts tier limits for a specific default API based on global usage.
        This method is designed to be called periodically (e.g., by a scheduled Cloud Function).
        """
        logger.info(f"Initiating dynamic limit adjustment for API: {api_id}")
        
        # Get global limits for this API (if any, from global_api_configs)
        global_api_config = await self.firestore_manager.get_global_config_document(
            collection_name="global_api_configs",
            document_id=api_id
        )
        if not global_api_config:
            logger.warning(f"No global config found for API {api_id}. Skipping dynamic adjustment.")
            return

        # Define a global limit for this API (e.g., from config or the global_api_config itself)
        # For demonstration, let's assume a hypothetical global monthly cap for *all* default APIs
        # or a specific one can be defined in global_api_config
        global_monthly_cap = global_api_config.get('global_monthly_cap', 100000) # Example cap
        global_daily_cap = global_api_config.get('global_daily_cap', 5000) # Example cap

        if global_monthly_cap <= 0 and global_daily_cap <= 0:
            logger.info(f"No global caps defined for API {api_id}. Skipping dynamic adjustment.")
            return

        current_total_monthly_usage = await self._get_total_default_api_usage(api_id, "month")
        current_total_daily_usage = await self._get_total_default_api_usage(api_id, "day")

        # Calculate usage percentages
        monthly_usage_percentage = (current_total_monthly_usage / global_monthly_cap) * 100 if global_monthly_cap > 0 else 0
        daily_usage_percentage = (current_total_daily_usage / global_daily_cap) * 100 if global_daily_cap > 0 else 0

        logger.info(f"API {api_id} - Monthly Usage: {monthly_usage_percentage:.2f}%, Daily Usage: {daily_usage_percentage:.2f}%")

        # Get current tier limits
        current_limits_doc = await self.firestore_manager.get_global_config("api_limits")
        current_tier_limits = current_limits_doc.get('limits', {}) if current_limits_doc else {}
        
        updated_tier_limits = current_tier_limits.copy()
        
        # Define adjustment rules (these would ideally be configurable in Firestore)
        adjustment_rules = {
            "monthly": [
                {"threshold": 80, "factor": 0.75}, # If 80% used, reduce remaining by 25%
                {"threshold": 90, "factor": 0.50}, # If 90% used, reduce remaining by 50%
            ],
            "daily": [
                {"threshold": 70, "factor": 0.80},
                {"threshold": 90, "factor": 0.40},
            ]
        }

        for tier_name, limits in updated_tier_limits.items():
            original_monthly = limits.get('monthly_calls', -1)
            original_daily = limits.get('daily_calls', -1)
            
            # Apply monthly adjustment
            new_monthly_adjustment = 0
            for rule in adjustment_rules["monthly"]:
                if monthly_usage_percentage >= rule["threshold"]:
                    # Calculate how much to reduce. If original was 100, and factor is 0.75, new limit is 75.
                    # So adjustment is 75 - 100 = -25
                    if original_monthly != -1:
                        new_monthly_adjustment = int(original_monthly * rule["factor"]) - original_monthly
                    break # Apply only the highest threshold rule

            # Apply daily adjustment
            new_daily_adjustment = 0
            for rule in adjustment_rules["daily"]:
                if daily_usage_percentage >= rule["threshold"]:
                    if original_daily != -1:
                        new_daily_adjustment = int(original_daily * rule["factor"]) - original_daily
                    break

            # Update the dynamic adjustments for this tier
            updated_tier_limits[tier_name]['dynamic_monthly_adjustment'] = new_monthly_adjustment
            updated_tier_limits[tier_name]['dynamic_daily_adjustment'] = new_daily_adjustment
            logger.debug(f"Tier {tier_name}: Monthly Adj: {new_monthly_adjustment}, Daily Adj: {new_daily_adjustment}")

        # Persist updated dynamic adjustments to Firestore
        await self.firestore_manager.set_global_config("api_limits", {"limits": updated_tier_limits})
        logger.info(f"Dynamic API limits updated for all tiers for API {api_id}.")

    async def _monitor_global_api_usage_task(self):
        """
        Background task to periodically monitor global API usage and trigger dynamic adjustments.
        This would typically be run by a scheduler (e.g., Cloud Scheduler + Cloud Function).
        """
        logger.info("Starting _monitor_global_api_usage_task...")
        # Get all global APIs to monitor
        global_apis = await self.get_global_api_configs()
        for api in global_apis:
            api_id = api.get('id')
            if api_id:
                try:
                    await self._adjust_tier_limits_dynamically(api_id)
                except Exception as e:
                    logger.error(f"Error during dynamic adjustment for API {api_id}: {e}", exc_info=True)
            else:
                logger.warning(f"Global API config found without an 'id': {api}")
        logger.info("Finished _monitor_global_api_usage_task.")

