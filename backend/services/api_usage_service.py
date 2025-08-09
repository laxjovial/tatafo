# backend/services/api_usage_service.py

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
import uuid
from fastapi import HTTPException, status, Depends # Ensure 'Depends' is imported for injection
from firebase_admin import firestore  # Or from google.cloud import firestore

# Import FirestoreManager and ConfigManager
from database.firestore_manager import FirestoreManager
from config.config_manager import config_manager
from utils.user_manager import UserManager

# Import Pydantic models for global API config
from backend.models.admin_models import GlobalApiConfigCreate, GlobalApiConfigUpdate, ApiCallLimitUpdate
from backend.models.user_models import UserProfile

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ApiUsageService:
    """
    Manages API call limits, usage tracking, and dynamic distribution for default APIs.
    Also handles user-defined API configurations and overrides.
    """
    # Inject FirestoreManager here
    def __init__(self, firestore_manager: FirestoreManager = Depends(FirestoreManager)):
        self.firestore_manager = firestore_manager
        # You might also want to inject config_manager if it's a dependency,
        # otherwise access it globally as before if that's the design.
        self.config_manager = config_manager # Keep this if it's accessed globally

        # Additional initialization for API usage tracking
        self.api_limits = {} # Will be loaded from Firestore
        self.user_api_call_counts = {} # In-memory cache for daily/monthly counts

        logger.info("ApiUsageService initialized.")

    async def _load_api_limits(self):
        """Loads API limits configuration from Firestore."""
        global_api_limits = await self.firestore_manager.get_global_config("api_limits")
        if global_api_limits and "limits" in global_api_limits:
            self.api_limits = global_api_limits["limits"]
            logger.info("API limits loaded from Firestore.")
        else:
            logger.warning("No API limits found in Firestore, using default empty configuration.")
            # Define sensible default limits if none are found in Firestore
            self.api_limits = {
                "free": {"monthly_calls": 100, "daily_calls": 10, "dynamic_monthly_adjustment": 0, "dynamic_daily_adjustment": 0},
                "pro": {"monthly_calls": 1000, "daily_calls": 100, "dynamic_monthly_adjustment": 0, "dynamic_daily_adjustment": 0},
                "premium": {"monthly_calls": 5000, "daily_calls": 500, "dynamic_monthly_adjustment": 0, "dynamic_daily_adjustment": 0},
                "admin": {"monthly_calls": float('inf'), "daily_calls": float('inf'), "dynamic_monthly_adjustment": 0, "dynamic_daily_adjustment": 0}
            }
            # Optionally, save these defaults to Firestore
            await self.firestore_manager.set_global_config("api_limits", {"limits": self.api_limits})


    def get_user_current_call_count_key(self, user_id: str, api_id: str = "llm_calls") -> str:
        """Generates a key for tracking user's API call count for the current day/month."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        current_month = datetime.now(timezone.utc).strftime("%Y-%m")
        return f"{user_id}:{api_id}:{today}:{current_month}"

    async def check_and_track_usage(self, user_id: str, user_tier: str, api_id: str = "llm_calls") -> bool:
        """
        Checks if the user has exceeded their API limits for the given tier and tracks usage.
        Returns True if allowed, False otherwise.
        """
        await self._load_api_limits() # Ensure limits are loaded

        tier_limits = self.api_limits.get(user_tier, {})
        monthly_limit = tier_limits.get("monthly_calls", 0) + tier_limits.get("dynamic_monthly_adjustment", 0)
        daily_limit = tier_limits.get("daily_calls", 0) + tier_limits.get("dynamic_daily_adjustment", 0)

        if monthly_limit == float('inf') and daily_limit == float('inf'):
            return True # Admin or unlimited tier

        user_call_key = self.get_user_current_call_count_key(user_id, api_id)
        current_daily_count = self.user_api_call_counts.get(user_call_key, 0)
        current_monthly_count = self.user_api_call_counts.get(user_call_key, 0) # For simplicity, using same key for now

        # Fetch actual counts from Firestore for persistence
        user_usage_doc = await self.firestore_manager.get_doc(
            f"artifacts/{config_manager.get_app_id()}/user_usage",
            user_id
        )
        current_daily_count_db = user_usage_doc.get("daily_counts", {}).get(api_id, {}).get(datetime.now(timezone.utc).strftime("%Y-%m-%d"), 0)
        current_monthly_count_db = user_usage_doc.get("monthly_counts", {}).get(api_id, {}).get(datetime.now(timezone.utc).strftime("%Y-%m"), 0)

        if current_daily_count_db >= daily_limit and daily_limit != float('inf'):
            logger.warning(f"User {user_id} ({user_tier}) exceeded daily limit for {api_id}.")
            return False
        if current_monthly_count_db >= monthly_limit and monthly_limit != float('inf'):
            logger.warning(f"User {user_id} ({user_tier}) exceeded monthly limit for {api_id}.")
            return False

        # Increment usage in Firestore (atomic update recommended for production)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        month_str = datetime.now(timezone.utc).strftime("%Y-%m")
        
        await self.firestore_manager.update_doc(
            f"artifacts/{config_manager.get_app_id()}/user_usage",
            user_id,
            {
                f"daily_counts.{api_id}.{today_str}": firestore.Increment(1),
                f"monthly_counts.{api_id}.{month_str}": firestore.Increment(1),
                "last_activity": datetime.now(timezone.utc)
            },
            merge=True # Use merge to create fields if they don't exist
        )
        logger.info(f"User {user_id} ({user_tier}) usage tracked for {api_id}.")

        # Update in-memory cache
        self.user_api_call_counts[user_call_key] = current_daily_count + 1 # Simple increment for cache

        return True

    async def get_global_api_configs(self) -> List[Dict[str, Any]]:
        """Retrieves all global API configurations."""
        try:
            configs = await self.firestore_manager.get_collection("global_configs/global_api_configs")
            return configs if configs else []
        except Exception as e:
            logger.error(f"Error getting global API configs: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve global API configurations.")

    async def add_global_api_config(self, api_config: GlobalApiConfigCreate) -> Dict[str, Any]:
        """Adds a new global API configuration."""
        api_dict = api_config.model_dump()
        api_id = api_dict.get("id", str(uuid.uuid4()))
        api_dict["id"] = api_id
        try:
            await self.firestore_manager.add_doc(f"global_configs/global_api_configs", api_dict, doc_id=api_id)
            logger.info(f"Added global API config: {api_id}")
            return api_dict
        except Exception as e:
            logger.error(f"Error adding global API config: {e}")
            raise HTTPException(status_code=500, detail="Failed to add global API configuration.")

    async def update_global_api_config(self, api_id: str, api_config: GlobalApiConfigUpdate) -> Dict[str, Any]:
        """Updates an existing global API configuration."""
        update_data = api_config.model_dump(exclude_unset=True)
        try:
            await self.firestore_manager.update_doc(f"global_configs/global_api_configs", api_id, update_data)
            updated_config = await self.firestore_manager.get_doc(f"global_configs/global_api_configs", api_id)
            logger.info(f"Updated global API config: {api_id}")
            return updated_config
        except Exception as e:
            logger.error(f"Error updating global API config {api_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to update global API configuration.")

    async def delete_global_api_config(self, api_id: str):
        """Deletes a global API configuration."""
        try:
            success = await self.firestore_manager.delete_doc(f"global_configs/global_api_configs", api_id)
            if not success:
                raise HTTPException(status_code=404, detail="Global API config not found.")
            logger.info(f"Deleted global API config: {api_id}")
            return {"message": "Global API config deleted successfully."}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting global API config {api_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete global API configuration.")

    async def update_api_call_limits(self, update_data: ApiCallLimitUpdate) -> Dict[str, Any]:
        """Updates the global API call limits for tiers."""
        try:
            # Fetch existing limits
            existing_limits_doc = await self.firestore_manager.get_global_config("api_limits")
            existing_limits = existing_limits_doc.get("limits", {}) if existing_limits_doc else {}

            # Apply updates
            for tier_update in update_data.tiers:
                tier_name = tier_update.tier_name
                if tier_name not in existing_limits:
                    existing_limits[tier_name] = {"monthly_calls": 0, "daily_calls": 0, "dynamic_monthly_adjustment": 0, "dynamic_daily_adjustment": 0}

                if tier_update.monthly_calls is not None:
                    existing_limits[tier_name]["monthly_calls"] = tier_update.monthly_calls
                if tier_update.daily_calls is not None:
                    existing_limits[tier_name]["daily_calls"] = tier_update.daily_calls

            await self.firestore_manager.set_global_config("api_limits", {"limits": existing_limits})
            self.api_limits = existing_limits # Update in-memory cache
            logger.info("API call limits updated successfully.")
            return {"message": "API call limits updated successfully", "new_limits": existing_limits}
        except Exception as e:
            logger.error(f"Error updating API call limits: {e}")
            raise HTTPException(status_code=500, detail="Failed to update API call limits.")


    async def _adjust_tier_limits_dynamically(self, api_id: str):
        """
        Adjusts API tier limits dynamically based on global API usage data.
        This is a placeholder for a more sophisticated dynamic adjustment logic.
        """
        logger.info(f"Initiating dynamic limit adjustment for API: {api_id}")

        global_api_config = await self.firestore_manager.get_doc("global_configs/global_api_configs", api_id)
        if not global_api_config or not global_api_config.get("dynamic_adjustment_enabled", False):
            logger.info(f"Dynamic adjustment not enabled for API {api_id} or config not found.")
            return

        current_month_str = datetime.now(timezone.utc).strftime("%Y-%m")
        # Fetch aggregated usage for the current month for this API
        # This part assumes you have aggregated usage data stored somewhere in Firestore,
        # e.g., in 'global_usage/api_id/monthly_counts/{year-month}'
        # For demonstration, let's assume we retrieve a total count.
        # In a real system, this would involve querying aggregate data.
        total_monthly_usage_doc = await self.firestore_manager.get_doc(
            f"global_usage/{api_id}/monthly_counts",
            current_month_str
        )
        total_monthly_calls = total_monthly_usage_doc.get("total_calls", 0) if total_monthly_usage_doc else 0

        logger.debug(f"Total monthly usage for {api_id}: {total_monthly_calls}")

        # Get current tier limits to adjust
        current_api_limits_doc = await self.firestore_manager.get_global_config("api_limits")
        updated_tier_limits = current_api_limits_doc.get("limits", {}) if current_api_limits_doc else {}

        # Simple example: if total usage is low, increase limits; if high, decrease.
        # This logic should be replaced with a proper algorithm based on your business rules.
        for tier_name, limits in updated_tier_limits.items():
            # For this example, let's just make a dummy adjustment
            # In production, this would be a more complex calculation
            new_monthly_adjustment = 0
            new_daily_adjustment = 0

            # Example: give a bonus if total usage is under a certain threshold (e.g., 50% of sum of all tier's base limits)
            # This is highly simplified
            base_monthly_sum = sum(v.get("monthly_calls", 0) for k, v in updated_tier_limits.items() if k != "admin")
            if base_monthly_sum > 0 and total_monthly_calls < (base_monthly_sum * 0.5):
                new_monthly_adjustment = int(limits.get("monthly_calls", 0) * 0.1) # 10% bonus
                new_daily_adjustment = int(limits.get("daily_calls", 0) * 0.1)
            elif total_monthly_calls > (base_monthly_sum * 0.9):
                new_monthly_adjustment = -int(limits.get("monthly_calls", 0) * 0.05) # 5% penalty
                new_daily_adjustment = -int(limits.get("daily_calls", 0) * 0.05)

            # Ensure adjustments don't go below zero or a predefined minimum
            new_monthly_adjustment = max(new_monthly_adjustment, -limits.get("monthly_calls", 0))
            new_daily_adjustment = max(new_daily_adjustment, -limits.get("daily_calls", 0))


            # Update dynamic adjustments for this tier
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
