# backend/services/user_service.py

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

# Import the FirestoreManager and UserManager for actual data operations
from database.firestore_manager import firestore_manager
from utils.user_manager import get_current_user, get_user_tier_capability, _TIER_HIERARCHY, _RBAC_CAPABILITIES

logger = logging.getLogger(__name__)

class UserService:
    """
    Manages user data operations, interacting with Firestore via FirestoreManager.
    This service acts as an intermediary between API routes and the database logic.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UserService, cls).__new__(cls)
            # Ensure FirestoreManager is initialized when UserService is created
            # This is implicitly handled by importing firestore_manager, but explicit
            # access ensures its singleton initialization
            _ = firestore_manager
        return cls._instance

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a user's profile from Firestore.
        Includes subscription details.
        """
        logger.info(f"Fetching user profile for {user_id} from Firestore.")
        user_profile = await firestore_manager.get_user_data(user_id) # Use Firestore to get user data

        if user_profile:
            # For demonstration, let's ensure subscription dates are calculated if missing or stale
            # In a real app, these would ideally be managed by a subscription service
            if "subscription_start_date" not in user_profile or "subscription_end_date" not in user_profile:
                # Default subscription info for users without explicit data
                user_profile["subscription_start_date"] = "N/A"
                user_profile["subscription_end_date"] = "N/A"
                user_profile["days_left"] = "N/A"
                user_profile["next_subscription_date"] = "N/A"
            else:
                try:
                    start_date = datetime.strptime(user_profile["subscription_start_date"], "%Y-%m-%d").date()
                    end_date = datetime.strptime(user_profile["subscription_end_date"], "%Y-%m-%d").date()
                    today = datetime.now().date()
                    
                    if end_date >= today:
                        user_profile["days_left"] = (end_date - today).days
                        user_profile["next_subscription_date"] = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
                    else:
                        user_profile["days_left"] = 0
                        user_profile["next_subscription_date"] = "Subscription Expired"
                except ValueError:
                    logger.warning(f"Invalid date format for user {user_id}. Dates not updated.")
                    user_profile["days_left"] = "Error"
                    user_profile["next_subscription_date"] = "Error"

            logger.info(f"User profile for {user_id} retrieved from Firestore.")
            return user_profile
        
        logger.warning(f"User profile for {user_id} not found in Firestore.")
        return None

    async def update_user_profile(self, user_id: str, data: Dict[str, Any]) -> None:
        """Updates a user's profile in Firestore."""
        logger.info(f"Updating user profile for {user_id} in Firestore.")
        await firestore_manager.update_user_data(user_id, data)
        logger.info(f"User profile for {user_id} updated in Firestore.")

    async def get_all_user_profiles(self) -> List[Dict[str, Any]]:
        """Retrieves all user profiles from Firestore."""
        logger.info("Fetching all user profiles from Firestore.")
        profiles = await firestore_manager.get_all_user_profiles()
        logger.info(f"Retrieved {len(profiles)} user profiles from Firestore.")
        return profiles

    # You can add more user-related methods here (e.g., create_user, delete_user)
    # which would then call methods in firestore_manager and/or Firebase Auth Admin SDK.

# Instantiate the UserService as a singleton
user_service = UserService()

