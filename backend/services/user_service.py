# backend/services/user_service.py

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

# Import config_manager for configuration settings
from config.config_manager import config_manager

logger = logging.getLogger(__name__)

# Mock user data for demonstration purposes.
# In a real application, this would be fetched from a database (e.g., Firestore).
_mock_users = {
    "default_user_token": {
        "user_id": "default_user_token",
        "username": "GuestUser",
        "email": "guest@example.com",
        "tier": "free",
        "roles": ["user"],
        "subscription_start_date": None,
        "subscription_end_date": None,
        "days_left": 0,
        "next_subscription_date": None
    },
    "mock_free_token": {
        "user_id": "mock_free_token",
        "username": "FreeUser",
        "email": "free@example.com",
        "tier": "free",
        "roles": ["user"],
        "subscription_start_date": None,
        "subscription_end_date": None,
        "days_left": 0,
        "next_subscription_date": None
    },
    "mock_basic_token": { # Added 'basic' tier for more granularity
        "user_id": "mock_basic_token",
        "username": "BasicUser",
        "email": "basic@example.com",
        "tier": "basic",
        "roles": ["user"],
        "subscription_start_date": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
        "subscription_end_date": (datetime.now() + timedelta(days=20)).strftime("%Y-%m-%d"),
        "days_left": 20,
        "next_subscription_date": (datetime.now() + timedelta(days=21)).strftime("%Y-%m-%d")
    },
    "mock_pro_token": {
        "user_id": "mock_pro_token",
        "username": "ProUser",
        "email": "pro@example.com",
        "tier": "pro",
        "roles": ["user"],
        "subscription_start_date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
        "subscription_end_date": (datetime.now() + timedelta(days=25)).strftime("%Y-%m-%d"),
        "days_left": 25,
        "next_subscription_date": (datetime.now() + timedelta(days=26)).strftime("%Y-%m-%d")
    },
    "mock_premium_token": {
        "user_id": "mock_premium_token",
        "username": "PremiumUser",
        "email": "premium@example.com",
        "tier": "premium",
        "roles": ["user"],
        "subscription_start_date": (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d"),
        "subscription_end_date": (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d"),
        "days_left": 45,
        "next_subscription_date": (datetime.now() + timedelta(days=46)).strftime("%Y-%m-%d")
    },
    "mock_admin_token": {
        "user_id": "mock_admin_token",
        "username": "AdminUser",
        "email": "admin@example.com",
        "tier": "admin",
        "roles": ["user", "admin"],
        "subscription_start_date": "N/A", # Admins typically don't have subscriptions
        "subscription_end_date": "N/A",
        "days_left": "N/A",
        "next_subscription_date": "N/A"
    },
    "mock_dev_token": { # Added 'dev' role for data analysis capability
        "user_id": "mock_dev_token",
        "username": "DeveloperUser",
        "email": "dev@example.com",
        "tier": "pro", # Devs might be pro tier or custom
        "roles": ["user", "dev"],
        "subscription_start_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        "subscription_end_date": (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d"),
        "days_left": 60,
        "next_subscription_date": (datetime.now() + timedelta(days=61)).strftime("%Y-%m-%d")
    }
}

# Tier hierarchy for RBAC checks (moved here for centralized access)
_TIER_HIERARCHY = {
    "free": 0,
    "basic": 1,
    "user": 1, # 'user' role is often equivalent to 'basic' tier for access level
    "pro": 2,
    "premium": 3,
    "admin": 99 # Highest level
}

# RBAC Capabilities (will be dynamically loaded from Firestore, but mocked here for initial setup)
_RBAC_CAPABILITIES = {
    'capabilities': {
        'general_chat_enabled': {'default': True, 'roles': {}},
        'llm_temperature_control_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'llm_default_temperature': {'default': 0.7, 'roles': {'free': 0.5, 'user': 0.6, 'pro': 0.7, 'premium': 0.8, 'admin': 0.9}},
        'llm_max_temperature': {'default': 1.0, 'roles': {'pro': 0.8, 'premium': 0.9, 'admin': 1.0}},
        'llm_model_selection_enabled': {'default': False, 'roles': {'premium': True, 'admin': True}},
        'web_search_enabled': {'default': False, 'roles': {'user': True, 'pro': True, 'premium': True, 'admin': True}},
        'data_analysis_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True, 'dev': True}},
        'summarization_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'summarization_max_input_chars': {'default': 5000, 'roles': {'pro': 10000, 'premium': 20000, 'admin': 50000}},
        'chart_generation_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'sentiment_analysis_enabled': {'default': False, 'roles': {'user': True, 'pro': True, 'premium': True, 'admin': True}},
        'document_upload_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'document_query_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'document_query_max_results_k': {'default': 3, 'roles': {'pro': 5, 'premium': 10, 'admin': 20}},
        'chart_export_enabled': {'default': False, 'roles': {'premium': True, 'admin': True}},
        'finance_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'historical_data_access': {'default': False, 'roles': {'premium': True, 'admin': True}},
        'crypto_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'news_tool_access': {'default': False, 'roles': {'user': True, 'pro': True, 'premium': True, 'admin': True}},
        'medical_tool_access': {'default': False, 'roles': {'premium': True, 'admin': True}},
        'legal_tool_access': {'default': False, 'roles': {'premium': True, 'admin': True}},
        'entertainment_tool_access': {'default': False, 'roles': {'user': True, 'pro': True, 'premium': True, 'admin': True}},
        'weather_tool_access': {'default': False, 'roles': {'user': True, 'pro': True, 'premium': True, 'admin': True}},
        'travel_tool_access': {'default': False, 'roles': {'premium': True, 'admin': True}},
        'sports_tool_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
        'analytics_access': {'default': False, 'roles': {'admin': True, 'premium': True, 'pro': True}},
        'analytics_charts_enabled': {'default': False, 'roles': {'admin': True, 'premium': True, 'pro': False}},
        'analytics_user_specific_access': {'default': False, 'roles': {'user': True, 'pro': True, 'premium': True, 'admin': True}},
    }
}


class UserManager:
    """
    Manages user-related operations, including authentication state and RBAC.
    This is a mock implementation for demonstration.
    In a real application, this would interact with Firebase Authentication and Firestore.
    """
    _current_user_data: Dict[str, Any] = {} # Stores the currently "logged in" user's data

    def get_user_by_token(self, user_token: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves user data by token.
        In a real app, this would query Firestore.
        """
        user_data = _mock_users.get(user_token)
        if user_data:
            logger.info(f"Retrieved mock user data for token: {user_token}")
            # Ensure subscription dates are up-to-date for mock users
            if user_data.get("subscription_start_date") and user_data.get("subscription_end_date"):
                try:
                    start_date = datetime.strptime(user_data["subscription_start_date"], "%Y-%m-%d").date()
                    end_date = datetime.strptime(user_data["subscription_end_date"], "%Y-%m-%d").date()
                    today = datetime.now().date()
                    
                    if end_date >= today:
                        user_data["days_left"] = (end_date - today).days
                        user_data["next_subscription_date"] = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
                    else:
                        user_data["days_left"] = 0
                        user_data["next_subscription_date"] = "Subscription Expired"
                except ValueError:
                    logger.warning(f"Invalid date format for user {user_token}. Dates not updated.")
            return user_data
        logger.warning(f"User not found for token: {user_token}")
        return None

    def get_current_user(self) -> Dict[str, Any]:
        """
        Returns the data of the currently "logged in" user from session state.
        In a real app, this would check Firebase Auth state.
        """
        return self._current_user_data

    def set_current_user(self, user_data: Dict[str, Any]):
        """
        Sets the currently "logged in" user data in session state.
        """
        self._current_user_data = user_data
        logger.info(f"Current user set to: {user_data.get('username', 'N/A')}")

    def clear_current_user(self):
        """
        Clears the currently "logged in" user data.
        """
        self._current_user_data = {}
        logger.info("Current user cleared.")

    def get_user_tier_capability(self, user_token: Optional[str], capability_key: str, default_value: Any = None) -> Any:
        """
        Checks if a user (identified by token) has a specific capability based on their tier and roles.
        If user_token is None or invalid, it defaults to 'free' tier capabilities.
        """
        user_info = self.get_user_by_token(user_token)
        user_tier = user_info.get('tier', 'free') if user_info else 'free'
        user_roles = user_info.get('roles', []) if user_info else []

        # Admins always have access to boolean capabilities
        if "admin" in user_roles:
            if isinstance(default_value, bool): return True
            if isinstance(default_value, (int, float)): return float('inf') # For numerical caps, admin gets max
            return default_value # For other types, return default

        capability_config = _RBAC_CAPABILITIES.get('capabilities', {}).get(capability_key)
        if not capability_config:
            logger.warning(f"Capability '{capability_key}' not found in RBAC configuration. Returning default value: {default_value}")
            return default_value

        # Check role-specific overrides first
        for role in user_roles:
            if role in capability_config.get('roles', {}):
                return capability_config['roles'][role]
        
        # Fallback to tier-specific default if no role override
        # This part is more complex for numerical values where tier might define a value directly
        if capability_key in ['llm_default_temperature', 'llm_max_temperature', 'summarization_max_input_chars', 'document_query_max_results_k']:
            # For numerical capabilities, get the value specific to the user's tier
            tier_value = capability_config.get('roles', {}).get(user_tier) # Check if tier itself is a 'role' for this cap
            if tier_value is not None:
                return tier_value
            # If not explicitly defined for the tier, fall back to the general default for the capability
            return capability_config.get('default', default_value)
        
        # For boolean capabilities, check general default if no role override applies
        return capability_config.get('default', default_value)

# Instantiate the UserManager as a singleton
user_manager = UserManager()

# Expose functions and _TIER_HIERARCHY directly for easier import in other modules
get_current_user = user_manager.get_current_user
set_current_user = user_manager.set_current_user
clear_current_user = user_manager.clear_current_user
get_user_tier_capability = user_manager.get_user_tier_capability
_TIER_HIERARCHY = _TIER_HIERARCHY # Expose the hierarchy
_RBAC_CAPABILITIES = _RBAC_CAPABILITIES # Expose capabilities for direct access if needed (e.g., in main_app)

