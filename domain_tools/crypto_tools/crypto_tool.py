# domain_tools/crypto_tools/crypto_tool.py

import logging
from typing import Optional, Dict, Any
from langchain_core.tools import tool

# Import the new flexible API request function
from shared_tools.historical_data_tool import make_api_request

# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability

# Import UserProfile for type hinting
from backend.models.user_models import UserProfile

logger = logging.getLogger(__name__)

class CryptoTools:
    """
    A collection of tools for cryptocurrency-related operations, including prices,
    historical data, and general information.
    It integrates with external APIs and provides fallback mechanisms.
    """
    def __init__(self, config_manager, firestore_manager, log_event, document_tools):
        self.config_manager = config_manager
        self.firestore_manager = firestore_manager
        self.log_event = log_event
        self.document_tools = document_tools

    @tool
    async def crypto_get_crypto_price(self, crypto_id: str, vs_currencies: str = "usd", user_context: UserProfile = None, provider: str = "coingecko", user_api_keys: list = []) -> str:
        """
        Retrieves the current price of a cryptocurrency.
        """
        if user_context is None:
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: crypto_get_crypto_price called for crypto_id: '{crypto_id}', vs_currencies: '{vs_currencies}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'crypto_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to crypto tools is not enabled for your current tier."
        
        params = {"ids": crypto_id.lower(), "vs_currencies": vs_currencies.lower()}
        api_data = make_api_request(
            provider_name=provider,
            function_name="get_crypto_price",
            params=params,
            user_api_keys=user_api_keys,
        )

        if api_data:
            return str(api_data)
        else:
            return f"Could not retrieve live cryptocurrency price for {crypto_id.capitalize()}."

    @tool
    async def crypto_get_historical_crypto_price(self, crypto_id: str, date: str, vs_currency: str = "usd", user_context: UserProfile = None, provider: str = "coingecko", user_api_keys: list = []) -> str:
        """
        Retrieves the historical price of a cryptocurrency for a specific date.
        """
        if user_context is None:
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: crypto_get_historical_crypto_price called for crypto_id: '{crypto_id}', date: '{date}', vs_currency: '{vs_currency}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'historical_data_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to historical data is not enabled for your current tier."

        params = {"id": crypto_id.lower(), "date": date, "localization": "false"}
        api_data = make_api_request(
            provider_name=provider,
            function_name="get_historical_crypto_price",
            params=params,
            user_api_keys=user_api_keys,
        )

        if api_data:
            return str(api_data)
        else:
            return f"Could not retrieve historical price for {crypto_id.capitalize()} on {date}."
