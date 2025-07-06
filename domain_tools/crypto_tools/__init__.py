# domain_tools/crypto_tools/__init__.py

import logging
from typing import Optional, Dict, Any

# Import individual tool functions from the crypto_tool module
from .crypto_tool import (
    get_crypto_price,
    get_crypto_info,
    get_historical_crypto_price
)

logger = logging.getLogger(__name__)

class CryptoTools:
    """
    A collection of crypto-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any):
        """
        Initializes the CryptoTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
        """
        self.config_manager = config_manager
        self.log_event = log_event
        logger.info("CryptoTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_token.

    async def get_crypto_price(self, crypto_id: str, vs_currencies: str = "usd", user_token: str = "default") -> str:
        """
        Retrieves the current price of a cryptocurrency in one or more specified fiat currencies or other cryptocurrencies.
        """
        return await get_crypto_price(crypto_id=crypto_id, vs_currencies=vs_currencies, user_token=user_token)

    async def get_crypto_info(self, crypto_id: str, user_token: str = "default") -> str:
        """
        Retrieves general information about a cryptocurrency, such as its description, genesis date, and market cap rank.
        """
        return await get_crypto_info(crypto_id=crypto_id, user_token=user_token)

    async def get_historical_crypto_price(self, crypto_id: str, date: str, vs_currency: str = "usd", user_token: str = "default") -> str:
        """
        Retrieves the historical price of a cryptocurrency for a specific date.
        """
        return await get_historical_crypto_price(crypto_id=crypto_id, date=date, vs_currency=vs_currency, user_token=user_token)

