# domain_tools/finance_tools/__init__.py

import logging
from typing import Optional, Dict, Any

# Import individual tool functions from the finance_tool module
from .finance_tool import (
    get_stock_price,
    get_company_overview,
    get_forex_exchange_rate,
    get_historical_stock_prices # Note: This function was cut off in the provided content, assuming it exists.
)

logger = logging.getLogger(__name__)

class FinanceTools:
    """
    A collection of finance-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any):
        """
        Initializes the FinanceTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
        """
        self.config_manager = config_manager
        self.log_event = log_event
        logger.info("FinanceTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_token.

    async def get_stock_price(self, symbol: str, user_token: str = "default") -> str:
        """
        Retrieves the current stock price for a given stock symbol.
        """
        return await get_stock_price(symbol=symbol, user_token=user_token)

    async def get_company_overview(self, symbol: str, user_token: str = "default") -> str:
        """
        Retrieves a detailed overview of a company based on its stock symbol.
        """
        return await get_company_overview(symbol=symbol, user_token=user_token)

    async def get_forex_exchange_rate(self, from_currency: str, to_currency: str, user_token: str = "default") -> str:
        """
        Retrieves the current exchange rate between two currencies.
        """
        return await get_forex_exchange_rate(from_currency=from_currency, to_currency=to_currency, user_token=user_token)

    async def get_historical_stock_prices(self, symbol: str, date: str, user_token: str = "default") -> str:
        """
        Retrieves the historical stock prices (open, high, low, close, volume) for a given symbol on a specific date.
        """
        return await get_historical_stock_prices(symbol=symbol, date=date, user_token=user_token)

