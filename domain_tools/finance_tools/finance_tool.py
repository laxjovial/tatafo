# domain_tools/finance_tools/finance_tool.py

import logging
import json
from typing import Optional, Dict, Any
from langchain_core.tools import tool

# Import the new flexible API request function
from shared_tools.historical_data_tool import make_api_request

# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability

# Import UserProfile for type hinting
from backend.models.user_models import UserProfile

logger = logging.getLogger(__name__)

class FinanceTools:
    """
    A collection of tools for finance-related operations, including stock prices,
    historical data, company overviews, and forex exchange rates.
    It integrates with external APIs and provides fallback mechanisms.
    """
    def __init__(self, config_manager, firestore_manager, log_event, document_tools):
        self.config_manager = config_manager
        self.firestore_manager = firestore_manager
        self.log_event = log_event
        self.document_tools = document_tools

    @tool
    async def finance_get_stock_price(self, symbol: str, user_context: UserProfile, provider: str = "alphavantage", user_api_keys: list = []) -> str:
        """
        Retrieves the current stock price for a given stock symbol.
        """
        logger.info(f"Tool: finance_get_stock_price called for symbol: '{symbol}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'finance_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to finance tools is not enabled for your current tier."
        
        params = {"symbol": symbol}
        api_data = make_api_request(
            provider_name=provider,
            function_name="get_stock_price",
            params=params,
            user_api_keys=user_api_keys,
        )

        if api_data:
            try:
                price = api_data.get("price")
                currency = api_data.get("currency", "USD")
                timestamp = api_data.get("timestamp")
                if price:
                    return f"The current price of {symbol.upper()} is {price} {currency} (as of {timestamp})."
                else:
                    return f"Could not retrieve complete live stock price for {symbol.upper()}."
            except (ValueError, TypeError) as e:
                return f"Error parsing live data for {symbol.upper()}."
        else:
            return f"Could not retrieve live stock price for {symbol.upper()}."

    @tool
    async def finance_get_historical_stock_prices(self, symbol: str, start_date: str, end_date: str, user_context: UserProfile, provider: str = "alphavantage", user_api_keys: list = []) -> str:

        """
        Retrievels historical daily stock prices for a given stock symbol within a date range.
        """
        logger.info(f"Tool: finance_get_historical_stock_prices called for symbol: '{symbol}' from {start_date} to {end_date} by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'historical_data_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to historical data is not enabled for your current tier."

        params = {"symbol": symbol, "start_date": start_date, "end_date": end_date}
        api_data = make_api_request(
            provider_name=provider,
            function_name="get_historical_stock_prices",
            params=params,
            user_api_keys=user_api_keys,
        )

        if api_data:
            try:
                # Assuming api_data is a list of dictionaries
                sorted_prices = sorted(api_data, key=lambda x: x.get('date', ''))
                if sorted_prices:
                    response_str = f"Historical prices for {symbol.upper()}:\n"
                    for data in sorted_prices:
                        response_str += (
                            f"  Date: {data.get('date', 'N/A')}\n"
                            f"    Open: {data.get('open', 'N/A')}\n"
                            f"    High: {data.get('high', 'N/A')}\n"
                            f"    Low: {data.get('low', 'N/A')}\n"
                            f"    Close: {data.get('close', 'N/A')}\n"
                            f"    Volume: {data.get('volume', 'N/A')}\n"
                        )
                    return response_str
                else:
                    return f"No historical prices found for {symbol.upper()} within the specified date range. Please try again or check the symbol/dates."
            except (json.JSONDecodeError, TypeError):
                return "Error: Could not parse historical data from the shared tool."
        else:
            return f"Could not retrieve historical stock prices for {symbol.upper()}."


    @tool
    async def finance_get_company_overview(self, symbol: str, user_context: UserProfile, provider: str = "alphavantage", user_api_keys: list = []) -> str:
        """
        Retrieves a company's overview, including its description, sector, and market capitalization.
        """
        logger.info(f"Tool: finance_get_company_overview called for symbol: '{symbol}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'finance_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to finance tools is not enabled for your current tier."
        
        params = {"symbol": symbol}
        api_data = make_api_request(
            provider_name=provider,
            function_name="get_company_overview",
            params=params,
            user_api_keys=user_api_keys,
        )

        if api_data:
            return str(api_data)
        else:
            return f"Could not retrieve company overview for {symbol.upper()}."

    @tool
    async def finance_get_forex_exchange_rate(self, from_currency: str, to_currency: str, user_context: UserProfile, provider: str = "alphavantage", user_api_keys: list = []) -> str:
        """
        Retrieves the current exchange rate between two currencies.
        """
        logger.info(f"Tool: finance_get_forex_exchange_rate called for {from_currency} to {to_currency} by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'finance_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to finance tools is not enabled for your current tier."
        
        params = {"from_currency": from_currency, "to_currency": to_currency}
        api_data = make_api_request(
            provider_name=provider,
            function_name="get_forex_exchange_rate",
            params=params,
            user_api_keys=user_api_keys,
        )

        if api_data:
            return str(api_data)
        else:
            return f"Could not retrieve exchange rate for {from_currency.upper()} to {to_currency.upper()}."
