# domain_tools/finance_tools/__init__.py

import logging
from typing import Optional, Dict, Any

# Import individual tool functions from the finance_tool module
from .finance_tool import (
    get_stock_price,
    get_historical_stock_prices,
    get_company_overview,
    get_forex_exchange_rate, # Added this line
    finance_search_web,
    finance_query_uploaded_docs,
    finance_summarize_document_by_path
)

logger = logging.getLogger(__name__)

class FinanceTools:
    """
    A collection of finance-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, firestore_manager: Any, config_manager: Any, log_event: Any, document_tools: Any):
        """
        Initializes the FinanceTools with necessary dependencies.

        Args:
            firestore_manager (Any): The FirestoreManager instance.
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
            document_tools (Any): The DocumentTools instance for document querying.
        """
        self.firestore_manager = firestore_manager
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools # Store DocumentTools instance
        logger.info("FinanceTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_token.

    async def get_stock_price(self, symbol: str, user_token: str = "default") -> str:
        """
        Retrieves the current stock price for a given stock symbol.
        """
        return await get_stock_price(symbol=symbol, user_token=user_token)

    async def get_historical_stock_prices(self, symbol: str, user_token: str = "default") -> str:
        """
        Retrieves historical daily stock prices for a given stock symbol.
        """
        return await get_historical_stock_prices(symbol=symbol, user_token=user_token)

    async def get_company_overview(self, symbol: str, user_token: str = "default") -> str:
        """
        Retrieves a company's overview, including its description, sector, and market capitalization.
        """
        return await get_company_overview(symbol=symbol, user_token=user_token)

    async def get_forex_exchange_rate(self, from_currency: str, to_currency: str, user_token: str = "default") -> str:
        """
        Retrieves the current exchange rate between two currencies.
        """
        return await get_forex_exchange_rate(from_currency=from_currency, to_currency=to_currency, user_token=user_token)

    async def finance_search_web(self, query: str, user_token: str = "default", max_chars: int = 2000) -> str:
        """
        Searches the web for general finance information using a smart search fallback mechanism.
        """
        return await finance_search_web(query=query, user_token=user_token, max_chars=max_chars)

    async def finance_query_uploaded_docs(self, query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed finance documents for a user.
        """
        # This now calls the DocumentTools instance
        return await self.document_tools.query_uploaded_docs(
            query_text=query,
            user_token=user_token,
            collection_name="finance", # Specific collection for finance documents
            export=export,
            k=k
        )

    async def finance_summarize_document_by_path(self, file_path_str: str) -> str:
        """
        Summarizes a document related to finance (e.g., financial reports, market analysis) located at the given file path.
        """
        return await finance_summarize_document_by_path(file_path_str=file_path_str)
