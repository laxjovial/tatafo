# domain_tools/crypto_tools/__init__.py

import logging
from typing import Optional, Dict, Any

# Import individual tool functions from the crypto_tool module
from .crypto_tool import (
    get_crypto_price,
    get_crypto_info,
    get_historical_crypto_price,
    crypto_search_web, # Added
    crypto_query_uploaded_docs, # Added
    crypto_summarize_document_by_path # Added
)

logger = logging.getLogger(__name__)

class CryptoTools:
    """
    A collection of crypto-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: Any): # Added document_tools
        """
        Initializes the CryptoTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
            document_tools (Any): The DocumentTools instance for document querying. # Added
        """
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools # Stored
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

    async def crypto_search_web(self, query: str, user_token: str = "default", max_chars: int = 2000) -> str:
        """
        Searches the web for cryptocurrency-related information.
        """
        return await crypto_search_web(query=query, user_token=user_token, max_chars=max_chars) # Call the function from crypto_tool.py

    async def crypto_query_uploaded_docs(self, query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed cryptocurrency documents for a user using vector similarity search.
        """
        # This now calls the DocumentTools instance
        return await self.document_tools.query_uploaded_docs(
            query_text=query,
            user_token=user_token,
            collection_name="crypto", # Specific collection for crypto documents
            export=export,
            k=k
        )

    async def crypto_summarize_document_by_path(self, file_path_str: str) -> str:
        """
        Summarizes a document related to cryptocurrency or blockchain located at the given file path.
        """
        return await crypto_summarize_document_by_path(file_path_str=file_path_str) # Call the function from crypto_tool.py

