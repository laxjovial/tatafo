# domain_tools/crypto_tools/__init__.py

import logging
from typing import Optional, Dict, Any, List

# Import individual tool functions from the crypto_tool module
from .crypto_tool import (
    get_crypto_price,
    get_crypto_info,
    get_historical_crypto_prices, # Corrected: Plural form to match crypto_tool.py
    crypto_search_web,
    crypto_query_uploaded_docs,
    crypto_summarize_document_by_path,
    get_crypto_id_by_symbol
)

logger = logging.getLogger(__name__)

# Ensure UserProfile is imported if needed for type hinting
try:
    from backend.models.user_models import UserProfile
except ImportError:
    # Fallback for environments where backend models might not be directly in path for this file
    class UserProfile: # Dummy class for type hinting in absence of real import
        user_id: str = "dummy"
        tier: str = "free"
        roles: List[str] = []
        # Add other attributes as needed for context
        pass

class CryptoTools:
    """
    A collection of crypto-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: Any):
        """
        Initializes the CryptoTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
            document_tools (Any): The DocumentTools instance for document querying.
        """
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools
        logger.info("CryptoTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_token.

    async def get_crypto_price(
        self,
        crypto_id: str,
        vs_currencies: str = "usd",
        user_context: Optional[UserProfile] = None,
        provider: str = "coingecko",
        user_api_keys: list = []
    ) -> str:
        """
        Retrieves the current price of a cryptocurrency.
        """
        # Call the standalone function
        return await get_crypto_price(
            crypto_id=crypto_id,
            vs_currencies=vs_currencies,
            user_context=user_context,
            provider=provider,
            user_api_keys=user_api_keys
        )
    
    async def get_crypto_info(self, crypto_id: str, user_context: Optional[UserProfile] = None) -> str:
        """
        Retrieves general information about a cryptocurrency.
        """
        return await get_crypto_info(crypto_id=crypto_id, user_context=user_context)

    # Corrected method to match the plural function name and signature
    async def get_historical_crypto_prices(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 30,
        user_context: Optional[UserProfile] = None,
        provider: str = "coingecko",
        user_api_keys: list = []
    ) -> str:
        """
        Retrieves historical prices for a cryptocurrency.
        """
        # Call the standalone plural function
        return await get_historical_crypto_prices(
            coin_id=coin_id,
            vs_currency=vs_currency,
            days=days,
            user_context=user_context,
            provider=provider,
            user_api_keys=user_api_keys
        )
    
    async def crypto_search_web(self, query: str, user_context: Optional[UserProfile] = None, max_chars: int = 2000) -> str:
        """
        Searches the web for cryptocurrency-related information.
        """
        return await crypto_search_web(query=query, user_context=user_context, max_chars=max_chars)

    async def crypto_query_uploaded_docs(
        self,
        query: str,
        user_context: Optional[UserProfile] = None,
        export: Optional[bool] = False,
        k: int = 5
    ) -> str:
        """
        Queries previously uploaded and indexed cryptocurrency documents for a user using vector similarity search.
        """
        # This now calls the DocumentTools instance
        return await self.document_tools.query_uploaded_docs(
            query_text=query,
            user_context=user_context,
            collection_name="crypto", # Specific collection for crypto documents
            export=export,
            k=k
        )

    async def crypto_summarize_document_by_path(self, file_path_str: str, user_context: Optional[UserProfile] = None) -> str:
        """
        Summarizes a document related to cryptocurrency or blockchain located at the given file path.
        """
        return await crypto_summarize_document_by_path(file_path_str=file_path_str, user_context=user_context)

    async def get_crypto_id_by_symbol(self, symbol: str, user_context: Optional[UserProfile] = None, provider: str = "coingecko", user_api_keys: list = []) -> str:
        """
        Looks up the cryptocurrency ID by its symbol.
        """
        return await get_crypto_id_by_symbol(symbol=symbol, user_context=user_context, provider=provider, user_api_keys=user_api_keys)
