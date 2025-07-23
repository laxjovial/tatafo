# domain_tools/crypto_tools/crypto_tool.py

import logging
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
from datetime import datetime, timezone, timedelta
import requests

# Import the new flexible API request function
from shared_tools.historical_data_tool import make_api_request

# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability

# Import UserProfile for type hinting
from backend.models.user_models import UserProfile

# Import config_manager and analytics_tracker for direct use in standalone tool functions
from config.config_manager import config_manager
from utils import analytics_tracker

# Import DocumentTools and scrape_web for wrapping
from domain_tools.document_tools.document_tool import DocumentTools
from shared_tools.scrapper_tool import scrape_web # Corrected import here


logger = logging.getLogger(__name__)

# --- Standalone Tool Functions ---

@tool
async def get_crypto_price(
    crypto_id: str,
    vs_currencies: str = "usd",
    user_context: Optional[UserProfile] = None,
    provider: str = "coingecko",
    user_api_keys: list = []
) -> str:
    """
    Retrieves the current price of a cryptocurrency.

    This tool fetches real-time cryptocurrency prices from specified providers.
    It supports multiple `vs_currencies` (e.g., "usd", "eur", "jpy") as a comma-separated string.
    The `provider` argument allows selection between supported API providers
    (e.g., "coingecko", "alphavantage").
    Requires 'crypto_tool_access' capability.

    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
        vs_currencies (str, optional): A comma-separated string of currencies
                                       to compare against (e.g., "usd,eur"). Defaults to "usd".
        user_context (UserProfile, optional): The user's profile for RBAC checks. Defaults to None.
        provider (str, optional): The API provider to use. Defaults to "coingecko".
        user_api_keys (list, optional): List of user-provided API keys (e.g., from Streamlit secrets).

    Returns:
        str: A JSON string containing the cryptocurrency price information,
             or an error message.
    """
    if user_context is None:
        user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

    logger.info(f"Tool: get_crypto_price called for crypto_id: '{crypto_id}', vs_currencies: '{vs_currencies}', provider: '{provider}', user: '{user_context.user_id}'")

    if not get_user_tier_capability(user_context.user_id, 'crypto_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "permission_denied",
                  {"tool_name": "get_crypto_price", "crypto_id": crypto_id, "provider": provider})
        return "Error: Access to crypto tools is not enabled for your current tier."
    
    params = {"ids": crypto_id.lower(), "vs_currencies": vs_currencies.lower()}
    api_data = await make_api_request(
        provider_name=provider,
        function_name="get_crypto_price",
        params=params,
        user_api_keys=user_api_keys,
        domain="crypto"
    )

    if api_data:
        if crypto_id.lower() in api_data:
            price_info = api_data[crypto_id.lower()]
            result_str = f"Current price of {crypto_id.capitalize()}: "
            for currency, price in price_info.items():
                result_str += f"{price} {currency.upper()}, "
            result_str = result_str.rstrip(', ') + "."
            analytics_tracker.log_event(user_context.user_id, "tool_usage", "success",
                      {"tool_name": "get_crypto_price", "crypto_id": crypto_id, "provider": provider, "price_data": price_info},
                      success=True)
            return result_str
        else:
            analytics_tracker.log_event(user_context.user_id, "tool_usage", "no_data",
                      {"tool_name": "get_crypto_price", "crypto_id": crypto_id, "provider": provider, "message": "Crypto ID not found in API response."},
                      success=False)
            return f"Could not retrieve live cryptocurrency price for {crypto_id.capitalize()}."
    else:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "api_error",
                  {"tool_name": "get_crypto_price", "crypto_id": crypto_id, "provider": provider, "error_message": "No data from API"},
                  success=False)
        return f"Could not retrieve live cryptocurrency price for {crypto_id.capitalize()}."


@tool
async def get_crypto_info(
    crypto_id: str,
    user_context: Optional[UserProfile] = None,
    provider: str = "coingecko",
    user_api_keys: list = []
) -> str:
    """
    Retrieves detailed information about a cryptocurrency.

    This tool fetches comprehensive details such as description, genesis date,
    market cap rank, hashing algorithm, and official website.
    Requires 'crypto_tool_access' capability.

    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., "bitcoin", "ethereum").
        user_context (UserProfile, optional): The user's profile for RBAC checks. Defaults to None.
        provider (str, optional): The API provider to use. Defaults to "coingecko".
        user_api_keys (list, optional): List of user-provided API keys (e.g., from Streamlit secrets).

    Returns:
        str: A formatted string containing the cryptocurrency information,
             or an error message.
    """
    if user_context is None:
        user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

    logger.info(f"Tool: get_crypto_info called for crypto_id: '{crypto_id}', provider: '{provider}', user: '{user_context.user_id}'")

    if not get_user_tier_capability(user_context.user_id, 'crypto_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "permission_denied",
                           {"tool_name": "get_crypto_info", "crypto_id": crypto_id, "provider": provider},
                           success=False)
        return "Error: Access to crypto tools is not enabled for your current tier."
    
    params = {"id": crypto_id.lower()}
    api_data = await make_api_request(
        provider_name=provider,
        function_name="get_crypto_info",
        params=params,
        user_api_keys=user_api_keys,
        domain="crypto"
    )

    if api_data:
        name = api_data.get("name", "N/A")
        symbol = api_data.get("symbol", "N/A").upper()
        description = api_data.get("description", {}).get("en", "No description available.")
        genesis_date = api_data.get("genesis_date", "N/A")
        market_cap_rank = api_data.get("market_cap_rank", "N/A")
        hashing_algorithm = api_data.get("hashing_algorithm", "N/A")
        homepage = api_data.get("links", {}).get("homepage", [])
        website = homepage[0] if homepage else "N/A"

        info_str = (
            f"**Information for {name} ({symbol}):**\n"
            f"- **Description:** {description.split('.')[0]}...\n"
            f"- **Genesis Date:** {genesis_date}\n"
            f"- **Market Cap Rank:** {market_cap_rank}\n"
            f"- **Hashing Algorithm:** {hashing_algorithm}\n"
            f"- **Official Website:** {website}"
        )
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "success",
                           {"tool_name": "get_crypto_info", "crypto_id": crypto_id, "provider": provider, "summary": f"Fetched info for {name}"},
                           success=True)
        return info_str
    else:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "no_data",
                           {"tool_name": "get_crypto_info", "crypto_id": crypto_id, "provider": provider, "message": "No data from API or coin not found."},
                           success=False)
        return f"Could not retrieve complete live crypto information for {crypto_id.capitalize()}."


@tool
async def get_historical_crypto_price(
    crypto_id: str,
    date: str,
    vs_currency: str = "usd",
    user_context: Optional[UserProfile] = None,
    provider: str = "coingecko",
    user_api_keys: list = []
) -> str:
    """
    Retrieves the historical price of a cryptocurrency for a specific date.
    """
    if user_context is None:
        user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

    logger.info(f"Tool: get_historical_crypto_price called for crypto_id: '{crypto_id}', date: '{date}', vs_currency: '{vs_currency}' by user: {user_context.user_id}")

    if not get_user_tier_capability(user_context.user_id, 'historical_data_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "permission_denied",
                           {"tool_name": "get_historical_crypto_price", "crypto_id": crypto_id, "provider": provider},
                           success=False)
        return "Error: Access to historical data is not enabled for your current tier."

    params = {"id": crypto_id.lower(), "date": date, "localization": "false"}
    api_data = await make_api_request(
        provider_name=provider,
        function_name="get_historical_crypto_price",
        params=params,
        user_api_keys=user_api_keys,
        domain="crypto"
    )

    if api_data and api_data.get("market_data", {}).get("current_price", {}).get(vs_currency.lower()):
        price = api_data["market_data"]["current_price"][vs_currency.lower()]
        market_cap = api_data["market_data"].get("market_cap", {}).get(vs_currency.lower(), "N/A")
        total_volume = api_data["market_data"].get("total_volumes", {}).get(vs_currency.lower(), "N/A")
        
        result_str = (
            f"Historical price for {crypto_id.capitalize()} on {date}:\n"
            f"- **Price:** {price} {vs_currency.upper()}\n"
            f"- **Market Cap:** {market_cap} {vs_currency.upper()}\n"
            f"- **Total Volume:** {total_volume} {vs_currency.upper()}"
        )
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "success",
                           {"tool_name": "get_historical_crypto_price", "crypto_id": crypto_id, "date": date, "provider": provider, "price": price},
                           success=True)
        return result_str
    else:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "no_data",
                           {"tool_name": "get_historical_crypto_price", "crypto_id": crypto_id, "date": date, "provider": provider, "message": "No historical data from API."},
                           success=False)
        return f"Could not retrieve historical price for {crypto_id.capitalize()} on {date}."


@tool
async def get_crypto_id_by_symbol(
    symbol: str,
    user_context: Optional[UserProfile] = None,
    provider: str = "coingecko",
    user_api_keys: list = []
) -> str:
    """
    Looks up the CoinGecko ID for a given cryptocurrency symbol.
    This ID is often required for other CoinGecko API calls.
    Requires 'crypto_tool_access' capability.

    Args:
        symbol (str): The cryptocurrency symbol (e.g., "BTC", "ETH", "SOL").
        user_context (UserProfile): The user's profile for RBAC checks and logging.
        provider (str): The API provider to use. Defaults to "coingecko".
        user_api_keys (list): List of user-provided API keys.

    Returns:
        str: The CoinGecko ID for the symbol, or an error message if not found.
    """
    if user_context is None:
        user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

    logger.info(f"Tool: get_crypto_id_by_symbol called for symbol: '{symbol}' by user: {user_context.user_id}")

    if not get_user_tier_capability(user_context.user_id, 'crypto_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "permission_denied",
                           {"tool_name": "get_crypto_id_by_symbol", "symbol": symbol, "provider": provider},
                           success=False)
        return "Error: Access to crypto tools is not enabled for your current tier."
    
    params = {}
    api_data = await make_api_request(
        provider_name=provider,
        function_name="get_crypto_id_by_symbol",
        params=params,
        user_api_keys=user_api_keys,
        domain="crypto"
    )

    if api_data and isinstance(api_data, list):
        found_id = next((item["id"] for item in api_data if item["symbol"].lower() == symbol.lower()), None)
        if found_id:
            analytics_tracker.log_event(user_context.user_id, "tool_usage", "success",
                           {"tool_name": "get_crypto_id_by_symbol", "symbol": symbol, "found_id": found_id},
                           success=True)
            return f"The CoinGecko ID for symbol {symbol.upper()} is: {found_id}."
        else:
            analytics_tracker.log_event(user_context.user_id, "tool_usage", "no_data",
                           {"tool_name": "get_crypto_id_by_symbol", "symbol": symbol, "message": "Symbol not found in CoinGecko list."},
                           success=False)
            return f"Could not find CoinGecko ID for symbol {symbol.upper()}. Please check the symbol and try again."
    else:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "api_error",
                           {"tool_name": "get_crypto_id_by_symbol", "symbol": symbol, "provider": provider, "error_message": "Invalid API response for coins/list."},
                           success=False)
        return f"Error fetching CoinGecko ID for symbol {symbol.upper()}."


@tool
async def crypto_search_web(
    query: str,
    user_context: Optional[UserProfile] = None,
    max_chars: int = 2000
) -> str:
    """
    Searches the web for cryptocurrency-related information using a smart search fallback mechanism.
    This tool wraps the generic `scrape_web` tool, providing a crypto-specific interface.
    Requires 'web_search_enabled' capability.
    
    Args:
        query (str): The crypto-related search query (e.g., "latest news on Ethereum 2.0", "how to buy Solana").
        user_context (UserProfile, optional): The user's profile for RBAC checks and logging. Defaults to None.
        max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
    
    Returns:
        str: A string containing relevant information from the web.
    """
    if user_context is None:
        user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

    logger.info(f"Tool: crypto_search_web called with query: '{query}' for user: '{user_context.user_id}'")
    
    if not get_user_tier_capability(user_context.user_id, 'web_search_enabled', False, user_tier=user_context.tier, user_roles=user_context.roles):
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "permission_denied",
                           {"tool_name": "crypto_search_web", "query": query},
                           success=False)
        return "Error: Web search is not enabled for your current tier."

    try:
        result = await scrape_web(query=query, user_context=user_context, max_chars=max_chars) # Changed user_token to user_context to align with scrape_web signature
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "success",
                           {"tool_name": "crypto_search_web", "query": query, "result_length": len(result)},
                           success=True)
        return result
    except Exception as e:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "error",
                           {"tool_name": "crypto_search_web", "query": query, "error": str(e)},
                           success=False)
        return f"Error during web search: {e}"


@tool
async def crypto_query_uploaded_docs(
    query: str,
    user_context: Optional[UserProfile] = None,
    export: Optional[bool] = False,
    k: int = 5,
    document_tools: Optional[DocumentTools] = None # Pass DocumentTools instance
) -> str:
    """
    Queries previously uploaded and indexed cryptocurrency documents for a user using vector similarity search.
    This tool wraps the generic `DocumentTools.document_query_uploaded_docs` tool, fixing the section to "crypto".
    Requires 'document_query_enabled' capability.
    
    Args:
        query (str): The search query to find relevant crypto documents (e.g., "whitepaper for project X", "my crypto portfolio balance").
        user_context (UserProfile, optional): The user's profile for RBAC checks and logging. Defaults to None.
        export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
        k (int): The number of top relevant documents to retrieve. Defaults to 5.
        document_tools (DocumentTools, optional): The DocumentTools instance. Required for this function.

    Returns:
        str: A string containing the combined content of the relevant document chunks,
             or a message indicating no data/results found, or the export path if exported.
    """
    if user_context is None:
        user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

    logger.info(f"Tool: crypto_query_uploaded_docs called with query: '{query}' for user: '{user_context.user_id}'")
    
    if not get_user_tier_capability(user_context.user_id, 'document_query_enabled', False, user_tier=user_context.tier, user_roles=user_context.roles):
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "permission_denied",
                           {"tool_name": "crypto_query_uploaded_docs", "query": query},
                           success=False)
        return "Error: Document querying is not enabled for your current tier."

    if not document_tools:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "error",
                           {"tool_name": "crypto_query_uploaded_docs", "query": query, "error": "DocumentTools instance not provided."},
                           success=False)
        return "Error: Document tools are not initialized. Cannot query uploaded documents."
    
    try:
        result = await document_tools.document_query_uploaded_docs(
            query_text=query, # Renamed from query to query_text as per DocumentTools
            user_context=user_context,
            section="crypto",
            export=export,
            k=k
        )
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "success",
                           {"tool_name": "crypto_query_uploaded_docs", "query": query, "result_length": len(result)},
                           success=True)
        return result
    except Exception as e:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "error",
                           {"tool_name": "crypto_query_uploaded_docs", "query": query, "error": str(e)},
                           success=False)
        return f"Error querying uploaded documents: {e}"


@tool
async def crypto_summarize_document_by_path(
    file_path_str: str,
    user_context: Optional[UserProfile] = None,
    document_tools: Optional[DocumentTools] = None # Pass DocumentTools instance
) -> str:
    """
    Summarizes a document related to cryptocurrency or blockchain located at the given file path.
    This tool wraps the generic `DocumentTools.document_summarize_document_by_path` tool.
    Requires 'summarization_enabled' capability.
    
    Args:
        file_path_str (str): The full path to the document file to be summarized.
                                Example: "uploads/default/crypto/bitcoin_whitepaper.pdf"
        user_context (UserProfile, optional): The user's profile for RBAC checks and logging. Defaults to None.
        document_tools (DocumentTools, optional): The DocumentTools instance. Required for this function.
        
    Returns:
        str: A concise summary of the document content.
    """
    if user_context is None:
        user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

    logger.info(f"Tool: crypto_summarize_document_by_path called for file: '{file_path_str}' by user: '{user_context.user_id}'")
    
    if not get_user_tier_capability(user_context.user_id, 'summarization_enabled', False, user_tier=user_context.tier, user_roles=user_context.roles):
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "permission_denied",
                           {"tool_name": "crypto_summarize_document_by_path", "file_path": file_path_str},
                           success=False)
        return "Error: Document summarization is not enabled for your current tier."

    if not document_tools:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "error",
                           {"tool_name": "crypto_summarize_document_by_path", "file_path": file_path_str, "error": "DocumentTools instance not provided."},
                           success=False)
        return "Error: Document tools are not initialized. Cannot summarize documents."

    try:
        result = await document_tools.document_summarize_document_by_path(
            file_path_str=file_path_str,
            user_context=user_context
        )
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "success",
                           {"tool_name": "crypto_summarize_document_by_path", "file_path": file_path_str, "result_length": len(result)},
                           success=True)
        return result
    except Exception as e:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "error",
                           {"tool_name": "crypto_summarize_document_by_path", "file_path": file_path_str, "error": str(e)},
                           success=False)
        return f"Error summarizing document: {e}"


# --- CryptoTools Class (Wrapper) ---
class CryptoTools:
    """
    A collection of tools for cryptocurrency-related operations.
    This class now acts primarily as a wrapper to expose the standalone tool functions
    as methods, ensuring a consistent interface.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: DocumentTools):
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools
        logger.info("CryptoTools initialized.")

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
        return await get_crypto_price(
            crypto_id=crypto_id,
            vs_currencies=vs_currencies,
            user_context=user_context,
            provider=provider,
            user_api_keys=user_api_keys
        )

    async def get_crypto_info(
        self,
        crypto_id: str,
        user_context: Optional[UserProfile] = None,
        provider: str = "coingecko",
        user_api_keys: list = []
    ) -> str:
        """
        Retrieves detailed information about a cryptocurrency.
        """
        return await get_crypto_info(
            crypto_id=crypto_id,
            user_context=user_context,
            provider=provider,
            user_api_keys=user_api_keys
        )

    async def get_historical_crypto_price(
        self,
        crypto_id: str,
        date: str,
        vs_currency: str = "usd",
        user_context: Optional[UserProfile] = None,
        provider: str = "coingecko",
        user_api_keys: list = []
    ) -> str:
        """
        Retrieves the historical price of a cryptocurrency for a specific date.
        """
        return await get_historical_crypto_price(
            crypto_id=crypto_id,
            date=date,
            vs_currency=vs_currency,
            user_context=user_context,
            provider=provider,
            user_api_keys=user_api_keys
        )

    async def get_crypto_id_by_symbol(
        self,
        symbol: str,
        user_context: Optional[UserProfile] = None,
        provider: str = "coingecko",
        user_api_keys: list = []
    ) -> str:
        """
        Looks up the CoinGecko ID for a given cryptocurrency symbol.
        """
        return await get_crypto_id_by_symbol(
            symbol=symbol,
            user_context=user_context,
            provider=provider,
            user_api_keys=user_api_keys
        )

    async def crypto_search_web(
        self,
        query: str,
        user_context: Optional[UserProfile] = None,
        max_chars: int = 2000
    ) -> str:
        """
        Searches the web for cryptocurrency-related information.
        """
        return await crypto_search_web(
            query=query,
            user_context=user_context,
            max_chars=max_chars
        )

    async def crypto_query_uploaded_docs(
        self,
        query: str,
        user_context: Optional[UserProfile] = None,
        export: Optional[bool] = False,
        k: int = 5
    ) -> str:
        """
        Queries previously uploaded and indexed cryptocurrency documents for a user.
        """
        # Pass the document_tools instance explicitly to the standalone function
        return await crypto_query_uploaded_docs(
            query=query,
            user_context=user_context,
            export=export,
            k=k,
            document_tools=self.document_tools
        )

    async def crypto_summarize_document_by_path(
        self,
        file_path_str: str,
        user_context: Optional[UserProfile] = None
    ) -> str:
        """
        Summarizes a document related to cryptocurrency or blockchain located at the given file path.
        """
        # Pass the document_tools instance explicitly to the standalone function
        return await crypto_summarize_document_by_path(
            file_path_str=file_path_str,
            user_context=user_context,
            document_tools=self.document_tools
        )


# CLI Test (optional)
if __name__ == "__main__":
    import asyncio
    from unittest.mock import MagicMock, AsyncMock, patch, ANY
    import shutil
    import os
    import sys
    from pathlib import Path
    
    try:
        from shared_tools.vector_utils import BASE_VECTOR_DIR
    except ImportError:
        BASE_VECTOR_DIR = Path("./mock_vector_dir")
        
    try:
        from database.firestore_manager import FirestoreManager
    except ImportError:
        class FirestoreManager: pass

    try:
        from shared_tools.cloud_storage_utils import CloudStorageUtilsWrapper
    except ImportError:
        class CloudStorageUtilsWrapper: pass

    try:
        from shared_tools.vector_utils import VectorUtilsWrapper
    except ImportError:
        class VectorUtilsWrapper: pass

    try:
        from domain_tools.document_tools.document_tool import DocumentTools 
    except ImportError:
        class DocumentTools:
            def __init__(self, *args, **kwargs): pass
            async def document_query_uploaded_docs(self, query_text, user_context, section, export, k): return f"Mocked document query for {section} with query '{query_text}'"
            async def document_summarize_document_by_path(self, file_path_str, user_context): return f"Mocked summary of {file_path_str}"

    try:
        from shared_tools.scraper_tool import scrape_web # For patching scrape_web
    except ImportError:
        async def scrape_web(*args, **kwargs): return "Mocked web search results."

    # Mock UserProfile
    mock_user_free_profile = UserProfile(user_id="mock_free_token", username="FreeUser", email="free@example.com", tier="free", roles=["user"])
    mock_user_pro_profile = UserProfile(user_id="mock_pro_token", username="ProUser", email="pro@example.com", tier="pro", roles=["user"])

    logging.basicConfig(level=logging.INFO)

    class MockSecrets:
        def __init__(self):
            self.coingecko_api_key = "MOCK_COINGECKO_API_KEY_LIVE"
            self.serpapi_api_key = "MOCK_SERPAPI_KEY_LIVE"
            self.openai_api_key = "sk-mock-openai-key-12345"
            self.google_api_key = "AIzaSy-mock-google-key"

        def get(self, key, default=None):
            return getattr(self, key, default)
    
    class MockConfigManager:
        _instance = None
        _is_loaded = False
        def __init__(self):
            if MockConfigManager._instance is not None:
                raise Exception("ConfigManager is a singleton. Use get_instance().")
            MockConfigManager._instance = self
            self._config_data = {
                'llm': {'max_summary_input_chars': 10000},
                'rag': {'chunk_size': 500, 'chunk_overlap': 50, 'max_query_results_k': 10},
                'web_scraping': {
                    'user_agent': 'Mozilla/5.0 (Test; Python)',
                    'timeout_seconds': 1
                },
                'tiers': {},
                'default_user_tier': 'free',
                'default_user_roles': ['user'],
                'api_defaults': {
                    'crypto': 'coingecko',
                    'web_search': 'serpapi',
                    'document_summarization_llm': 'openai'
                },
                'analytics': {
                    'enabled': True,
                    'log_tool_usage': True,
                    'log_query_failures': True
                }
            }
            self._api_providers_data = {
                "crypto": {
                    "coingecko": {
                        "base_url": "https://api.coingecko.com/api/v3",
                        "api_key_name": "coingecko_api_key",
                        "api_key_param_name": "x_cg_demo_api_key",
                        "functions": {
                            "get_crypto_price": {
                                "endpoint": "/simple/price",
                                "required_params": ["ids", "vs_currencies"],
                                "optional_params": ["include_market_cap", "include_24hr_vol", "include_24hr_change", "include_last_updated_at"],
                                "response_path": [],
                                "data_map": {}
                            },
                            "get_crypto_info": {
                                "endpoint": "/coins/{id}",
                                "path_params": ["id"],
                                "required_params": [],
                                "response_path": [],
                                "data_map": {
                                    "name": "name",
                                    "symbol": "symbol",
                                    "description": "description.en",
                                    "genesis_date": "genesis_date",
                                    "market_cap_rank": "market_cap_rank",
                                    "hashing_algorithm": "hashing_algorithm",
                                    "website": "links.homepage.0"
                                }
                            },
                            "get_historical_crypto_price": {
                                "endpoint": "/coins/{id}/history",
                                "path_params": ["id"],
                                "required_params": ["date", "vs_currency"],
                                "response_path": [],
                                "data_map": {
                                    "price": "market_data.current_price.{vs_currency}",
                                    "market_cap": "market_data.market_cap.{vs_currency}",
                                    "volume": "market_data.total_volumes.{vs_currency}"
                                }
                            },
                             "get_crypto_id_by_symbol": {
                                "endpoint": "/coins/list",
                                "required_params": [],
                                "optional_params": [],
                                "response_path": [],
                                "data_map": {}
                            }
                        }
                    }
                },
                "web_search": {
                    "serpapi": {
                        "base_url": "https://serpapi.com/search",
                        "api_key_name": "serpapi_api_key",
                        "api_key_param_name": "api_key",
                        "functions": {
                            "scrape_web": {
                                "required_params": ["q"],
                                "optional_params": ["engine"],
                                "response_path": ["organic_results"],
                                "data_map": {
                                    "title": "title",
                                    "link": "link",
                                    "snippet": "snippet"
                                }
                            }
                        }
                    }
                },
                "document_summarization_llm": {
                    "openai": {
                        "base_url": "https://api.openai.com/v1/chat/completions",
                        "api_key_name": "openai_api_key",
                        "functions": {
                            "summarize_document": {
                                "endpoint": "",
                                "required_params": [],
                                "optional_params": [],
                                "response_path": ["choices", 0, "message", "content"],
                                "data_map": {}
                            }
                        }
                    }
                }
            }
            self._is_loaded = True
        
        def get(self, key, default=None):
            parts = key.split('.')
            val = self._config_data
            for part in parts:
                if isinstance(val, dict) and part in val:
                    val = val[part]
                else:
                    return default
            return val
        
        def get_secret(self, key, default=None):
            mock_secrets_instance = MockSecrets()
            return mock_secrets_instance.get(key, default)

        def set_secret(self, key, value):
            pass
        
        def get_api_provider_config(self, domain: str, provider_name: str) -> Optional[Dict[str, Any]]:
            return self._api_providers_data.get(domain, {}).get(provider_name)

        def get_domain_api_providers(self, domain: str) -> Dict[str, Any]:
            return self._api_providers_data.get(domain, {})


    class MockUserManager:
        _mock_users = {
            "mock_free_token": {"user_id": "mock_free_token", "username": "FreeUser", "email": "free@example.com", "tier": "free", "roles": ["user"]},
            "mock_pro_token": {"user_id": "mock_pro_token", "username": "ProUser", "email": "pro@example.com", "tier": "pro", "roles": ["user"]},
            "mock_premium_token": {"user_id": "mock_premium_token", "username": "PremiumUser", "email": "premium@example.com", "tier": "premium", "roles": ["user"]},
            "mock_admin_token": {"user_id": "mock_admin_token", "username": "AdminUser", "email": "admin@example.com", "tier": "admin", "roles": ["user", "admin"]},
        }
        _rbac_capabilities = {
            'capabilities': {
                'crypto_tool_access': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'historical_data_access': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
                'document_upload_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
                'document_query_enabled': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'web_search_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
                'summarization_enabled': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'llm_default_provider': {
                    'default': 'gemini',
                    'tiers': {'pro': 'gemini', 'premium': 'openai', 'admin': 'gemini'}
                },
                'llm_default_model_name': {
                    'default': 'gemini-1.5-flash',
                    'tiers': {'pro': 'gemini-1.5-flash', 'premium': 'gpt-4o', 'admin': 'gemini-1.5-flash'}
                },
                'llm_default_temperature': {
                    'default': 0.7,
                    'tiers': {'pro': 0.5, 'premium': 0.3, 'admin': 0.7}
                },
            }
        }
        _tier_hierarchy = {
            "free": 0, "user": 1, "basic": 2, "pro": 3, "premium": 4, "admin": 99
        }

        def get_user_tier_capability(self, user_id: str, capability_key: str, default_value: Any = None, user_tier: Optional[str] = None, user_roles: Optional[List[str]] = None) -> Any:
            if user_tier is None or user_roles is None:
                user_info = self._mock_users.get(user_id, {})
                user_tier = user_info.get('tier', 'free')
                user_roles = user_info.get('roles', [])

            if "admin" in user_roles:
                if isinstance(default_value, bool): return True
                if isinstance(default_value, (int, float)): return float('inf')
                return default_value
            
            capability_config = self._rbac_capabilities.get('capabilities', {}).get(capability_key)
            if not capability_config:
                return default_value

            for role in user_roles:
                if role in capability_config.get('roles', {}):
                    return capability_config['roles'][role]
            
            if user_tier in capability_config.get('tiers', {}):
                return capability_config['tiers'][user_tier]

            return capability_config.get('default', default_value)


    import streamlit as st_mock
    if not hasattr(st_mock, 'secrets'):
        st_mock.secrets = MockSecrets()
    
    if 'config.config_manager' not in sys.modules:
        sys.modules['config.config_manager'] = MagicMock()
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager
    
    if 'utils.user_manager' not in sys.modules:
        sys.modules['utils.user_manager'] = MagicMock()
    sys.modules['utils.user_manager'].get_user_tier_capability = MockUserManager().get_user_tier_capability

    mock_firestore_manager_for_analytics = MagicMock(spec=FirestoreManager)
    mock_firestore_manager_for_analytics.collection.return_value.add = AsyncMock(return_value=MagicMock(id="mock_doc_id"))

    mock_auth_for_analytics = MagicMock()
    mock_auth_for_analytics.currentUser = MagicMock(uid="mock_user_123")
    
    with patch.dict(sys.modules, {'firebase_admin.firestore': MagicMock(firestore=MagicMock())}):
        sys.modules['firebase_admin.firestore'].firestore.CollectionReference = MagicMock()
        sys.modules['firebase_admin'].firestore.DocumentReference = MagicMock()
        
        analytics_tracker.initialize_analytics(
            mock_firestore_manager_for_analytics,
            mock_auth_for_analytics,
            "test_app_id_for_analytics",
            "mock_user_123"
        )

        original_requests_get = requests.get
        original_requests_post = requests.post

        def mock_requests_dynamic(method, url, params=None, headers=None, json=None, timeout=None):
            logger.info(f"Mocking {method} API request to {url} with params: {params or json}")
            if "api.coingecko.com/api/v3" in url:
                if "/simple/price" in url:
                    ids = params.get("ids", "").lower()
                    vs_currencies = params.get("vs_currencies", "").lower()
                    if ids == "bitcoin" and vs_currencies == "usd":
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "bitcoin": {
                                "usd": 65000.00,
                                "usd_market_cap": 1280000000000,
                                "usd_24hr_vol": 35000000000,
                                "usd_24hr_change": 2.5,
                                "last_updated_at": int(datetime.now(timezone.utc).timestamp())
                            }
                        }
                        return mock_response
                    elif ids == "ethereum" and vs_currencies == "usd":
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "ethereum": {
                                "usd": 3500.00,
                                "usd_market_cap": 420000000000,
                                "usd_24hr_vol": 15000000000,
                                "usd_24hr_change": 1.8,
                                "last_updated_at": int(datetime.now(timezone.utc).timestamp())
                            }
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {}
                        return mock_response
                elif "/coins/list" in url:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = [
                        {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"},
                        {"id": "ethereum", "symbol": "eth", "name": "Ethereum"},
                        {"id": "solana", "symbol": "sol", "name": "Solana"},
                        {"id": "dogecoin", "symbol": "doge", "name": "Dogecoin"},
                    ]
                    return mock_response
                elif "/coins/" in url and "/history" not in url and "/market_chart" not in url:
                    crypto_id_from_url = url.split("/coins/")[1].split("/")[0].lower()
                    if crypto_id_from_url == "bitcoin":
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "id": "bitcoin", "symbol": "btc", "name": "Bitcoin",
                            "description": {"en": "Bitcoin is a decentralized digital currency, without a central bank or single administrator, that can be sent from user to user on the peer-to-peer bitcoin network without the need for intermediaries."},
                            "genesis_date": "2009-01-03", "market_cap_rank": 1,
                            "hashing_algorithm": "SHA-256",
                            "links": {"homepage": ["https://bitcoin.org/en/", "other.link"]}
                        }
                        return mock_response
                    elif crypto_id_from_url == "nonexistentcoin":
                        mock_response = MagicMock()
                        mock_response.status_code = 404
                        mock_response.json.return_value = {"error": "coin not found"}
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 404
                        mock_response.json.return_value = {"error": "coin not found"}
                        return mock_response
                elif "/coins/" in url and "/history" in url:
                    crypto_id_from_url = url.split("/coins/")[1].split("/history")[0].lower()
                    date = params.get("date")
                    vs_currency = params.get("vs_currency", "usd").lower()
                    if crypto_id_from_url == "bitcoin" and date == (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%d-%m-%Y"):
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "id": "bitcoin", "symbol": "btc", "name": "Bitcoin",
                            "market_data": {
                                "current_price": {vs_currency: 64500.00},
                                "market_cap": {vs_currency: 1270000000000},
                                "total_volumes": {vs_currency: 34000000000}
                            }
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {}
                        return mock_response
            
            if "serpapi.com/search" in url:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "organic_results": [
                        {"title": "Mock Search Result 1", "link": "http://example.com/1", "snippet": f"Snippet for {params.get('q', 'crypto')} result 1."},
                        {"title": "Mock Search Result 2", "link": "http://example.com/2", "snippet": f"Snippet for {params.get('q', 'crypto')} result 2."}
                    ]
                }
                return mock_response

            if "api.openai.com/v1/chat/completions" in url:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Mocked LLM summary content."}}]
                }
                return mock_response

            if method == "GET":
                return original_requests_get(url, params=params, headers=headers, timeout=timeout)
            elif method == "POST":
                return original_requests_post(url, json=json, headers=headers, timeout=timeout)
            else:
                raise NotImplementedError(f"Mock for method {method} not implemented.")

        requests.get = MagicMock(side_effect=lambda url, params=None, headers=None, timeout=None: mock_requests_dynamic("GET", url, params, headers, timeout=timeout))
        requests.post = MagicMock(side_effect=lambda url, json=None, headers=None, timeout=None: mock_requests_dynamic("POST", url, json=json, headers=headers, timeout=timeout))

        mock_firestore_manager_instance = MagicMock(spec=FirestoreManager)
        mock_cloud_storage_utils_instance = MagicMock(spec=CloudStorageUtilsWrapper)
        mock_vector_utils_instance = MagicMock(spec=VectorUtilsWrapper)
        
        mock_document_tools_instance = DocumentTools(
            config_manager=sys.modules['config.config_manager'].config_manager,
            firestore_manager=mock_firestore_manager_instance,
            cloud_storage_utils=mock_cloud_storage_utils_instance,
            vector_utils=mock_vector_utils_instance,
            log_event=analytics_tracker.log_event
        )

        crypto_tools_instance = CryptoTools(
            config_manager=sys.modules['config.config_manager'].config_manager,
            log_event=analytics_tracker.log_event,
            document_tools=mock_document_tools_instance
        )

        async def run_crypto_tests(crypto_tools_instance):
            print("\n--- Testing crypto_tool functions with Live API Simulation and Analytics ---")

            # Test 1: get_crypto_price (success)
            print("\n--- Test 1: get_crypto_price (Success) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            result_crypto_price = await crypto_tools_instance.get_crypto_price("bitcoin", user_context=mock_user_pro_profile)
            print(f"Crypto Price: {result_crypto_price}")
            assert "Current price of Bitcoin: 65000.0 USD" in result_crypto_price
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "get_crypto_price"
            assert logged_data["success"] is True
            print("Test 1 Passed.")

            # Test 2: get_crypto_info (success)
            print("\n--- Test 2: get_crypto_info (Success) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            result_crypto_info = await crypto_tools_instance.get_crypto_info("bitcoin", user_context=mock_user_pro_profile)
            print(f"Crypto Info: {result_crypto_info}")
            assert "Information for Bitcoin (BTC):" in result_crypto_info
            assert "Genesis Date: 2009-01-03" in result_crypto_info
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "get_crypto_info"
            assert logged_data["success"] is True
            print("Test 2 Passed.")

            # Test 3: get_crypto_info (API failure - coin not found)
            print("\n--- Test 3: get_crypto_info (API Failure - Coin Not Found) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            result_crypto_info_fail = await crypto_tools_instance.get_crypto_info("nonexistentcoin", user_context=mock_user_pro_profile)
            print(f"Crypto Info (API Error): {result_crypto_info_fail}")
            assert "Could not retrieve complete live crypto information for Nonexistentcoin." in result_crypto_info_fail
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "get_crypto_info"
            assert logged_data["success"] is False
            assert "No data from API or coin not found." in logged_data["message"]
            print("Test 3 Passed.")

            # Test 4: get_historical_crypto_price (RBAC denied)
            print("\n--- Test 4: get_historical_crypto_price (RBAC Denied) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            result_historical_rbac_denied = await crypto_tools_instance.get_historical_crypto_price("ethereum", "2023-01-01", user_context=mock_user_free_profile)
            print(f"Historical Crypto Price (Free User, RBAC Denied): {result_historical_rbac_denied}")
            assert "Error: Access to historical data is not enabled" in result_historical_rbac_denied
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["status"] == "permission_denied"
            print("Test 4 Passed.")

            # Test 5: get_historical_crypto_price (Success)
            print("\n--- Test 5: get_historical_crypto_price (Success) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            yesterday_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%d-%m-%Y")
            result_historical_success = await crypto_tools_instance.get_historical_crypto_price("bitcoin", yesterday_date, user_context=mock_user_pro_profile)
            print(f"Historical Crypto Price (Success): {result_historical_success}")
            assert "Historical price for Bitcoin on" in result_historical_success
            assert "Price: 64500.0 USD" in result_historical_success
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["status"] == "success"
            print("Test 5 Passed.")

            # Test 6: get_crypto_id_by_symbol (success)
            print("\n--- Test 6: get_crypto_id_by_symbol (Success) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            result_id = await crypto_tools_instance.get_crypto_id_by_symbol("btc", user_context=mock_user_pro_profile)
            print(f"Crypto ID: {result_id}")
            assert "The CoinGecko ID for symbol BTC is: bitcoin." in result_id
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["status"] == "success"
            print("Test 6 Passed.")

            # Test 7: crypto_search_web (generic tool)
            print("\n--- Test 7: crypto_search_web (Generic Tool) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            result_web_search = await crypto_tools_instance.crypto_search_web("best crypto wallets", user_context=mock_user_pro_profile)
            print(f"Web Search Result: {result_web_search[:100]}...")
            assert "Mocked web search results." in result_web_search
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "crypto_search_web"
            assert logged_data["success"] is True
            print("Test 7 Passed.")

            # Test 8: crypto_query_uploaded_docs (generic tool via DocumentTools)
            print("\n--- Test 8: crypto_query_uploaded_docs (Generic Tool via DocumentTools) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            result_doc_query = await crypto_tools_instance.crypto_query_uploaded_docs("whitepaper details", user_context=mock_user_pro_profile)
            print(f"Document Query Result: {result_doc_query}")
            assert "Mocked document query for crypto with query 'whitepaper details'" in result_doc_query
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "crypto_query_uploaded_docs"
            assert logged_data["success"] is True
            print("Test 8 Passed.")

            # Test 9: crypto_summarize_document_by_path (generic tool via DocumentTools)
            print("\n--- Test 9: crypto_summarize_document_by_path (Generic Tool via DocumentTools) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            test_user_pro_dir = Path("uploads") / mock_user_pro_profile.user_id
            dummy_file_path = test_user_pro_dir / "crypto" / "dummy_whitepaper.txt"
            dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_file_path.write_text("This is a dummy whitepaper content for testing summarization.")

            result_summarize = await crypto_tools_instance.crypto_summarize_document_by_path(str(dummy_file_path), user_context=mock_user_pro_profile)
            print(f"Summarize Result: {result_summarize}")
            assert "Mocked summary of uploads" in result_summarize
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "crypto_summarize_document_by_path"
            assert logged_data["success"] is True
            print("Test 9 Passed.")

            print("\nAll crypto_tool tests with live API simulation and analytics considerations completed.")

        if __name__ == "__main__":
            asyncio.run(run_crypto_tests(crypto_tools_instance))

        requests.get = original_requests_get
        requests.post = original_requests_post

        test_user_dirs = [Path("uploads") / mock_user_pro_profile.user_id, BASE_VECTOR_DIR / mock_user_pro_profile.user_id]
        for d in test_user_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print(f"Cleaned up {d}")
