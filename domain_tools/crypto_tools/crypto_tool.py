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
    async def crypto_get_historical_crypto_price(self, crypto_id: str, date: str, vs_currency: str = "usd", user_context: UserProfile = None, provider: str = "coingecko", user_api_keys: list = []) -> 
        if user_context is None:
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: crypto_get_historical_crypto_price called for crypto_id: '{crypto_id}', date: '{date}', vs_currency: '{vs_currency}' by user: {user_context.user_id}")

        # Use the historical_data_tool to get the data
        from shared_tools.historical_data_tool import HistoricalDataTools

        historical_data_json = await HistoricalDataTools.historical_get_data(
            domain="historical_crypto",
            identifier=crypto_id,
            start_date=date,
            end_date=date,
            user_context=user_context,
            vs_currency=vs_currency
        )

        if historical_data_json.startswith("Error:"):
            return historical_data_json

        try:
            historical_prices = json.loads(historical_data_json)
            if historical_prices:
                # The historical_data_tool returns a list of data points. For a single day, we'll take the first one.
                data = historical_prices[0]
                response_str = (
                    f"Historical Price for {crypto_id.capitalize()} on {date}:\n"
                    f"  Price: {data.get('price')} {vs_currency.upper()}\n"
                )
                return response_str
            else:
                return f"No historical price found for {crypto_id.capitalize()} on {date}. Please try again or check the ID/date."
        except (json.JSONDecodeError, IndexError):
            return "Error: Could not parse historical data from the shared tool."

    @tool
    async def crypto_get_crypto_id_by_symbol(self, symbol: str, user_context: UserProfile = None) -> str:
        """
        Looks up the CoinGecko ID for a given cryptocurrency symbol.
        This ID is often required for other CoinGecko API calls.

        Args:
            symbol (str): The cryptocurrency symbol (e.g., "BTC", "ETH", "SOL").
            user_context (UserProfile): The user's profile for RBAC checks and logging.

        Returns:
            str: The CoinGecko ID for the symbol, or an error message if not found.
        """
        if user_context is None: # For CLI testing without full UserProfile
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: crypto_get_crypto_id_by_symbol called for symbol: '{symbol}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'crypto_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to crypto tools is not enabled for your current tier."
        
        # The /coins/list endpoint takes no params, but we pass symbol for internal logic
        params = {"symbol": symbol.lower()} 
        api_data = await self._make_dynamic_api_request("crypto", "get_crypto_id_by_symbol", params, user_context)

        if api_data and api_data.get("id"):
            return f"The CoinGecko ID for symbol {symbol.upper()} is: {api_data['id']}."
        else:
            return f"Could not find CoinGecko ID for symbol {symbol.upper()}. Please check the symbol and try again."


    # --- Existing Generic Tools (now methods of CryptoTools) ---
    # These functions wrap existing shared tools or DocumentTools methods.
    # They will pass the user_context down if the wrapped tool supports it.

    @tool
    async def crypto_search_web(self, query: str, user_context: UserProfile, max_chars: int = 2000) -> str:
        """
        Searches the web for cryptocurrency-related information using a smart search fallback mechanism.
        This tool wraps the generic `scrape_web` tool, providing a crypto-specific interface.
        
        Args:
            query (str): The crypto-related search query (e.g., "latest news on Ethereum 2.0", "how to buy Solana").
            user_context (UserProfile): The user's profile for RBAC checks and logging.
            max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
        
        Returns:
            str: A string containing relevant information from the web.
        """
        logger.info(f"Tool: crypto_search_web called with query: '{query}' for user: '{user_context.user_id}'")
        # scrape_web is a standalone function, ensure it handles its own RBAC/logging if applicable
        # For now, it's assumed LLMService wrapper handles its API limit check.
        return await scrape_web(query=query, user_token=user_context.user_id, max_chars=max_chars) # Pass user_token for scrape_web's internal logging

    @tool
    async def crypto_query_uploaded_docs(self, query: str, user_context: UserProfile, export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed cryptocurrency documents for a user using vector similarity search.
        This tool wraps the generic `DocumentTools.document_query_uploaded_docs` tool, fixing the section to "crypto".
        
        Args:
            query (str): The search query to find relevant crypto documents (e.g., "whitepaper for project X", "my crypto portfolio balance").
            user_context (UserProfile): The user's profile for RBAC checks and logging.
            export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
            k (int): The number of top relevant documents to retrieve. Defaults to 5.
        
        Returns:
            str: A string containing the combined content of the relevant document chunks,
                 or a message indicating no data/results found, or the export path if exported.
        """
        logger.info(f"Tool: crypto_query_uploaded_docs called with query: '{query}' for user: '{user_context.user_id}'")
        if not self.document_tools:
            return "Error: Document tools are not initialized. Cannot query uploaded documents."
        
        # Call the actual document_query_uploaded_docs from the DocumentTools instance
        return await self.document_tools.document_query_uploaded_docs(
            query=query,
            user_context=user_context, # Pass user_context directly
            section="crypto", # Specify the section for crypto documents
            export=export,
            k=k
        )

    @tool
    async def crypto_summarize_document_by_path(self, file_path_str: str, user_context: UserProfile) -> str:
        """
        Summarizes a document related to cryptocurrency or blockchain located at the given file path.
        The file path should be accessible by the system (e.g., in the 'uploads' directory).
        This tool wraps the generic `DocumentTools.document_summarize_document_by_path` tool.
        
        Args:
            file_path_str (str): The full path to the document file to be summarized.
                                Example: "uploads/default/crypto/bitcoin_whitepaper.pdf"
            user_context (UserProfile): The user's profile for RBAC checks and logging.
        
        Returns:
            str: A concise summary of the document content.
        """
        logger.info(f"Tool: crypto_summarize_document_by_path called for file: '{file_path_str}' by user: '{user_context.user_id}'")
        if not self.document_tools:
            return "Error: Document tools are not initialized. Cannot summarize documents."

        # Call the actual document_summarize_document_by_path from the DocumentTools instance
        return await self.document_tools.document_summarize_document_by_path(
            file_path_str=file_path_str,
            user_context=user_context # Pass user_context directly
        )


# CLI Test (optional)
if __name__ == "__main__":
    import asyncio
    from unittest.mock import MagicMock, AsyncMock, patch, ANY
    import shutil
    import os
    import sys
    from shared_tools.vector_utils import BASE_VECTOR_DIR # For cleanup
    from database.firestore_manager import FirestoreManager # For mocking
    from shared_tools.cloud_storage_utils import CloudStorageUtilsWrapper # For mocking
    from shared_tools.vector_utils import VectorUtilsWrapper # For mocking
    from domain_tools.document_tools.document_tool import DocumentTools # For mocking
    from backend.models.user_models import UserProfile # For mock user_context
    from langchain_core.messages import HumanMessage, AIMessage # For mocking LLM in summarizer

    logging.basicConfig(level=logging.INFO)

    # Mock UserProfile for testing
    mock_user_pro_profile = UserProfile(user_id="mock_pro_token", username="ProUser", email="pro@example.com", tier="pro", roles=["user"])
    mock_user_free_profile = UserProfile(user_id="mock_free_token", username="FreeUser", email="free@example.com", tier="free", roles=["user"])
    mock_user_premium_profile = UserProfile(user_id="mock_premium_token", username="PremiumUser", email="premium@example.com", tier="premium", roles=["user"])
    mock_user_admin_profile = UserProfile(user_id="mock_admin_token", username="AdminUser", email="admin@example.com", tier="admin", roles=["user", "admin"])


    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.coingecko_api_key = "MOCK_COINGECKO_API_KEY_LIVE"
            self.serpapi_api_key = "MOCK_SERPAPI_KEY_LIVE" # For scrape_web
            self.openai_api_key = "sk-mock-openai-key-12345" # For summarizer
            self.google_api_key = "AIzaSy-mock-google-key" # For summarizer

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
                    'timeout_seconds': 1 # Short timeout for mocks
                },
                'tiers': {},
                'default_user_tier': 'free',
                'default_user_roles': ['user'],
                'api_defaults': { # Mock api_defaults
                    'crypto': 'coingecko',
                    'web_search': 'serpapi',
                    'document_summarization_llm': 'openai'
                },
                'analytics': { # Mock analytics settings
                    'enabled': True,
                    'log_tool_usage': True,
                    'log_query_failures': True
                }
            }
            self._api_providers_data = { # Mock api_providers_data for crypto
                "crypto": {
                    "coingecko": {
                        "base_url": "https://api.coingecko.com/api/v3",
                        "api_key_name": "coingecko_api_key",
                        "api_key_param_name": "x_cg_demo_api_key", # For CoinGecko's demo key
                        "functions": {
                            "get_crypto_price": {
                                "endpoint": "/simple/price",
                                "required_params": ["ids", "vs_currencies"],
                                "optional_params": ["include_market_cap", "include_24hr_vol", "include_24hr_change", "include_last_updated_at"],
                                "response_path": [], # Root is the data, special handling in _make_dynamic_api_request
                                "data_map": {} # Special handling in _make_dynamic_api_request
                            },
                            "get_crypto_info": {
                                "endpoint": "/coins/{id}", # Path parameter
                                "path_params": ["id"],
                                "required_params": [],
                                "response_path": [], # Root is the data
                                "data_map": {
                                    "name": "name",
                                    "symbol": "symbol",
                                    "description": "description.en", # Nested path
                                    "genesis_date": "genesis_date",
                                    "market_cap_rank": "market_cap_rank",
                                    "hashing_algorithm": "hashing_algorithm",
                                    "website": "links.homepage.0" # Nested path, first item in list
                                }
                            },
                            "get_historical_crypto_price": {
                                "endpoint": "/coins/{id}/history", # Path parameter
                                "path_params": ["id"],
                                "required_params": ["date", "vs_currency"],
                                "response_path": [], # Root is the data
                                "data_map": {
                                    "price": "market_data.current_price.{vs_currency}", # Dynamic key
                                    "market_cap": "market_data.market_cap.{vs_currency}",
                                    "volume": "market_data.total_volumes.{vs_currency}"
                                }
                            },
                             "get_crypto_id_by_symbol": {
                                "endpoint": "/coins/list", # Endpoint for listing all coins
                                "required_params": [], # No required params for the list endpoint itself
                                "optional_params": [],
                                "response_path": [], # Root is a list, special handling in _make_dynamic_api_request
                                "data_map": {} # Special handling in _make_dynamic_api_request
                            }
                        }
                    }
                },
                "web_search": { # Mock for web search (SerpAPI)
                    "serpapi": {
                        "base_url": "https://serpapi.com/search",
                        "api_key_name": "serpapi_api_key",
                        "api_key_param_name": "api_key",
                        "functions": {
                            "scrape_web": { # This function name should match the tool name
                                "required_params": ["q"],
                                "optional_params": ["engine"],
                                "response_path": ["organic_results"], # Example path for search results
                                "data_map": { # Simplified mapping for search results
                                    "title": "title",
                                    "link": "link",
                                    "snippet": "snippet"
                                }
                            }
                        }
                    }
                },
                "document_summarization_llm": { # Mock for summarization LLM
                    "openai": {
                        "base_url": "https://api.openai.com/v1/chat/completions",
                        "api_key_name": "openai_api_key",
                        "functions": {
                            "summarize_document": { # This function name should match the tool name
                                "endpoint": "", # No specific endpoint for chat completions
                                "required_params": [],
                                "optional_params": [],
                                "response_path": ["choices", 0, "message", "content"],
                                "data_map": {} # No specific mapping needed for direct content
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


    # Mock user_manager.get_user_tier_capability for testing RBAC
    # This mock is for the standalone get_user_tier_capability function
    # which is now imported directly by tools.
    class MockUserManager:
        _mock_users = {
            "mock_free_token": {"user_id": "mock_free_token", "username": "FreeUser", "email": "free@example.com", "tier": "free", "roles": ["user"]},
            "mock_pro_token": {"user_id": "mock_pro_token", "username": "ProUser", "email": "pro@example.com", "tier": "pro", "roles": ["user"]},
            "mock_premium_token": {"user_id": "mock_premium_token", "username": "PremiumUser", "email": "premium@example.com", "tier": "premium", "roles": ["user"]},
            "mock_admin_token": {"user_id": "mock_admin_token", "username": "AdminUser", "email": "admin@example.com", "tier": "admin", "roles": ["user", "admin"]},
        }
        _rbac_capabilities = { # This now mirrors the _RBAC_CAPABILITIES_CONFIG in utils/user_manager.py
            'capabilities': {
                'crypto_tool_access': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'document_upload_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
                'document_query_enabled': { # Added for document tool
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'web_search_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
                'summarization_enabled': { # For summarize_document
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'llm_default_provider': { # For summarize_document
                    'default': 'gemini',
                    'tiers': {'pro': 'gemini', 'premium': 'openai', 'admin': 'gemini'}
                },
                'llm_default_model_name': { # For summarize_document
                    'default': 'gemini-1.5-flash',
                    'tiers': {'pro': 'gemini-1.5-flash', 'premium': 'gpt-4o', 'admin': 'gemini-1.5-flash'}
                },
                'llm_default_temperature': { # For summarize_document
                    'default': 0.7,
                    'tiers': {'pro': 0.5, 'premium': 0.3, 'admin': 0.7}
                },
            }
        }
        _tier_hierarchy = {
            "free": 0, "user": 1, "basic": 2, "pro": 3, "premium": 4, "admin": 99
        }

        def get_user_tier_capability(self, user_id: str, capability_key: str, default_value: Any = None, user_tier: Optional[str] = None, user_roles: Optional[List[str]] = None) -> Any:
            # If user_tier/user_roles are provided, use them directly (from UserProfile)
            # Otherwise, try to look up from _mock_users
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

            # Check roles first
            for role in user_roles:
                if role in capability_config.get('roles', {}):
                    return capability_config['roles'][role]
            
            # Then check tiers
            if user_tier in capability_config.get('tiers', {}):
                return capability_config['tiers'][user_tier]

            return capability_config.get('default', default_value)


    # Patch the actual imports for testing
    import streamlit as st_mock
    if not hasattr(st_mock, 'secrets'):
        st_mock.secrets = MockSecrets()
    
    # Patch config_manager and user_manager in their respective modules
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager # Also patch the class if needed by other modules
    
    # Patch the standalone get_user_tier_capability function in utils.user_manager
    # This is crucial for the tools to use the mock during their CLI tests.
    sys.modules['utils.user_manager'].get_user_tier_capability = MockUserManager().get_user_tier_capability

    # Mock analytics_tracker
    mock_analytics_tracker_db = MagicMock()
    mock_analytics_tracker_auth = MagicMock()
    mock_analytics_tracker_auth.currentUser = MagicMock(uid="mock_user_123")
    mock_analytics_tracker_db.collection.return_value.add = AsyncMock(return_value=MagicMock(id="mock_doc_id"))

    # Patch firebase_admin.firestore for the local import within log_event
    with patch.dict(sys.modules, {'firebase_admin.firestore': MagicMock(firestore=MagicMock())}):
        sys.modules['firebase_admin.firestore'].firestore.CollectionReference = MagicMock()
        sys.modules['firebase_admin'].firestore.DocumentReference = MagicMock()
        
        # Initialize the actual analytics_tracker with mocks
        analytics_tracker.initialize_analytics(
            mock_analytics_tracker_db,
            mock_analytics_tracker_auth,
            "test_app_id_for_analytics",
            "mock_user_123"
        )

        # Mock requests.get for external API calls
        original_requests_get = requests.get

        def mock_requests_get_dynamic(url, params=None, headers=None, timeout=None):
            # Simulate CoinGecko responses
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
                        mock_response.json.return_value = {} # Empty for unknown coins
                        return mock_response
                elif "/coins/list" in url: # get_crypto_id_by_symbol
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = [
                        {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"},
                        {"id": "ethereum", "symbol": "eth", "name": "Ethereum"},
                        {"id": "solana", "symbol": "sol", "name": "Solana"},
                        {"id": "dogecoin", "symbol": "doge", "name": "Dogecoin"},
                    ]
                    return mock_response
                elif "/coins/" in url and "/history" not in url: # get_crypto_info
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
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 404
                        mock_response.json.return_value = {"error": "coin not found"}
                        return mock_response
                elif "/coins/" in url and "/history" in url: # get_historical_crypto_price
                    crypto_id_from_url = url.split("/coins/")[1].split("/history")[0].lower()
                    date = params.get("date")
                    vs_currency = params.get("vs_currency", "usd").lower()
                    if crypto_id_from_url == "bitcoin" and date == (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d"):
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "id": "bitcoin", "symbol": "btc", "name": "Bitcoin",
                            "market_data": {
                                "current_price": {vs_currency: 64500.00},
                                "market_cap": {vs_currency: 1270000000000},
                                "total_volume": {vs_currency: 34000000000}
                            }
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {} # No data for this date/crypto
                        return mock_response
            
            # Simulate scrape_web's internal requests.get if needed (SerpAPI)
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

            # Mock LLM for summarizer (if it uses requests.post for an API)
            if "api.openai.com/v1/chat/completions" in url:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Mocked LLM summary content."}}]
                }
                return mock_response

            return original_requests_get(url, params=params, headers=headers, timeout=timeout)

        requests.get = MagicMock(side_effect=mock_requests_get_dynamic)
        requests.post = MagicMock(side_effect=mock_requests_get_dynamic) # For OpenAI chat completions

        # Mock FirestoreManager, CloudStorageUtilsWrapper, VectorUtilsWrapper, DocumentTools for init
        mock_firestore_manager = MagicMock(spec=FirestoreManager)
        mock_cloud_storage_utils = MagicMock(spec=CloudStorageUtilsWrapper)
        mock_vector_utils = MagicMock(spec=VectorUtilsWrapper)
        
        # Create a mock DocumentTools instance
        mock_document_tools = MagicMock(spec=DocumentTools)
        mock_document_tools.document_query_uploaded_docs = AsyncMock(return_value="Mocked document query results for crypto.")
        mock_document_tools.document_summarize_document_by_path = AsyncMock(return_value="Mocked summary of dummy_file.txt")

        # Instantiate CryptoTools with mocks
        crypto_tools_instance = CryptoTools(
            config_manager=sys.modules['config.config_manager'].config_manager,
            log_event=analytics_tracker.log_event, # Pass the actual (mocked) log_event
            document_tools=mock_document_tools
        )

        async def run_crypto_tests(crypto_tools_instance):
            print("\n--- Testing crypto_tool functions with Live API Simulation and Analytics ---")

            # Test 1: crypto_get_crypto_price (success)
            print("\n--- Test 1: crypto_get_crypto_price (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock() # Reset mock call count
            result_crypto_price = await crypto_tools_instance.crypto_get_crypto_price("bitcoin", user_context=mock_user_pro_profile)
            print(f"Crypto Price: {result_crypto_price}")
            assert "Current price of Bitcoin: 65000.0 USD" in result_crypto_price
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called() # Analytics should NOT be logged for success here
            print("Test 1 Passed.")

            # Test 2: crypto_get_crypto_info (API failure - coin not found)
            print("\n--- Test 2: crypto_get_crypto_info (API Failure - Coin Not Found) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_crypto_info = await crypto_tools_instance.crypto_get_crypto_info("nonexistentcoin", user_context=mock_user_pro_profile)
            print(f"Crypto Info (API Error): {result_crypto_info}")
            assert "Could not retrieve complete live crypto information for Nonexistentcoin." in result_crypto_info
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once() # Analytics should be logged for failure
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "crypto_get_crypto_info"
            assert logged_data["success"] is False
            assert "coin not found" in logged_data["error_message"]
            print("Test 2 Passed.")

            # Test 3: crypto_get_historical_crypto_price (RBAC denied)
            print("\n--- Test 3: crypto_get_historical_crypto_price (RBAC Denied) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_historical_rbac_denied = await crypto_tools_instance.crypto_get_historical_crypto_price("ethereum", "2023-01-01", user_context=mock_user_free_profile)
            print(f"Historical Crypto Price (Free User, RBAC Denied): {result_historical_rbac_denied}")
            assert "Error: Access to crypto tools is not enabled for your current tier." in result_historical_rbac_denied
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called() # RBAC check happens before _make_dynamic_api_request
            print("Test 3 Passed.")

            # Test 4: crypto_get_crypto_id_by_symbol (success)
            print("\n--- Test 4: crypto_get_crypto_id_by_symbol (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_id = await crypto_tools_instance.crypto_get_crypto_id_by_symbol("btc", user_context=mock_user_pro_profile)
            print(f"Crypto ID: {result_id}")
            assert "The CoinGecko ID for symbol BTC is: bitcoin." in result_id
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called() # Analytics should NOT be logged for success here
            print("Test 4 Passed.")

            # Test 5: crypto_search_web (generic tool)
            print("\n--- Test 5: crypto_search_web (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_web_search = await crypto_tools_instance.crypto_search_web("best crypto wallets", user_context=mock_user_pro_profile)
            print(f"Web Search Result: {result_web_search[:100]}...")
            assert "Search results for best crypto wallets" in result_web_search
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called() # Analytics for scrape_web is handled by its own internal logging or LLMService wrapper
            print("Test 5 Passed.")

            # Test 6: crypto_query_uploaded_docs (generic tool via DocumentTools)
            print("\n--- Test 6: crypto_query_uploaded_docs (Generic Tool via DocumentTools) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_doc_query = await crypto_tools_instance.crypto_query_uploaded_docs("whitepaper details", user_context=mock_user_pro_profile)
            print(f"Document Query Result: {result_doc_query}")
            assert "Mocked document query results for crypto." in result_doc_query
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once() # Analytics logged by DocumentTools mock
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "document_query_uploaded_docs"
            assert logged_data["success"] is True
            print("Test 6 Passed.")

            # Test 7: crypto_summarize_document_by_path (generic tool via DocumentTools)
            print("\n--- Test 7: crypto_summarize_document_by_path (Generic Tool via DocumentTools) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            # Create a dummy file for summarization test
            test_user_pro_dir = Path("uploads") / mock_user_pro_profile.user_id
            dummy_file_path = test_user_pro_dir / "crypto" / "dummy_whitepaper.txt"
            dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_file_path.write_text("This is a dummy whitepaper content for testing summarization.")

            result_summarize = await crypto_tools_instance.crypto_summarize_document_by_path(str(dummy_file_path), user_context=mock_user_pro_profile)
            print(f"Summarize Result: {result_summarize}")
            assert "Mocked summary of dummy_file.txt" in result_summarize # Check for mock summary from DocumentTools
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once() # Now logged by DocumentTools mock
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "document_summarize_document_by_path"
            assert logged_data["success"] is True
            print("Test 7 Passed.")

            print("\nAll crypto_tool tests with live API simulation and analytics considerations completed.")

        # Ensure tests are only run when the script is executed directly
        if __name__ == "__main__":
            asyncio.run(run_crypto_tests(crypto_tools_instance))

        # Restore original requests.get and requests.post
        requests.get = original_requests_get
        requests.post = original_requests_get # Restore to original get if post was patched to get

        # Clean up dummy files and directories
        test_user_dirs = [Path("uploads") / mock_user_pro_profile.user_id, BASE_VECTOR_DIR / mock_user_pro_profile.user_id]
        for d in test_user_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print(f"Cleaned up {d}")


