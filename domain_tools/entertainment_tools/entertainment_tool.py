# domain_tools/entertainment_tools/entertainment_tool.py

import logging
import requests
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta, timezone # Import timezone for consistent datetime objects

# Import generic tools
from langchain_core.tools import tool
from shared_tools.scrapper_tool import scrape_web # This tool is a standalone function

# Import config_manager to access API configurations and secrets
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability
# Import date_parser for date format flexibility
from utils.date_parser import parse_date_to_yyyymmdd

# Import analytics_tracker (for logging failures within _make_dynamic_api_request)
from utils import analytics_tracker

# Import UserProfile for type hinting
from backend.models.user_models import UserProfile

# Import DocumentTools for wrapping document related tools
from domain_tools.document_tools.document_tool import DocumentTools


logger = logging.getLogger(__name__)

# --- Generic API Request Helper (copied for standalone tool file, ideally in shared utils) ---
# This helper is designed to work with the structure defined in api_providers.yml

async def make_api_request(
    provider_name: str,
    function_name: str,
    params: Dict[str, Any],
    user_api_keys: List[str],
    domain: str,
    user_id: str = "default_user",
    additional_headers: Optional[Dict[str, str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Makes a dynamic API request based on the provider configuration from config_manager.
    Handles API key injection, URL construction, and response parsing.
    """
    provider_config = config_manager.get_api_provider_config(domain, provider_name)
    if not provider_config:
        logger.error(f"Provider config not found for domain: {domain}, provider: {provider_name}")
        analytics_tracker.log_event(user_id, "api_request_failed", "config_missing",
                                    {"domain": domain, "provider": provider_name, "function": function_name}, success=False)
        return None

    base_url = provider_config.get("base_url")
    if not base_url:
        logger.error(f"Base URL not found for provider: {provider_name}")
        analytics_tracker.log_event(user_id, "api_request_failed", "base_url_missing",
                                    {"domain": domain, "provider": provider_name, "function": function_name}, success=False)
        return None

    function_config = provider_config.get("functions", {}).get(function_name)
    if not function_config:
        logger.error(f"Function config not found for function: {function_name} in provider: {provider_name}")
        analytics_tracker.log_event(user_id, "api_request_failed", "function_config_missing",
                                    {"domain": domain, "provider": provider_name, "function": function_name}, success=False)
        return None

    endpoint = function_config.get("endpoint", "")
    method = function_config.get("method", "GET").upper()
    api_key_name = provider_config.get("api_key_name")
    api_key_param_name = provider_config.get("api_key_param_name")
    response_path = function_config.get("response_path", [])

    # Prepare request parameters
    request_params = {k: v for k, v in params.items() if k in function_config.get("required_params", []) or k in function_config.get("optional_params", [])}

    # Inject API key
    api_key_value = None
    if user_api_keys:
        # Prioritize user-provided keys if they match the required api_key_name
        for key_dict in user_api_keys:
            if key_dict.get("name") == api_key_name:
                api_key_value = key_dict.get("value")
                break
    
    if not api_key_value:
        # Fallback to backend secrets if not provided by user or not found in user_api_keys
        api_key_value = config_manager.get_secret(api_key_name)
    
    if api_key_value and api_key_param_name:
        request_params[api_key_param_name] = api_key_value
    
    # Construct URL, handling path parameters
    full_url = base_url + endpoint
    if function_config.get("path_params"):
        for param in function_config["path_params"]:
            if param in request_params:
                full_url = full_url.replace(f"{{{param}}}", str(request_params.pop(param))) # Remove from query params
            else:
                logger.warning(f"Missing path parameter '{param}' for {function_name} in {provider_name}. URL might be malformed.")

    headers = additional_headers or {}
    # Default to application/json for POST if not specified
    if method == "POST" and "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"

    try:
        response = None
        if method == "GET":
            response = requests.get(full_url, params=request_params, headers=headers, timeout=config_manager.get("web_scraping.timeout_seconds", 30))
        elif method == "POST":
            response = requests.post(full_url, json=request_params, headers=headers, timeout=config_manager.get("web_scraping.timeout_seconds", 30))
        else:
            logger.error(f"Unsupported HTTP method: {method}")
            analytics_tracker.log_event(user_id, "api_request_failed", "unsupported_method",
                                        {"domain": domain, "provider": provider_name, "function": function_name, "method": method}, success=False)
            return None

        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()

        # Navigate response path
        result = data
        for key in response_path:
            if isinstance(result, dict) and key in result:
                result = result[key]
            elif isinstance(result, list) and isinstance(key, int) and len(result) > key:
                result = result[key]
            else:
                logger.warning(f"Could not navigate to response_path {response_path} for {function_name}. Returning full data.")
                result = data
                break
        
        analytics_tracker.log_event(user_id, "api_request_success", "api_call",
                                    {"domain": domain, "provider": provider_name, "function": function_name, "status_code": response.status_code}, success=True)
        return result

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error for {provider_name} {function_name}: {e.response.status_code} - {e.response.text}")
        analytics_tracker.log_event(user_id, "api_request_failed", "http_error",
                                    {"domain": domain, "provider": provider_name, "function": function_name, "status_code": e.response.status_code, "error": str(e)}, success=False)
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error for {provider_name} {function_name}: {e}")
        analytics_tracker.log_event(user_id, "api_request_failed", "connection_error",
                                    {"domain": domain, "provider": provider_name, "function": function_name, "error": str(e)}, success=False)
        return None
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout error for {provider_name} {function_name}: {e}")
        analytics_tracker.log_event(user_id, "api_request_failed", "timeout_error",
                                    {"domain": domain, "provider": provider_name, "function": function_name, "error": str(e)}, success=False)
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during API request to {provider_name} {function_name}: {e}")
        analytics_tracker.log_event(user_id, "api_request_failed", "request_error",
                                    {"domain": domain, "provider": provider_name, "function": function_name, "error": str(e)}, success=False)
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for {provider_name} {function_name}: {e}. Response: {response.text if response else 'N/A'}")
        analytics_tracker.log_event(user_id, "api_request_failed", "json_decode_error",
                                    {"domain": domain, "provider": provider_name, "function": function_name, "error": str(e)}, success=False)
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred during API request to {provider_name} {function_name}.")
        analytics_tracker.log_event(user_id, "api_request_failed", "unexpected_error",
                                    {"domain": domain, "provider": provider_name, "function": function_name, "error": str(e)}, success=False)
        return None


# --- Standalone Tool Functions ---

@tool
async def search_movies(
    query: str,
    user_context: Optional[UserProfile] = None,
    provider: str = "tmdb", # Example provider
    user_api_keys: List[str] = []
) -> str:
    """
    Searches for movies based on a query.
    This tool integrates with movie database APIs.
    Requires 'entertainment_tool_access' capability.

    Args:
        query (str): The movie search query (e.g., "Inception", "action movies 2023").
        user_context (UserProfile, optional): The user's profile for RBAC checks. Defaults to None.
        provider (str, optional): The API provider to use. Defaults to "tmdb" (example).
        user_api_keys (list, optional): List of user-provided API keys (e.g., from Streamlit secrets).

    Returns:
        str: A JSON string containing the search results, or an error message.
    """
    if user_context is None:
        user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

    logger.info(f"Tool: search_movies called for query: '{query}', provider: '{provider}', user: '{user_context.user_id}'")

    if not get_user_tier_capability(user_context.user_id, 'entertainment_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "permission_denied",
                                    {"tool_name": "search_movies", "query": query, "provider": provider}, success=False)
        return "Error: Access to entertainment tools is not enabled for your current tier."
    
    params = {"query": query}
    
    api_data = await make_api_request(
        provider_name=provider,
        function_name="search_movies",
        params=params,
        user_api_keys=user_api_keys,
        domain="entertainment",
        user_id=user_context.user_id
    )

    if api_data:
        results = []
        for item in api_data.get("results", [])[:3]: # Limit to top 3 for brevity
            title = item.get("title", "N/A")
            release_date = item.get("release_date", "N/A")
            overview = item.get("overview", "N/A")
            results.append(f"- **{title}** ({release_date})\n  {overview[:150]}...\n") # Truncate overview
        
        if results:
            result_str = f"Movie search results for '{query}':\n" + "\n".join(results)
            analytics_tracker.log_event(user_context.user_id, "tool_usage", "success",
                                        {"tool_name": "search_movies", "query": query, "provider": provider, "num_results": len(api_data.get("results", []))}, success=True)
            return result_str
        else:
            analytics_tracker.log_event(user_context.user_id, "tool_usage", "no_data",
                                        {"tool_name": "search_movies", "query": query, "provider": provider, "message": "No movies found."}, success=False)
            return f"No movies found for query: '{query}'."
    else:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "api_error",
                                    {"tool_name": "search_movies", "query": query, "provider": provider, "message": "API call failed."}, success=False)
        return "Error: Could not search movies at this time. Please try again later."


@tool
async def search_tv_shows(
    query: str,
    user_context: Optional[UserProfile] = None,
    provider: str = "tmdb", # Example provider
    user_api_keys: List[str] = []
) -> str:
    """
    Searches for TV shows based on a query.
    This tool integrates with TV show database APIs.
    Requires 'entertainment_tool_access' capability.

    Args:
        query (str): The TV show search query (e.g., "The Crown", "sci-fi series").
        user_context (UserProfile, optional): The user's profile for RBAC checks. Defaults to None.
        provider (str, optional): The API provider to use. Defaults to "tmdb" (example).
        user_api_keys (list, optional): List of user-provided API keys (e.g., from Streamlit secrets).

    Returns:
        str: A JSON string containing the search results, or an error message.
    """
    if user_context is None:
        user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

    logger.info(f"Tool: search_tv_shows called for query: '{query}', provider: '{provider}', user: '{user_context.user_id}'")

    if not get_user_tier_capability(user_context.user_id, 'entertainment_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "permission_denied",
                                    {"tool_name": "search_tv_shows", "query": query, "provider": provider}, success=False)
        return "Error: Access to entertainment tools is not enabled for your current tier."
    
    params = {"query": query}
    
    api_data = await make_api_request(
        provider_name=provider,
        function_name="search_tv_shows",
        params=params,
        user_api_keys=user_api_keys,
        domain="entertainment",
        user_id=user_context.user_id
    )

    if api_data:
        results = []
        for item in api_data.get("results", [])[:3]: # Limit to top 3 for brevity
            name = item.get("name", "N/A")
            first_air_date = item.get("first_air_date", "N/A")
            overview = item.get("overview", "N/A")
            results.append(f"- **{name}** (First Air Date: {first_air_date})\n  {overview[:150]}...\n") # Truncate overview
        
        if results:
            result_str = f"TV Show search results for '{query}':\n" + "\n".join(results)
            analytics_tracker.log_event(user_context.user_id, "tool_usage", "success",
                                        {"tool_name": "search_tv_shows", "query": query, "provider": provider, "num_results": len(api_data.get("results", []))}, success=True)
            return result_str
        else:
            analytics_tracker.log_event(user_context.user_id, "tool_usage", "no_data",
                                        {"tool_name": "search_tv_shows", "query": query, "provider": provider, "message": "No TV shows found."}, success=False)
            return f"No TV shows found for query: '{query}'."
    else:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "api_error",
                                    {"tool_name": "search_tv_shows", "query": query, "provider": provider, "message": "API call failed."}, success=False)
        return "Error: Could not search TV shows at this time. Please try again later."


@tool
async def entertainment_search_web(
    query: str,
    user_context: Optional[UserProfile] = None,
    max_chars: int = 2000
) -> str:
    """
    Searches the web for general entertainment-related information using a smart search fallback mechanism.
    This tool wraps the generic `scrape_web` tool, providing an entertainment-specific interface.
    Requires 'web_search_enabled' capability.
    
    Args:
        query (str): The entertainment-related search query (e.g., "new movie releases", "best sci-fi books").
        user_context (UserProfile, optional): The user's profile for RBAC checks and logging. Defaults to None.
        max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
    
    Returns:
        str: A string containing relevant information from the web.
    """
    if user_context is None:
        user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

    logger.info(f"Tool: entertainment_search_web called with query: '{query}' for user: '{user_context.user_id}'")
    
    if not get_user_tier_capability(user_context.user_id, 'web_search_enabled', False, user_tier=user_context.tier, user_roles=user_context.roles):
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "permission_denied",
                                    {"tool_name": "entertainment_search_web", "query": query}, success=False)
        return "Error: Web search is not enabled for your current tier."

    try:
        # Call the standalone scrape_web function
        result = await scrape_web(query=query, user_context=user_context, max_chars=max_chars)
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "success",
                                    {"tool_name": "entertainment_search_web", "query": query, "result_length": len(result)}, success=True)
        return result
    except Exception as e:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "error",
                                    {"tool_name": "entertainment_search_web", "query": query, "error": str(e)}, success=False)
        return f"Error during entertainment web search: {e}"


@tool
async def entertainment_query_uploaded_docs(
    query: str,
    user_context: Optional[UserProfile] = None,
    export: Optional[bool] = False,
    k: int = 5,
    document_tools: Optional[DocumentTools] = None # Accept DocumentTools instance
) -> str:
    """
    Queries previously uploaded and indexed entertainment documents for a user using vector similarity search.
    This tool wraps the generic `DocumentTools.document_query_uploaded_docs` tool, fixing the section to "entertainment".
    Requires 'document_query_enabled' capability.
    
    Args:
        query (str): The search query to find relevant entertainment documents (e.g., "movie script analysis", "fan fiction plot summary").
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

    logger.info(f"Tool: entertainment_query_uploaded_docs called with query: '{query}' for user: '{user_context.user_id}'")
    
    if not get_user_tier_capability(user_context.user_id, 'document_query_enabled', False, user_tier=user_context.tier, user_roles=user_context.roles):
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "permission_denied",
                                    {"tool_name": "entertainment_query_uploaded_docs", "query": query}, success=False)
        return "Error: Document querying is not enabled for your current tier."
    
    if not document_tools:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "error",
                                    {"tool_name": "entertainment_query_uploaded_docs", "query": query, "error": "DocumentTools instance not provided."}, success=False)
        return "Error: Document tools are not initialized. Cannot query uploaded documents."

    try:
        result = await document_tools.document_query_uploaded_docs(
            query_text=query, # Using query_text as per DocumentTools signature
            user_context=user_context,
            section="entertainment",
            export=export,
            k=k
        )
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "success",
                                    {"tool_name": "entertainment_query_uploaded_docs", "query": query, "result_length": len(result)}, success=True)
        return result
    except Exception as e:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "error",
                                    {"tool_name": "entertainment_query_uploaded_docs", "query": query, "error": str(e)}, success=False)
        return f"Error querying uploaded entertainment documents: {e}"


@tool
async def entertainment_summarize_document_by_path(
    file_path_str: str,
    user_context: Optional[UserProfile] = None,
    document_tools: Optional[DocumentTools] = None # Accept DocumentTools instance
) -> str:
    """
    Summarizes a document related to entertainment located at the given file path.
    This tool wraps the generic `DocumentTools.document_summarize_document_by_path` tool.
    Requires 'summarization_enabled' capability.
    
    Args:
        file_path_str (str): The full path to the document file to be summarized.
                                Example: "uploads/default/entertainment/movie_review.pdf"
        user_context (UserProfile, optional): The user's profile for RBAC checks and logging. Defaults to None.
        document_tools (DocumentTools, optional): The DocumentTools instance. Required for this function.
        
    Returns:
        str: A concise summary of the document content.
    """
    if user_context is None:
        user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

    logger.info(f"Tool: entertainment_summarize_document_by_path called for file: '{file_path_str}' by user: '{user_context.user_id}'")
    
    if not get_user_tier_capability(user_context.user_id, 'summarization_enabled', False, user_tier=user_context.tier, user_roles=user_context.roles):
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "permission_denied",
                                    {"tool_name": "entertainment_summarize_document_by_path", "file_path": file_path_str}, success=False)
        return "Error: Document summarization is not enabled for your current tier."

    if not document_tools:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "error",
                                    {"tool_name": "entertainment_summarize_document_by_path", "file_path": file_path_str, "error": "DocumentTools instance not provided."}, success=False)
        return "Error: Document tools are not initialized. Cannot summarize documents."

    try:
        result = await document_tools.document_summarize_document_by_path(
            file_path_str=file_path_str,
            user_context=user_context
        )
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "success",
                                    {"tool_name": "entertainment_summarize_document_by_path", "file_path": file_path_str, "result_length": len(result)}, success=True)
        return result
    except Exception as e:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "error",
                                    {"tool_name": "entertainment_summarize_document_by_path", "file_path": file_path_str, "error": str(e)}, success=False)
        return f"Error summarizing document: {e}"


# --- EntertainmentTools Class (Wrapper) ---
class EntertainmentTools:
    """
    A collection of tools for entertainment-related operations.
    This class acts primarily as a wrapper to expose the standalone tool functions
    as methods, ensuring a consistent interface.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: DocumentTools):
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools
        logger.info("EntertainmentTools initialized.")

    async def search_movies(
        self,
        query: str,
        user_context: Optional[UserProfile] = None,
        provider: str = "tmdb",
        user_api_keys: List[str] = []
    ) -> str:
        """
        Searches for movies.
        """
        return await search_movies(
            query=query,
            user_context=user_context,
            provider=provider,
            user_api_keys=user_api_keys
        )

    async def search_tv_shows(
        self,
        query: str,
        user_context: Optional[UserProfile] = None,
        provider: str = "tmdb",
        user_api_keys: List[str] = []
    ) -> str:
        """
        Searches for TV shows.
        """
        return await search_tv_shows(
            query=query,
            user_context=user_context,
            provider=provider,
            user_api_keys=user_api_keys
        )

    async def entertainment_search_web(
        self,
        query: str,
        user_context: Optional[UserProfile] = None,
        max_chars: int = 2000
    ) -> str:
        """
        Searches the web for general entertainment-related information.
        """
        return await entertainment_search_web(
            query=query,
            user_context=user_context,
            max_chars=max_chars
        )

    async def entertainment_query_uploaded_docs(
        self,
        query: str,
        user_context: Optional[UserProfile] = None,
        export: Optional[bool] = False,
        k: int = 5
    ) -> str:
        """
        Queries previously uploaded and indexed entertainment documents for a user.
        """
        return await entertainment_query_uploaded_docs(
            query=query,
            user_context=user_context,
            export=export,
            k=k,
            document_tools=self.document_tools
        )

    async def entertainment_summarize_document_by_path(
        self,
        file_path_str: str,
        user_context: Optional[UserProfile] = None
    ) -> str:
        """
        Summarizes a document related to entertainment located at the given file path.
        """
        return await entertainment_summarize_document_by_path(
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
        from shared_tools.scrapper_tool import scrape_web # For patching scrape_web
    except ImportError:
        async def scrape_web(*args, **kwargs): return "Mocked web search results."

    # Mock UserProfile
    mock_user_free_profile = UserProfile(user_id="mock_free_token", username="FreeUser", email="free@example.com", tier="free", roles=["user"])
    mock_user_pro_profile = UserProfile(user_id="mock_pro_token", username="ProUser", email="pro@example.com", tier="pro", roles=["user"])

    logging.basicConfig(level=logging.INFO)

    class MockSecrets:
        def __init__(self):
            self.tmdb_api_key = "MOCK_TMDB_KEY" # Placeholder
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
                    'entertainment': 'tmdb',
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
                "entertainment": {
                    "tmdb": {
                        "base_url": "https://api.themoviedb.org/3",
                        "api_key_name": "tmdb_api_key",
                        "api_key_param_name": "api_key",
                        "functions": {
                            "search_movies": {
                                "endpoint": "/search/movie",
                                "required_params": ["query"],
                                "optional_params": ["language"],
                                "response_path": ["results"],
                                "data_map": {}
                            },
                            "search_tv_shows": {
                                "endpoint": "/search/tv",
                                "required_params": ["query"],
                                "optional_params": ["language"],
                                "response_path": ["results"],
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
                'entertainment_tool_access': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'web_search_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
                'document_query_enabled': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
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
            if "api.themoviedb.org/3" in url:
                if "/search/movie" in url:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "results": [
                            {"title": "Mock Movie 1", "release_date": "2023-01-01", "overview": "Overview of mock movie 1."},
                            {"title": "Mock Movie 2", "release_date": "2023-02-01", "overview": "Overview of mock movie 2."}
                        ]
                    }
                    return mock_response
                elif "/search/tv" in url:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "results": [
                            {"name": "Mock TV Show 1", "first_air_date": "2022-03-01", "overview": "Overview of mock TV show 1."},
                            {"name": "Mock TV Show 2", "first_air_date": "2022-04-01", "overview": "Overview of mock TV show 2."}
                        ]
                    }
                    return mock_response
            
            if "serpapi.com/search" in url:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "organic_results": [
                        {"title": "Mock Web Search Result 1", "link": "http://example.com/web1", "snippet": f"Snippet for {params.get('q', 'entertainment')} web search result 1."},
                        {"title": "Mock Web Search Result 2", "link": "http://example.com/web2", "snippet": f"Snippet for {params.get('q', 'entertainment')} web search result 2."}
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

        entertainment_tools_instance = EntertainmentTools(
            config_manager=sys.modules['config.config_manager'].config_manager,
            log_event=analytics_tracker.log_event,
            document_tools=mock_document_tools_instance
        )

        async def run_entertainment_tests(entertainment_tools_instance):
            print("\n--- Testing entertainment_tool functions with Live API Simulation and Analytics ---")

            # Test 1: search_movies (success)
            print("\n--- Test 1: search_movies (Success) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            result_movies = await entertainment_tools_instance.search_movies(
                query="action movies",
                user_context=mock_user_pro_profile
            )
            print(f"Movie Search Result: {result_movies}")
            assert "Movie search results for 'action movies':" in result_movies
            assert "Mock Movie 1" in result_movies
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "search_movies"
            assert logged_data["success"] is True
            print("Test 1 Passed.")

            # Test 2: search_tv_shows (success)
            print("\n--- Test 2: search_tv_shows (Success) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            result_tv_shows = await entertainment_tools_instance.search_tv_shows(
                query="drama series",
                user_context=mock_user_pro_profile
            )
            print(f"TV Show Search Result: {result_tv_shows}")
            assert "TV Show search results for 'drama series':" in result_tv_shows
            assert "Mock TV Show 1" in result_tv_shows
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "search_tv_shows"
            assert logged_data["success"] is True
            print("Test 2 Passed.")

            # Test 3: entertainment_search_web (generic tool)
            print("\n--- Test 3: entertainment_search_web (Generic Tool) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            result_web_search = await entertainment_tools_instance.entertainment_search_web("top grossing films", user_context=mock_user_pro_profile)
            print(f"Web Search Result: {result_web_search[:100]}...")
            assert "Mocked web search results." in result_web_search
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "entertainment_search_web"
            assert logged_data["success"] is True
            print("Test 3 Passed.")

            # Test 4: entertainment_query_uploaded_docs (generic tool via DocumentTools)
            print("\n--- Test 4: entertainment_query_uploaded_docs (Generic Tool via DocumentTools) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            result_doc_query = await entertainment_tools_instance.entertainment_query_uploaded_docs("plot summary of new sci-fi book", user_context=mock_user_pro_profile)
            print(f"Document Query Result: {result_doc_query}")
            assert "Mocked document query for entertainment with query 'plot summary of new sci-fi book'" in result_doc_query
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "entertainment_query_uploaded_docs"
            assert logged_data["success"] is True
            print("Test 4 Passed.")

            # Test 5: entertainment_summarize_document_by_path (generic tool via DocumentTools)
            print("\n--- Test 5: entertainment_summarize_document_by_path (Generic Tool via DocumentTools) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            test_user_pro_dir = Path("uploads") / mock_user_pro_profile.user_id
            dummy_file_path = test_user_pro_dir / "entertainment" / "dummy_movie_review.pdf"
            dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_file_path.write_text("This is a dummy movie review content for testing summarization.")

            result_summarize = await entertainment_tools_instance.entertainment_summarize_document_by_path(str(dummy_file_path), user_context=mock_user_pro_profile)
            print(f"Summarize Result: {result_summarize}")
            assert "Mocked summary of uploads" in result_summarize
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "entertainment_summarize_document_by_path"
            assert logged_data["success"] is True
            print("Test 5 Passed.")

            print("\nAll entertainment_tool tests with live API simulation and analytics considerations completed.")

        if __name__ == "__main__":
            asyncio.run(run_entertainment_tests(entertainment_tools_instance))

        requests.get = original_requests_get
        requests.post = original_requests_post

        test_user_dirs = [Path("uploads") / mock_user_pro_profile.user_id, BASE_VECTOR_DIR / mock_user_pro_profile.user_id]
        for d in test_user_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print(f"Cleaned up {d}")
