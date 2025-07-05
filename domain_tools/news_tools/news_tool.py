# domain_tools/news_tools/news_tool.py

import logging
import requests
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta

# Import generic tools
from langchain_core.tools import tool
from shared_tools.query_uploaded_docs_tool import QueryUploadedDocs
from shared_tools.scraper_tool import scrape_web
from shared_tools.doc_summarizer import summarize_document

# Import config_manager to access API configurations and secrets
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability
# Import date_parser for date format flexibility
from utils.date_parser import parse_date_to_yyyymmdd

logger = logging.getLogger(__name__)

# --- Generic API Request Helper (copied for standalone tool file, ideally in shared utils) ---

def _get_nested_value(data: Dict[str, Any], path: List[str]):
    """Helper to get a value from a nested dictionary using a list of keys."""
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and key.isdigit(): # Handle list indices
            try:
                current = current[int(key)]
            except (IndexError, ValueError):
                return None
        else:
            return None
    return current

def _make_dynamic_api_request(
    domain: str,
    function_name: str,
    params: Dict[str, Any],
    user_token: str
) -> Optional[Dict[str, Any]]:
    """
    Makes an API request to the dynamically configured provider for a given domain and function.
    Handles API key retrieval, request construction, and basic error handling.
    Returns parsed JSON data or None on failure (triggering mock fallback).
    """
    # Get the default active API provider for the domain from config.yml
    active_provider_name = config_manager.get(f"api_defaults.{domain}")
    if not active_provider_name:
        logger.error(f"No default API provider configured for domain '{domain}'.")
        return None

    # Get the full configuration for the active provider from api_providers.yml
    provider_config = config_manager.get_api_provider_config(domain, active_provider_name)
    if not provider_config:
        logger.error(f"Configuration for API provider '{active_provider_name}' in domain '{domain}' not found in api_providers.yml.")
        return None

    base_url = provider_config.get("base_url")
    api_key_name = provider_config.get("api_key_name")
    api_key = config_manager.get_secret(api_key_name) if api_key_name else None

    # Special handling for Amadeus which uses client_id and client_secret for token
    if active_provider_name == "amadeus":
        api_secret_name = provider_config.get("api_secret_name")
        api_secret = config_manager.get_secret(api_secret_name) if api_secret_name else None
        token_endpoint = provider_config.get("token_endpoint")

        if not api_key or not api_secret or not token_endpoint:
            logger.warning(f"Amadeus API credentials (client_id/secret) or token_endpoint missing. Cannot make live Amadeus call.")
            return None
        
        # Get Amadeus access token (simplified for demonstration)
        try:
            token_response = requests.post(
                token_endpoint,
                data={'grant_type': 'client_credentials', 'client_id': api_key, 'client_secret': api_secret},
                timeout=5
            )
            token_response.raise_for_status()
            access_token = token_response.json().get('access_token')
            if not access_token:
                logger.error("Failed to get Amadeus access token.")
                return None
            headers = {"Authorization": f"Bearer {access_token}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting Amadeus access token: {e}")
            return None
    else:
        headers = {} # No special headers by default

    if not base_url:
        logger.error(f"Base URL not configured for API provider '{active_provider_name}' in domain '{domain}'.")
        return None

    function_details = provider_config.get("functions", {}).get(function_name)
    if not function_details:
        logger.error(f"Function '{function_name}' not configured for API provider '{active_provider_name}' in domain '{domain}'.")
        return None

    endpoint = function_details.get("endpoint")
    function_param = function_details.get("function_param") # For Alpha Vantage style 'function' param
    path_params = function_details.get("path_params", []) # For ExchangeRate-API style path params

    if not endpoint and not function_param:
        logger.error(f"Neither 'endpoint' nor 'function_param' defined for function '{function_name}'.")
        return None

    # Construct URL
    full_url = f"{base_url}{endpoint}" if endpoint else base_url

    # Add path parameters to URL if specified
    for p_param in path_params:
        if p_param in params:
            value = str(params.pop(p_param))
            full_url = full_url.replace(f"{{{p_param}}}", value)
        else:
            logger.warning(f"Missing path parameter '{p_param}' for function '{function_name}'.")
            return None # Cannot construct URL without required path params

    # Construct query parameters
    query_params = {}
    if function_param:
        query_params["function"] = function_param # Alpha Vantage specific

    # Add API key if it's a query param (not in path or header)
    if api_key_name and active_provider_name not in ["amadeus", "exchangerate_api"]: # Amadeus handled by headers, ExchangeRate by path
        param_name_in_url = provider_config.get("api_key_param_name", api_key_name.replace("_api_key", ""))
        if api_key: # Only add if key exists
            query_params[param_name_in_url] = api_key 
    elif active_provider_name == "exchangerate_api" and api_key:
        pass # Key is a path parameter, already handled above

    for param_key in function_details.get("required_params", []) + function_details.get("optional_params", []):
        if param_key in params:
            query_params[param_key] = params[param_key]
        elif param_key in function_details.get("required_params", []):
            logger.warning(f"Missing required parameter '{param_key}' for function '{function_name}'.")
            return None # Missing required param, cannot proceed

    try:
        logger.info(f"Making API call to: {full_url} with params: {query_params}")
        response = requests.get(full_url, params=query_params, headers=headers, timeout=config_manager.get("web_scraping.timeout_seconds", 15))
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        raw_data = response.json()
        
        # Check for API-specific error messages in the response body
        if "Error Message" in raw_data: # Alpha Vantage specific
            logger.error(f"API Error from {active_provider_name}: {raw_data['Error Message']}")
            return None
        if "Note" in raw_data and "Thank you for using Alpha Vantage!" in raw_data["Note"]: # Alpha Vantage rate limit
            logger.warning(f"API rate limit hit for {active_provider_name}: {raw_data['Note']}")
            return None
        if raw_data.get("status") == "error": # NewsAPI specific
            logger.error(f"API Error from {active_provider_name}: {raw_data.get('message', 'Unknown error')}")
            return None
        if raw_data.get("Error"): # OMDBAPI specific
            logger.error(f"API Error from {active_provider_name}: {raw_data.get('Error')}")
            return None
        if raw_data.get("status") and raw_data["status"].get("error_code"): # CoinGecko error
            logger.error(f"API Error from {active_provider_name}: {raw_data['status'].get('error_message', 'Unknown CoinGecko error')}")
            return None
        if raw_data.get("result") == "error": # ExchangeRate-API error
            logger.error(f"API Error from {active_provider_name}: {raw_data.get('error-type', 'Unknown ExchangeRate-API error')}")
            return None


        # Extract data based on response_path
        data_to_map = raw_data
        response_path = function_details.get("response_path")
        if response_path:
            data_to_map = _get_nested_value(raw_data, response_path)
            if data_to_map is None:
                logger.warning(f"Response path '{'.'.join(response_path)}' not found in API response from {active_provider_name}. Raw data: {raw_data}")
                return None

        # Apply data mapping
        mapped_data = {}
        data_map = function_details.get("data_map", {})
        if isinstance(data_to_map, list): # For lists of items (e.g., news articles, historical data)
            mapped_data_list = []
            for item in data_to_map:
                mapped_item = {}
                for mapped_key, original_key_path in data_map.items():
                    if isinstance(original_key_path, list): # Handle nested paths in data_map
                        mapped_item[mapped_key] = _get_nested_value(item, original_key_path)
                    elif '.' in str(original_key_path): # Handle dot-separated paths in data_map
                        mapped_item[mapped_key] = _get_nested_value(item, original_key_path.split('.'))
                    else: # Direct key or list index
                        if isinstance(original_key_path, int) and isinstance(item, list):
                            try: mapped_item[mapped_key] = item[original_key_path]
                            except IndexError: mapped_item[mapped_key] = None
                        else:
                            mapped_item[mapped_key] = item.get(original_key_path)
                mapped_data_list.append(mapped_item)
            return {"data": mapped_data_list} # Wrap list in a dict for consistent return
        elif isinstance(data_to_map, dict) and function_name == "get_historical_stock_prices" and active_provider_name == "alphavantage":
            # Special handling for Alpha Vantage TIME_SERIES_DAILY where keys are dates
            processed_data = {}
            for date_key, values in data_to_map.items():
                mapped_values = {}
                for mapped_key, original_key_path in data_map.items():
                    if isinstance(original_key_path, list):
                        mapped_values[mapped_key] = _get_nested_value(values, original_key_path)
                    elif '.' in str(original_key_path):
                        mapped_values[mapped_key] = _get_nested_value(values, original_key_path.split('.'))
                    else:
                        mapped_values[mapped_key] = values.get(original_key_path)
                processed_data[date_key] = mapped_values
            return {"data": processed_data}
        else: # For single object responses
            # Special handling for CoinGecko simple price, where response is { "bitcoin": { "usd": 20000 } }
            if function_name == "get_crypto_price" and active_provider_name == "coingecko":
                # params will contain 'ids' and 'vs_currencies'
                crypto_id = params.get("ids", "").lower()
                currency = params.get("vs_currencies", "").lower()
                if crypto_id in raw_data and currency in raw_data[crypto_id]:
                    mapped_data["price"] = raw_data[crypto_id][currency]
                    if f"{currency}_market_cap" in raw_data[crypto_id]:
                        mapped_data["market_cap"] = raw_data[crypto_id][f"{currency}_market_cap"]
                    if f"{currency}_24hr_vol" in raw_data[crypto_id]:
                        mapped_data["vol_24hr"] = raw_data[crypto_id][f"{currency}_24hr_vol"]
                    if f"{currency}_24hr_change" in raw_data[crypto_id]:
                        mapped_data["change_24hr"] = raw_data[crypto_id][f"{currency}_24hr_change"]
                    if "last_updated_at" in raw_data[crypto_id]:
                        mapped_data["last_updated"] = raw_data[crypto_id]["last_updated_at"]
                    return mapped_data
                else:
                    logger.warning(f"CoinGecko simple price response unexpected for {crypto_id}/{currency}: {raw_data}")
                    return None
            
            for mapped_key, original_key_path in data_map.items():
                if isinstance(original_key_path, list):
                    mapped_data[mapped_key] = _get_nested_value(data_to_map, original_key_path)
                elif '.' in str(original_key_path):
                    mapped_data[mapped_key] = _get_nested_value(data_to_map, original_key_path.split('.'))
                else:
                    mapped_data[mapped_key] = data_to_map.get(original_key_path)
            return mapped_data

    except requests.exceptions.Timeout:
        logger.error(f"API request to {active_provider_name} timed out for function '{function_name}'.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making API request to {active_provider_name} for function '{function_name}': {e}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from {active_provider_name} for function '{function_name}'.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during API call to {active_provider_name} for '{function_name}': {e}", exc_info=True)
        return None


# --- Mock Data for Fallback ---
_mock_news_data = {
    "top_headlines": [
        {
            "title": "Global Markets Rally Amid Tech Sector Growth",
            "description": "Stock markets worldwide saw significant gains today, driven by strong performances in the technology sector.",
            "source": "Financial Times",
            "published_at": (datetime.now() - timedelta(hours=2)).isoformat(),
            "url": "http://example.com/mock-tech-rally"
        },
        {
            "title": "New Climate Accord Signed by Major Nations",
            "description": "Leaders from over 50 countries have signed a landmark agreement aimed at reducing carbon emissions.",
            "source": "Reuters",
            "published_at": (datetime.now() - timedelta(hours=5)).isoformat(),
            "url": "http://example.com/mock-climate-accord"
        }
    ],
    "everything_search": [
        {
            "title": "Innovations in AI Healthcare",
            "description": "Recent advancements in artificial intelligence are transforming healthcare diagnostics.",
            "source": "Tech Health Daily",
            "published_at": (datetime.now() - timedelta(days=1)).isoformat(),
            "url": "http://example.com/mock-ai-health"
        },
        {
            "title": "Impact of Interest Rate Hikes on Housing Market",
            "description": "Economists debate the long-term effects of rising interest rates on housing affordability.",
            "source": "Economy Today",
            "published_at": (datetime.now() - timedelta(days=3)).isoformat(),
            "url": "http://example.com/mock-housing-rates"
        }
    ]
}

@tool
def get_latest_news(query: Optional[str] = None, category: Optional[str] = None, country: Optional[str] = None, user_token: str = "default") -> str:
    """
    Retrieves the latest news headlines. Can filter by a specific query, category (e.g., 'business', 'technology'),
    or country (2-letter ISO code, e.g., 'us', 'gb').
    Falls back to mock data if API key is missing or API call fails.

    Args:
        query (str, optional): A keyword or phrase to search for in the article title and description.
        category (str, optional): The category to filter by (e.g., 'business', 'entertainment', 'general', 'health', 'science', 'sports', 'technology').
        country (str, optional): The 2-letter ISO country code (e.g., 'us', 'gb', 'ng').
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of news headlines, or an error/fallback message.
    """
    logger.info(f"Tool: get_latest_news called with query='{query}', category='{category}', country='{country}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'news_tool_access', False):
        return "Error: Access to news tools is not enabled for your current tier."
    
    params = {}
    if query: params["q"] = query
    if category: params["category"] = category.lower()
    if country: params["country"] = country.lower()

    # NewsAPI uses 'top-headlines' for categories/countries and 'everything' for general queries
    # We will prioritize 'top-headlines' if category or country is provided.
    function_name = "get_top_headlines"
    if query and not (category or country):
        function_name = "search_everything" # Use 'everything' endpoint for general queries

    api_data = _make_dynamic_api_request(
        "news", function_name,
        params,
        user_token
    )

    if api_data and api_data.get("data"): # 'data' key because _make_dynamic_api_request wraps lists
        articles = api_data["data"]
        if articles:
            response_str = "Latest News Headlines:\n"
            for i, article in enumerate(articles[:5]): # Limit to top 5 articles
                published_at = datetime.fromisoformat(article.get("published_at", "")).strftime("%Y-%m-%d %H:%M") if article.get("published_at") else "N/A"
                response_str += (
                    f"{i+1}. Title: {article.get('title', 'N/A')}\n"
                    f"   Source: {article.get('source', 'N/A')}\n"
                    f"   Published: {published_at}\n"
                    f"   Description: {article.get('description', 'N/A')}\n"
                    f"   URL: {article.get('url', 'N/A')}\n\n"
                )
            return response_str
        else:
            return f"No live news found for your criteria (query='{query}', category='{category}', country='{country}'). Falling back to mock data."

    # Fallback to mock data
    mock_articles = _mock_news_data.get("top_headlines") if not query else _mock_news_data.get("everything_search")
    if mock_articles:
        response_str = "Latest News Headlines (Mock Data Fallback):\n"
        for i, article in enumerate(mock_articles[:2]): # Limit mock to top 2
            published_at = datetime.fromisoformat(article.get("published_at", "")).strftime("%Y-%m-%d %H:%M") if article.get("published_at") else "N/A"
            response_str += (
                f"{i+1}. Title: {article.get('title', 'N/A')}\n"
                f"   Source: {article.get('source', 'N/A')}\n"
                f"   Published: {published_at}\n"
                f"   Description: {article.get('description', 'N/A')}\n"
                f"   URL: {article.get('url', 'N/A')}\n\n"
            )
        return response_str
    else:
        return f"News information not found for your criteria. (API/Mock Fallback Failed)"


# --- Existing Generic Tools (not directly using external APIs, but can be used in news context) ---

@tool
def news_search_web(query: str, user_token: str = "default", max_chars: int = 2000) -> str:
    """
    Searches the web for news-related information using a smart search fallback mechanism.
    This tool wraps the generic `scrape_web` tool, providing a news-specific interface.
    
    Args:
        query (str): The news-related search query (e.g., "latest political developments", "sports news today").
        user_token (str): The unique identifier for the user. Defaults to "default".
        max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
    
    Returns:
        str: A string containing relevant information from the web.
    """
    logger.info(f"Tool: news_search_web called with query: '{query}' for user: '{user_token}'")
    return scrape_web(query=query, user_token=user_token, max_chars=max_chars)

@tool
def news_query_uploaded_docs(query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
    """
    Queries previously uploaded and indexed news documents for a user using vector similarity search.
    This tool wraps the generic `QueryUploadedDocs` tool, fixing the section to "news".
    
    Args:
        query (str): The search query to find relevant news documents (e.g., "summary of recent economic reports", "details on the new policy").
        user_token (str): The unique identifier for the user. Defaults to "default".
        export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
        k (int): The number of top relevant documents to retrieve. Defaults to 5.
    
    Returns:
        str: A string containing the combined content of the relevant document chunks,
             or a message indicating no data/results found, or the export path if exported.
    """
    logger.info(f"Tool: news_query_uploaded_docs called with query: '{query}' for user: '{user_token}'")
    return QueryUploadedDocs(query=query, user_token=user_token, section="news", export=export, k=k)

@tool
def news_summarize_document_by_path(file_path_str: str) -> str:
    """
    Summarizes a document related to news or current events located at the given file path.
    The file path should be accessible by the system (e.g., in the 'uploads' directory).
    This tool wraps the generic `summarize_document` tool.
    
    Args:
        file_path_str (str): The full path to the document file to be summarized.
                              Example: "uploads/default/news/daily_briefing.pdf"
    
    Returns:
        str: A concise summary of the document content.
    """
    logger.info(f"Tool: news_summarize_document_by_path called for file: '{file_path_str}'")
    file_path = Path(file_path_str)
    if not file_path.exists():
        logger.error(f"Document not found at '{file_path_str}' for summarization.")
        return f"Error: Document not found at '{file_path_str}'."
    
    try:
        summary = summarize_document(file_path)
        return f"Summary of '{file_path.name}':\n{summary}"
    except ValueError as e:
        logger.error(f"Error summarizing document '{file_path_str}': {e}")
        return f"Error summarizing document: {e}"
    except Exception as e:
        logger.critical(f"An unexpected error occurred during summarization of '{file_path_str}': {e}", exc_info=True)
        return f"An unexpected error occurred during summarization: {e}"


# CLI Test (optional)
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch
    import shutil
    import os
    from shared_tools.vector_utils import BASE_VECTOR_DIR # For cleanup
    from shared_tools.python_interpreter_tool import python_interpreter_with_rbac # For testing REPL

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.news_api_key = "MOCK_NEWS_API_KEY"
            self.openai_api_key = "sk-mock-openai-key-12345"
            self.google_api_key = "AIzaSy-mock-google-key"
            self.firebase_config = "{}"
            self.serpapi_api_key = "MOCK_SERPAPI_KEY" # For scrape_web

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
                    'news': 'newsapi'
                }
            }
            self._api_providers_data = { # Mock api_providers_data for news
                "news": {
                    "newsapi": {
                        "base_url": "https://newsapi.org/v2",
                        "api_key_name": "news_api_key",
                        "api_key_param_name": "apiKey",
                        "functions": {
                            "get_top_headlines": {
                                "endpoint": "/top-headlines",
                                "required_params": [],
                                "optional_params": ["q", "category", "country", "sources"],
                                "response_path": ["articles"],
                                "data_map": {
                                    "title": "title",
                                    "description": "description",
                                    "url": "url",
                                    "source": "source.name", # Nested path
                                    "published_at": "publishedAt"
                                }
                            },
                            "search_everything": {
                                "endpoint": "/everything",
                                "required_params": ["q"],
                                "optional_params": ["sources", "domains", "from", "to", "language", "sort_by", "page_size", "page"],
                                "response_path": ["articles"],
                                "data_map": {
                                    "title": "title",
                                    "description": "description",
                                    "url": "url",
                                    "source": "source.name",
                                    "published_at": "publishedAt"
                                }
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


    # Mock user_manager.get_current_user and get_user_tier_capability for testing RBAC
    class MockUserManager:
        _mock_users = {
            "mock_free_token": {"user_id": "mock_free_token", "username": "FreeUser", "email": "free@example.com", "tier": "free", "roles": ["user"]},
            "mock_pro_token": {"user_id": "mock_pro_token", "username": "ProUser", "email": "pro@example.com", "tier": "pro", "roles": ["user"]},
            "mock_premium_token": {"user_id": "mock_premium_token", "username": "PremiumUser", "email": "premium@example.com", "tier": "premium", "roles": ["user"]},
            "mock_admin_token": {"user_id": "mock_admin_token", "username": "AdminUser", "email": "admin@example.com", "tier": "admin", "roles": ["user", "admin"]},
        }
        _rbac_capabilities = {
            'capabilities': {
                'news_tool_access': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'data_analysis_enabled': { # For python interpreter
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'web_search_max_results': {
                    'default': 2,
                    'tiers': {'pro': 7, 'premium': 15}
                },
                'web_search_limit_chars': {
                    'default': 500,
                    'tiers': {'pro': 3000, 'premium': 10000}
                }
            }
        }
        _tier_hierarchy = {
            "free": 0, "user": 1, "basic": 2, "pro": 3, "premium": 4, "admin": 99
        }

        def get_current_user(self) -> Dict[str, Any]:
            return getattr(self, '_current_mock_user', {})

        def get_user_tier_capability(self, user_token: Optional[str], capability_key: str, default_value: Any = None) -> Any:
            user_info = self._mock_users.get(user_token, {})
            user_id = user_info.get('user_id')
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
    
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager
    sys.modules['utils.user_manager'] = MockUserManager()
    sys.modules['utils.user_manager'].get_user_tier_capability = MockUserManager().get_user_tier_capability # Patch the function directly

    # Mock requests.get for external API calls
    original_requests_get = requests.get

    def mock_requests_get_dynamic(url, params, headers, timeout):
        # Simulate NewsAPI responses
        if "newsapi.org/v2" in url:
            if "/top-headlines" in url:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "status": "ok",
                    "totalResults": 2,
                    "articles": [
                        {
                            "source": {"id": "bbc-news", "name": "BBC News"},
                            "author": "BBC News",
                            "title": "Mock Headline 1: Global Economy Update",
                            "description": "A brief on the current state of the global economy.",
                            "url": "http://mock.news/article1",
                            "urlToImage": "http://mock.news/image1.jpg",
                            "publishedAt": (datetime.now() - timedelta(hours=1)).isoformat(),
                            "content": "Content of article 1."
                        },
                        {
                            "source": {"id": "cnn", "name": "CNN"},
                            "author": "CNN",
                            "title": "Mock Headline 2: Political Developments",
                            "description": "Key political events unfolding.",
                            "url": "http://mock.news/article2",
                            "urlToImage": "http://mock.news/image2.jpg",
                            "publishedAt": (datetime.now() - timedelta(hours=3)).isoformat(),
                            "content": "Content of article 2."
                        }
                    ]
                }
                return mock_response
            elif "/everything" in url:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "status": "ok",
                    "totalResults": 1,
                    "articles": [
                        {
                            "source": {"id": "reuters", "name": "Reuters"},
                            "author": "Reuters",
                            "title": f"Mock Search Result for '{params.get('q', 'query')}': Latest Tech Innovations",
                            "description": "Details about new tech breakthroughs.",
                            "url": "http://mock.news/search_article",
                            "urlToImage": "http://mock.news/search_image.jpg",
                            "publishedAt": (datetime.now() - timedelta(days=1)).isoformat(),
                            "content": "Content of search article."
                        }
                    ]
                }
                return mock_response
            
            # Simulate NewsAPI error
            if "invalid_api_key" in params.get("apiKey", ""):
                 mock_response = MagicMock()
                 mock_response.status_code = 401
                 mock_response.json.return_value = {"status": "error", "code": "apiKeyInvalid", "message": "Your API key is invalid or incorrect."}
                 return mock_response

        # Simulate scrape_web's internal requests.get if needed
        if "google.com/search" in url or "example.com" in url: # Mock for scrape_web
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = f"<html><body><h1>Search results for {params.get('q', 'news')}</h1><p>Some news snippet from web search.</p></body></html>"
            return mock_response

        return original_requests_get(url, params=params, headers=headers, timeout=timeout)

    requests.get = mock_requests_get_dynamic

    test_user_pro = "mock_pro_token"
    test_user_free = "mock_free_token"

    print("\n--- Testing news_tool functions ---")

    # Test get_latest_news (top headlines)
    print("\n--- Testing get_latest_news (Top Headlines) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result_headlines = get_latest_news(category="technology", country="us", user_token=test_user_pro)
    print(f"Latest Tech News (Pro User, API):\n{result_headlines[:500]}...")
    assert "Latest News Headlines:" in result_headlines
    assert "Mock Headline 1: Global Economy Update" in result_headlines # Mock returns general headlines
    print("Test 1 Passed.")

    # Test get_latest_news (search everything)
    print("\n--- Testing get_latest_news (Search Everything) ---")
    result_search = get_latest_news(query="AI in medicine", user_token=test_user_pro)
    print(f"News Search 'AI in medicine' (Pro User, API):\n{result_search[:500]}...")
    assert "Latest News Headlines:" in result_search
    assert "Mock Search Result for 'AI in medicine': Latest Tech Innovations" in result_search
    print("Test 2 Passed.")

    # Test get_latest_news (fallback)
    print("\n--- Testing get_latest_news (Fallback) ---")
    with patch('domain_tools.news_tools.news_tool._make_dynamic_api_request', return_value=None):
        result_fallback = get_latest_news(query="climate change", user_token=test_user_pro)
        print(f"News Search 'climate change' (Pro User, Fallback):\n{result_fallback[:500]}...")
        assert "Latest News Headlines (Mock Data Fallback):" in result_fallback
    print("Test 3 Passed.")

    # Test RBAC for news_tool_access (e.g., get_latest_news for free user)
    print("\n--- Testing RBAC for news_tool_access (Free User) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result_rbac_denied = get_latest_news(query="sports", user_token=test_user_free)
    print(f"News (Free User, RBAC Denied): {result_rbac_denied}")
    assert "Error: Access to news tools is not enabled for your current tier." in result_rbac_denied
    print("Test 4 Passed.")

    # Test news_search_web
    print("\n--- Testing news_search_web ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    search_web_query = "recent political news"
    search_web_result = news_search_web(search_web_query, user_token=test_user_pro)
    print(f"Web Search Result for '{search_web_query}':\n{search_web_result[:500]}...")
    assert "Search results for recent political news" in search_web_result
    print("Test 5 Passed.")

    # Test news_summarize_document_by_path (requires a dummy file)
    print("\n--- Testing news_summarize_document_by_path ---")
    dummy_upload_dir = Path("uploads") / test_user_pro / "news"
    dummy_upload_dir.mkdir(parents=True, exist_ok=True)
    dummy_file_path = dummy_upload_dir / "news_briefing.txt"
    with open(dummy_file_path, "w") as f:
        f.write("This is a sample news briefing. It covers the latest economic indicators and a new government policy.")
    
    result_summary = news_summarize_document_by_path(str(dummy_file_path))
    print(f"News Briefing Summary (Pro User): {result_summary}")
    assert "Mock summary of the provided text." in result_summary
    assert "economic indicators" in result_summary
    print("Test 6 Passed.")

    print("\nAll news_tool tests completed.")

    # Restore original requests.get
    requests.get = original_requests_get

    # Clean up dummy files and directories
    test_user_dirs = [Path("uploads") / test_user_pro, BASE_VECTOR_DIR / test_user_pro]
    for d in test_user_dirs:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            print(f"Cleaned up {d}")
