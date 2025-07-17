# shared_tools/scraper_tool.py

import requests
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from bs4 import BeautifulSoup
import json
import os
import yaml

from langchain_core.tools import tool

from config.config_manager import config_manager
from utils.user_manager import get_user_tier_capability # Removed get_current_user
from backend.models.user_models import UserProfile # Import UserProfile

logger = logging.getLogger(__name__)

# --- Helper Function to get API Keys for Search Engines ---
def _get_search_api_key(api_name: str) -> Optional[str]:
    """
    Retrieves the API key for a given search API from secrets.
    """
    if api_name == "serpapi":
        return config_manager.get_secret("serpapi_api_key")
    elif api_name == "google_custom_search":
        return config_manager.get_secret("google_custom_search_api_key")
    return None

# --- Web Scraping Tool ---
@tool
def scrape_web(query: str, user_context: Optional[UserProfile] = None, max_chars: Optional[int] = None) -> str:
    """
    Searches the web for information using a smart search fallback mechanism.
    It attempts to use configured search APIs (like SerpAPI or Google Custom Search) first.
    If no API key is available or the API call fails, it falls back to direct web scraping
    of a general search engine (e.g., Google Search results page).

    Args:
        query (str): The search query.
        user_context (UserProfile, optional): The user's profile for RBAC capability checks.
                                              Defaults to None.
        max_chars (int, optional): Maximum characters for the returned snippet.
                                   If not provided, it will be determined by user's tier capability.

    Returns:
        str: A string containing relevant information from the web, or an error message.
    """
    user_id = user_context.user_id if user_context else "default"
    user_tier = user_context.tier if user_context else "default"

    logger.info(f"Tool: scrape_web called with query: '{query}' for user: '{user_id}' (tier: {user_tier})")

    if not query:
        return "Please provide a non-empty query."

    # Get user's allowed max_chars from RBAC capabilities if not explicitly provided
    if max_chars is None:
        max_chars = get_user_tier_capability(user_id, 'web_search_limit_chars', config_manager.get('web_scraping.max_search_results', 500), user_tier=user_tier)
    
    # Get max search results allowed by user's tier
    max_results_allowed = get_user_tier_capability(user_id, 'web_search_max_results', config_manager.get('web_scraping.max_search_results', 5), user_tier=user_tier)

    headers = {
        "User-Agent": config_manager.get("web_scraping.user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "DNT": "1", # Do Not Track
        "Connection": "keep-alive",
    }
    timeout = config_manager.get("web_scraping.timeout_seconds", 15)

    search_results = []

    # --- Attempt to use configured Search APIs first (e.g., SerpAPI, Google Custom Search) ---
    search_apis = config_manager.get("api_configs", [])
    for api_config_file in search_apis:
        api_path = Path(f"data/{api_config_file}")
        if not api_path.exists():
            continue
        try:
            with open(api_path, "r") as f:
                full_api_config = yaml.safe_load(f) or {}
                for api_info in full_api_config.get('search_apis', []):
                    api_name = api_info.get("name")
                    api_type = api_info.get("type")
                    if api_type == "search":
                        api_key = _get_search_api_key(api_name.lower())
                        if api_key:
                            logger.info(f"Attempting to use {api_name} for web search.")
                            try:
                                if api_name.lower() == "serpapi":
                                    params = {
                                        "api_key": api_key,
                                        "q": query,
                                        "engine": "google",
                                        "num": min(10, max_results_allowed)
                                    }
                                    response = requests.get("https://serpapi.com/search", params=params, timeout=timeout)
                                    response.raise_for_status()
                                    data = response.json()
                                    if "organic_results" in data:
                                        for res in data["organic_results"][:max_results_allowed]:
                                            search_results.append({
                                                "title": res.get("title"),
                                                "link": res.get("link"),
                                                "snippet": res.get("snippet")
                                            })
                                        if search_results:
                                            logger.info(f"Successfully fetched {len(search_results)} results from SerpAPI.")
                                            return _format_search_results(search_results, max_chars)

                                elif api_name.lower() == "google_custom_search":
                                    cx = config_manager.get_secret("google_custom_search_cx")
                                    if not cx:
                                        logger.warning("Google Custom Search CX not found in secrets. Skipping.")
                                        continue
                                    params = {
                                        "key": api_key,
                                        "cx": cx,
                                        "q": query,
                                        "num": min(10, max_results_allowed)
                                    }
                                    response = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=timeout)
                                    response.raise_for_status()
                                    data = response.json()
                                    if "items" in data:
                                        for res in data["items"][:max_results_allowed]:
                                            search_results.append({
                                                "title": res.get("title"),
                                                "link": res.get("link"),
                                                "snippet": res.get("snippet")
                                            })
                                        if search_results:
                                            logger.info(f"Successfully fetched {len(search_results)} results from Google Custom Search.")
                                            return _format_search_results(search_results, max_chars)

                            except requests.exceptions.RequestException as req_e:
                                logger.warning(f"API search with {api_name} failed: {req_e}. Falling back to direct scraping.")
                            except Exception as e:
                                logger.warning(f"Error processing {api_name} response: {e}. Falling back to direct scraping.")
        except Exception as e:
            logger.error(f"Error loading API config from {api_path}: {e}")
            continue

    # --- Fallback to direct Google Search scraping if no API works or is configured ---
    logger.info("Falling back to direct Google Search scraping.")
    try:
        search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
        response = requests.get(search_url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        
        for g in soup.find_all('div', class_='g')[:max_results_allowed]:
            title_tag = g.find('h3')
            link_tag = g.find('a')
            snippet_tag = g.find('div', class_='VwiC3b')

            title = title_tag.get_text() if title_tag else "No Title"
            link = link_tag['href'] if link_tag and 'href' in link_tag.attrs else "No Link"
            snippet = snippet_tag.get_text() if snippet_tag else "No Snippet"
            
            search_results.append({"title": title, "link": link, "snippet": snippet})

        if search_results:
            logger.info(f"Successfully scraped {len(search_results)} results from Google Search.")
            return _format_search_results(search_results, max_chars)
        else:
            logger.warning("No search results found via direct scraping.")
            return "No relevant information found on the web."

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to scrape web: {e}", exc_info=True)
        return f"Failed to perform web search due to a network error: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during web scraping: {e}", exc_info=True)
        return f"An unexpected error occurred during web search: {e}"

def _format_search_results(results: List[Dict[str, str]], max_chars: int) -> str:
    """
    Formats the list of search results into a readable string, truncating snippets.
    """
    formatted_output = []
    for i, res in enumerate(results):
        snippet = res.get("snippet", "No snippet available.")
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "..."

        formatted_output.append(
            f"Result {i+1}:\n"
            f"Title: {res.get('title', 'N/A')}\n"
            f"Link: {res.get('link', 'N/A')}\n"
            f"Snippet: {snippet}\n"
            f"---"
        )
    return "\n".join(formatted_output)

# CLI Test (optional)
if __name__ == "__main__":
    import shutil
    from unittest.mock import MagicMock
    import sys
    import yaml
    from datetime import datetime, timedelta

    logging.basicConfig(level=logging.INFO)

    # Mock classes for testing environment
    class MockSecrets:
        def __init__(self):
            self._secrets = {}

        def set_secret(self, key, value):
            self._secrets[key] = value

        def get_secret(self, key, default=None):
            return self._secrets.get(key, default)

    class MockConfigManager:
        def __init__(self):
            self._config = {
                "llm": {"temperature": 0.7, "provider": "openai", "model_name": "gpt-3.5-turbo"},
                "web_scraping": {"max_search_results": 5, "max_search_chars": 500, "user_agent": "Mozilla/5.0", "timeout_seconds": 15},
                "api_configs": ["mock_search_apis.yaml"] # Reference the dummy config
            }
        
        def get(self, key, default=None):
            keys = key.split('.')
            val = self._config
            for k in keys:
                if isinstance(val, dict) and k in val:
                    val = val[k]
                else:
                    return default
            return val

        def get_secret(self, key, default=None):
            return st_mock.secrets.get_secret(key, default)

        # These methods are for the purpose of the test setup, not for direct use by scrape_web
        def get_api_provider_config(self, domain: str, provider_name: str) -> Optional[Dict[str, Any]]:
            return None 

        def get_domain_api_providers(self, domain: str) -> Dict[str, Any]:
            return {}

    # Mock user_manager.get_user_tier_capability for testing RBAC
    _mock_users_for_test = {
        "mock_free_token": UserProfile(user_id="mock_free_token", username="FreeUser", email="free@example.com", tier="free", roles=["user"]),
        "mock_pro_token": UserProfile(user_id="mock_pro_token", username="ProUser", email="pro@example.com", tier="pro", roles=["user"]),
        "mock_premium_token": UserProfile(user_id="mock_premium_token", username="PremiumUser", email="premium@example.com", tier="premium", roles=["user"]),
        "mock_admin_token": UserProfile(user_id="mock_admin_token", username="AdminUser", email="admin@example.com", tier="admin", roles=["user", "admin"]),
    }
    _rbac_capabilities_for_test = {
        'capabilities': {
            'web_search_enabled': {
                'default': False,
                'tiers': {'free': True, 'basic': True, 'pro': True, 'premium': True, 'admin': True}
            },
            'web_search_limit_chars': {
                'default': 500,
                'tiers': {'basic': 1000, 'pro': 3000, 'premium': 5000, 'admin': float('inf')} # Admin gets inf chars
            },
            'web_search_max_results': {
                'default': 2,
                'tiers': {'basic': 5, 'pro': 7, 'premium': 10, 'admin': 15} # Admin gets max results
            }
        }
    }

    def get_user_tier_capability_mock(user_id_or_profile: Any, capability_key: str, default_value: Any = None, user_tier: Optional[str] = None, user_roles: Optional[List[str]] = None) -> Any:
        if isinstance(user_id_or_profile, UserProfile):
            user_info = user_id_or_profile
            user_tier = user_info.tier
            user_roles = user_info.roles
        elif isinstance(user_id_or_profile, str) and user_id_or_profile in _mock_users_for_test:
            user_info = _mock_users_for_test[user_id_or_profile]
            user_tier = user_info.tier
            user_roles = user_info.roles
        else:
            user_tier = "free" # Default to free tier for unknown user IDs
            user_roles = ["user"]

        if "admin" in user_roles:
            if capability_key == 'web_search_limit_chars': return float('inf')
            if capability_key == 'web_search_max_results': return 15 # Or a high number
            if capability_key == 'web_search_enabled': return True

        capability_config = _rbac_capabilities_for_test.get('capabilities', {}).get(capability_key)
        if not capability_config:
            return default_value

        if user_tier in capability_config.get('tiers', {}):
            return capability_config['tiers'][user_tier]
        
        return capability_config.get('default', default_value)


    # Patch the actual imports for testing
    import streamlit as st_mock
    if not hasattr(st_mock, 'secrets'):
        st_mock.secrets = MockSecrets()
    
    # Patch config_manager directly in its module to affect the import within scrape_web
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager

    # Patch get_user_tier_capability directly in the module it's imported from for tests
    sys.modules['utils.user_manager'].get_user_tier_capability = get_user_tier_capability_mock
    sys.modules['backend.models.user_models'].UserProfile = UserProfile # Ensure UserProfile is available for mocks

    # Setup dummy API YAML for testing search APIs
    dummy_data_dir = Path("data")
    dummy_data_dir.mkdir(exist_ok=True)
    dummy_search_apis_path = dummy_data_dir / "mock_search_apis.yaml"
    with open(dummy_search_apis_path, "w") as f:
        f.write("""
search_apis:
  - name: "SerpAPI"
    type: "search"
    endpoint: "https://serpapi.com/search"
    key_name: "api_key"
    key_value: "load_from_secrets.serpapi_api_key"
    headers: {}
    default_params:
      engine: "google"
    query_param: "q"

  - name: "Google Custom Search"
    type: "search"
    endpoint: "https://www.googleapis.com/customsearch/v1"
    key_name: "key"
    key_value: "load_from_secrets.google_custom_search_api_key"
    headers: {}
    default_params:
      cx: "load_from_secrets.google_custom_search_cx"
    query_param: "q"
""")
    print("Dummy mock_search_apis.yaml created for testing.")

    test_user_free_profile = _mock_users_for_test["mock_free_token"]
    test_user_pro_profile = _mock_users_for_test["mock_pro_token"]
    test_user_premium_profile = _mock_users_for_test["mock_premium_token"]
    test_user_admin_profile = _mock_users_for_test["mock_admin_token"]


    # Mock requests.get for external API calls
    original_requests_get = requests.get

    class MockSerpAPIResponse:
        def __init__(self, query, num_results=3):
            self.status_code = 200
            self._query = query
            self._num_results = num_results
        
        def json(self):
            results = []
            for i in range(self._num_results):
                results.append({
                    "title": f"Mock SerpAPI Result {i+1} for {self._query}",
                    "link": f"http://mockserpapi.com/result{i+1}",
                    "snippet": f"This is a mock snippet for SerpAPI result {i+1}. It contains information about {self._query}." * 2
                })
            return {"organic_results": results}
        
        def raise_for_status(self):
            pass

    class MockGoogleCSEResponse:
        def __init__(self, query, num_results=3):
            self.status_code = 200
            self._query = query
            self._num_results = num_results
        
        def json(self):
            items = []
            for i in range(self._num_results):
                items.append({
                    "title": f"Mock Google CSE Result {i+1} for {self._query}",
                    "link": f"http://mockgooglecse.com/item{i+1}",
                    "snippet": f"This is a mock snippet for Google CSE item {i+1}. It provides details about {self._query}." * 2
                })
            return {"items": items}
        
        def raise_for_status(self):
            pass

    class MockDirectScrapeResponse:
        def __init__(self, query, num_results=3):
            self.status_code = 200
            self._query = query
            self._num_results = num_results
            self.text = self._generate_html()

        def _generate_html(self):
            html_content = "<html><body>"
            for i in range(self._num_results):
                html_content += f"""
                <div class="g">
                    <div class="rc">
                        <h3 class="LC20lb DKV0Md"><a href="http://mockdirectscrape.com/page{i+1}">Mock Direct Scrape Title {i+1} for {self._query}</a></h3>
                        <div class="VwiC3b yXK7L AjY5ze fxKbKc">
                            <span>This is a mock snippet from direct scraping result {i+1}. It has details on {self._query}.</span>
                        </div>
                    </div>
                </div>
                """
            html_content += "</body></html>"
            return html_content
        
        def raise_for_status(self):
            pass

    def mock_requests_get_side_effect(url, params=None, headers=None, timeout=None):
        if "serpapi.com" in url:
            query = params.get("q", "")
            num = params.get("num", 10)
            return MockSerpAPIResponse(query, num_results=num)
        elif "googleapis.com/customsearch" in url:
            query = params.get("q", "")
            num = params.get("num", 10)
            return MockGoogleCSEResponse(query, num_results=num)
        elif "google.com/search" in url:
            query = url.split("q=")[1].split("&")[0]
            query = requests.utils.unquote(query)
            return MockDirectScrapeResponse(query, num_results=3)
        raise requests.exceptions.RequestException(f"Unexpected URL: {url}")

    requests.get = MagicMock(side_effect=mock_requests_get_side_effect)


    print("\n--- Testing scrape_web function ---")

    # Test 1: Pro user, default max_chars and max_results
    print("\n--- Test 1: Pro user, default max_chars and max_results ---")
    result1 = scrape_web("latest AI news", user_context=test_user_pro_profile)
    print(f"Result for 'latest AI news' (Pro user):\n{result1[:500]}...")
    assert "Mock SerpAPI Result 1" in result1
    assert len(result1.split("---")) >= 1
    print("Test 1 Passed.")

    # Test 2: Premium user, explicit max_chars
    print("\n--- Test 2: Premium user, explicit max_chars (200) ---")
    result2 = scrape_web("quantum computing breakthroughs", user_context=test_user_premium_profile, max_chars=200)
    print(f"Result for 'quantum computing breakthroughs' (Premium user, max_chars=200):\n{result2[:500]}...")
    assert "Mock SerpAPI Result 1" in result2
    expected_truncated_snippet_part = ("This is a mock snippet for SerpAPI result 1. It contains information about quantum computing breakthroughs." * 2)[:200]
    assert expected_truncated_snippet_part in result2
    print("Test 2 Passed.")

    # Test 3: Free user, should fall back to default max_chars (500) and max_results (2)
    print("\n--- Test 3: Free user, default max_chars and max_results ---")
    result3 = scrape_web("sustainable energy solutions", user_context=test_user_free_profile)
    print(f"Result for 'sustainable energy solutions' (Free user):\n{result3[:500]}...")
    assert "Mock SerpAPI Result 1" in result3
    assert result3.count("Result") == 2
    print("Test 3 Passed.")

    # Test 4: Admin user, should get max capabilities
    print("\n--- Test 4: Admin user, max capabilities ---")
    result4 = scrape_web("space exploration future", user_context=test_user_admin_profile)
    print(f"Result for 'space exploration future' (Admin user):\n{result4[:500]}...")
    assert "Mock SerpAPI Result 1" in result4
    assert len(result4.split("---")) >= 1
    print("Test 4 Passed.")

    # Test 5: No API key, fallback to direct scraping
    print("\n--- Test 5: No API key, fallback to direct scraping ---")
    st_mock.secrets.set_secret('serpapi_api_key', None)
    st_mock.secrets.set_secret('google_custom_search_api_key', None)
    
    result5 = scrape_web("historical events", user_context=test_user_free_profile) # Use free user for basic tier tests if no specific basic mock
    print(f"Result for 'historical events' (Direct Scrape Fallback):\n{result5[:500]}...")
    assert "Mock Direct Scrape Title 1" in result5
    assert "Mock SerpAPI Result" not in result5
    print("Test 5 Passed.")

    # Restore original API keys
    st_mock.secrets.set_secret('serpapi_api_key', "MOCK_SERPAPI_KEY_123")
    st_mock.secrets.set_secret('google_custom_search_api_key', "MOCK_GOOGLE_CSE_KEY_456")

    # Test 6: Empty query
    print("\n--- Test 6: Empty query ---")
    result6 = scrape_web("", user_context=test_user_pro_profile)
    print(f"Result for empty query: {result6}")
    assert "Please provide a non-empty query." in result6
    print("Test 6 Passed.")

    # Test 7: Error during API call
    print("\n--- Test 7: Error during API call ---")
    def mock_error_requests_get(*args, **kwargs):
        raise requests.exceptions.RequestException("Simulated network error")
    requests.get = MagicMock(side_effect=mock_error_requests_get)
    result7 = scrape_web("error test", user_context=test_user_pro_profile)
    print(f"Result for error test: {result7}")
    assert "Failed to perform web search due to a network error" in result7 or "An unexpected error occurred" in result7
    print("Test 7 Passed.")

    # Restore original requests.get
    requests.get = original_requests_get

    print("\nAll scrape_web tests passed.")

    # Clean up dummy files and directories
    if dummy_data_dir.exists():
        dummy_search_apis_path.unlink(missing_ok=True)
        if not os.listdir(dummy_data_dir):
            os.rmdir(dummy_data_dir)
