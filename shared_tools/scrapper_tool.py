# shared_tools/scraper_tool.py

import requests
import logging
from typing import Optional, List, Dict, Any
from bs4 import BeautifulSoup
import json

from langchain_core.tools import tool

# Import config_manager and user_manager for RBAC checks
from config.config_manager import config_manager
from utils.user_manager import get_user_tier_capability, get_current_user

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
    # Add other search API key retrieval logic here if needed
    return None

# --- Web Scraping Tool ---
@tool
def scrape_web(query: str, user_token: str = "default", max_chars: Optional[int] = None) -> str:
    """
    Searches the web for information using a smart search fallback mechanism.
    It attempts to use configured search APIs (like SerpAPI or Google Custom Search) first.
    If no API key is available or the API call fails, it falls back to direct web scraping
    of a general search engine (e.g., Google Search results page).

    Args:
        query (str): The search query.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks.
        max_chars (int, optional): Maximum characters for the returned snippet.
                                   If not provided, it will be determined by user's tier capability.

    Returns:
        str: A string containing relevant information from the web, or an error message.
    """
    logger.info(f"Tool: scrape_web called with query: '{query}' for user: '{user_token}'")

    # Get user's allowed max_chars from RBAC capabilities if not explicitly provided
    if max_chars is None:
        max_chars = get_user_tier_capability(user_token, 'web_search_max_chars', config_manager.get('web_scraping.max_search_results', 500))
    
    # Get max search results allowed by user's tier
    max_results_allowed = get_user_tier_capability(user_token, 'web_search_max_results', config_manager.get('web_scraping.max_search_results', 5))

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
                # Look for 'search_apis' section within each config file
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
                                        "num": min(10, max_results_allowed) # SerpAPI 'num' param
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
                                    # Requires a CX (Custom Search Engine ID)
                                    cx = config_manager.get_secret("google_custom_search_cx")
                                    if not cx:
                                        logger.warning("Google Custom Search CX not found in secrets. Skipping.")
                                        continue
                                    params = {
                                        "key": api_key,
                                        "cx": cx,
                                        "q": query,
                                        "num": min(10, max_results_allowed) # Google CSE 'num' param
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
        response.raise_for_status() # Raise an exception for HTTP errors

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Google search results often have div with class 'g' or 'tF2CMy'
        # This is a simplified selector and might need adjustment over time
        for g in soup.find_all('div', class_='g')[:max_results_allowed]:
            title_tag = g.find('h3')
            link_tag = g.find('a')
            snippet_tag = g.find('div', class_='VwiC3b') # or 'lEBKkf' or similar

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
        # Truncate snippet to max_chars
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

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.openai = {"api_key": "sk-mock-openai-key-12345"}
            self.google = {"api_key": "AIzaSy-mock-google-key"}
            self.serpapi_api_key = "MOCK_SERPAPI_KEY_123" # Mock SerpAPI key
            self.google_custom_search_api_key = "MOCK_GOOGLE_CSE_KEY_456" # Mock Google CSE key
            self.google_custom_search_cx = "MOCK_GOOGLE_CSE_CX_789" # Mock Google CSE CX
            self.user_tokens = {
                "free_user_token": "mock_free_token",
                "pro_user_token": "mock_pro_token",
                "premium_user_token": "mock_premium_token",
                "admin_user_token": "mock_admin_token"
            }
            self.firebase_config = "{}" # Mock empty config for Firebase if not set

        def get(self, key, default=None):
            parts = key.split('.')
            val = self
            for part in parts:
                if hasattr(val, part):
                    val = getattr(val, part)
                elif isinstance(val, dict) and part in val:
                    val = val[part]
                else:
                    return default
            return val
    
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
                    'timeout_seconds': 5,
                    'max_search_results': 5 # Default for config
                },
                'tiers': {}, # This will be overridden by tiers.yaml
                'default_user_tier': 'free',
                'default_user_roles': ['user'],
                'api_configs': [ # Mock API config files
                    "mock_search_apis.yaml"
                ]
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
            return st.secrets.get(key, default)

        def set_secret(self, key, value):
            # Allow setting secrets for testing purposes
            setattr(st.secrets, key, value)


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
                'web_search_enabled': {
                    'default': False,
                    'roles': {'user': True, 'basic': True, 'pro': True, 'premium': True, 'admin': True}
                },
                'web_search_max_chars': {
                    'default': 500,
                    'roles': {'basic': 1000, 'pro': 3000, 'premium': 5000, 'admin': 10000}
                },
                'web_search_max_results': { # This is a new capability, define it here for mock
                    'default': 2,
                    'roles': {'basic': 5, 'pro': 7, 'premium': 10, 'admin': 15}
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

            for role in user_roles:
                if role in capability_config.get('roles', {}):
                    return capability_config['roles'][role]
            
            return capability_config.get('default', default_value)

    # Patch the actual imports for testing
    import streamlit as st_mock
    if not hasattr(st_mock, 'secrets'):
        st_mock.secrets = MockSecrets()
    
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager
    sys.modules['utils.user_manager'] = MockUserManager()
    sys.modules['utils.user_manager']._RBAC_CAPABILITIES = MockUserManager()._rbac_capabilities
    sys.modules['utils.user_manager']._TIER_HIERARCHY = MockUserManager()._tier_hierarchy

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

    test_user_free = sys.modules['utils.user_manager']._mock_users["mock_free_token"]['user_id']
    test_user_basic = sys.modules['utils.user_manager']._mock_users["mock_free_token"]['user_id'] # Use free for basic tier tests
    test_user_pro = sys.modules['utils.user_manager']._mock_users["mock_pro_token"]['user_id']
    test_user_premium = sys.modules['utils.user_manager']._mock_users["mock_premium_token"]['user_id']
    test_user_admin = sys.modules['utils.user_manager']._mock_users["mock_admin_token"]['user_id']

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
            return MockDirectScrapeResponse(query, num_results=3) # Default to 3 for direct scrape mock
        raise requests.exceptions.RequestException(f"Unexpected URL: {url}")

    requests.get = MagicMock(side_effect=mock_requests_get_side_effect)


    print("\n--- Testing scrape_web function ---")

    # Test 1: Pro user, default max_chars and max_results
    print("\n--- Test 1: Pro user, default max_chars and max_results ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result1 = scrape_web("latest AI news", user_token=test_user_pro)
    print(f"Result for 'latest AI news' (Pro user):\n{result1[:500]}...")
    # Pro user max_chars should be 3000, max_results 7
    assert "Mock SerpAPI Result 1" in result1
    assert len(result1.split("---")) >= 1 # At least one result
    print("Test 1 Passed.")

    # Test 2: Premium user, explicit max_chars
    print("\n--- Test 2: Premium user, explicit max_chars (200) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_premium
    result2 = scrape_web("quantum computing breakthroughs", user_token=test_user_premium, max_chars=200)
    print(f"Result for 'quantum computing breakthroughs' (Premium user, max_chars=200):\n{result2[:500]}...")
    # Premium user max_chars should be 5000, but overridden to 200. Max results 10.
    assert "Mock SerpAPI Result 1" in result2
    assert len(result2.split("This is a mock snippet for SerpAPI result 1. It contains information about quantum computing breakthroughs.This is a mock snippet for SerpAPI result 1. It contains information about quantum computing breakthroughs." * 2)) == 1 # Should be truncated
    print("Test 2 Passed.")

    # Test 3: Free user, should fall back to default max_chars (500) and max_results (2)
    print("\n--- Test 3: Free user, default max_chars and max_results ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result3 = scrape_web("sustainable energy solutions", user_token=test_user_free)
    print(f"Result for 'sustainable energy solutions' (Free user):\n{result3[:500]}...")
    # Free user max_chars should be 500, max_results 2
    assert "Mock SerpAPI Result 1" in result3
    assert "Mock SerpAPI Result 3" not in result3 # Should be limited to 2 results
    print("Test 3 Passed.")

    # Test 4: Admin user, should get max capabilities
    print("\n--- Test 4: Admin user, max capabilities ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_admin
    result4 = scrape_web("space exploration future", user_token=test_user_admin)
    print(f"Result for 'space exploration future' (Admin user):\n{result4[:500]}...")
    # Admin max_chars should be 10000, max_results 15
    assert "Mock SerpAPI Result 1" in result4
    assert "Mock SerpAPI Result 10" in result4 # Should get more results
    print("Test 4 Passed.")

    # Test 5: No API key, fallback to direct scraping
    print("\n--- Test 5: No API key, fallback to direct scraping ---")
    sys.modules['streamlit'].secrets.serpapi_api_key = None # Temporarily disable SerpAPI key
    sys.modules['streamlit'].secrets.google_custom_search_api_key = None # Temporarily disable Google CSE key
    sys.modules['utils.user_manager']._current_mock_user = test_user_basic # Use basic user
    result5 = scrape_web("historical events", user_token=test_user_basic)
    print(f"Result for 'historical events' (Direct Scrape Fallback):\n{result5[:500]}...")
    assert "Mock Direct Scrape Title 1" in result5 # Should indicate direct scrape
    assert "Mock SerpAPI Result" not in result5 # Should not use SerpAPI
    print("Test 5 Passed.")

    # Restore original API keys
    sys.modules['streamlit'].secrets.serpapi_api_key = "MOCK_SERPAPI_KEY_123"
    sys.modules['streamlit'].secrets.google_custom_search_api_key = "MOCK_GOOGLE_CSE_KEY_456"

    # Test 6: Empty query
    print("\n--- Test 6: Empty query ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result6 = scrape_web("", user_token=test_user_pro)
    print(f"Result for empty query: {result6}")
    assert "No relevant information found on the web." in result6 or "Failed to perform web search" in result6
    print("Test 6 Passed.")

    # Test 7: Error during API call
    print("\n--- Test 7: Error during API call ---")
    def mock_error_requests_get(*args, **kwargs):
        raise requests.exceptions.RequestException("Simulated network error")
    requests.get = MagicMock(side_effect=mock_error_requests_get)
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result7 = scrape_web("error test", user_token=test_user_pro)
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
