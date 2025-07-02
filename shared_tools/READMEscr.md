Implementing shared_tools/ Directory - Part 3: scraper_tool.py
We will now review and update shared_tools/scraper_tool.py to ensure it adheres to our new architectural standards, including using config_manager for all configurations and integrating RBAC checks for capabilities like max_chars and max_search_results.

1. shared_tools/scraper_tool.py (Updated for RBAC and ConfigManager)
shared_tools/scraper_tool.py (Updated for RBAC and ConfigManager)
Jul 2, 7:49 AM

Open

Key Changes in shared_tools/scraper_tool.py:

RBAC for Capabilities:

max_chars: If not explicitly provided, the max_chars for the snippet is now determined by get_user_tier_capability(user_token, 'web_search_max_chars', default_value).

max_results_allowed: A new variable max_results_allowed is introduced, also determined by get_user_tier_capability(user_token, 'web_search_max_results', default_value). This limits the number of search results retrieved from APIs or scraped directly, based on the user's tier.

config_manager Usage:

user_agent and timeout_seconds are now consistently retrieved from config_manager.get().

_get_search_api_key helper function uses config_manager.get_secret() to retrieve API keys for SerpAPI and Google Custom Search.

The tool now iterates through api_configs defined in data/config.yml to dynamically check for available search APIs (e.g., SerpAPI, Google Custom Search) and attempts to use them first. This makes the search API preference configurable.

Improved Fallback: The logic explicitly tries configured search APIs first. If they fail or no API key is available, it gracefully falls back to direct Google Search scraping.

Robust Error Handling: More specific try-except blocks are used to catch requests.exceptions.RequestException and general Exception during both API calls and direct scraping.

Test Suite Enhancement: The if __name__ == "__main__": block has been significantly expanded to include:

Comprehensive mocks for st.secrets, config_manager, and user_manager to ensure consistent and isolated testing.

Mocks for requests.get to simulate responses from SerpAPI, Google Custom Search, and direct scraping, allowing for testing the fallback mechanism without actual external calls.

Tests for different user tiers (Pro, Premium, Free, Admin) to verify RBAC max_chars and max_results limits.

Tests for invalid queries, missing API keys, and simulated network errors.

