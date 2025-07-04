# domain_tools/news_tools/news_tool.py

import logging
from typing import List, Dict
from datetime import datetime, timedelta
from langchain_core.tools import tool

# Import config_manager for API keys
from config.config_manager import config_manager
# Import user_manager for RBAC checks within the tool
from utils.user_manager import get_user_tier_capability

logger = logging.getLogger(__name__)

@tool
def get_general_news(query: str, user_token: str) -> List[Dict[str, str]]:
    """
    Fetches general news articles based on a query.
    This tool requires 'news_tool_access' capability.

    Args:
        query (str): The search query for news articles (e.g., "technology", "global economy").
        user_token (str): The user's authentication token for RBAC checks.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each representing a news article
                               with 'title', 'description', and 'url'.
                               Returns an empty list or an error message if access is denied
                               or news cannot be fetched.
    """
    logger.info(f"Attempting to fetch general news for query: '{query}' for user: {user_token}")

    # RBAC Check: Ensure the user has 'news_tool_access' capability
    if not get_user_tier_capability(user_token, 'news_tool_access', False):
        logger.warning(f"User {user_token} attempted to use 'get_general_news' without 'news_tool_access' capability.")
        return {"error": "Access Denied: You do not have permission to access news tools."}

    # In a real scenario, you would use a news API here (e.g., NewsAPI, GNews API).
    # For now, we'll use mock data.
    # news_api_key = config_manager.get_secret("news_api_key")
    # if not news_api_key:
    #     logger.error("News API key not found in secrets for get_general_news.")
    #     return {"error": "News API key is not configured."}

    # Example of how a real API call might look (commented out):
    # try:
    #     url = f"https://newsapi.org/v2/everything?q={query}&apiKey={news_api_key}&pageSize=5"
    #     response = requests.get(url)
    #     response.raise_for_status() # Raise an exception for bad status codes
    #     data = response.json()
    #     articles = []
    #     for article in data.get('articles', []):
    #         articles.append({
    #             "title": article.get('title'),
    #             "description": article.get('description'),
    #             "url": article.get('url')
    #         })
    #     logger.info(f"Successfully fetched {len(articles)} news articles for query: '{query}'.")
    #     return articles
    # except requests.exceptions.RequestException as e:
    #     logger.error(f"Error fetching news for query '{query}': {e}")
    #     return {"error": f"Failed to fetch news: {e}"}
    # except Exception as e:
    #     logger.error(f"An unexpected error occurred while fetching news for query '{query}': {e}")
    #     return {"error": f"An unexpected error occurred: {e}"}

    # --- Mock Data for Demonstration ---
    mock_news_data = {
        "technology": [
            {"title": "New AI Breakthrough in Robotics", "description": "Researchers announce a significant leap in AI-powered robotics.", "url": "http://example.com/ai-robotics"},
            {"title": "Quantum Computing Progress", "description": "Latest developments in quantum computing hardware.", "url": "http://example.com/quantum-progress"},
            {"title": "Cybersecurity Trends for 2025", "description": "Experts predict major shifts in cybersecurity threats and defenses.", "url": "http://example.com/cybersecurity-trends"}
        ],
        "global economy": [
            {"title": "Inflation Concerns Rise Globally", "description": "Central banks grapple with persistent inflationary pressures.", "url": "http://example.com/global-inflation"},
            {"title": "Emerging Markets Show Resilience", "description": "Despite headwinds, some emerging economies demonstrate strong growth.", "url": "http://example.com/emerging-markets"},
            {"title": "Oil Prices Fluctuate Amid Geopolitical Tensions", "description": "Geopolitical events continue to impact global oil markets.", "url": "http://example.com/oil-prices"}
        ],
        "health": [
            {"title": "New Vaccine Shows Promising Results", "description": "Clinical trials for a novel vaccine yield positive outcomes.", "url": "http://example.com/new-vaccine"},
            {"title": "Mental Health Awareness Campaigns", "description": "Governments launch initiatives to promote mental well-being.", "url": "http://example.com/mental-health"}
        ]
    }

    normalized_query = query.lower()
    for key, articles in mock_news_data.items():
        if normalized_query in key or key in normalized_query:
            logger.info(f"Returning mock news data for query: '{query}'.")
            return articles

    logger.warning(f"No mock news data found for query: '{query}'.")
    return {"message": f"No news found for '{query}'. Try 'technology', 'global economy', or 'health'."}

# Example Usage (for testing the tool directly)
if __name__ == "__main__":
    # This block is for direct testing of the tool.
    # In a real application, this tool would be called by the LLM agent.

    # Mock user token for testing purposes. In a real scenario, this would come from authentication.
    # Ensure 'news_tool_access' is enabled for this mock user in your test setup or config.
    mock_user_token_with_access = "test_user_with_news_access"
    mock_user_token_without_access = "test_user_without_news_access"

    # You might need to temporarily modify user_manager or config to simulate capabilities
    # For a quick test, let's assume the capability is hardcoded or mocked in user_manager for __main__
    # (This is not how it works in the actual app, where it reads from Firestore/config)
    
    print("\n--- Testing with access ---")
    news_tech = get_general_news("technology", mock_user_token_with_access)
    print(f"Technology News: {news_tech}")

    news_economy = get_general_news("global economy", mock_user_token_with_access)
    print(f"Global Economy News: {news_economy}")

    news_health = get_general_news("health", mock_user_token_with_access)
    print(f"Health News: {news_health}")

    news_unknown = get_general_news("sports", mock_user_token_with_access)
    print(f"Sports News (expected mock fail): {news_unknown}")

    print("\n--- Testing without access ---")
    news_denied = get_general_news("technology", mock_user_token_without_access)
    print(f"News with denied access: {news_denied}")

    # To make the __main__ block work for RBAC, you'd typically mock get_user_tier_capability
    # For a simple test, you can temporarily add a print statement inside get_user_tier_capability
    # or ensure your test user has the capability defined in a test config.
    # For the purpose of this code generation, we assume the user_manager will correctly
    # return the capability based on the `user_token`.
