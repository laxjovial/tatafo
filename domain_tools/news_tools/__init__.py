# domain_tools/news_tools/__init__.py

import logging
from typing import Optional, Dict, Any, List

# Import individual tool functions from the news_tool module
from .news_tool import (
    get_top_headlines,
    search_news_articles,
    news_search_web,
    news_query_uploaded_docs,
    news_summarize_document_by_path # Added
)

logger = logging.getLogger(__name__)

class NewsTools:
    """
    A collection of news-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: Any): # Added document_tools
        """
        Initializes the NewsTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
            document_tools (Any): The DocumentTools instance for document querying. # Added
        """
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools # Stored
        logger.info("NewsTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_token.

    async def get_top_headlines(self, category: Optional[str] = None, country: Optional[str] = None, user_token: str = "default") -> str:
        """
        Retrieves the top news headlines.
        """
        return await get_top_headlines(category=category, country=country, user_token=user_token)

    async def search_news_articles(self, query: str, from_date: Optional[str] = None, to_date: Optional[str] = None, user_token: str = "default") -> str:
        """
        Searches for news articles matching a specific query, optionally within a date range.
        """
        return await search_news_articles(query=query, from_date=from_date, to_date=to_date, user_token=user_token)

    async def news_search_web(self, query: str, user_token: str = "default", max_chars: int = 2000) -> str:
        """
        Searches the web for general news information using a smart search fallback mechanism.
        """
        return await news_search_web(query=query, user_token=user_token, max_chars=max_chars)

    async def news_query_uploaded_docs(self, query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed news documents for a user.
        """
        # This now calls the DocumentTools instance
        return await self.document_tools.query_uploaded_docs(
            query_text=query,
            user_token=user_token,
            collection_name="news", # Specific collection for news documents
            export=export,
            k=k
        )

    async def news_summarize_document_by_path(self, file_path_str: str) -> str:
        """
        Summarizes a document related to news or current events located at the given file path.
        """
        return await news_summarize_document_by_path(file_path_str=file_path_str) # Call the function from news_tool.py

