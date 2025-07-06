# domain_tools/news_tools/__init__.py

import logging
from typing import Any, Optional # <--- ENSURE THIS LINE IS PRESENT

# Import individual tool functions from the news_tool module
from .news_tool import (
    get_top_headlines,
    search_news,
    news_search_web,
    news_query_uploaded_docs,
    news_summarize_document_by_path
)

logger = logging.getLogger(__name__)

class NewsTools:
    """
    A collection of news-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: Any):
        """
        Initializes the NewsTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
            document_tools (Any): The DocumentTools instance for document querying.
        """
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools
        logger.info("NewsTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_token.

    async def get_top_headlines(self, category: Optional[str] = None, country: Optional[str] = None, user_token: str = "default") -> str:
        """
        Retrieves the top news headlines. Can be filtered by category and country.
        """
        return await get_top_headlines(category=category, country=country, user_token=user_token)

    async def search_news(self, query: str, from_date: Optional[str] = None, to_date: Optional[str] = None, language: str = "en", user_token: str = "default") -> str:
        """
        Searches for news articles matching a specific query. Can filter by date range and language.
        """
        return await search_news(query=query, from_date=from_date, to_date=to_date, language=language, user_token=user_token)

    async def news_search_web(self, query: str, user_token: str = "default", max_chars: int = 2000) -> str:
        """
        Searches the web for general news-related information.
        """
        return await news_search_web(query=query, user_token=user_token, max_chars=max_chars)

    async def news_query_uploaded_docs(self, query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed news documents for a user using vector similarity search.
        """
        # This now calls the DocumentTools instance
        return await self.document_tools.document_query_uploaded_docs(
            query=query,
            user_token=user_token,
            section="news", # Specific collection for news documents
            export=export,
            k=k
        )

    async def news_summarize_document_by_path(self, file_path_str: str) -> str:
        """
        Summarizes a document related to news or current events located at the given file path.
        """
        # This now calls the DocumentTools instance
        return await self.document_tools.document_summarize_document_by_path(file_path_str=file_path_str)

    def get_tools(self):
        """
        Returns a list of tool functions exposed by this class.
        """
        return [
            self.get_top_headlines,
            self.search_news,
            self.news_search_web,
            self.news_query_uploaded_docs,
            self.news_summarize_document_by_path
        ]
