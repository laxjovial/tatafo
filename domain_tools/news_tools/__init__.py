# domain_tools/news_tools/__init__.py

import logging
from typing import Any, Optional, List # <--- ENSURE THIS LINE IS PRESENT

# Import individual tool functions from the news_tool module
from .news_tool import (
    get_top_headlines,
    search_news,
    news_search_web,
    news_query_uploaded_docs,
    news_summarize_document_by_path
)

# Import UserProfile for type hinting
from backend.models.user_models import UserProfile # Added UserProfile import

# Import DocumentTools for type hinting in the NewsTools class
from domain_tools.document_tools.document_tool import DocumentTools


logger = logging.getLogger(__name__)

class NewsTools:
    """
    A collection of news-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: DocumentTools):
        """
        Initializes the NewsTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
            document_tools (DocumentTools): The DocumentTools instance for document querying.
        """
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools
        logger.info("NewsTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_context.

    async def get_top_headlines(self, category: Optional[str] = None, country: Optional[str] = None, user_context: Optional[UserProfile] = None, provider: str = "newsapi", user_api_keys: List[str] = []) -> str:
        """
        Retrieves the current top headlines from a news API.
        """
        return await get_top_headlines(
            category=category,
            country=country,
            user_context=user_context,
            provider=provider,
            user_api_keys=user_api_keys
        )

    async def search_news(self, query: str, from_date: Optional[str] = None, to_date: Optional[str] = None, language: str = "en", user_context: Optional[UserProfile] = None, provider: str = "newsapi", user_api_keys: List[str] = []) -> str:
        """
        Searches for news articles matching a specific query.
        """
        return await search_news(
            query=query,
            from_date=from_date,
            to_date=to_date,
            language=language,
            user_context=user_context,
            provider=provider,
            user_api_keys=user_api_keys
        )

    async def news_search_web(self, query: str, user_context: Optional[UserProfile] = None, max_chars: int = 2000) -> str:
        """
        Searches the web for general news-related information.
        """
        return await news_search_web(
            query=query,
            user_context=user_context,
            max_chars=max_chars
        )

    async def news_query_uploaded_docs(self, query: str, user_context: Optional[UserProfile] = None, export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed news documents for a user using vector similarity search.
        """
        # This now calls the standalone function, passing the DocumentTools instance
        return await news_query_uploaded_docs(
            query=query,
            user_context=user_context,
            export=export,
            k=k,
            document_tools=self.document_tools # Pass the instance
        )

    async def news_summarize_document_by_path(self, file_path_str: str, user_context: Optional[UserProfile] = None) -> str:
        """
        Summarizes a document related to news or current events located at the given file path.
        """
        # This now calls the standalone function, passing the DocumentTools instance
        return await news_summarize_document_by_path(
            file_path_str=file_path_str,
            user_context=user_context,
            document_tools=self.document_tools # Pass the instance
        )

    def get_tools(self):
        """
        Returns a list of tool functions exposed by this class.
        This is typically used for registering tools with an LLM agent.
        """
        # When registering with an LLM, you want the actual tool functions,
        # which are the standalone ones in news_tool.py.
        # This assumes the LLM integration can directly use these decorated functions.
        return [
            get_top_headlines,
            search_news,
            news_search_web,
            news_query_uploaded_docs,
            news_summarize_document_by_path
        ]
