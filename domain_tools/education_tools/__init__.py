# domain_tools/education_tools/__init__.py

import logging
from typing import Any, Optional # Import Optional

from .education_tool import (
    search_educational_resources,
    education_search_web,
    education_query_uploaded_docs,
    education_summarize_document_by_path
)

logger = logging.getLogger(__name__)

class EducationTools:
    """
    A collection of education-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: Any):
        """
        Initializes the EducationTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
            document_tools (Any): The DocumentTools instance for document querying and summarization.
        """
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools
        logger.info("EducationTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_token.

    async def search_educational_resources(self, query: str, resource_type: Optional[str] = None, user_token: str = "default") -> str:
        """
        Searches for educational resources based on a query, optionally filtered by resource type.
        """
        return await search_educational_resources(query=query, resource_type=resource_type, user_token=user_token)

    async def education_search_web(self, query: str, user_token: str = "default", max_chars: int = 2000) -> str:
        """
        Searches the web for general education-related information.
        """
        return await education_search_web(query=query, user_token=user_token, max_chars=max_chars)

    async def education_query_uploaded_docs(self, query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed educational documents for a user using vector similarity search.
        """
        # This now calls the DocumentTools instance
        return await self.document_tools.document_query_uploaded_docs(
            query=query,
            user_token=user_token,
            section="education", # Specific collection for education documents
            export=export,
            k=k
        )

    async def education_summarize_document_by_path(self, file_path_str: str) -> str:
        """
        Summarizes a document related to education located at the given file path.
        """
        # This now calls the DocumentTools instance
        return await self.document_tools.document_summarize_document_by_path(file_path_str=file_path_str)

    def get_tools(self):
        """
        Returns a list of tool functions exposed by this class.
        """
        return [
            self.search_educational_resources,
            self.education_search_web,
            self.education_query_uploaded_docs,
            self.education_summarize_document_by_path
        ]

