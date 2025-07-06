# domain_tools/education_tools/__init__.py

import logging
from typing import Optional, Dict, Any, List

# Import individual tool functions from the education_tool module
from .education_tool import (
    search_educational_courses,
    get_school_info,
    find_educational_resources,
    education_search_web, # Added
    education_query_uploaded_docs, # Added
    education_summarize_document_by_path # Added
)

logger = logging.getLogger(__name__)

class EducationTools:
    """
    A collection of education-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: Any): # Added document_tools
        """
        Initializes the EducationTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
            document_tools (Any): The DocumentTools instance for document querying. # Added
        """
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools # Stored
        logger.info("EducationTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_token.

    async def search_educational_courses(self, query: str, level: Optional[str] = None, provider: Optional[str] = None, user_token: str = "default") -> str:
        """
        Searches for educational courses based on a query, optionally filtered by level and provider.
        """
        return await search_educational_courses(query=query, level=level, provider=provider, user_token=user_token)

    async def get_school_info(self, school_name: str, location: Optional[str] = None, user_token: str = "default") -> str:
        """
        Retrieves information about a specific school, college, or university.
        """
        return await get_school_info(school_name=school_name, location=location, user_token=user_token)

    async def find_educational_resources(self, topic: str, resource_type: Optional[str] = None, user_token: str = "default") -> str:
        """
        Finds educational resources (e.g., tutorials, documentaries, articles) on a specific topic.
        """
        return await find_educational_resources(topic=topic, resource_type=resource_type, user_token=user_token)

    async def education_search_web(self, query: str, user_token: str = "default", max_chars: int = 2000) -> str:
        """
        Searches the web for general educational information using a smart search fallback mechanism.
        """
        return await education_search_web(query=query, user_token=user_token, max_chars=max_chars)

    async def education_query_uploaded_docs(self, query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed educational documents for a user.
        """
        # This now calls the DocumentTools instance
        return await self.document_tools.query_uploaded_docs(
            query_text=query,
            user_token=user_token,
            collection_name="education", # Specific collection for education documents
            export=export,
            k=k
        )

    async def education_summarize_document_by_path(self, file_path_str: str) -> str:
        """
        Summarizes a document related to education or academic topics located at the given file path.
        """
        return await education_summarize_document_by_path(file_path_str=file_path_str) # Call the function from education_tool.py

