# domain_tools/education_tools/__init__.py

import logging
from typing import Optional, Dict, Any, List

# Import individual tool functions from the education_tool module
from .education_tool import (
    search_educational_courses,
    get_school_info,
    find_educational_resources # Note: This function was cut off in the provided content, assuming it exists.
)

logger = logging.getLogger(__name__)

class EducationTools:
    """
    A collection of education-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any):
        """
        Initializes the EducationTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
        """
        self.config_manager = config_manager
        self.log_event = log_event
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

