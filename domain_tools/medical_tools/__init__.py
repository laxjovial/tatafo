# domain_tools/medical_tools/__init__.py

import logging
from typing import Optional, Dict, Any, List

# Import individual tool functions from the medical_tool module
from .medical_tool import (
    get_drug_info,
    check_symptoms,
    get_hospital_info,
    medical_search_web, # Added
    medical_query_uploaded_docs, # Added
    medical_summarize_document_by_path # Added
)

logger = logging.getLogger(__name__)

class MedicalTools:
    """
    A collection of medical-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: Any): # Added document_tools
        """
        Initializes the MedicalTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
            document_tools (Any): The DocumentTools instance for document querying. # Added
        """
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools # Stored
        logger.info("MedicalTools initialized.")

    # Expose individual tool functions as methods of this class
    # These methods will simply call the underlying functions,
    # passing the required arguments including user_token.

    async def get_drug_info(self, drug_name: str, user_token: str = "default") -> str:
        """
        Retrieves information about a specific drug, including its uses, side effects, and dosage.
        """
        return await get_drug_info(drug_name=drug_name, user_token=user_token)

    async def check_symptoms(self, symptoms: List[str], user_token: str = "default") -> str:
        """
        Checks a list of symptoms and suggests possible medical conditions and recommendations.
        """
        return await check_symptoms(symptoms=symptoms, user_token=user_token)

    async def get_hospital_info(self, hospital_name: str, location: Optional[str] = None, user_token: str = "default") -> str:
        """
        Retrieves information about a specific hospital or medical center.
        """
        return await get_hospital_info(hospital_name=hospital_name, location=location, user_token=user_token)

    async def medical_search_web(self, query: str, user_token: str = "default", max_chars: int = 2000) -> str:
        """
        Searches the web for medical or health-related information.
        """
        return await medical_search_web(query=query, user_token=user_token, max_chars=max_chars) # Call the function from medical_tool.py

    async def medical_query_uploaded_docs(self, query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed medical documents for a user using vector similarity search.
        """
        # This now calls the DocumentTools instance
        return await self.document_tools.query_uploaded_docs(
            query_text=query,
            user_token=user_token,
            collection_name="medical", # Specific collection for medical documents
            export=export,
            k=k
        )

    async def medical_summarize_document_by_path(self, file_path_str: str) -> str:
        """
        Summarizes a document related to medicine or health located at the given file path.
        """
        return await medical_summarize_document_by_path(file_path_str=file_path_str) # Call the function from medical_tool.py

