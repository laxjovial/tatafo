# domain_tools/medical_tools/__init__.py

import logging
from typing import Optional, Dict, Any, List

# Import individual tool functions from the medical_tool module
from .medical_tool import (
    get_drug_info,
    check_symptoms,
    get_hospital_info
)

logger = logging.getLogger(__name__)

class MedicalTools:
    """
    A collection of medical-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """
    def __init__(self, config_manager: Any, log_event: Any):
        """
        Initializes the MedicalTools with necessary dependencies.

        Args:
            config_manager (Any): The configuration manager instance.
            log_event (Any): The analytics logging function.
        """
        self.config_manager = config_manager
        self.log_event = log_event
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

