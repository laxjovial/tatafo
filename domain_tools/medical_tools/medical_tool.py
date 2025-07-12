# domain_tools/medical_tools/medical_tool.py

import logging
import requests
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta, timezone # Import timezone for consistent datetime objects

# Import generic tools
from langchain_core.tools import tool
from shared_tools.scrapper_tool import scrape_web # This tool is a standalone function

# Import config_manager to access API configurations and secrets
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability
# Import date_parser for date format flexibility
from utils.date_parser import parse_date_to_yyyymmdd

# Import analytics_tracker (for logging failures within _make_dynamic_api_request)
from utils import analytics_tracker

# Import UserProfile for type hinting
from backend.models.user_models import UserProfile

logger = logging.getLogger(__name__)

# --- Generic API Request Helper (copied for standalone tool file, ideally in shared utils) ---
# This helper is designed to work with the structure defined in api_providers.yml

def _get_nested_value(data: Dict[str, Any], path: List[str]):
    """Helper to get a value from a nested dictionary using a list of keys."""
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and isinstance(key, str) and key.isdigit(): # Handle list indices
            try:
                current = current[int(key)]
            except (IndexError, ValueError):
                return None
        else:
            return None
    return current

class MedicalTools:
    """
    A collection of tools for medical-related operations, including drug information,
    symptom checking, and hospital information.
    It integrates with external APIs and provides fallback mechanisms.
    """
    def __init__(self, config_manager, log_event, document_tools):
        self.config_manager = config_manager
        self.log_event = log_event # For direct logging if needed, but _make_dynamic_api_request handles tool usage
        self.document_tools = document_tools # For medical_query_uploaded_docs and medical_summarize_document_by_path

    async def _make_dynamic_api_request(
        self,
        domain: str,
        function_name: str,
        params: Dict[str, Any],
        user_context: UserProfile # Changed from user_token to user_context
    ) -> Optional[Dict[str, Any]]:
        """
        Makes an API request to the dynamically configured provider for a given domain and function.
        Handles API key retrieval, request construction, and basic error handling.
        Returns parsed JSON data or None on failure (triggering generic fallback message).
        Logs tool usage analytics for *failures* via analytics_tracker.
        Success logging is handled by LLMService's wrapped_tool_executor.
        """
        user_id = user_context.user_id # Get user_id from UserProfile

        # Get the default active API provider for the domain from data/config.yml
        active_provider_name = self.config_manager.get(f"api_defaults.{domain}")
        if not active_provider_name:
            logger.error(f"No default API provider configured for domain '{domain}'.")
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=f"No default API provider configured for domain '{domain}'."
            )
            return None

        # Get the full configuration for the active provider from api_providers.yml
        provider_config = self.config_manager.get_api_provider_config(domain, active_provider_name)
        if not provider_config:
            logger.error(f"Configuration for API provider '{active_provider_name}' in domain '{domain}' not found in api_providers.yml.")
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=f"API provider config '{active_provider_name}' not found for domain '{domain}'."
            )
            return None

        base_url = provider_config.get("base_url")
        api_key_name = provider_config.get("api_key_name")
        api_key = self.config_manager.get_secret(api_key_name) if api_key_name else None

        headers = {} # No special headers by default for most medical APIs

        if not base_url:
            logger.error(f"Base URL not configured for API provider '{active_provider_name}' in domain '{domain}'.")
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=f"Base URL not configured for '{active_provider_name}'."
            )
            return None

        function_details = provider_config.get("functions", {}).get(function_name)
        if not function_details:
            logger.error(f"Function '{function_name}' not configured for API provider '{active_provider_name}' in domain '{domain}'.")
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=f"Function '{function_name}' not configured for '{active_provider_name}'."
            )
            return None

        endpoint = function_details.get("endpoint")
        path_params_config = function_details.get("path_params", [])

        # Construct URL
        full_url = f"{base_url}{endpoint}" if endpoint else base_url

        # Add path parameters to URL if specified
        for p_param in path_params_config:
            if p_param in params:
                value = str(params.pop(p_param)) # Remove from params after using for path
                full_url = full_url.replace(f"{{{p_param}}}", value)
            else:
                error_msg = f"Missing path parameter '{p_param}' for function '{function_name}'."
                logger.warning(error_msg)
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_id=user_id,
                    success=False,
                    error_message=error_msg
                )
                return None # Cannot construct URL without required path params

        # Construct query parameters
        query_params = {}

        # Add API key if it's a query param
        if api_key_name and api_key:
            param_name_in_url = provider_config.get("api_key_param_name", api_key_name.replace("_api_key", ""))
            query_params[param_name_in_url] = api_key 

        for param_key in function_details.get("required_params", []) + function_details.get("optional_params", []):
            if param_key in params:
                # Special handling for list parameters like 'symptoms'
                if isinstance(params[param_key], list):
                    query_params[param_key] = ",".join(params[param_key]) # Join list into comma-separated string
                else:
                    query_params[param_key] = params[param_key]
            elif param_key in function_details.get("required_params", []):
                error_msg = f"Missing required parameter '{param_key}' for function '{function_name}'."
                logger.warning(error_msg)
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_id=user_id,
                    success=False,
                    error_message=error_msg
                )
                return None # Missing required param, cannot proceed

        try:
            logger.info(f"Making API call to: {full_url} with params: {query_params}")
            response = requests.get(full_url, params=query_params, headers=headers, timeout=self.config_manager.get("web_scraping.timeout_seconds", 15))
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            raw_data = response.json()
            
            # Check for API-specific error messages in the response body
            api_error_message = None
            if raw_data.get("status") == "error": # Generic error status
                api_error_message = f"API Error from {active_provider_name}: {raw_data.get('message', 'Unknown error')}"
            elif raw_data.get("Error"): # Generic error key
                api_error_message = f"API Error from {active_provider_name}: {raw_data['Error']}"
            elif raw_data.get("message") == "Not Found": # Generic "Not Found" message
                api_error_message = f"API Error from {active_provider_name}: Resource not found."
            elif raw_data.get("code") and raw_data.get("message"): # Common error pattern
                api_error_message = f"API Error from {active_provider_name} (Code: {raw_data['code']}): {raw_data['message']}"

            if api_error_message:
                logger.error(api_error_message)
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_id=user_id,
                    success=False,
                    error_message=api_error_message
                )
                return None


            # Extract data based on response_path
            data_to_map = raw_data
            response_path = function_details.get("response_path")
            if response_path:
                data_to_map = _get_nested_value(raw_data, response_path)
                if data_to_map is None:
                    error_msg = f"Response path '{'.'.join(response_path)}' not found in API response from {active_provider_name}. Raw data: {raw_data}"
                    logger.warning(error_msg)
                    await analytics_tracker.log_tool_usage(
                        tool_name=f"{domain}_{function_name}",
                        tool_params=params,
                        user_id=user_id,
                        success=False,
                        error_message=error_msg
                    )
                    return None

            # Apply data mapping
            mapped_data = {}
            data_map = function_details.get("data_map", {})
            
            if isinstance(data_to_map, list): # For lists of items (e.g., multiple search results)
                mapped_data_list = []
                for item in data_to_map:
                    mapped_item = {}
                    for mapped_key, original_key_path in data_map.items():
                        if isinstance(original_key_path, list):
                            mapped_item[mapped_key] = _get_nested_value(item, original_key_path)
                        elif isinstance(original_key_path, str) and '.' in original_key_path:
                            mapped_item[mapped_key] = _get_nested_value(item, original_key_path.split('.'))
                        else:
                            mapped_item[mapped_key] = item.get(original_key_path)
                    mapped_data_list.append(mapped_item)
                final_result = {"data": mapped_data_list} # Wrap list in a dict for consistent return
            else: # For single object responses
                for mapped_key, original_key_path in data_map.items():
                    if isinstance(original_key_path, list):
                        mapped_data[mapped_key] = _get_nested_value(data_to_map, original_key_path)
                    elif isinstance(original_key_path, str) and '.' in original_key_path:
                        mapped_data[mapped_key] = _get_nested_value(data_to_map, original_key_path.split('.'))
                    else:
                        mapped_data[mapped_key] = data_to_map.get(original_key_path)
                final_result = mapped_data

            # Success logging is handled by LLMService's wrapped_tool_executor, not here.
            return final_result

        except requests.exceptions.Timeout:
            error_msg = f"API request to {active_provider_name} timed out for function '{function_name}'."
            logger.error(error_msg)
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=error_msg
            )
            return None
        except requests.exceptions.RequestException as e:
            error_msg = f"Error making API request to {active_provider_name} for function '{function_name}': {e}"
            logger.error(error_msg)
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=str(e) # Ensure error message is string
            )
            return None
        except json.JSONDecodeError:
            error_msg = f"Failed to decode JSON response from {active_provider_name} for function '{function_name}'."
            logger.error(error_msg)
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=error_msg
            )
            return None
        except Exception as e:
            error_msg = f"An unexpected error occurred during API call to {active_provider_name} for '{function_name}': {e}"
            logger.error(error_msg, exc_info=True)
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_id=user_id,
                success=False,
                error_message=error_msg
            )
            return None

    @tool
    async def medical_get_drug_info(self, drug_name: str, user_context: UserProfile = None) -> str:
        """
        Retrieves information about a specific drug, including its uses, side effects, and dosage.
        Falls back to a generic message if API key is missing or API call fails.

        Args:
            drug_name (str): The name of the drug (e.g., "Paracetamol", "Ibuprofen").
            user_context (UserProfile): The user's profile for RBAC checks and logging.

        Returns:
            str: A formatted string of drug information, or an error/fallback message.
        """
        if user_context is None: # For CLI testing without full UserProfile
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: medical_get_drug_info called for drug: '{drug_name}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'medical_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to medical tools is not enabled for your current tier."
        
        params = {"name": drug_name}
        api_data = await self._make_dynamic_api_request("medical", "get_drug_info", params, user_context)

        if api_data:
            try:
                name = api_data.get("name")
                generic_name = api_data.get("generic_name")
                uses = api_data.get("uses")
                side_effects = api_data.get("side_effects")
                dosage_adult = api_data.get("dosage_adult")
                interactions = api_data.get("interactions")

                if name and uses:
                    response_str = (
                        f"Information for Drug: {name}\n"
                    )
                    if generic_name:
                        response_str += f"  Generic Name: {generic_name}\n"
                    response_str += (
                        f"  Uses: {uses}\n"
                        f"  Side Effects: {side_effects}\n"
                    )
                    if dosage_adult:
                        response_str += f"  Adult Dosage: {dosage_adult}\n"
                    if interactions:
                        response_str += f"  Interactions: {interactions}\n"
                    return response_str
                else:
                    logger.warning(f"Live API data for drug '{drug_name}' is incomplete. Raw: {api_data}")
                    return f"Could not retrieve complete live drug information for '{drug_name}'. Please try again or check the name."
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing live drug info data for '{drug_name}': {e}")
                return f"Error parsing live data for '{drug_name}'. Please try again."
        else:
            return f"Could not retrieve live drug information for '{drug_name}'. The API call failed or returned no data. Please ensure your API key is valid and try again."


    @tool
    async def medical_check_symptoms(self, symptoms: List[str], user_context: UserProfile = None) -> str:
        """
        Checks a list of symptoms and suggests possible medical conditions and recommendations.
        Falls back to a generic message if API key is missing or API call fails.

        Args:
            symptoms (List[str]): A list of symptoms (e.g., ["fever", "cough", "sore throat"]).
            user_context (UserProfile): The user's profile for RBAC checks and logging.

        Returns:
            str: A formatted string of possible conditions and recommendations, or an error/fallback message.
        """
        if user_context is None: # For CLI testing without full UserProfile
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: medical_check_symptoms called for symptoms: {symptoms} by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'medical_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to medical tools is not enabled for your current tier."
        
        params = {"symptoms": symptoms} # Pass as list, _make_dynamic_api_request will join for query param
        api_data = await self._make_dynamic_api_request("medical", "check_symptoms", params, user_context)

        if api_data:
            try:
                possible_conditions = api_data.get("possible_conditions")
                recommendations = api_data.get("recommendations")

                if possible_conditions and recommendations:
                    response_str = (
                        f"Based on symptoms: {', '.join(symptoms)}:\n"
                        f"  Possible Conditions: {', '.join(possible_conditions)}\n"
                        f"  Recommendations: {recommendations}\n"
                    )
                    return response_str
                else:
                    logger.warning(f"Live API data for symptoms '{symptoms}' is incomplete. Raw: {api_data}")
                    return f"Could not retrieve complete live symptom check information for '{', '.join(symptoms)}'. Please try again."
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing live symptom check data for '{symptoms}': {e}")
                return f"Error parsing live data for '{', '.join(symptoms)}'. Please try again."
        else:
            return f"Could not retrieve live symptom information for '{', '.join(symptoms)}'. The API call failed or returned no data. Please ensure your API key is valid and try again."


    @tool
    async def medical_get_hospital_info(self, hospital_name: str, location: Optional[str] = None, user_context: UserProfile = None) -> str:
        """
        Retrieves information about a specific hospital or medical center.
        Falls back to a generic message if API key is missing or API call fails.

        Args:
            hospital_name (str): The full or partial name of the hospital (e.g., "General Hospital", "City Medical Center").
            location (str, optional): The city or region where the hospital is located.
            user_context (UserProfile): The user's profile for RBAC checks and logging.

        Returns:
            str: A formatted string of hospital information, or an error/fallback message.
        """
        if user_context is None: # For CLI testing without full UserProfile
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: medical_get_hospital_info called for hospital: '{hospital_name}', location: '{location}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'medical_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            return "Error: Access to medical tools is not enabled for your current tier."
        
        params = {"name": hospital_name}
        if location: params["location"] = location

        api_data = await self._make_dynamic_api_request("medical", "get_hospital_info", params, user_context)

        if api_data:
            try:
                name = api_data.get("name")
                loc = api_data.get("location")
                specialties = api_data.get("specialties")
                contact = api_data.get("contact")
                website = api_data.get("website")

                if name and loc:
                    response_str = (
                        f"Information for Hospital: {name}\n"
                        f"  Location: {loc}\n"
                    )
                    if specialties:
                        response_str += f"  Specialties: {', '.join(specialties)}\n"
                    if contact:
                        response_str += f"  Contact: {contact}\n"
                    if website:
                        response_str += f"  Website: {website}\n"
                    return response_str
                else:
                    logger.warning(f"Live API data for hospital '{hospital_name}' is incomplete. Raw: {api_data}")
                    return f"Could not retrieve complete live hospital information for '{hospital_name}'. Please try again or check the name/location."
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing live hospital info data for '{hospital_name}': {e}")
                return f"Error parsing live data for '{hospital_name}'. Please try again."
        else:
            return f"Could not retrieve live hospital information for '{hospital_name}'. The API call failed or returned no data. Please ensure your API key is valid and try again."


    # --- Existing Generic Tools (now methods of MedicalTools) ---
    # These functions wrap existing shared tools or DocumentTools methods.
    # They will pass the user_context down if the wrapped tool supports it.

    @tool
    async def medical_search_web(self, query: str, user_context: UserProfile, max_chars: int = 2000) -> str:
        """
        Searches the web for medical or health-related information using a smart search fallback mechanism.
        This tool wraps the generic `scrape_web` tool, providing a medical-specific interface.
        
        Args:
            query (str): The medical-related search query (e.g., "symptoms of common cold", "latest cancer research").
            user_context (UserProfile): The user's profile for RBAC checks and logging.
            max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
        
        Returns:
            str: A string containing relevant information from the web.
        """
        logger.info(f"Tool: medical_search_web called with query: '{query}' for user: '{user_context.user_id}'")
        # scrape_web is a standalone function, ensure it handles its own RBAC/logging if applicable
        # For now, it's assumed LLMService wrapper handles its API limit check.
        return await scrape_web(query=query, user_token=user_context.user_id, max_chars=max_chars) # Pass user_token for scrape_web's internal logging

    @tool
    async def medical_query_uploaded_docs(self, query: str, user_context: UserProfile, export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed medical documents for a user using vector similarity search.
        This tool wraps the generic `DocumentTools.document_query_uploaded_docs` tool, fixing the section to "medical".
        
        Args:
            query (str): The search query to find relevant medical documents (e.g., "my lab results explanation", "patient history for John Doe").
            user_context (UserProfile): The user's profile for RBAC checks and logging.
            export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
            k (int): The number of top relevant documents to retrieve. Defaults to 5.
        
        Returns:
            str: A string containing the combined content of the relevant document chunks,
                 or a message indicating no data/results found, or the export path if exported.
        """
        logger.info(f"Tool: medical_query_uploaded_docs called with query: '{query}' for user: '{user_context.user_id}'")
        if not self.document_tools:
            return "Error: Document tools are not initialized. Cannot query uploaded documents."
        
        # Call the actual document_query_uploaded_docs from the DocumentTools instance
        return await self.document_tools.document_query_uploaded_docs(
            query=query,
            user_context=user_context, # Pass user_context directly
            section="medical", # Specify the section for medical documents
            export=export,
            k=k
        )

    @tool
    async def medical_summarize_document_by_path(self, file_path_str: str, user_context: UserProfile) -> str:
        """
        Summarizes a document related to medicine or health located at the given file path.
        The file path should be accessible by the system (e.g., in the 'uploads' directory).
        This tool wraps the generic `DocumentTools.document_summarize_document_by_path` tool.
        
        Args:
            file_path_str (str): The full path to the document file to be summarized.
                                Example: "uploads/default/medical/patient_report.pdf"
            user_context (UserProfile): The user's profile for RBAC checks and logging.
        
        Returns:
            str: A concise summary of the document content.
        """
        logger.info(f"Tool: medical_summarize_document_by_path called for file: '{file_path_str}' by user: '{user_context.user_id}'")
        if not self.document_tools:
            return "Error: Document tools are not initialized. Cannot summarize documents."

        # Call the actual document_summarize_document_by_path from the DocumentTools instance
        return await self.document_tools.document_summarize_document_by_path(
            file_path_str=file_path_str,
            user_context=user_context # Pass user_context directly
        )


# CLI Test (optional)
if __name__ == "__main__":
    import asyncio
    from unittest.mock import MagicMock, AsyncMock, patch, ANY
    import shutil
    import os
    import sys
    from shared_tools.vector_utils import BASE_VECTOR_DIR # For cleanup
    from database.firestore_manager import FirestoreManager # For mocking
    from shared_tools.cloud_storage_utils import CloudStorageUtilsWrapper # For mocking
    from shared_tools.vector_utils import VectorUtilsWrapper # For mocking
    from domain_tools.document_tools.document_tool import DocumentTools # For mocking
    from backend.models.user_models import UserProfile # For mock user_context
    from langchain_core.messages import HumanMessage, AIMessage # For mocking LLM in summarizer

    logging.basicConfig(level=logging.INFO)

    # Mock UserProfile for testing
    mock_user_pro_profile = UserProfile(user_id="mock_pro_token", username="ProUser", email="pro@example.com", tier="pro", roles=["user"])
    mock_user_free_profile = UserProfile(user_id="mock_free_token", username="FreeUser", email="free@example.com", tier="free", roles=["user"])
    mock_user_premium_profile = UserProfile(user_id="mock_premium_token", username="PremiumUser", email="premium@example.com", tier="premium", roles=["user"])
    mock_user_admin_profile = UserProfile(user_id="mock_admin_token", username="AdminUser", email="admin@example.com", tier="admin", roles=["user", "admin"])


    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.medical_api_key = "MOCK_MEDICAL_API_KEY_LIVE"
            self.serpapi_api_key = "MOCK_SERPAPI_KEY_LIVE" # For scrape_web
            self.openai_api_key = "sk-mock-openai-key-12345" # For summarizer
            self.google_api_key = "AIzaSy-mock-google-key" # For summarizer

        def get(self, key, default=None):
            return getattr(self, key, default)
    
    class MockConfigManager:
        _instance = None
        _is_loaded = False
        def __init__(self):
            if MockConfigManager._instance is not None:
                raise Exception("ConfigManager is a singleton. Use get_instance().")
            MockConfigManager._instance = self
            self._config_data = {
                'llm': {'max_summary_input_chars': 10000},
                'rag': {'chunk_size': 500, 'chunk_overlap': 50, 'max_query_results_k': 10},
                'web_scraping': {
                    'user_agent': 'Mozilla/5.0 (Test; Python)',
                    'timeout_seconds': 1 # Short timeout for mocks
                },
                'tiers': {},
                'default_user_tier': 'free',
                'default_user_roles': ['user'],
                'api_defaults': { # Mock api_defaults
                    'medical': 'mock_medical_api', # Using a mock name for clarity
                    'web_search': 'serpapi',
                    'document_summarization_llm': 'openai'
                },
                'analytics': { # Mock analytics settings
                    'enabled': True,
                    'log_tool_usage': True,
                    'log_query_failures': True
                }
            }
            self._api_providers_data = { # Mock api_providers_data for medical
                "medical": {
                    "mock_medical_api": { # Using a mock name for clarity
                        "base_url": "https://api.mock-medical.com/v1", # Example mock API base URL
                        "api_key_name": "medical_api_key",
                        "api_key_param_name": "api_key",
                        "functions": {
                            "get_drug_info": {
                                "endpoint": "/drugs",
                                "required_params": ["name"],
                                "response_path": ["data", 0], # Assuming first result is most relevant
                                "data_map": {
                                    "name": "name",
                                    "generic_name": "generic_name",
                                    "uses": "uses",
                                    "side_effects": "side_effects",
                                    "dosage_adult": "dosage_adult",
                                    "interactions": "interactions"
                                }
                            },
                            "check_symptoms": {
                                "endpoint": "/symptoms/check",
                                "required_params": ["symptoms"],
                                "response_path": ["data", 0],
                                "data_map": {
                                    "possible_conditions": "conditions",
                                    "recommendations": "recommendations"
                                }
                            },
                            "get_hospital_info": {
                                "endpoint": "/hospitals",
                                "required_params": ["name"],
                                "optional_params": ["location"],
                                "response_path": ["data", 0],
                                "data_map": {
                                    "name": "name",
                                    "location": "location",
                                    "specialties": "specialties",
                                    "contact": "contact",
                                    "website": "website"
                                }
                            }
                        }
                    }
                },
                "web_search": { # Mock for web search (SerpAPI)
                    "serpapi": {
                        "base_url": "https://serpapi.com/search",
                        "api_key_name": "serpapi_api_key",
                        "api_key_param_name": "api_key",
                        "functions": {
                            "scrape_web": { # This function name should match the tool name
                                "required_params": ["q"],
                                "optional_params": ["engine"],
                                "response_path": ["organic_results"], # Example path for search results
                                "data_map": { # Simplified mapping for search results
                                    "title": "title",
                                    "link": "link",
                                    "snippet": "snippet"
                                }
                            }
                        }
                    }
                },
                "document_summarization_llm": { # Mock for summarization LLM
                    "openai": {
                        "base_url": "https://api.openai.com/v1/chat/completions",
                        "api_key_name": "openai_api_key",
                        "functions": {
                            "summarize_document": { # This function name should match the tool name
                                "endpoint": "", # No specific endpoint for chat completions
                                "required_params": [],
                                "optional_params": [],
                                "response_path": ["choices", 0, "message", "content"],
                                "data_map": {} # No specific mapping needed for direct content
                            }
                        }
                    }
                }
            }
            self._is_loaded = True
        
        def get(self, key, default=None):
            parts = key.split('.')
            val = self._config_data
            for part in parts:
                if isinstance(val, dict) and part in val:
                    val = val[part]
                else:
                    return default
            return val
        
        def get_secret(self, key, default=None):
            mock_secrets_instance = MockSecrets()
            return mock_secrets_instance.get(key, default)

        def set_secret(self, key, value):
            pass
        
        def get_api_provider_config(self, domain: str, provider_name: str) -> Optional[Dict[str, Any]]:
            return self._api_providers_data.get(domain, {}).get(provider_name)

        def get_domain_api_providers(self, domain: str) -> Dict[str, Any]:
            return self._api_providers_data.get(domain, {})


    # Mock user_manager.get_user_tier_capability for testing RBAC
    # This mock is for the standalone get_user_tier_capability function
    # which is now imported directly by tools.
    class MockUserManager:
        _mock_users = {
            "mock_free_token": {"user_id": "mock_free_token", "username": "FreeUser", "email": "free@example.com", "tier": "free", "roles": ["user"]},
            "mock_pro_token": {"user_id": "mock_pro_token", "username": "ProUser", "email": "pro@example.com", "tier": "pro", "roles": ["user"]},
            "mock_premium_token": {"user_id": "mock_premium_token", "username": "PremiumUser", "email": "premium@example.com", "tier": "premium", "roles": ["user"]},
            "mock_admin_token": {"user_id": "mock_admin_token", "username": "AdminUser", "email": "admin@example.com", "tier": "admin", "roles": ["user", "admin"]},
        }
        _rbac_capabilities = { # This now mirrors the _RBAC_CAPABILITIES_CONFIG in utils/user_manager.py
            'capabilities': {
                'medical_tool_access': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'document_query_enabled': { # Added for document tool
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'web_search_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
                'summarization_enabled': { # For summarize_document
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'llm_default_provider': { # For summarize_document
                    'default': 'gemini',
                    'tiers': {'pro': 'gemini', 'premium': 'openai', 'admin': 'gemini'}
                },
                'llm_default_model_name': { # For summarize_document
                    'default': 'gemini-1.5-flash',
                    'tiers': {'pro': 'gemini-1.5-flash', 'premium': 'gpt-4o', 'admin': 'gemini-1.5-flash'}
                },
                'llm_default_temperature': { # For summarize_document
                    'default': 0.7,
                    'tiers': {'pro': 0.5, 'premium': 0.3, 'admin': 0.7}
                },
            }
        }
        _tier_hierarchy = {
            "free": 0, "user": 1, "basic": 2, "pro": 3, "premium": 4, "admin": 99
        }

        def get_user_tier_capability(self, user_id: str, capability_key: str, default_value: Any = None, user_tier: Optional[str] = None, user_roles: Optional[List[str]] = None) -> Any:
            # If user_tier/user_roles are provided, use them directly (from UserProfile)
            # Otherwise, try to look up from _mock_users
            if user_tier is None or user_roles is None:
                user_info = self._mock_users.get(user_id, {})
                user_tier = user_info.get('tier', 'free')
                user_roles = user_info.get('roles', [])

            if "admin" in user_roles:
                if isinstance(default_value, bool): return True
                if isinstance(default_value, (int, float)): return float('inf')
                return default_value
            
            capability_config = self._rbac_capabilities.get('capabilities', {}).get(capability_key)
            if not capability_config:
                return default_value

            # Check roles first
            for role in user_roles:
                if role in capability_config.get('roles', {}):
                    return capability_config['roles'][role]
            
            # Then check tiers
            if user_tier in capability_config.get('tiers', {}):
                return capability_config['tiers'][user_tier]

            return capability_config.get('default', default_value)


    # Patch the actual imports for testing
    import streamlit as st_mock
    if not hasattr(st_mock, 'secrets'):
        st_mock.secrets = MockSecrets()
    
    # Patch config_manager and user_manager in their respective modules
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager # Also patch the class if needed by other modules
    
    # Patch the standalone get_user_tier_capability function in utils.user_manager
    # This is crucial for the tools to use the mock during their CLI tests.
    sys.modules['utils.user_manager'].get_user_tier_capability = MockUserManager().get_user_tier_capability

    # Mock analytics_tracker
    mock_analytics_tracker_db = MagicMock()
    mock_analytics_tracker_auth = MagicMock()
    mock_analytics_tracker_auth.currentUser = MagicMock(uid="mock_user_123")
    mock_analytics_tracker_db.collection.return_value.add = AsyncMock(return_value=MagicMock(id="mock_doc_id"))

    # Patch firebase_admin.firestore for the local import within log_event
    with patch.dict(sys.modules, {'firebase_admin.firestore': MagicMock(firestore=MagicMock())}):
        sys.modules['firebase_admin.firestore'].firestore.CollectionReference = MagicMock()
        sys.modules['firebase_admin'].firestore.DocumentReference = MagicMock()
        
        # Initialize the actual analytics_tracker with mocks
        analytics_tracker.initialize_analytics(
            mock_analytics_tracker_db,
            mock_analytics_tracker_auth,
            "test_app_id_for_analytics",
            "mock_user_123"
        )

        # Mock requests.get for external API calls
        original_requests_get = requests.get

        def mock_requests_get_dynamic(url, params=None, headers=None, timeout=None):
            # Simulate hypothetical Medical API responses
            if "api.mock-medical.com/v1" in url:
                if "/drugs" in url:
                    name = params.get("name", "").lower()
                    if "paracetamol" in name:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "data": [{
                                "name": "Paracetamol",
                                "generic_name": "Acetaminophen",
                                "uses": "Pain relief and fever reduction.",
                                "side_effects": "Rare: allergic reactions, skin rash.",
                                "dosage_adult": "500-1000 mg every 4-6 hours.",
                                "interactions": "Warfarin."
                            }]
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"data": []} # No data found
                        return mock_response
                elif "/symptoms/check" in url:
                    symptoms_str = params.get("symptoms", "")
                    symptoms = [s.strip().lower() for s in symptoms_str.split(',')]
                    if "fever" in symptoms and "cough" in symptoms:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "data": [{
                                "conditions": ["Common Cold", "Flu"],
                                "recommendations": "Rest and fluids."
                            }]
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"data": []} # No data found
                        return mock_response
                elif "/hospitals" in url:
                    name = params.get("name", "").lower()
                    location = params.get("location", "").lower() if params.get("location") else ""
                    if "general hospital" in name and "london" in location:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "data": [{
                                "name": "General Hospital London",
                                "location": "London, UK",
                                "specialties": ["Emergency Medicine", "Cardiology"],
                                "contact": "+44 20 1234 5678",
                                "website": "http://example.com/ghl"
                            }]
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"data": []} # No data found
                        return mock_response
            
            # Simulate scrape_web's internal requests.get if needed (SerpAPI)
            if "serpapi.com/search" in url:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "organic_results": [
                        {"title": "Mock Search Result 1", "link": "http://example.com/1", "snippet": f"Snippet for {params.get('q', 'medical')} result 1."},
                        {"title": "Mock Search Result 2", "link": "http://example.com/2", "snippet": f"Snippet for {params.get('q', 'medical')} result 2."}
                    ]
                }
                return mock_response
            
            # Mock LLM for summarizer (if it uses requests.post for an API)
            if "api.openai.com/v1/chat/completions" in url:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Mocked LLM summary content."}}]
                }
                return mock_response

            return original_requests_get(url, params=params, headers=headers, timeout=timeout)

        requests.get = MagicMock(side_effect=mock_requests_get_dynamic)
        requests.post = MagicMock(side_effect=mock_requests_get_dynamic) # For OpenAI chat completions

        # Mock FirestoreManager, CloudStorageUtilsWrapper, VectorUtilsWrapper, DocumentTools for init
        mock_firestore_manager = MagicMock(spec=FirestoreManager)
        mock_cloud_storage_utils = MagicMock(spec=CloudStorageUtilsWrapper)
        mock_vector_utils = MagicMock(spec=VectorUtilsWrapper)
        
        # Create a mock DocumentTools instance
        mock_document_tools = MagicMock(spec=DocumentTools)
        mock_document_tools.document_query_uploaded_docs = AsyncMock(return_value="Mocked document query results for medical.")
        mock_document_tools.document_summarize_document_by_path = AsyncMock(return_value="Mocked summary of dummy_file.txt")

        # Instantiate MedicalTools with mocks
        medical_tools_instance = MedicalTools(
            config_manager=sys.modules['config.config_manager'].config_manager,
            log_event=analytics_tracker.log_event, # Pass the actual (mocked) log_event
            document_tools=mock_document_tools
        )

        async def run_medical_tests(medical_tools_instance):
            print("\n--- Testing medical_tool functions with Live API Simulation and Analytics ---")

            # Test 1: medical_get_drug_info (success)
            print("\n--- Test 1: medical_get_drug_info (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock() # Reset mock call count
            result_drug_info = await medical_tools_instance.medical_get_drug_info("Paracetamol", user_context=mock_user_pro_profile)
            print(f"Drug Info: {result_drug_info}")
            assert "Information for Drug: Paracetamol" in result_drug_info
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called() # Analytics should NOT be logged for success here
            print("Test 1 Passed.")

            # Test 2: medical_check_symptoms (API failure - no data found)
            print("\n--- Test 2: medical_check_symptoms (API Failure - No Data Found) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_symptom_check = await medical_tools_instance.medical_check_symptoms(["rash", "joint pain"], user_context=mock_user_pro_profile)
            print(f"Symptom Check (API Error): {result_symptom_check}")
            assert "Could not retrieve complete live symptom check information for 'rash, joint pain'." in result_symptom_check
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once() # Analytics should be logged for failure
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "medical_check_symptoms"
            assert logged_data["success"] is False
            assert "Response path 'data.0' not found" in logged_data["error_message"]
            print("Test 2 Passed.")

            # Test 3: medical_get_hospital_info (RBAC denied)
            print("\n--- Test 3: medical_get_hospital_info (RBAC Denied) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_hospital_rbac_denied = await medical_tools_instance.medical_get_hospital_info("St. Jude's", user_context=mock_user_free_profile)
            print(f"Hospital Info (Free User, RBAC Denied): {result_hospital_rbac_denied}")
            assert "Error: Access to medical tools is not enabled for your current tier." in result_hospital_rbac_denied
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called() # RBAC check happens before _make_dynamic_api_request
            print("Test 3 Passed.")

            # Test 4: medical_search_web (generic tool)
            print("\n--- Test 4: medical_search_web (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_web_search = await medical_tools_instance.medical_search_web("causes of high blood pressure", user_context=mock_user_pro_profile)
            print(f"Web Search Result: {result_web_search[:100]}...")
            assert "Search results for causes of high blood pressure" in result_web_search
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called() # Analytics for scrape_web is handled by its own internal logging or LLMService wrapper
            print("Test 4 Passed.")

            # Test 5: medical_query_uploaded_docs (generic tool via DocumentTools)
            print("\n--- Test 5: medical_query_uploaded_docs (Generic Tool via DocumentTools) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_doc_query = await medical_tools_instance.medical_query_uploaded_docs("patient history", user_context=mock_user_pro_profile)
            print(f"Document Query Result: {result_doc_query}")
            assert "Mocked document query results for medical." in result_doc_query
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once() # Analytics logged by DocumentTools mock
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "document_query_uploaded_docs"
            assert logged_data["success"] is True
            print("Test 5 Passed.")

            # Test 6: medical_summarize_document_by_path (generic tool via DocumentTools)
            print("\n--- Test 6: medical_summarize_document_by_path (Generic Tool via DocumentTools) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            # Create a dummy file for summarization test
            test_user_pro_dir = Path("uploads") / mock_user_pro_profile.user_id
            dummy_file_path = test_user_pro_dir / "medical" / "dummy_report.txt"
            dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_file_path.write_text("This is a dummy medical report content for testing summarization.")

            result_summarize = await medical_tools_instance.medical_summarize_document_by_path(str(dummy_file_path), user_context=mock_user_pro_profile)
            print(f"Summarize Result: {result_summarize}")
            assert "Mocked summary of dummy_file.txt" in result_summarize # Check for mock summary from DocumentTools
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once() # Now logged by DocumentTools mock
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "document_summarize_document_by_path"
            assert logged_data["success"] is True
            print("Test 6 Passed.")

            print("\nAll medical_tool tests with live API simulation and analytics considerations completed.")

        # Ensure tests are only run when the script is executed directly
        if __name__ == "__main__":
            asyncio.run(run_medical_tests(medical_tools_instance))

        # Restore original requests.get
        requests.get = original_requests_get
        requests.post = original_requests_get # Restore post if it was patched to get

        # Clean up dummy files and directories
        test_user_dirs = [Path("uploads") / mock_user_pro_profile.user_id, BASE_VECTOR_DIR / mock_user_pro_profile.user_id]
        for d in test_user_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print(f"Cleaned up {d}")
