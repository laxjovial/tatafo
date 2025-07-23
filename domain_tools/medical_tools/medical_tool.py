# domain_tools/medical_tools/medical_tool.py

import logging
import requests
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta, timezone # Import timezone for consistent datetime objects

# Import generic tools
from langchain_core.tools import tool
from shared_tools.scrapper_tool import scrape_web # Corrected: from scraper_tool, not scrapper_tool

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

# Import DocumentTools for wrapping document related tools
from domain_tools.document_tools.document_tool import DocumentTools


logger = logging.getLogger(__name__)

# --- Generic API Request Helper (copied for standalone tool file, ideally in shared utils) ---
# This helper is designed to work with the structure defined in api_providers.yml

async def make_api_request(
    provider_name: str,
    function_name: str,
    params: Dict[str, Any],
    user_api_keys: List[str],
    domain: str,
    user_id: str = "default_user",
    additional_headers: Optional[Dict[str, str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Makes a dynamic API request based on the provider configuration from config_manager.
    Handles API key injection, URL construction, and response parsing.
    """
    provider_config = config_manager.get_api_provider_config(domain, provider_name)
    if not provider_config:
        logger.error(f"Provider config not found for domain: {domain}, provider: {provider_name}")
        analytics_tracker.log_event(user_id, "api_request_failed", "config_missing",
                                    {"domain": domain, "provider": provider_name, "function": function_name}, success=False)
        return None

    base_url = provider_config.get("base_url")
    if not base_url:
        logger.error(f"Base URL not found for provider: {provider_name}")
        analytics_tracker.log_event(user_id, "api_request_failed", "base_url_missing",
                                    {"domain": domain, "provider": provider_name, "function": function_name}, success=False)
        return None

    function_config = provider_config.get("functions", {}).get(function_name)
    if not function_config:
        logger.error(f"Function config not found for function: {function_name} in provider: {provider_name}")
        analytics_tracker.log_event(user_id, "api_request_failed", "function_config_missing",
                                    {"domain": domain, "provider": provider_name, "function": function_name}, success=False)
        return None

    endpoint = function_config.get("endpoint", "")
    method = function_config.get("method", "GET").upper()
    api_key_name = provider_config.get("api_key_name")
    api_key_param_name = provider_config.get("api_key_param_name")
    response_path = function_config.get("response_path", [])

    # Prepare request parameters
    request_params = {k: v for k, v in params.items() if k in function_config.get("required_params", []) or k in function_config.get("optional_params", [])}

    # Inject API key
    api_key_value = None
    if user_api_keys:
        # Prioritize user-provided keys if they match the required api_key_name
        for key_dict in user_api_keys:
            if key_dict.get("name") == api_key_name:
                api_key_value = key_dict.get("value")
                break
    
    if not api_key_value:
        # Fallback to backend secrets if not provided by user or not found in user_api_keys
        api_key_value = config_manager.get_secret(api_key_name)
    
    if api_key_value and api_key_param_name:
        request_params[api_key_param_name] = api_key_value
    
    # Construct URL, handling path parameters
    full_url = base_url + endpoint
    if function_config.get("path_params"):
        for param in function_config["path_params"]:
            if param in request_params:
                full_url = full_url.replace(f"{{{param}}}", str(request_params.pop(param))) # Remove from query params
            else:
                logger.warning(f"Missing path parameter '{param}' for {function_name} in {provider_name}. URL might be malformed.")

    headers = additional_headers or {}
    # Default to application/json for POST if not specified
    if method == "POST" and "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"

    try:
        response = None
        if method == "GET":
            response = requests.get(full_url, params=request_params, headers=headers, timeout=config_manager.get("web_scraping.timeout_seconds", 30))
        elif method == "POST":
            response = requests.post(full_url, json=request_params, headers=headers, timeout=config_manager.get("web_scraping.timeout_seconds", 30))
        else:
            logger.error(f"Unsupported HTTP method: {method}")
            analytics_tracker.log_event(user_id, "api_request_failed", "unsupported_method",
                                        {"domain": domain, "provider": provider_name, "function": function_name, "method": method}, success=False)
            return None

        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()

        # Navigate response path
        result = data
        for key in response_path:
            if isinstance(result, dict) and key in result:
                result = result[key]
            elif isinstance(result, list) and isinstance(key, int) and len(result) > key:
                result = result[key]
            else:
                logger.warning(f"Could not navigate to response_path {response_path} for {function_name}. Returning full data.")
                result = data
                break
        
        analytics_tracker.log_event(user_id, "api_request_success", "api_call",
                                    {"domain": domain, "provider": provider_name, "function": function_name, "status_code": response.status_code}, success=True)
        return result

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error for {provider_name} {function_name}: {e.response.status_code} - {e.response.text}")
        analytics_tracker.log_event(user_id, "api_request_failed", "http_error",
                                    {"domain": domain, "provider": provider_name, "function": function_name, "status_code": e.response.status_code, "error": str(e)}, success=False)
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error for {provider_name} {function_name}: {e}")
        analytics_tracker.log_event(user_id, "api_request_failed", "connection_error",
                                    {"domain": domain, "provider": provider_name, "function": function_name, "error": str(e)}, success=False)
        return None
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout error for {provider_name} {function_name}: {e}")
        analytics_tracker.log_event(user_id, "api_request_failed", "timeout_error",
                                    {"domain": domain, "provider": provider_name, "function": function_name, "error": str(e)}, success=False)
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during API request to {provider_name} {function_name}: {e}")
        analytics_tracker.log_event(user_id, "api_request_failed", "request_error",
                                    {"domain": domain, "provider": provider_name, "function": function_name, "error": str(e)}, success=False)
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for {provider_name} {function_name}: {e}. Response: {response.text if response else 'N/A'}")
        analytics_tracker.log_event(user_id, "api_request_failed", "json_decode_error",
                                    {"domain": domain, "provider": provider_name, "function": function_name, "error": str(e)}, success=False)
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred during API request to {provider_name} {function_name}.")
        analytics_tracker.log_event(user_id, "api_request_failed", "unexpected_error",
                                    {"domain": domain, "provider": provider_name, "function": function_name, "error": str(e)}, success=False)
        return None

# --- Standalone Tool Functions ---

@tool
async def get_drug_info(
    drug_name: str,
    user_context: Optional[UserProfile] = None,
    provider: str = "openfda",
    user_api_keys: List[str] = []
) -> str:
    """
    Retrieves information about a specific drug, including its uses, side effects, and warnings.
    This tool integrates with the OpenFDA API.
    Requires 'medical_tool_access' capability.

    Args:
        drug_name (str): The name of the drug to search for (e.g., "aspirin", "paracetamol").
        user_context (UserProfile, optional): The user's profile for RBAC checks. Defaults to None.
        provider (str, optional): The API provider to use. Defaults to "openfda".
        user_api_keys (list, optional): List of user-provided API keys (e.g., from Streamlit secrets).

    Returns:
        str: A JSON string containing the drug information, or an error message.
    """
    if user_context is None:
        user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

    logger.info(f"Tool: get_drug_info called for drug: '{drug_name}', provider: '{provider}', user: '{user_context.user_id}'")

    if not get_user_tier_capability(user_context.user_id, 'medical_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "permission_denied",
                                    {"tool_name": "get_drug_info", "drug_name": drug_name, "provider": provider}, success=False)
        return "Error: Access to medical tools is not enabled for your current tier."
    
    # OpenFDA doesn't typically require an API key for basic searches unless rate limits are hit
    # but we include the parameter for consistency if a key ever becomes necessary.
    params = {"search": f"openfda.brand_name:\"{drug_name}\"+OR+openfda.generic_name:\"{drug_name}\"", "limit": 1}
    api_data = await make_api_request(
        provider_name=provider,
        function_name="get_drug_info",
        params=params,
        user_api_keys=user_api_keys,
        domain="medical",
        user_id=user_context.user_id
    )

    if api_data and api_data.get("results"):
        drug_info = api_data["results"][0]
        
        brand_name = drug_info.get("openfda", {}).get("brand_name", ["N/A"])[0]
        generic_name = drug_info.get("openfda", {}).get("generic_name", ["N/A"])[0]
        purpose = drug_info.get("purpose", [{}])[0].get("description", ["N/A"])[0] if drug_info.get("purpose") else "N/A"
        indications_and_usage = drug_info.get("indications_and_usage", ["N/A"])[0]
        warnings = drug_info.get("warnings", ["N/A"])[0]
        adverse_reactions = drug_info.get("adverse_reactions", ["N/A"])[0]

        result_str = (
            f"**Drug Information for {brand_name} (Generic: {generic_name}):**\n"
            f"- **Purpose:** {purpose}\n"
            f"- **Indications and Usage:** {indications_and_usage}\n"
            f"- **Warnings:** {warnings}\n"
            f"- **Adverse Reactions:** {adverse_reactions}"
        )
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "success",
                                    {"tool_name": "get_drug_info", "drug_name": drug_name, "provider": provider, "summary": f"Fetched info for {brand_name}"}, success=True)
        return result_str
    else:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "no_data",
                                    {"tool_name": "get_drug_info", "drug_name": drug_name, "provider": provider, "message": "No drug information found."}, success=False)
        return f"Could not retrieve information for the drug: {drug_name.capitalize()}. Please check the name and try again."


@tool
async def check_symptoms(
    symptoms: List[str],
    user_context: Optional[UserProfile] = None,
    provider: str = "apimedic", # Example provider, replace with actual if using a symptom checker API
    user_api_keys: List[str] = []
) -> str:
    """
    Checks common conditions or potential causes based on a list of symptoms.
    This tool is illustrative and would typically integrate with a dedicated
    symptom checker API (e.g., Infermedica, Apimedic).
    Requires 'medical_tool_access' capability.

    Args:
        symptoms (List[str]): A list of symptoms (e.g., ["fever", "cough", "headache"]).
        user_context (UserProfile, optional): The user's profile for RBAC checks. Defaults to None.
        provider (str, optional): The API provider to use. Defaults to "apimedic".
        user_api_keys (list, optional): List of user-provided API keys.

    Returns:
        str: A formatted string suggesting potential conditions or an error message.
    """
    if user_context is None:
        user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

    logger.info(f"Tool: check_symptoms called for symptoms: '{symptoms}', provider: '{provider}', user: '{user_context.user_id}'")

    if not get_user_tier_capability(user_context.user_id, 'medical_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "permission_denied",
                                    {"tool_name": "check_symptoms", "symptoms": symptoms, "provider": provider}, success=False)
        return "Error: Access to medical tools is not enabled for your current tier."
    
    # In a real scenario, this would format symptoms for the chosen API
    # and call make_api_request. For now, it's a mock response.
    # Example for a mock API that accepts a comma-separated symptom list:
    # params = {"symptoms": ",".join(symptoms)}
    # api_data = await make_api_request(provider, "check_symptoms", params, user_api_keys, "medical", user_context.user_id)

    # Mock response for demonstration
    mock_conditions = []
    if "fever" in symptoms and "cough" in symptoms:
        mock_conditions.append("Common Cold")
    if "headache" in symptoms and "nausea" in symptoms:
        mock_conditions.append("Migraine")
    if not mock_conditions:
        mock_conditions.append("No specific conditions immediately identifiable from these symptoms. Please consult a medical professional.")

    result_str = f"Based on the symptoms '{', '.join(symptoms)}', potential conditions include: {', '.join(mock_conditions)}."
    
    analytics_tracker.log_event(user_context.user_id, "tool_usage", "success",
                                {"tool_name": "check_symptoms", "symptoms": symptoms, "provider": provider, "result": result_str}, success=True)
    return result_str


@tool
async def get_hospital_info(
    hospital_name: str,
    user_context: Optional[UserProfile] = None,
    provider: str = "google_places", # Example provider for geographical/local search
    user_api_keys: List[str] = []
) -> str:
    """
    Retrieves information about a specific hospital, such as its address, contact details,
    and a brief overview. This tool would typically integrate with a geographical
    data API like Google Places API or a dedicated healthcare directory API.
    Requires 'medical_tool_access' capability.

    Args:
        hospital_name (str): The name of the hospital (e.g., "Mayo Clinic", "General Hospital").
        user_context (UserProfile, optional): The user's profile for RBAC checks. Defaults to None.
        provider (str, optional): The API provider to use. Defaults to "google_places".
        user_api_keys (list, optional): List of user-provided API keys.

    Returns:
        str: A formatted string containing hospital information or an error message.
    """
    if user_context is None:
        user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

    logger.info(f"Tool: get_hospital_info called for hospital: '{hospital_name}', provider: '{provider}', user: '{user_context.user_id}'")

    if not get_user_tier_capability(user_context.user_id, 'medical_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "permission_denied",
                                    {"tool_name": "get_hospital_info", "hospital_name": hospital_name, "provider": provider}, success=False)
        return "Error: Access to medical tools is not enabled for your current tier."

    # Example: Google Places API - Text Search
    # The actual implementation would call make_api_request with appropriate parameters
    # For now, we'll use a mock response.
    
    # params = {"query": f"{hospital_name} hospital", "type": "hospital"}
    # api_data = await make_api_request(provider, "place_search", params, user_api_keys, "medical", user_context.user_id)

    mock_data = {
        "Mayo Clinic": {
            "address": "200 1st St SW, Rochester, MN 55905, USA",
            "phone": "+1 507-284-2511",
            "website": "https://www.mayoclinic.org/",
            "rating": 4.7
        },
        "General Hospital": {
            "address": "123 Main St, Anytown, USA",
            "phone": "+1 555-123-4567",
            "website": "http://generalhospital.org",
            "rating": 3.9
        }
    }

    info = mock_data.get(hospital_name.replace(" Hospital", "")) # Simple matching
    if info:
        result_str = (
            f"**Information for {hospital_name}:**\n"
            f"- **Address:** {info.get('address', 'N/A')}\n"
            f"- **Phone:** {info.get('phone', 'N/A')}\n"
            f"- **Website:** {info.get('website', 'N/A')}\n"
            f"- **Rating:** {info.get('rating', 'N/A')}/5"
        )
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "success",
                                    {"tool_name": "get_hospital_info", "hospital_name": hospital_name, "provider": provider, "info": info}, success=True)
        return result_str
    else:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "no_data",
                                    {"tool_name": "get_hospital_info", "hospital_name": hospital_name, "provider": provider, "message": "No information found for hospital."}, success=False)
        return f"Could not retrieve information for hospital: {hospital_name}. Please check the name and try again."


@tool
async def medical_search_web(
    query: str,
    user_context: Optional[UserProfile] = None,
    max_chars: int = 2000
) -> str:
    """
    Searches the web for medical or health-related information using a smart search fallback mechanism.
    This tool wraps the generic `scrape_web` tool, providing a medical-specific interface.
    Requires 'web_search_enabled' capability.
    
    Args:
        query (str): The medical-related search query (e.g., "latest research on cancer treatment", "symptoms of flu").
        user_context (UserProfile, optional): The user's profile for RBAC checks and logging. Defaults to None.
        max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
    
    Returns:
        str: A string containing relevant information from the web.
    """
    if user_context is None:
        user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

    logger.info(f"Tool: medical_search_web called with query: '{query}' for user: '{user_context.user_id}'")
    
    if not get_user_tier_capability(user_context.user_id, 'web_search_enabled', False, user_tier=user_context.tier, user_roles=user_context.roles):
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "permission_denied",
                                    {"tool_name": "medical_search_web", "query": query}, success=False)
        return "Error: Web search is not enabled for your current tier."

    try:
        # Call the standalone scrape_web function
        result = await scrape_web(query=query, user_context=user_context, max_chars=max_chars)
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "success",
                                    {"tool_name": "medical_search_web", "query": query, "result_length": len(result)}, success=True)
        return result
    except Exception as e:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "error",
                                    {"tool_name": "medical_search_web", "query": query, "error": str(e)}, success=False)
        return f"Error during medical web search: {e}"


@tool
async def medical_query_uploaded_docs(
    query: str,
    user_context: Optional[UserProfile] = None,
    export: Optional[bool] = False,
    k: int = 5,
    document_tools: Optional[DocumentTools] = None # Accept DocumentTools instance
) -> str:
    """
    Queries previously uploaded and indexed medical documents for a user using vector similarity search.
    This tool wraps the generic `DocumentTools.document_query_uploaded_docs` tool, fixing the section to "medical".
    Requires 'document_query_enabled' capability.
    
    Args:
        query (str): The search query to find relevant medical documents (e.g., "patient history for John Doe", "latest clinical trials").
        user_context (UserProfile, optional): The user's profile for RBAC checks and logging. Defaults to None.
        export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
        k (int): The number of top relevant documents to retrieve. Defaults to 5.
        document_tools (DocumentTools, optional): The DocumentTools instance. Required for this function.

    Returns:
        str: A string containing the combined content of the relevant document chunks,
             or a message indicating no data/results found, or the export path if exported.
    """
    if user_context is None:
        user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

    logger.info(f"Tool: medical_query_uploaded_docs called with query: '{query}' for user: '{user_context.user_id}'")
    
    if not get_user_tier_capability(user_context.user_id, 'document_query_enabled', False, user_tier=user_context.tier, user_roles=user_context.roles):
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "permission_denied",
                                    {"tool_name": "medical_query_uploaded_docs", "query": query}, success=False)
        return "Error: Document querying is not enabled for your current tier."

    if not document_tools:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "error",
                                    {"tool_name": "medical_query_uploaded_docs", "query": query, "error": "DocumentTools instance not provided."}, success=False)
        return "Error: Document tools are not initialized. Cannot query uploaded documents."
    
    try:
        result = await document_tools.document_query_uploaded_docs(
            query_text=query, # Using query_text as per DocumentTools signature
            user_context=user_context,
            section="medical",
            export=export,
            k=k
        )
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "success",
                                    {"tool_name": "medical_query_uploaded_docs", "query": query, "result_length": len(result)}, success=True)
        return result
    except Exception as e:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "error",
                                    {"tool_name": "medical_query_uploaded_docs", "query": query, "error": str(e)}, success=False)
        return f"Error querying uploaded medical documents: {e}"


@tool
async def medical_summarize_document_by_path(
    file_path_str: str,
    user_context: Optional[UserProfile] = None,
    document_tools: Optional[DocumentTools] = None # Accept DocumentTools instance
) -> str:
    """
    Summarizes a document related to medicine or health located at the given file path.
    This tool wraps the generic `DocumentTools.document_summarize_document_by_path` tool.
    Requires 'summarization_enabled' capability.
    
    Args:
        file_path_str (str): The full path to the document file to be summarized.
                                Example: "uploads/default/medical/patient_notes.pdf"
        user_context (UserProfile, optional): The user's profile for RBAC checks and logging. Defaults to None.
        document_tools (DocumentTools, optional): The DocumentTools instance. Required for this function.
        
    Returns:
        str: A concise summary of the document content.
    """
    if user_context is None:
        user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

    logger.info(f"Tool: medical_summarize_document_by_path called for file: '{file_path_str}' by user: '{user_context.user_id}'")
    
    if not get_user_tier_capability(user_context.user_id, 'summarization_enabled', False, user_tier=user_context.tier, user_roles=user_context.roles):
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "permission_denied",
                                    {"tool_name": "medical_summarize_document_by_path", "file_path": file_path_str}, success=False)
        return "Error: Document summarization is not enabled for your current tier."

    if not document_tools:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "error",
                                    {"tool_name": "medical_summarize_document_by_path", "file_path": file_path_str, "error": "DocumentTools instance not provided."}, success=False)
        return "Error: Document tools are not initialized. Cannot summarize documents."

    try:
        result = await document_tools.document_summarize_document_by_path(
            file_path_str=file_path_str,
            user_context=user_context
        )
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "success",
                                    {"tool_name": "medical_summarize_document_by_path", "file_path": file_path_str, "result_length": len(result)}, success=True)
        return result
    except Exception as e:
        analytics_tracker.log_event(user_context.user_id, "tool_usage", "error",
                                    {"tool_name": "medical_summarize_document_by_path", "file_path": file_path_str, "error": str(e)}, success=False)
        return f"Error summarizing document: {e}"


# --- MedicalTools Class (Wrapper) ---
class MedicalTools:
    """
    A collection of tools for medical-related operations.
    This class acts primarily as a wrapper to expose the standalone tool functions
    as methods, ensuring a consistent interface.
    """
    def __init__(self, config_manager: Any, log_event: Any, document_tools: DocumentTools):
        self.config_manager = config_manager
        self.log_event = log_event
        self.document_tools = document_tools
        logger.info("MedicalTools initialized.")

    async def get_drug_info(
        self,
        drug_name: str,
        user_context: Optional[UserProfile] = None,
        provider: str = "openfda",
        user_api_keys: List[str] = []
    ) -> str:
        """
        Retrieves information about a specific drug.
        """
        return await get_drug_info(
            drug_name=drug_name,
            user_context=user_context,
            provider=provider,
            user_api_keys=user_api_keys
        )

    async def check_symptoms(
        self,
        symptoms: List[str],
        user_context: Optional[UserProfile] = None,
        provider: str = "apimedic",
        user_api_keys: List[str] = []
    ) -> str:
        """
        Checks common conditions or potential causes based on a list of symptoms.
        """
        return await check_symptoms(
            symptoms=symptoms,
            user_context=user_context,
            provider=provider,
            user_api_keys=user_api_keys
        )

    async def get_hospital_info(
        self,
        hospital_name: str,
        user_context: Optional[UserProfile] = None,
        provider: str = "google_places",
        user_api_keys: List[str] = []
    ) -> str:
        """
        Retrieves information about a specific hospital.
        """
        return await get_hospital_info(
            hospital_name=hospital_name,
            user_context=user_context,
            provider=provider,
            user_api_keys=user_api_keys
        )

    async def medical_search_web(
        self,
        query: str,
        user_context: Optional[UserProfile] = None,
        max_chars: int = 2000
    ) -> str:
        """
        Searches the web for medical or health-related information.
        """
        return await medical_search_web(
            query=query,
            user_context=user_context,
            max_chars=max_chars
        )

    async def medical_query_uploaded_docs(
        self,
        query: str,
        user_context: Optional[UserProfile] = None,
        export: Optional[bool] = False,
        k: int = 5
    ) -> str:
        """
        Queries previously uploaded and indexed medical documents for a user.
        """
        # Pass the document_tools instance explicitly to the standalone function
        return await medical_query_uploaded_docs(
            query=query,
            user_context=user_context,
            export=export,
            k=k,
            document_tools=self.document_tools
        )

    async def medical_summarize_document_by_path(
        self,
        file_path_str: str,
        user_context: Optional[UserProfile] = None
    ) -> str:
        """
        Summarizes a document related to medicine or health located at the given file path.
        """
        # Pass the document_tools instance explicitly to the standalone function
        return await medical_summarize_document_by_path(
            file_path_str=file_path_str,
            user_context=user_context,
            document_tools=self.document_tools
        )


# CLI Test (optional)
if __name__ == "__main__":
    import asyncio
    from unittest.mock import MagicMock, AsyncMock, patch, ANY
    import shutil
    import os
    import sys
    from pathlib import Path
    
    try:
        from shared_tools.vector_utils import BASE_VECTOR_DIR
    except ImportError:
        BASE_VECTOR_DIR = Path("./mock_vector_dir")
        
    try:
        from database.firestore_manager import FirestoreManager
    except ImportError:
        class FirestoreManager: pass

    try:
        from shared_tools.cloud_storage_utils import CloudStorageUtilsWrapper
    except ImportError:
        class CloudStorageUtilsWrapper: pass

    try:
        from shared_tools.vector_utils import VectorUtilsWrapper
    except ImportError:
        class VectorUtilsWrapper: pass

    try:
        from domain_tools.document_tools.document_tool import DocumentTools 
    except ImportError:
        class DocumentTools:
            def __init__(self, *args, **kwargs): pass
            async def document_query_uploaded_docs(self, query_text, user_context, section, export, k): return f"Mocked document query for {section} with query '{query_text}'"
            async def document_summarize_document_by_path(self, file_path_str, user_context): return f"Mocked summary of {file_path_str}"

    try:
        from shared_tools.scraper_tool import scrape_web # For patching scrape_web
    except ImportError:
        async def scrape_web(*args, **kwargs): return "Mocked web search results."

    # Mock UserProfile
    mock_user_free_profile = UserProfile(user_id="mock_free_token", username="FreeUser", email="free@example.com", tier="free", roles=["user"])
    mock_user_pro_profile = UserProfile(user_id="mock_pro_token", username="ProUser", email="pro@example.com", tier="pro", roles=["user"])

    logging.basicConfig(level=logging.INFO)

    class MockSecrets:
        def __init__(self):
            self.openfda_api_key = "MOCK_OPENFDA_API_KEY" # Placeholder
            self.serpapi_api_key = "MOCK_SERPAPI_KEY_LIVE"
            self.openai_api_key = "sk-mock-openai-key-12345"
            self.google_api_key = "AIzaSy-mock-google-key"

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
                    'timeout_seconds': 1
                },
                'tiers': {},
                'default_user_tier': 'free',
                'default_user_roles': ['user'],
                'api_defaults': {
                    'medical': 'openfda',
                    'web_search': 'serpapi',
                    'document_summarization_llm': 'openai'
                },
                'analytics': {
                    'enabled': True,
                    'log_tool_usage': True,
                    'log_query_failures': True
                }
            }
            self._api_providers_data = {
                "medical": {
                    "openfda": {
                        "base_url": "https://api.fda.gov",
                        "api_key_name": "openfda_api_key",
                        "api_key_param_name": "api_key",
                        "functions": {
                            "get_drug_info": {
                                "endpoint": "/drug/ndc.json", # Simplified for mock
                                "required_params": ["search"],
                                "optional_params": ["limit"],
                                "response_path": [],
                                "data_map": {}
                            }
                        }
                    },
                    "apimedic": { # Placeholder for a symptom checker
                        "base_url": "https://api.apimedic.com",
                        "api_key_name": "apimedic_api_key",
                        "api_key_param_name": "api_key",
                        "functions": {
                            "check_symptoms": {
                                "endpoint": "/symptoms",
                                "required_params": ["symptoms"],
                                "response_path": [],
                                "data_map": {}
                            }
                        }
                    },
                    "google_places": { # Placeholder for hospital info
                        "base_url": "https://maps.googleapis.com/maps/api/place",
                        "api_key_name": "google_api_key",
                        "api_key_param_name": "key",
                        "functions": {
                            "place_search": {
                                "endpoint": "/textsearch/json",
                                "required_params": ["query"],
                                "optional_params": ["type"],
                                "response_path": ["results"],
                                "data_map": {}
                            }
                        }
                    }
                },
                "web_search": {
                    "serpapi": {
                        "base_url": "https://serpapi.com/search",
                        "api_key_name": "serpapi_api_key",
                        "api_key_param_name": "api_key",
                        "functions": {
                            "scrape_web": {
                                "required_params": ["q"],
                                "optional_params": ["engine"],
                                "response_path": ["organic_results"],
                                "data_map": {
                                    "title": "title",
                                    "link": "link",
                                    "snippet": "snippet"
                                }
                            }
                        }
                    }
                },
                "document_summarization_llm": {
                    "openai": {
                        "base_url": "https://api.openai.com/v1/chat/completions",
                        "api_key_name": "openai_api_key",
                        "functions": {
                            "summarize_document": {
                                "endpoint": "",
                                "required_params": [],
                                "optional_params": [],
                                "response_path": ["choices", 0, "message", "content"],
                                "data_map": {}
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


    class MockUserManager:
        _mock_users = {
            "mock_free_token": {"user_id": "mock_free_token", "username": "FreeUser", "email": "free@example.com", "tier": "free", "roles": ["user"]},
            "mock_pro_token": {"user_id": "mock_pro_token", "username": "ProUser", "email": "pro@example.com", "tier": "pro", "roles": ["user"]},
            "mock_premium_token": {"user_id": "mock_premium_token", "username": "PremiumUser", "email": "premium@example.com", "tier": "premium", "roles": ["user"]},
            "mock_admin_token": {"user_id": "mock_admin_token", "username": "AdminUser", "email": "admin@example.com", "tier": "admin", "roles": ["user", "admin"]},
        }
        _rbac_capabilities = {
            'capabilities': {
                'medical_tool_access': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'web_search_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
                'document_query_enabled': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'summarization_enabled': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'llm_default_provider': {
                    'default': 'gemini',
                    'tiers': {'pro': 'gemini', 'premium': 'openai', 'admin': 'gemini'}
                },
                'llm_default_model_name': {
                    'default': 'gemini-1.5-flash',
                    'tiers': {'pro': 'gemini-1.5-flash', 'premium': 'gpt-4o', 'admin': 'gemini-1.5-flash'}
                },
                'llm_default_temperature': {
                    'default': 0.7,
                    'tiers': {'pro': 0.5, 'premium': 0.3, 'admin': 0.7}
                },
            }
        }
        _tier_hierarchy = {
            "free": 0, "user": 1, "basic": 2, "pro": 3, "premium": 4, "admin": 99
        }

        def get_user_tier_capability(self, user_id: str, capability_key: str, default_value: Any = None, user_tier: Optional[str] = None, user_roles: Optional[List[str]] = None) -> Any:
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

            for role in user_roles:
                if role in capability_config.get('roles', {}):
                    return capability_config['roles'][role]
            
            if user_tier in capability_config.get('tiers', {}):
                return capability_config['tiers'][user_tier]

            return capability_config.get('default', default_value)


    import streamlit as st_mock
    if not hasattr(st_mock, 'secrets'):
        st_mock.secrets = MockSecrets()
    
    if 'config.config_manager' not in sys.modules:
        sys.modules['config.config_manager'] = MagicMock()
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager
    
    if 'utils.user_manager' not in sys.modules:
        sys.modules['utils.user_manager'] = MagicMock()
    sys.modules['utils.user_manager'].get_user_tier_capability = MockUserManager().get_user_tier_capability

    mock_firestore_manager_for_analytics = MagicMock(spec=FirestoreManager)
    mock_firestore_manager_for_analytics.collection.return_value.add = AsyncMock(return_value=MagicMock(id="mock_doc_id"))

    mock_auth_for_analytics = MagicMock()
    mock_auth_for_analytics.currentUser = MagicMock(uid="mock_user_123")
    
    with patch.dict(sys.modules, {'firebase_admin.firestore': MagicMock(firestore=MagicMock())}):
        sys.modules['firebase_admin.firestore'].firestore.CollectionReference = MagicMock()
        sys.modules['firebase_admin'].firestore.DocumentReference = MagicMock()
        
        analytics_tracker.initialize_analytics(
            mock_firestore_manager_for_analytics,
            mock_auth_for_analytics,
            "test_app_id_for_analytics",
            "mock_user_123"
        )

        original_requests_get = requests.get
        original_requests_post = requests.post

        def mock_requests_dynamic(method, url, params=None, headers=None, json=None, timeout=None):
            logger.info(f"Mocking {method} API request to {url} with params: {params or json}")
            if "api.fda.gov" in url:
                if "/drug/ndc.json" in url:
                    search_query = params.get("search", "").lower()
                    if "aspirin" in search_query:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "results": [
                                {
                                    "openfda": {"brand_name": ["Aspirin"], "generic_name": ["ASPIRIN"]},
                                    "purpose": [{"description": ["For temporary relief of minor aches and pains."]}],
                                    "indications_and_usage": ["Used for pain relief, fever reduction, and anti-inflammatory purposes."],
                                    "warnings": ["Reye's syndrome warning. Consult doctor before use."],
                                    "adverse_reactions": ["Stomach upset, heartburn."]
                                }
                            ]
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"results": []}
                        return mock_response
            
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

            if "api.openai.com/v1/chat/completions" in url:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Mocked LLM summary content."}}]
                }
                return mock_response

            if method == "GET":
                return original_requests_get(url, params=params, headers=headers, timeout=timeout)
            elif method == "POST":
                return original_requests_post(url, json=json, headers=headers, timeout=timeout)
            else:
                raise NotImplementedError(f"Mock for method {method} not implemented.")

        requests.get = MagicMock(side_effect=lambda url, params=None, headers=None, timeout=None: mock_requests_dynamic("GET", url, params, headers, timeout=timeout))
        requests.post = MagicMock(side_effect=lambda url, json=None, headers=None, timeout=None: mock_requests_dynamic("POST", url, json=json, headers=headers, timeout=timeout))

        mock_firestore_manager_instance = MagicMock(spec=FirestoreManager)
        mock_cloud_storage_utils_instance = MagicMock(spec=CloudStorageUtilsWrapper)
        mock_vector_utils_instance = MagicMock(spec=VectorUtilsWrapper)
        
        mock_document_tools_instance = DocumentTools(
            config_manager=sys.modules['config.config_manager'].config_manager,
            firestore_manager=mock_firestore_manager_instance,
            cloud_storage_utils=mock_cloud_storage_utils_instance,
            vector_utils=mock_vector_utils_instance,
            log_event=analytics_tracker.log_event
        )

        medical_tools_instance = MedicalTools(
            config_manager=sys.modules['config.config_manager'].config_manager,
            log_event=analytics_tracker.log_event,
            document_tools=mock_document_tools_instance
        )

        async def run_medical_tests(medical_tools_instance):
            print("\n--- Testing medical_tool functions with Live API Simulation and Analytics ---")

            # Test 1: get_drug_info (success)
            print("\n--- Test 1: get_drug_info (Success) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            result_drug_info = await medical_tools_instance.get_drug_info("aspirin", user_context=mock_user_pro_profile)
            print(f"Drug Info: {result_drug_info}")
            assert "Drug Information for Aspirin (Generic: ASPIRIN):" in result_drug_info
            assert "For temporary relief of minor aches and pains." in result_drug_info
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "get_drug_info"
            assert logged_data["success"] is True
            print("Test 1 Passed.")

            # Test 2: get_drug_info (no data found)
            print("\n--- Test 2: get_drug_info (No Data Found) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            result_drug_info_fail = await medical_tools_instance.get_drug_info("nonexistentdrug", user_context=mock_user_pro_profile)
            print(f"Drug Info (No Data): {result_drug_info_fail}")
            assert "Could not retrieve information for the drug: Nonexistentdrug." in result_drug_info_fail
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "get_drug_info"
            assert logged_data["success"] is False
            assert "No drug information found." in logged_data["message"]
            print("Test 2 Passed.")

            # Test 3: check_symptoms (success)
            print("\n--- Test 3: check_symptoms (Success) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            result_symptoms = await medical_tools_instance.check_symptoms(["fever", "cough"], user_context=mock_user_pro_profile)
            print(f"Symptoms Check: {result_symptoms}")
            assert "Based on the symptoms 'fever, cough', potential conditions include: Common Cold." in result_symptoms
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "check_symptoms"
            assert logged_data["success"] is True
            print("Test 3 Passed.")

            # Test 4: get_hospital_info (success)
            print("\n--- Test 4: get_hospital_info (Success) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            result_hospital = await medical_tools_instance.get_hospital_info("Mayo Clinic", user_context=mock_user_pro_profile)
            print(f"Hospital Info: {result_hospital}")
            assert "Information for Mayo Clinic:" in result_hospital
            assert "200 1st St SW, Rochester, MN 55905, USA" in result_hospital
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "get_hospital_info"
            assert logged_data["success"] is True
            print("Test 4 Passed.")

            # Test 5: medical_search_web (generic tool)
            print("\n--- Test 5: medical_search_web (Generic Tool) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            result_web_search = await medical_tools_instance.medical_search_web("latest medical breakthroughs", user_context=mock_user_pro_profile)
            print(f"Web Search Result: {result_web_search[:100]}...")
            assert "Mocked web search results." in result_web_search
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "medical_search_web"
            assert logged_data["success"] is True
            print("Test 5 Passed.")

            # Test 6: medical_query_uploaded_docs (generic tool via DocumentTools)
            print("\n--- Test 6: medical_query_uploaded_docs (Generic Tool via DocumentTools) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            result_doc_query = await medical_tools_instance.medical_query_uploaded_docs("patient records for emergency", user_context=mock_user_pro_profile)
            print(f"Document Query Result: {result_doc_query}")
            assert "Mocked document query for medical with query 'patient records for emergency'" in result_doc_query
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "medical_query_uploaded_docs"
            assert logged_data["success"] is True
            print("Test 6 Passed.")

            # Test 7: medical_summarize_document_by_path (generic tool via DocumentTools)
            print("\n--- Test 7: medical_summarize_document_by_path (Generic Tool via DocumentTools) ---")
            mock_firestore_manager_for_analytics.collection.return_value.add.reset_mock()
            test_user_pro_dir = Path("uploads") / mock_user_pro_profile.user_id
            dummy_file_path = test_user_pro_dir / "medical" / "dummy_patient_notes.txt"
            dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_file_path.write_text("This is a dummy patient note content for testing summarization.")

            result_summarize = await medical_tools_instance.medical_summarize_document_by_path(str(dummy_file_path), user_context=mock_user_pro_profile)
            print(f"Summarize Result: {result_summarize}")
            assert "Mocked summary of uploads" in result_summarize
            mock_firestore_manager_for_analytics.collection.return_value.add.assert_called_once()
            args, kwargs = mock_firestore_manager_for_analytics.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "medical_summarize_document_by_path"
            assert logged_data["success"] is True
            print("Test 7 Passed.")

            print("\nAll medical_tool tests with live API simulation and analytics considerations completed.")

        if __name__ == "__main__":
            asyncio.run(run_medical_tests(medical_tools_instance))

        requests.get = original_requests_get
        requests.post = original_requests_post

        test_user_dirs = [Path("uploads") / mock_user_pro_profile.user_id, BASE_VECTOR_DIR / mock_user_pro_profile.user_id]
        for d in test_user_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print(f"Cleaned up {d}")
