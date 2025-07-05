# domain_tools/medical_tools/medical_tool.py

import logging
import requests
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

# Import generic tools
from langchain_core.tools import tool
from shared_tools.query_uploaded_docs_tool import QueryUploadedDocs
from shared_tools.scraper_tool import scrape_web
from shared_tools.doc_summarizer import summarize_document

# Import config_manager to access API configurations and secrets
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability
# Import date_parser for date format flexibility (not directly used by current tools, but available)
from utils.date_parser import parse_date_to_yyyymmdd 

logger = logging.getLogger(__name__)

# --- Generic API Request Helper (copied for standalone tool file, ideally in shared utils) ---

def _get_nested_value(data: Dict[str, Any], path: List[str]):
    """Helper to get a value from a nested dictionary using a list of keys."""
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and key.isdigit(): # Handle list indices
            try:
                current = current[int(key)]
            except (IndexError, ValueError):
                return None
        else:
            return None
    return current

def _make_dynamic_api_request(
    domain: str,
    function_name: str,
    params: Dict[str, Any],
    user_token: str
) -> Optional[Dict[str, Any]]:
    """
    Makes an API request to the dynamically configured provider for a given domain and function.
    Handles API key retrieval, request construction, and basic error handling.
    Returns parsed JSON data or None on failure (triggering mock fallback).
    """
    # Get the default active API provider for the domain from config.yml
    active_provider_name = config_manager.get(f"api_defaults.{domain}")
    if not active_provider_name:
        logger.error(f"No default API provider configured for domain '{domain}'.")
        return None

    # Get the full configuration for the active provider from api_providers.yml
    provider_config = config_manager.get_api_provider_config(domain, active_provider_name)
    if not provider_config:
        logger.error(f"Configuration for API provider '{active_provider_name}' in domain '{domain}' not found in api_providers.yml.")
        return None

    base_url = provider_config.get("base_url")
    api_key_name = provider_config.get("api_key_name")
    api_key = config_manager.get_secret(api_key_name) if api_key_name else None

    # Special handling for Amadeus which uses client_id and client_secret for token
    if active_provider_name == "amadeus":
        api_secret_name = provider_config.get("api_secret_name")
        api_secret = config_manager.get_secret(api_secret_name) if api_secret_name else None
        token_endpoint = provider_config.get("token_endpoint")

        if not api_key or not api_secret or not token_endpoint:
            logger.warning(f"Amadeus API credentials (client_id/secret) or token_endpoint missing. Cannot make live Amadeus call.")
            return None
        
        # Get Amadeus access token (simplified for demonstration)
        try:
            token_response = requests.post(
                token_endpoint,
                data={'grant_type': 'client_credentials', 'client_id': api_key, 'client_secret': api_secret},
                timeout=5
            )
            token_response.raise_for_status()
            access_token = token_response.json().get('access_token')
            if not access_token:
                logger.error("Failed to get Amadeus access token.")
                return None
            headers = {"Authorization": f"Bearer {access_token}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting Amadeus access token: {e}")
            return None
    else:
        headers = {} # No special headers by default

    if not base_url:
        logger.error(f"Base URL not configured for API provider '{active_provider_name}' in domain '{domain}'.")
        return None

    function_details = provider_config.get("functions", {}).get(function_name)
    if not function_details:
        logger.error(f"Function '{function_name}' not configured for API provider '{active_provider_name}' in domain '{domain}'.")
        return None

    endpoint = function_details.get("endpoint")
    function_param = function_details.get("function_param") # For Alpha Vantage style 'function' param
    path_params = function_details.get("path_params", []) # For ExchangeRate-API style path params

    if not endpoint and not function_param:
        logger.error(f"Neither 'endpoint' nor 'function_param' defined for function '{function_name}'.")
        return None

    # Construct URL
    full_url = f"{base_url}{endpoint}" if endpoint else base_url

    # Add path parameters to URL if specified
    for p_param in path_params:
        if p_param in params:
            # Ensure path parameters are correctly formatted (e.g., uppercase for currencies)
            value = str(params.pop(p_param))
            full_url = full_url.replace(f"{{{p_param}}}", value)
        else:
            logger.warning(f"Missing path parameter '{p_param}' for function '{function_name}'.")
            return None # Cannot construct URL without required path params

    # Construct query parameters
    query_params = {}
    if function_param:
        query_params["function"] = function_param # Alpha Vantage specific

    # Add API key if it's a query param (not in path or header)
    if api_key_name and active_provider_name not in ["amadeus", "exchangerate_api"]: # Amadeus handled by headers, ExchangeRate by path
        param_name_in_url = provider_config.get("api_key_param_name", api_key_name.replace("_api_key", ""))
        if api_key: # Only add if key exists
            query_params[param_name_in_url] = api_key 
    elif active_provider_name == "exchangerate_api" and api_key:
        # For ExchangeRate-API, the key is a path parameter, already handled above.
        # This 'elif' ensures we don't add it as a query param if it's already in the path.
        pass


    for param_key in function_details.get("required_params", []) + function_details.get("optional_params", []):
        if param_key in params:
            query_params[param_key] = params[param_key]
        elif param_key in function_details.get("required_params", []):
            logger.warning(f"Missing required parameter '{param_key}' for function '{function_name}'.")
            return None # Missing required param, cannot proceed

    try:
        logger.info(f"Making API call to: {full_url} with params: {query_params}")
        response = requests.get(full_url, params=query_params, headers=headers, timeout=config_manager.get("web_scraping.timeout_seconds", 15))
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        raw_data = response.json()
        
        # Check for API-specific error messages in the response body
        if "Error Message" in raw_data: # Alpha Vantage specific
            logger.error(f"API Error from {active_provider_name}: {raw_data['Error Message']}")
            return None
        if "Note" in raw_data and "Thank you for using Alpha Vantage!" in raw_data["Note"]: # Alpha Vantage rate limit
            logger.warning(f"API rate limit hit for {active_provider_name}: {raw_data['Note']}")
            return None
        if raw_data.get("status") == "error": # NewsAPI specific
            logger.error(f"API Error from {active_provider_name}: {raw_data.get('message', 'Unknown error')}")
            return None
        if raw_data.get("Error"): # OMDBAPI specific
            logger.error(f"API Error from {active_provider_name}: {raw_data.get('Error')}")
            return None
        if raw_data.get("status") and raw_data["status"].get("error_code"): # CoinGecko error
            logger.error(f"API Error from {active_provider_name}: {raw_data['status'].get('error_message', 'Unknown CoinGecko error')}")
            return None
        if raw_data.get("result") == "error": # ExchangeRate-API error
            logger.error(f"API Error from {active_provider_name}: {raw_data.get('error-type', 'Unknown ExchangeRate-API error')}")
            return None


        # Extract data based on response_path
        data_to_map = raw_data
        response_path = function_details.get("response_path")
        if response_path:
            data_to_map = _get_nested_value(raw_data, response_path)
            if data_to_map is None:
                logger.warning(f"Response path '{'.'.join(response_path)}' not found in API response from {active_provider_name}. Raw data: {raw_data}")
                return None

        # Apply data mapping
        mapped_data = {}
        data_map = function_details.get("data_map", {})
        if isinstance(data_to_map, list): # For lists of items (e.g., news articles, historical data)
            mapped_data_list = []
            for item in data_to_map:
                mapped_item = {}
                for mapped_key, original_key_path in data_map.items():
                    if isinstance(original_key_path, list): # Handle nested paths in data_map
                        mapped_item[mapped_key] = _get_nested_value(item, original_key_path)
                    elif '.' in str(original_key_path): # Handle dot-separated paths in data_map
                        mapped_item[mapped_key] = _get_nested_value(item, original_key_path.split('.'))
                    else: # Direct key or list index
                        if isinstance(original_key_path, int) and isinstance(item, list):
                            try: mapped_item[mapped_key] = item[original_key_path]
                            except IndexError: mapped_item[mapped_key] = None
                        else:
                            mapped_item[mapped_key] = item.get(original_key_path)
                mapped_data_list.append(mapped_item)
            return {"data": mapped_data_list} # Wrap list in a dict for consistent return
        elif isinstance(data_to_map, dict) and function_name == "get_historical_stock_prices" and active_provider_name == "alphavantage":
            # Special handling for Alpha Vantage TIME_SERIES_DAILY where keys are dates
            processed_data = {}
            for date_key, values in data_to_map.items():
                mapped_values = {}
                for mapped_key, original_key_path in data_map.items():
                    if isinstance(original_key_path, list):
                        mapped_values[mapped_key] = _get_nested_value(values, original_key_path)
                    elif '.' in str(original_key_path):
                        mapped_values[mapped_key] = _get_nested_value(values, original_key_path.split('.'))
                    else:
                        mapped_values[mapped_key] = values.get(original_key_path)
                processed_data[date_key] = mapped_values
            return {"data": processed_data}
        else: # For single object responses
            # Special handling for CoinGecko simple price, where response is { "bitcoin": { "usd": 20000 } }
            if function_name == "get_crypto_price" and active_provider_name == "coingecko":
                # params will contain 'ids' and 'vs_currencies'
                crypto_id = params.get("ids", "").lower()
                currency = params.get("vs_currencies", "").lower()
                if crypto_id in raw_data and currency in raw_data[crypto_id]:
                    mapped_data["price"] = raw_data[crypto_id][currency]
                    if f"{currency}_market_cap" in raw_data[crypto_id]:
                        mapped_data["market_cap"] = raw_data[crypto_id][f"{currency}_market_cap"]
                    if f"{currency}_24hr_vol" in raw_data[crypto_id]:
                        mapped_data["vol_24hr"] = raw_data[crypto_id][f"{currency}_24hr_vol"]
                    if f"{currency}_24hr_change" in raw_data[crypto_id]:
                        mapped_data["change_24hr"] = raw_data[crypto_id][f"{currency}_24hr_change"]
                    if "last_updated_at" in raw_data[crypto_id]:
                        mapped_data["last_updated"] = raw_data[crypto_id]["last_updated_at"]
                    return mapped_data
                else:
                    logger.warning(f"CoinGecko simple price response unexpected for {crypto_id}/{currency}: {raw_data}")
                    return None
            
            for mapped_key, original_key_path in data_map.items():
                if isinstance(original_key_path, list):
                    mapped_data[mapped_key] = _get_nested_value(data_to_map, original_key_path)
                elif '.' in str(original_key_path):
                    mapped_data[mapped_key] = _get_nested_value(data_to_map, original_key_path.split('.'))
                else:
                    mapped_data[mapped_key] = data_to_map.get(original_key_path)
            return mapped_data

    except requests.exceptions.Timeout:
        logger.error(f"API request to {active_provider_name} timed out for function '{function_name}'.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making API request to {active_provider_name} for function '{function_name}': {e}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from {active_provider_name} for function '{function_name}'.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during API call to {active_provider_name} for '{function_name}': {e}", exc_info=True)
        return None


# --- Mock Data for Fallback ---
_mock_medical_data = {
    "drug_info": {
        "aspirin": {
            "name": "Aspirin",
            "description": "Aspirin is a salicylate drug, often used as an analgesic to relieve minor aches and pains, as an antipyretic to reduce fever, and as an anti-inflammatory medication.",
            "side_effects": ["Stomach upset", "heartburn", "drowsiness"],
            "dosage": "300-600mg every 4-6 hours"
        },
        "ibuprofen": {
            "name": "Ibuprofen",
            "description": "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used for treating pain, fever, and inflammation.",
            "side_effects": ["Nausea", "vomiting", "headache"],
            "dosage": "200-400mg every 4-6 hours"
        }
    },
    "symptom_info": {
        "headache": {
            "name": "Headache",
            "causes": ["Stress", "dehydration", "lack of sleep", "migraine", "tension"],
            "treatment": "Rest, hydration, pain relievers (e.g., ibuprofen, aspirin)",
            "when_to_see_doctor": "Sudden severe headache, headache with fever/stiff neck, vision changes, weakness/numbness"
        },
        "fever": {
            "name": "Fever",
            "causes": ["Infections (viral, bacterial)", "inflammation", "medication side effect"],
            "treatment": "Rest, fluids, fever reducers (e.g., acetaminophen, ibuprofen)",
            "when_to_see_doctor": "High fever in infants, fever with rash, difficulty breathing, persistent fever"
        }
    }
}

@tool
def get_drug_info(drug_name: str, user_token: str = "default") -> str:
    """
    Retrieves information about a specific drug, including its description,
    common side effects, and typical dosage.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        drug_name (str): The name of the drug (e.g., "Aspirin", "Ibuprofen").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A string containing the drug information, or an error/fallback message.
    """
    logger.info(f"Tool: get_drug_info called for drug: {drug_name} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'medical_tool_access', False):
        return "Error: Access to medical tools is not enabled for your current tier."

    api_data = _make_dynamic_api_request(
        "medical", "get_drug_info",
        {"name": drug_name},
        user_token
    )

    if api_data:
        try:
            name = api_data.get("drug_name")
            description = api_data.get("description")
            side_effects = api_data.get("side_effects")
            dosage = api_data.get("dosage")

            if name and description:
                response_str = (
                    f"Information for {name}:\n"
                    f"  Description: {description}\n"
                )
                if side_effects:
                    response_str += f"  Side Effects: {', '.join(side_effects)}\n"
                if dosage:
                    response_str += f"  Typical Dosage: {dosage}\n"
                return response_str
            else:
                logger.warning(f"Live API data for {drug_name} is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live drug information for {drug_name}. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live drug info data for {drug_name}: {e}")
            return f"Error parsing live data for {drug_name}. Falling back to mock data."

    # Fallback to mock data
    mock_data = _mock_medical_data.get("drug_info", {}).get(drug_name.lower())
    if mock_data:
        response_str = (
            f"Information for {mock_data['name']} (Mock Data Fallback):\n"
            f"  Description: {mock_data['description']}\n"
        )
        if mock_data.get('side_effects'):
            response_str += f"  Side Effects: {', '.join(mock_data['side_effects'])}\n"
        if mock_data.get('dosage'):
            response_str += f"  Typical Dosage: {mock_data['dosage']}\n"
        return response_str
    else:
        return f"Drug information not found for '{drug_name}'. (API/Mock Fallback Failed)"


@tool
def get_symptom_info(symptom_name: str, user_token: str = "default") -> str:
    """
    Retrieves information about a specific medical symptom, including its common causes,
    suggested treatments, and when to seek professional medical attention.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        symptom_name (str): The name of the symptom (e.g., "Headache", "Fever").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A string containing the symptom information, or an error/fallback message.
    """
    logger.info(f"Tool: get_symptom_info called for symptom: {symptom_name} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'medical_tool_access', False):
        return "Error: Access to medical tools is not enabled for your current tier."

    api_data = _make_dynamic_api_request(
        "medical", "get_symptom_info",
        {"name": symptom_name},
        user_token
    )

    if api_data:
        try:
            name = api_data.get("symptom_name")
            causes = api_data.get("causes")
            treatment = api_data.get("treatment")
            when_to_see_doctor = api_data.get("when_to_see_doctor")

            if name and causes and treatment:
                response_str = (
                    f"Information for {name}:\n"
                    f"  Common Causes: {', '.join(causes)}\n"
                    f"  Suggested Treatment: {treatment}\n"
                )
                if when_to_see_doctor:
                    response_str += f"  When to see a doctor: {when_to_see_doctor}\n"
                return response_str
            else:
                logger.warning(f"Live API data for {symptom_name} is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live symptom information for {symptom_name}. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live symptom info data for {symptom_name}: {e}")
            return f"Error parsing live data for {symptom_name}. Falling back to mock data."

    # Fallback to mock data
    mock_data = _mock_medical_data.get("symptom_info", {}).get(symptom_name.lower())
    if mock_data:
        response_str = (
            f"Information for {mock_data['name']} (Mock Data Fallback):\n"
            f"  Common Causes: {', '.join(mock_data['causes'])}\n"
            f"  Suggested Treatment: {mock_data['treatment']}\n"
        )
        if mock_data.get('when_to_see_doctor'):
            response_str += f"  When to see a doctor: {mock_data['when_to_see_doctor']}\n"
        return response_str
    else:
        return f"Symptom information not found for '{symptom_name}'. (API/Mock Fallback Failed)"


# --- Existing Generic Tools (not directly using external APIs, but can be used in medical context) ---

@tool
def medical_search_web(query: str, user_token: str = "default", max_chars: int = 2000) -> str:
    """
    Searches the web for medical or health-related information using a smart search fallback mechanism.
    This tool wraps the generic `scrape_web` tool, providing a medical-specific interface.
    
    Args:
        query (str): The medical/health-related search query (e.g., "latest research on cancer treatment", "symptoms of common cold").
        user_token (str): The unique identifier for the user. Defaults to "default".
        max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
    
    Returns:
        str: A string containing relevant information from the web.
    """
    logger.info(f"Tool: medical_search_web called with query: '{query}' for user: '{user_token}'")
    return scrape_web(query=query, user_token=user_token, max_chars=max_chars)

@tool
def medical_query_uploaded_docs(query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
    """
    Queries previously uploaded and indexed medical documents for a user using vector similarity search.
    This tool wraps the generic `QueryUploadedDocs` tool, fixing the section to "medical".
    
    Args:
        query (str): The search query to find relevant medical documents (e.g., "what are the side effects of drug X", "summary of patient's medical history").
        user_token (str): The unique identifier for the user. Defaults to "default".
        export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
        k (int): The number of top relevant documents to retrieve. Defaults to 5.
    
    Returns:
        str: A string containing the combined content of the relevant document chunks,
             or a message indicating no data/results found, or the export path if exported.
    """
    logger.info(f"Tool: medical_query_uploaded_docs called with query: '{query}' for user: '{user_token}'")
    return QueryUploadedDocs(query=query, user_token=user_token, section="medical", export=export, k=k)

@tool
def medical_summarize_document_by_path(file_path_str: str) -> str:
    """
    Summarizes a document related to medical or health information located at the given file path.
    The file path should be accessible by the system (e.g., in the 'uploads' directory).
    This tool wraps the generic `summarize_document` tool.
    
    Args:
        file_path_str (str): The full path to the document file to be summarized.
                              Example: "uploads/default/medical/patient_record.pdf"
    
    Returns:
        str: A concise summary of the document content.
    """
    logger.info(f"Tool: medical_summarize_document_by_path called for file: '{file_path_str}'")
    file_path = Path(file_path_str)
    if not file_path.exists():
        logger.error(f"Document not found at '{file_path_str}' for summarization.")
        return f"Error: Document not found at '{file_path_str}'."
    
    try:
        summary = summarize_document(file_path)
        return f"Summary of '{file_path.name}':\n{summary}"
    except ValueError as e:
        logger.error(f"Error summarizing document '{file_path_str}': {e}")
        return f"Error summarizing document: {e}"
    except Exception as e:
        logger.critical(f"An unexpected error occurred during summarization of '{file_path_str}': {e}", exc_info=True)
        return f"An unexpected error occurred during summarization: {e}"


# CLI Test (optional)
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, patch
    import shutil
    import os
    from shared_tools.vector_utils import BASE_VECTOR_DIR # For cleanup
    from shared_tools.python_interpreter_tool import python_interpreter_with_rbac # For testing REPL

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.health_api_key = "MOCK_HEALTH_API_KEY"
            self.openai_api_key = "sk-mock-openai-key-12345"
            self.google_api_key = "AIzaSy-mock-google-key"
            self.firebase_config = "{}"
            self.serpapi_api_key = "MOCK_SERPAPI_KEY" # For scrape_web

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
                    'medical': 'health_api'
                }
            }
            self._api_providers_data = { # Mock api_providers_data for medical
                "medical": {
                    "health_api": {
                        "base_url": "https://api.example.com/health",
                        "api_key_name": "health_api_key",
                        "api_key_param_name": "apikey",
                        "functions": {
                            "get_drug_info": {
                                "endpoint": "/drugs",
                                "required_params": ["name"],
                                "response_path": ["data", 0], # Assuming first result is most relevant
                                "data_map": {
                                    "drug_name": "name",
                                    "description": "description",
                                    "side_effects": "side_effects",
                                    "dosage": "dosage"
                                }
                            },
                            "get_symptom_info": {
                                "endpoint": "/symptoms",
                                "required_params": ["name"],
                                "response_path": ["data", 0],
                                "data_map": {
                                    "symptom_name": "name",
                                    "causes": "causes",
                                    "treatment": "treatment",
                                    "when_to_see_doctor": "when_to_see_doctor"
                                }
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


    # Mock user_manager.get_current_user and get_user_tier_capability for testing RBAC
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
                'data_analysis_enabled': { # For python interpreter
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'web_search_max_results': {
                    'default': 2,
                    'tiers': {'pro': 7, 'premium': 15}
                },
                'web_search_limit_chars': {
                    'default': 500,
                    'tiers': {'pro': 3000, 'premium': 10000}
                }
            }
        }
        _tier_hierarchy = {
            "free": 0, "user": 1, "basic": 2, "pro": 3, "premium": 4, "admin": 99
        }

        def get_current_user(self) -> Dict[str, Any]:
            return getattr(self, '_current_mock_user', {})

        def get_user_tier_capability(self, user_token: Optional[str], capability_key: str, default_value: Any = None) -> Any:
            user_info = self._mock_users.get(user_token, {})
            user_id = user_info.get('user_id')
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
    
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager
    sys.modules['utils.user_manager'] = MockUserManager()
    sys.modules['utils.user_manager'].get_user_tier_capability = MockUserManager().get_user_tier_capability # Patch the function directly

    # Mock requests.get for external API calls
    original_requests_get = requests.get

    def mock_requests_get_dynamic(url, params, headers, timeout):
        # Simulate hypothetical Health API responses
        if "api.example.com/health" in url:
            if "/drugs" in url:
                drug_name = params.get("name", "").lower()
                if drug_name == "aspirin":
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "data": [{
                            "name": "Aspirin",
                            "description": "A salicylate drug used for pain, fever, and inflammation.",
                            "side_effects": ["Stomach upset", "heartburn"],
                            "dosage": "325mg daily"
                        }]
                    }
                    return mock_response
                elif drug_name == "ibuprofen":
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "data": [{
                            "name": "Ibuprofen",
                            "description": "A nonsteroidal anti-inflammatory drug (NSAID) for pain and inflammation.",
                            "side_effects": ["Nausea", "headache"],
                            "dosage": "200-400mg every 4-6 hours"
                        }]
                    }
                    return mock_response
                else: # No data found
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"data": []}
                    return mock_response
            elif "/symptoms" in url:
                symptom_name = params.get("name", "").lower()
                if symptom_name == "headache":
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "data": [{
                            "name": "Headache",
                            "causes": ["Stress", "dehydration"],
                            "treatment": "Rest, pain relievers",
                            "when_to_see_doctor": "Sudden severe pain, vision changes"
                        }]
                    }
                    return mock_response
                elif symptom_name == "fever":
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "data": [{
                            "name": "Fever",
                            "causes": ["Infection"],
                            "treatment": "Fluids, fever reducers",
                            "when_to_see_doctor": "High fever in infants, persistent fever"
                        }]
                    }
                    return mock_response
                else: # No data found
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"data": []}
                    return mock_response
        
        # Simulate scrape_web's internal requests.get if needed
        if "google.com/search" in url or "example.com" in url: # Mock for scrape_web
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = f"<html><body><h1>Search results for {params.get('q', 'medical')}</h1><p>Some medical news snippet.</p></body></html>"
            return mock_response

        return original_requests_get(url, params=params, headers=headers, timeout=timeout)

    requests.get = mock_requests_get_dynamic

    test_user_pro = "mock_pro_token"
    test_user_free = "mock_free_token"

    print("\n--- Testing medical_tool functions ---")

    # Test get_drug_info
    print("\n--- Testing get_drug_info ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result_aspirin = get_drug_info("Aspirin", user_token=test_user_pro)
    print(f"Aspirin Info (Pro User, API):\n{result_aspirin[:200]}...")
    assert "Information for Aspirin:" in result_aspirin
    assert "A salicylate drug" in result_aspirin
    print("Test 1 Passed.")

    # Test get_drug_info (fallback)
    print("\n--- Testing get_drug_info (Fallback) ---")
    with patch('domain_tools.medical_tools.medical_tool._make_dynamic_api_request', return_value=None):
        result_ibuprofen_fallback = get_drug_info("Ibuprofen", user_token=test_user_pro)
        print(f"Ibuprofen Info (Pro User, Fallback):\n{result_ibuprofen_fallback[:200]}...")
        assert "Information for Ibuprofen (Mock Data Fallback):" in result_ibuprofen_fallback
    print("Test 2 Passed.")

    # Test get_symptom_info
    print("\n--- Testing get_symptom_info ---")
    result_headache = get_symptom_info("Headache", user_token=test_user_pro)
    print(f"Headache Info (Pro User, API):\n{result_headache[:200]}...")
    assert "Information for Headache:" in result_headache
    assert "Common Causes: Stress, dehydration" in result_headache
    print("Test 3 Passed.")

    # Test get_symptom_info (fallback)
    print("\n--- Testing get_symptom_info (Fallback) ---")
    with patch('domain_tools.medical_tools.medical_tool._make_dynamic_api_request', return_value=None):
        result_fever_fallback = get_symptom_info("Fever", user_token=test_user_pro)
        print(f"Fever Info (Pro User, Fallback):\n{result_fever_fallback[:200]}...")
        assert "Information for Fever (Mock Data Fallback):" in result_fever_fallback
    print("Test 4 Passed.")

    # Test RBAC for medical_tool_access (e.g., get_drug_info for free user)
    print("\n--- Testing RBAC for medical_tool_access (Free User) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result_rbac_denied = get_drug_info("Aspirin", user_token=test_user_free)
    print(f"Aspirin Info (Free User, RBAC Denied): {result_rbac_denied}")
    assert "Error: Access to medical tools is not enabled for your current tier." in result_rbac_denied
    print("Test 5 Passed.")

    # Test medical_search_web
    print("\n--- Testing medical_search_web ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    search_query = "latest research on diabetes"
    search_result = medical_search_web(search_query, user_token=test_user_pro)
    print(f"Search Result for '{search_query}':\n{search_result[:500]}...")
    assert "Search results for latest research on diabetes" in search_result
    print("Test 6 Passed.")

    # Test medical_summarize_document_by_path (requires a dummy file)
    print("\n--- Testing medical_summarize_document_by_path ---")
    dummy_upload_dir = Path("uploads") / test_user_pro / "medical"
    dummy_upload_dir.mkdir(parents=True, exist_ok=True)
    dummy_file_path = dummy_upload_dir / "medical_report.txt"
    with open(dummy_file_path, "w") as f:
        f.write("This is a sample medical report. It discusses new findings in cardiology. The patient's heart rate was stable.")
    
    result_summary = medical_summarize_document_by_path(str(dummy_file_path))
    print(f"Medical Report Summary (Pro User): {result_summary}")
    assert "Mock summary of the provided text." in result_summary
    assert "cardiology" in result_summary
    print("Test 7 Passed.")

    print("\nAll medical_tool tests completed.")

    # Restore original requests.get
    requests.get = original_requests_get

    # Clean up dummy files and directories
    test_user_dirs = [Path("uploads") / test_user_pro, BASE_VECTOR_DIR / test_user_pro]
    for d in test_user_dirs:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            print(f"Cleaned up {d}")
