# domain_tools/legal_tools/legal_tool.py

import logging
import requests
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta

# Import generic tools
from langchain_core.tools import tool
# REMOVED: from shared_tools.query_uploaded_docs_tool import QueryUploadedDocs
from shared_tools.scraper_tool import scrape_web
from shared_tools.doc_summarizer import summarize_document

# Import config_manager to access API configurations and secrets
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability
# Import date_parser for date format flexibility
from utils.date_parser import parse_date_to_yyyymmdd
# Import analytics_tracker
from utils import analytics_tracker # Import the module

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

async def _make_dynamic_api_request( # Made async to await analytics_tracker.log_tool_usage
    domain: str,
    function_name: str,
    params: Dict[str, Any],
    user_token: str
) -> Optional[Dict[str, Any]]:
    """
    Makes an API request to the dynamically configured provider for a given domain and function.
    Handles API key retrieval, request construction, and basic error handling.
    Returns parsed JSON data or None on failure (triggering mock fallback).
    Logs tool usage analytics.
    """
    # Check if analytics is enabled for logging tool usage
    log_tool_usage_enabled = config_manager.get("analytics.log_tool_usage", False)

    # Get the default active API provider for the domain from data/config.yml
    active_provider_name = config_manager.get(f"api_defaults.{domain}")
    if not active_provider_name:
        logger.error(f"No default API provider configured for domain '{domain}'.")
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=f"No default API provider configured for domain '{domain}'."
            )
        return None

    # Get the full configuration for the active provider from api_providers.yml
    provider_config = config_manager.get_api_provider_config(domain, active_provider_name)
    if not provider_config:
        logger.error(f"Configuration for API provider '{active_provider_name}' in domain '{domain}' not found in api_providers.yml.")
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=f"API provider config '{active_provider_name}' not found for domain '{domain}'."
            )
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
            if log_tool_usage_enabled:
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_token=user_token,
                    success=False,
                    error_message="Amadeus API credentials or token endpoint missing."
                )
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
                if log_tool_usage_enabled:
                    await analytics_tracker.log_tool_usage(
                        tool_name=f"{domain}_{function_name}",
                        tool_params=params,
                        user_token=user_token,
                        success=False,
                        error_message="Failed to get Amadeus access token."
                    )
                return None
            headers = {"Authorization": f"Bearer {access_token}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting Amadeus access token: {e}")
            if log_tool_usage_enabled:
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_token=user_token,
                    success=False,
                    error_message=f"Error getting Amadeus access token: {e}"
                )
            return None
    else:
        headers = {} # No special headers by default

    if not base_url:
        logger.error(f"Base URL not configured for API provider '{active_provider_name}' in domain '{domain}'.")
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=f"Base URL not configured for '{active_provider_name}'."
            )
        return None

    function_details = provider_config.get("functions", {}).get(function_name)
    if not function_details:
        logger.error(f"Function '{function_name}' not configured for API provider '{active_provider_name}' in domain '{domain}'.")
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=f"Function '{function_name}' not configured for '{active_provider_name}'."
            )
        return None

    endpoint = function_details.get("endpoint")
    function_param = function_details.get("function_param") # For Alpha Vantage style 'function' param
    path_params = function_details.get("path_params", []) # For ExchangeRate-API style path params

    if not endpoint and not function_param:
        logger.error(f"Neither 'endpoint' nor 'function_param' defined for function '{function_name}'.")
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=f"Endpoint or function_param missing for '{function_name}'."
            )
        return None

    # Construct URL
    full_url = f"{base_url}{endpoint}" if endpoint else base_url

    # Add path parameters to URL if specified
    for p_param in path_params:
        if p_param in params:
            value = str(params.pop(p_param))
            full_url = full_url.replace(f"{{{p_param}}}", value)
        else:
            error_msg = f"Missing path parameter '{p_param}' for function '{function_name}'."
            logger.warning(error_msg)
            if log_tool_usage_enabled:
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_token=user_token,
                    success=False,
                    error_message=error_msg
                )
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
        pass # Key is a path parameter, already handled above

    for param_key in function_details.get("required_params", []) + function_details.get("optional_params", []):
        if param_key in params:
            query_params[param_key] = params[param_key]
        elif param_key in function_details.get("required_params", []):
            error_msg = f"Missing required parameter '{param_key}' for function '{function_name}'."
            logger.warning(error_msg)
            if log_tool_usage_enabled:
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_token=user_token,
                    success=False,
                    error_message=error_msg
                )
            return None # Missing required param, cannot proceed

    try:
        logger.info(f"Making API call to: {full_url} with params: {query_params}")
        response = requests.get(full_url, params=query_params, headers=headers, timeout=config_manager.get("web_scraping.timeout_seconds", 15))
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        raw_data = response.json()
        
        # Check for API-specific error messages in the response body
        api_error_message = None
        if "Error Message" in raw_data: # Alpha Vantage specific
            api_error_message = f"API Error from {active_provider_name}: {raw_data['Error Message']}"
        elif "Note" in raw_data and "Thank you for using Alpha Vantage!" in raw_data["Note"]: # Alpha Vantage rate limit
            api_error_message = f"API rate limit hit for {active_provider_name}: {raw_data['Note']}"
        elif raw_data.get("status") == "error": # NewsAPI specific
            api_error_message = f"API Error from {active_provider_name}: {raw_data.get('message', 'Unknown error')}"
        elif raw_data.get("Error"): # OMDBAPI specific
            api_error_message = f"API Error from {active_provider_name}: {raw_data.get('Error')}"
        elif raw_data.get("status") and raw_data["status"].get("error_code"): # CoinGecko error
            api_error_message = f"API Error from {active_provider_name}: {raw_data['status'].get('error_message', 'Unknown CoinGecko error')}"
        elif raw_data.get("result") == "error": # ExchangeRate-API error
            api_error_message = f"API Error from {active_provider_name}: {raw_data.get('error-type', 'Unknown ExchangeRate-API error')}"

        if api_error_message:
            logger.error(api_error_message)
            if log_tool_usage_enabled:
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_token=user_token,
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
                if log_tool_usage_enabled:
                    await analytics_tracker.log_tool_usage(
                        tool_name=f"{domain}_{function_name}",
                        tool_params=params,
                        user_token=user_token,
                        success=False,
                        error_message=error_msg
                    )
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
            final_result = {"data": mapped_data_list} # Wrap list in a dict for consistent return
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
            final_result = {"data": processed_data}
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
                    final_result = mapped_data
                else:
                    error_msg = f"CoinGecko simple price response unexpected for {crypto_id}/{currency}: {raw_data}"
                    logger.warning(error_msg)
                    if log_tool_usage_enabled:
                        await analytics_tracker.log_tool_usage(
                            tool_name=f"{domain}_{function_name}",
                            tool_params=params,
                            user_token=user_token,
                            success=False,
                            error_message=error_msg
                        )
                    return None
            
            for mapped_key, original_key_path in data_map.items():
                if isinstance(original_key_path, list):
                    mapped_data[mapped_key] = _get_nested_value(data_to_map, original_key_path)
                elif '.' in str(original_key_path):
                    mapped_data[mapped_key] = _get_nested_value(data_to_map, original_key_path.split('.'))
                else:
                    mapped_data[mapped_key] = data_to_map.get(original_key_path)
            final_result = mapped_data

        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=True
            )
        return final_result

    except requests.exceptions.Timeout:
        error_msg = f"API request to {active_provider_name} timed out for function '{function_name}'."
        logger.error(error_msg)
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=error_msg
            )
        return None
    except requests.exceptions.RequestException as e:
        error_msg = f"Error making API request to {active_provider_name} for function '{function_name}': {e}"
        logger.error(error_msg)
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=error_msg
            )
        return None
    except json.JSONDecodeError:
        error_msg = f"Failed to decode JSON response from {active_provider_name} for function '{function_name}'."
        logger.error(error_msg)
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=error_msg
            )
        return None
    except Exception as e:
        error_msg = f"An unexpected error occurred during API call to {active_provider_name} for '{function_name}': {e}"
        logger.error(error_msg, exc_info=True)
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=error_msg
            )
        return None


# --- Mock Data for Fallback ---
_mock_legal_data = {
    "legal_definitions": {
        "contract": {
            "term": "Contract",
            "definition": "A legally binding agreement between two or more parties that creates mutual obligations enforceable by law.",
            "key_elements": ["Offer", "Acceptance", "Consideration", "Intention to create legal relations", "Capacity", "Legality"]
        },
        "tort": {
            "term": "Tort",
            "definition": "A civil wrong that causes a claimant to suffer loss or harm, resulting in legal liability for the person who commits the tortious act.",
            "examples": ["Negligence", "Defamation", "Trespass"]
        }
    },
    "legal_precedents": {
        "donoghue_v_stevenson": {
            "case_name": "Donoghue v Stevenson",
            "year": 1932,
            "summary": "Established the modern concept of negligence in English law, including the 'neighbour principle'.",
            "relevance": "Foundation of consumer protection and product liability law."
        },
        "miranda_v_arizona": {
            "case_name": "Miranda v Arizona",
            "year": 1966,
            "summary": "Established the 'Miranda warnings' for custodial interrogations in US law.",
            "relevance": "Protects Fifth Amendment rights against self-incrimination."
        }
    },
    "legal_aid_info": {
        "london": {
            "city": "London, UK",
            "organizations": [
                {"name": "Citizens Advice London", "services": "Free, confidential advice on legal, money and other issues.", "contact": "0800 144 8848", "website": "https://www.citizensadvice.org.uk/"},
                {"name": "Legal Aid Agency (UK)", "services": "Government-funded legal aid for eligible cases.", "contact": "0345 345 4345", "website": "https://www.gov.uk/legal-aid"}
            ]
        },
        "new_york": {
            "city": "New York, USA",
            "organizations": [
                {"name": "Legal Aid Society (NYC)", "services": "Free legal services to low-income New Yorkers.", "contact": "(212) 577-3300", "website": "https://legalaidnyc.org/"},
                {"name": "New York City Bar Association", "services": "Lawyer referral service.", "contact": "(212) 382-6600", "website": "https://www.nycbar.org/"}
            ]
        }
    }
}

@tool
def get_legal_definition(term: str, user_token: str = "default") -> str:
    """
    Retrieves the legal definition and key elements of a specific legal term.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        term (str): The legal term to define (e.g., "Contract", "Tort", "Copyright").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of the legal definition, or an error/fallback message.
    """
    logger.info(f"Tool: get_legal_definition called for term: '{term}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'legal_tool_access', False):
        return "Error: Access to legal tools is not enabled for your current tier."
    
    params = {"term": term}
    api_data = asyncio.run(_make_dynamic_api_request("legal", "get_legal_definition", params, user_token))

    if api_data:
        try:
            defined_term = api_data.get("term")
            definition = api_data.get("definition")
            key_elements = api_data.get("key_elements")

            if defined_term and definition:
                response_str = (
                    f"Legal Definition of '{defined_term}':\n"
                    f"  Definition: {definition}\n"
                )
                if key_elements:
                    response_str += f"  Key Elements: {', '.join(key_elements)}\n"
                return response_str
            else:
                logger.warning(f"Live API data for term '{term}' is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live legal definition for '{term}'. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live legal definition data for '{term}': {e}")
            return f"Error parsing live data for '{term}'. Falling back to mock data."

    # Fallback to mock data
    mock_data_key = term.lower().replace(" ", "_")
    mock_data = _mock_legal_data.get("legal_definitions", {}).get(mock_data_key)
    if mock_data:
        response_str = (
            f"Legal Definition of '{mock_data['term']}' (Mock Data Fallback):\n"
            f"  Definition: {mock_data['definition']}\n"
        )
        if mock_data.get('key_elements'):
            response_str += f"  Key Elements: {', '.join(mock_data['key_elements'])}\n"
        return response_str
    else:
        return f"Legal definition for '{term}' not found. (API/Mock Fallback Failed)"


@tool
def search_legal_precedents(query: str, user_token: str = "default") -> str:
    """
    Searches for significant legal precedents or case law relevant to a query.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        query (str): The legal query to find precedents for (e.g., "negligence in product liability", "freedom of speech cases").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of relevant legal precedents, or an error/fallback message.
    """
    logger.info(f"Tool: search_legal_precedents called for query: '{query}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'legal_tool_access', False):
        return "Error: Access to legal tools is not enabled for your current tier."
    
    params = {"query": query}
    api_data = asyncio.run(_make_dynamic_api_request("legal", "search_legal_precedents", params, user_token))

    if api_data and api_data.get("data"):
        precedents = api_data["data"]
        if precedents:
            response_str = f"Relevant Legal Precedents for '{query}':\n"
            for i, precedent in enumerate(precedents[:5]): # Limit to top 5
                response_str += (
                    f"{i+1}. Case: {precedent.get('case_name', 'N/A')} ({precedent.get('year', 'N/A')})\n"
                    f"   Summary: {precedent.get('summary', 'N/A')}\n"
                    f"   Relevance: {precedent.get('relevance', 'N/A')}\n\n"
                )
            return response_str
        else:
            return f"No live legal precedents found for '{query}'. Falling back to mock data."

    # Fallback to mock data
    mock_precedents = _mock_legal_data.get("legal_precedents", {})
    filtered_mock_precedents = []
    for key, precedent in mock_precedents.items():
        if query.lower() in precedent.get("summary", "").lower() or query.lower() in precedent.get("relevance", "").lower():
            filtered_mock_precedents.append(precedent)
    
    if filtered_mock_precedents:
        response_str = f"Relevant Legal Precedents for '{query}' (Mock Data Fallback):\n"
        for i, precedent in enumerate(filtered_mock_precedents[:2]): # Limit mock to top 2
            response_str += (
                f"{i+1}. Case: {precedent.get('case_name', 'N/A')} ({precedent.get('year', 'N/A')})\n"
                f"   Summary: {precedent.get('summary', 'N/A')}\n"
                f"   Relevance: {precedent.get('relevance', 'N/A')}\n\n"
            )
        return response_str
    else:
        return f"Legal precedents for '{query}' not found. (API/Mock Fallback Failed)"


@tool
def get_legal_aid_info(location: str, user_token: str = "default") -> str:
    """
    Retrieves information about legal aid organizations or services available in a specific location.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        location (str): The city or region to search for legal aid (e.g., "London", "New York").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of legal aid information, or an error/fallback message.
    """
    logger.info(f"Tool: get_legal_aid_info called for location: '{location}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'legal_tool_access', False):
        return "Error: Access to legal tools is not enabled for your current tier."
    
    params = {"location": location}
    api_data = asyncio.run(_make_dynamic_api_request("legal", "get_legal_aid_info", params, user_token))

    if api_data and api_data.get("data"):
        organizations = api_data["data"]
        if organizations:
            response_str = f"Legal Aid Information for {location}:\n"
            for i, org in enumerate(organizations[:5]): # Limit to top 5
                response_str += (
                    f"{i+1}. Organization: {org.get('name', 'N/A')}\n"
                    f"   Services: {org.get('services', 'N/A')}\n"
                    f"   Contact: {org.get('contact', 'N/A')}\n"
                    f"   Website: {org.get('website', 'N/A')}\n\n"
                )
            return response_str
        else:
            return f"No live legal aid information found for '{location}'. Falling back to mock data."

    # Fallback to mock data
    mock_data_key = location.lower().replace(" ", "_")
    mock_data = _mock_legal_data.get("legal_aid_info", {}).get(mock_data_key)
    if mock_data and mock_data.get("organizations"):
        response_str = f"Legal Aid Information for {mock_data['city']} (Mock Data Fallback):\n"
        for i, org in enumerate(mock_data['organizations'][:2]): # Limit mock to top 2
            response_str += (
                f"{i+1}. Organization: {org.get('name', 'N/A')}\n"
                f"   Services: {org.get('services', 'N/A')}\n"
                f"   Contact: {org.get('contact', 'N/A')}\n"
                f"   Website: {org.get('website', 'N/A')}\n\n"
            )
        return response_str
    else:
        return f"Legal aid information not found for '{location}'. (API/Mock Fallback Failed)"


# --- Existing Generic Tools (not directly using external APIs, but can be used in legal context) ---

@tool
def legal_search_web(query: str, user_token: str = "default", max_chars: int = 2000) -> str:
    """
    Searches the web for legal information using a smart search fallback mechanism.
    This tool wraps the generic `scrape_web` tool, providing a legal-specific interface.
    
    Args:
        query (str): The legal-related search query (e.g., "copyright law for digital content", "employment rights in California").
        user_token (str): The unique identifier for the user. Defaults to "default".
        max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
    
    Returns:
        str: A string containing relevant information from the web.
    """
    logger.info(f"Tool: legal_search_web called with query: '{query}' for user: '{user_token}'")
    return scrape_web(query=query, user_token=user_token, max_chars=max_chars)

@tool
def legal_query_uploaded_docs(query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
    """
    Queries previously uploaded and indexed legal documents for a user using vector similarity search.
    This tool wraps the generic `QueryUploadedDocs` tool, fixing the section to "legal".
    
    Args:
        query (str): The search query to find relevant legal documents (e.g., "terms and conditions of service", "my legal contract details").
        user_token (str): The unique identifier for the user. Defaults to "default".
        export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
        k (int): The number of top relevant documents to retrieve. Defaults to 5.
    
    Returns:
        str: A string containing the combined content of the relevant document chunks,
             or a message indicating no data/results found, or the export path if exported.
    """
    logger.info(f"Tool: legal_query_uploaded_docs called with query: '{query}' for user: '{user_token}'")
    # This will be replaced by a call to self.document_tools.query_uploaded_docs
    # For now, keeping the original call for review purposes.
    return QueryUploadedDocs(query=query, user_token=user_token, section="legal", export=export, k=k)

@tool
def legal_summarize_document_by_path(file_path_str: str) -> str:
    """
    Summarizes a document related to legal matters located at the given file path.
    The file path should be accessible by the system (e.g., in the 'uploads' directory).
    This tool wraps the generic `summarize_document` tool.
    
    Args:
        file_path_str (str): The full path to the document file to be summarized.
                              Example: "uploads/default/legal/privacy_policy.pdf"
    
    Returns:
        str: A concise summary of the document content.
    """
    logger.info(f"Tool: legal_summarize_document_by_path called for file: '{file_path_str}'")
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
    import asyncio
    from unittest.mock import MagicMock, AsyncMock, patch
    import shutil
    import os
    import sys # Import sys for patching modules
    from shared_tools.vector_utils import BASE_VECTOR_DIR # For cleanup
    # from shared_tools.python_interpreter_tool import python_interpreter_with_rbac # For testing REPL

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.legal_api_key = "MOCK_LEGAL_API_KEY"
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
                    'legal': 'legal_api'
                },
                'analytics': { # Mock analytics settings
                    'enabled': True,
                    'log_tool_usage': True,
                    'log_query_failures': True
                }
            }
            self._api_providers_data = { # Mock api_providers_data for legal
                "legal": {
                    "legal_api": {
                        "base_url": "https://api.example.com/legal",
                        "api_key_name": "legal_api_key",
                        "api_key_param_name": "api_key",
                        "functions": {
                            "get_legal_definition": {
                                "endpoint": "/definitions",
                                "required_params": ["term"],
                                "response_path": ["data", 0],
                                "data_map": {
                                    "term": "term",
                                    "definition": "definition",
                                    "key_elements": "elements"
                                }
                            },
                            "search_legal_precedents": {
                                "endpoint": "/precedents/search",
                                "required_params": ["query"],
                                "response_path": ["data"],
                                "data_map": {
                                    "case_name": "name",
                                    "year": "year",
                                    "summary": "summary",
                                    "relevance": "relevance"
                                }
                            },
                            "get_legal_aid_info": {
                                "endpoint": "/aid",
                                "required_params": ["location"],
                                "response_path": ["data"],
                                "data_map": {
                                    "name": "organization_name",
                                    "services": "services_offered",
                                    "contact": "contact_info",
                                    "website": "website_url"
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
                'legal_tool_access': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'document_query_enabled': { # Added for document tool
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

    # Mock analytics_tracker
    mock_analytics_tracker_db = MagicMock()
    mock_analytics_tracker_auth = MagicMock()
    mock_analytics_tracker_auth.currentUser = MagicMock(uid="mock_user_123")
    mock_analytics_tracker_db.collection.return_value.add = AsyncMock(return_value=MagicMock(id="mock_doc_id"))

    # Patch firebase_admin.firestore for the local import within log_event
    with patch.dict(sys.modules, {'firebase_admin.firestore': MagicMock(firestore=MagicMock())}):
        sys.modules['firebase_admin.firestore'].firestore.CollectionReference = MagicMock()
        sys.modules['firebase_admin.firestore'].firestore.DocumentReference = MagicMock()
        
        # Initialize the actual analytics_tracker with mocks
        analytics_tracker.initialize_analytics(
            mock_analytics_tracker_db,
            mock_analytics_tracker_auth,
            "test_app_id_for_analytics",
            "mock_user_123"
        )

        # Mock requests.get for external API calls
        original_requests_get = requests.get

        def mock_requests_get_dynamic(url, params, headers, timeout):
            # Simulate hypothetical Legal API responses
            if "api.example.com/legal" in url:
                if "/definitions" in url:
                    term = params.get("term", "").lower()
                    if "contract" in term:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "data": [{
                                "term": "Contract",
                                "definition": "A legally binding agreement...",
                                "elements": ["Offer", "Acceptance"]
                            }]
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"data": []}
                        return mock_response
                elif "/precedents/search" in url:
                    query = params.get("query", "").lower()
                    if "negligence" in query:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "data": [{
                                "name": "Donoghue v Stevenson",
                                "year": 1932,
                                "summary": "Established negligence.",
                                "relevance": "Foundation of tort law."
                            }]
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"data": []}
                        return mock_response
                elif "/aid" in url:
                    location = params.get("location", "").lower()
                    if "london" in location:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "data": [{
                                "organization_name": "Citizens Advice London",
                                "services_offered": "Free legal advice.",
                                "contact_info": "0800 144 8848",
                                "website_url": "https://www.citizensadvice.org.uk/"
                            }]
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"data": []}
                        return mock_response
            
            # Simulate scrape_web's internal requests.get if needed
            if "google.com/search" in url or "example.com" in url: # Mock for scrape_web
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.text = f"<html><body><h1>Search results for {params.get('q', 'legal')}</h1><p>Some legal content from web search.</p></body></html>"
                return mock_response

            return original_requests_get(url, params=params, headers=headers, timeout=timeout)

        requests.get = mock_requests_get_dynamic

        test_user_pro = "mock_pro_token"
        test_user_free = "mock_free_token"

        # Mock for QueryUploadedDocs
        class MockQueryUploadedDocs:
            def __init__(self, query, user_token, section, export, k):
                self.query = query
                self.user_token = user_token
                self.section = section
                self.export = export
                self.k = k
            def __call__(self):
                return f"Mocked document query results for '{self.query}' in section '{self.section}'."

        # Mock for summarize_document
        class MockSummarizeDocument:
            def __call__(self, file_path):
                return f"Mocked summary of {file_path.name}"

        # Patch QueryUploadedDocs and summarize_document in the legal_tool module
        original_QueryUploadedDocs = sys.modules['domain_tools.legal_tools.legal_tool'].QueryUploadedDocs
        original_summarize_document = sys.modules['domain_tools.legal_tools.legal_tool'].summarize_document
        sys.modules['domain_tools.legal_tools.legal_tool'].QueryUploadedDocs = MockQueryUploadedDocs
        sys.modules['domain_tools.legal_tools.legal_tool'].summarize_document = MockSummarizeDocument()

        async def run_legal_tests():
            print("\n--- Testing legal_tool functions with Analytics ---")

            # Test get_legal_definition (success)
            print("\n--- Test 1: get_legal_definition (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock() # Reset mock call count
            result_definition = await get_legal_definition("Contract", user_token=test_user_pro)
            print(f"Legal Definition: {result_definition}")
            assert "Legal Definition of 'Contract':" in result_definition
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "legal_get_legal_definition"
            assert logged_data["success"] is True
            print("Test 1 Passed (and analytics logged success).")

            # Test search_legal_precedents (API failure - no data found)
            print("\n--- Test 2: search_legal_precedents (API Failure) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_precedents = await search_legal_precedents("patent infringement", user_token=test_user_pro)
            print(f"Legal Precedents (API Error): {result_precedents}")
            assert "No live legal precedents found for 'patent infringement'." in result_precedents
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "legal_search_legal_precedents"
            assert logged_data["success"] is False
            assert "Response path 'data' not found" in logged_data["error_message"] or "incomplete" in logged_data["error_message"]
            print("Test 2 Passed (and analytics logged failure).")

            # Test get_legal_aid_info (RBAC denied)
            print("\n--- Test 3: get_legal_aid_info (RBAC Denied) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_legal_aid_rbac_denied = await get_legal_aid_info("Paris", user_token=test_user_free)
            print(f"Legal Aid Info (Free User, RBAC Denied): {result_legal_aid_rbac_denied}")
            assert "Error: Access to legal tools is not enabled for your current tier." in result_legal_aid_rbac_denied
            # No analytics log expected here because RBAC check happens before _make_dynamic_api_request
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 3 Passed (RBAC correctly prevented call and no analytics logged).")

            # Test legal_search_web (generic tool, not using _make_dynamic_api_request)
            print("\n--- Test 4: legal_search_web (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_web_search = await legal_search_web("data privacy regulations EU", user_token=test_user_pro)
            print(f"Web Search Result: {result_web_search[:100]}...")
            assert "Search results for data privacy regulations EU" in result_web_search
            # Analytics for generic tools like scrape_web or summarize_document
            # would need to be integrated within those shared_tools themselves,
            # or wrapped by a higher-level agent logging.
            # For now, we are focusing on _make_dynamic_api_request.
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 4 Passed (no analytics expected for generic tool directly).")

            # Test 5: legal_query_uploaded_docs (generic tool)
            print("\n--- Test 5: legal_query_uploaded_docs (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            # Mock QueryUploadedDocs to simulate a response
            class MockQueryUploadedDocs:
                def __init__(self, query, user_token, section, export, k):
                    self.query = query
                    self.user_token = user_token
                    self.section = section
                    self.export = export
                    self.k = k
                def __call__(self):
                    return f"Mocked document query results for '{self.query}' in section '{self.section}'."
            
            # Temporarily replace QueryUploadedDocs with our mock
            original_QueryUploadedDocs_in_test = sys.modules['domain_tools.legal_tools.legal_tool'].QueryUploadedDocs
            sys.modules['domain_tools.legal_tools.legal_tool'].QueryUploadedDocs = MockQueryUploadedDocs

            result_doc_query = await legal_query_uploaded_docs("terms of service for a new app", user_token=test_user_pro)
            print(f"Document Query Result: {result_doc_query}")
            assert "Mocked document query results for 'terms of service for a new app' in section 'legal'." in result_doc_query
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 5 Passed (no analytics expected for generic tool directly, will be logged by DocumentTools).")
            sys.modules['domain_tools.legal_tools.legal_tool'].QueryUploadedDocs = original_QueryUploadedDocs_in_test # Restore original

            # Test 6: legal_summarize_document_by_path (generic tool)
            print("\n--- Test 6: legal_summarize_document_by_path (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            # Create a dummy file for summarization test
            dummy_file_path = Path("uploads") / test_user_pro / "legal" / "contract_draft.txt"
            dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_file_path.write_text("This is a dummy legal contract draft for testing summarization.")

            result_summarize = await legal_summarize_document_by_path(str(dummy_file_path))
            print(f"Summarize Result: {result_summarize}")
            assert "Mocked summary of contract_draft.txt" in result_summarize
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 6 Passed (no analytics expected for generic tool directly).")

            print("\nAll legal_tool tests with analytics considerations completed.")

        await run_legal_tests()

        # Restore original requests.get
        requests.get = original_requests_get

        # Clean up dummy files and directories
        test_user_dirs = [Path("uploads") / test_user_pro, BASE_VECTOR_DIR / test_user_pro]
        for d in test_user_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print(f"Cleaned up {d}")
