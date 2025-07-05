# domain_tools/education_tools/education_tool.py

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
        pass # Key is a path parameter, already handled above

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
_mock_education_data = {
    "course_search": [
        {
            "course_id": "CS101",
            "title": "Introduction to Computer Science",
            "description": "Fundamental concepts of computer science and programming.",
            "provider": "University of Tech",
            "level": "Beginner",
            "duration": "12 weeks",
            "cost": "Free",
            "url": "http://example.edu/cs101"
        },
        {
            "course_id": "MATH201",
            "title": "Calculus I",
            "description": "Introduction to differential and integral calculus.",
            "provider": "Online Learning Platform",
            "level": "Intermediate",
            "duration": "10 weeks",
            "cost": "$99",
            "url": "http://example.edu/math201"
        }
    ],
    "university_info": {
        "university_of_tech": {
            "name": "University of Tech",
            "location": "Tech City, CA",
            "ranking": "Top 50 Global",
            "programs": ["Computer Science", "Engineering", "Data Science"],
            "website": "http://example.edu/tech"
        },
        "state_university": {
            "name": "State University",
            "location": "Capital City, NY",
            "ranking": "Top 100 National",
            "programs": ["Arts", "Humanities", "Business"],
            "website": "http://example.edu/state"
        }
    },
    "educational_resource": {
        "khan_academy_math": {
            "title": "Khan Academy - Math",
            "type": "Online Platform",
            "description": "Free online courses and practice in math, science, and more.",
            "url": "https://www.khanacademy.org/math"
        },
        "wikipedia_physics": {
            "title": "Wikipedia - Physics",
            "type": "Encyclopedia",
            "description": "Comprehensive articles on various physics topics.",
            "url": "https://en.wikipedia.org/wiki/Physics"
        }
    }
}

@tool
def search_courses(query: str, level: Optional[str] = None, provider: Optional[str] = None, user_token: str = "default") -> str:
    """
    Searches for educational courses based on a query, optional difficulty level (e.g., 'Beginner', 'Intermediate', 'Advanced'),
    and optional course provider.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        query (str): The search query for courses (e.g., "python programming", "data science").
        level (str, optional): The difficulty level of the course.
        provider (str, optional): The name of the course provider (e.g., "Coursera", "edX", "University of Tech").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of course information, or an error/fallback message.
    """
    logger.info(f"Tool: search_courses called with query='{query}', level='{level}', provider='{provider}' by user: {user_token}")

    if not get_user_tier_capability(user_token, 'education_tool_access', False):
        return "Error: Access to education tools is not enabled for your current tier."
    
    params = {"query": query}
    if level: params["level"] = level
    if provider: params["provider"] = provider

    api_data = _make_dynamic_api_request(
        "education", "search_courses",
        params,
        user_token
    )

    if api_data and api_data.get("data"):
        courses = api_data["data"]
        if courses:
            response_str = "Found Educational Courses:\n"
            for i, course in enumerate(courses[:5]): # Limit to top 5 courses
                response_str += (
                    f"{i+1}. Title: {course.get('title', 'N/A')}\n"
                    f"   Provider: {course.get('provider', 'N/A')}\n"
                    f"   Level: {course.get('level', 'N/A')}\n"
                    f"   Duration: {course.get('duration', 'N/A')}\n"
                    f"   Cost: {course.get('cost', 'N/A')}\n"
                    f"   URL: {course.get('url', 'N/A')}\n\n"
                )
            return response_str
        else:
            return f"No live educational courses found for your criteria (query='{query}', level='{level}', provider='{provider}'). Falling back to mock data."

    # Fallback to mock data
    mock_courses = _mock_education_data.get("course_search", [])
    filtered_mock_courses = []
    for course in mock_courses:
        match = True
        if query and query.lower() not in course.get("title", "").lower() and query.lower() not in course.get("description", "").lower():
            match = False
        if level and course.get("level", "").lower() != level.lower():
            match = False
        if provider and course.get("provider", "").lower() != provider.lower():
            match = False
        if match:
            filtered_mock_courses.append(course)

    if filtered_mock_courses:
        response_str = "Found Educational Courses (Mock Data Fallback):\n"
        for i, course in enumerate(filtered_mock_courses[:2]): # Limit mock to top 2
            response_str += (
                f"{i+1}. Title: {course.get('title', 'N/A')}\n"
                f"   Provider: {course.get('provider', 'N/A')}\n"
                f"   Level: {course.get('level', 'N/A')}\n"
                f"   Duration: {course.get('duration', 'N/A')}\n"
                f"   Cost: {course.get('cost', 'N/A')}\n"
                f"   URL: {course.get('url', 'N/A')}\n\n"
            )
        return response_str
    else:
        return f"Educational course information not found for your criteria. (API/Mock Fallback Failed)"


@tool
def get_university_info(university_name: str, user_token: str = "default") -> str:
    """
    Retrieves information about a specific university or educational institution.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        university_name (str): The full or partial name of the university (e.g., "Stanford University", "MIT").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of university information, or an error/fallback message.
    """
    logger.info(f"Tool: get_university_info called for university: {university_name} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'education_tool_access', False):
        return "Error: Access to education tools is not enabled for your current tier."
    
    api_data = _make_dynamic_api_request(
        "education", "get_university_info",
        {"name": university_name},
        user_token
    )

    if api_data:
        try:
            name = api_data.get("name")
            location = api_data.get("location")
            ranking = api_data.get("ranking")
            programs = api_data.get("programs")
            website = api_data.get("website")

            if name and location:
                response_str = (
                    f"Information for {name}:\n"
                    f"  Location: {location}\n"
                )
                if ranking:
                    response_str += f"  Ranking: {ranking}\n"
                if programs:
                    response_str += f"  Key Programs: {', '.join(programs)}\n"
                if website:
                    response_str += f"  Website: {website}\n"
                return response_str
            else:
                logger.warning(f"Live API data for {university_name} is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live university information for {university_name}. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live university info data for {university_name}: {e}")
            return f"Error parsing live data for {university_name}. Falling back to mock data."

    # Fallback to mock data
    mock_data_key = university_name.lower().replace(" ", "_")
    mock_data = _mock_education_data.get("university_info", {}).get(mock_data_key)
    if mock_data:
        response_str = (
            f"Information for {mock_data['name']} (Mock Data Fallback):\n"
            f"  Location: {mock_data['location']}\n"
        )
        if mock_data.get('ranking'):
            response_str += f"  Ranking: {mock_data['ranking']}\n"
        if mock_data.get('programs'):
            response_str += f"  Key Programs: {', '.join(mock_data['programs'])}\n"
        if mock_data.get('website'):
            response_str += f"  Website: {mock_data['website']}\n"
        return response_str
    else:
        return f"University information not found for '{university_name}'. (API/Mock Fallback Failed)"


@tool
def get_educational_resource(resource_name: str, user_token: str = "default") -> str:
    """
    Retrieves information about a specific educational resource or platform.
    Falls back to mock data if API key is missing or API call fails.

    Args:
        resource_name (str): The name of the educational resource (e.g., "Khan Academy", "Coursera", "Wikipedia").
        user_token (str, optional): The unique identifier for the user. Defaults to "default".

    Returns:
        str: A formatted string of educational resource information, or an error/fallback message.
    """
    logger.info(f"Tool: get_educational_resource called for resource: {resource_name} by user: {user_token}")

    if not get_user_tier_capability(user_token, 'education_tool_access', False):
        return "Error: Access to education tools is not enabled for your current tier."
    
    api_data = _make_dynamic_api_request(
        "education", "get_educational_resource",
        {"name": resource_name},
        user_token
    )

    if api_data:
        try:
            title = api_data.get("title")
            resource_type = api_data.get("type")
            description = api_data.get("description")
            url = api_data.get("url")

            if title and description:
                response_str = (
                    f"Information for {title}:\n"
                )
                if resource_type:
                    response_str += f"  Type: {resource_type}\n"
                response_str += f"  Description: {description}\n"
                if url:
                    response_str += f"  URL: {url}\n"
                return response_str
            else:
                logger.warning(f"Live API data for {resource_name} is incomplete. Raw: {api_data}")
                return f"Could not retrieve complete live educational resource information for {resource_name}. Falling back to mock data."
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing live educational resource data for {resource_name}: {e}")
            return f"Error parsing live data for {resource_name}. Falling back to mock data."

    # Fallback to mock data
    mock_data_key = resource_name.lower().replace(" ", "_")
    mock_data = _mock_education_data.get("educational_resource", {}).get(mock_data_key)
    if mock_data:
        response_str = (
            f"Information for {mock_data['title']} (Mock Data Fallback):\n"
        )
        if mock_data.get('type'):
            response_str += f"  Type: {mock_data['type']}\n"
        response_str += f"  Description: {mock_data['description']}\n"
        if mock_data.get('url'):
            response_str += f"  URL: {mock_data['url']}\n"
        return response_str
    else:
        return f"Educational resource information not found for '{resource_name}'. (API/Mock Fallback Failed)"


# --- Existing Generic Tools (not directly using external APIs, but can be used in education context) ---

@tool
def education_search_web(query: str, user_token: str = "default", max_chars: int = 2000) -> str:
    """
    Searches the web for educational information using a smart search fallback mechanism.
    This tool wraps the generic `scrape_web` tool, providing an education-specific interface.
    
    Args:
        query (str): The education-related search query (e.g., "best online courses for AI", "history of ancient Rome").
        user_token (str): The unique identifier for the user. Defaults to "default".
        max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
    
    Returns:
        str: A string containing relevant information from the web.
    """
    logger.info(f"Tool: education_search_web called with query: '{query}' for user: '{user_token}'")
    return scrape_web(query=query, user_token=user_token, max_chars=max_chars)

@tool
def education_query_uploaded_docs(query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
    """
    Queries previously uploaded and indexed educational documents for a user using vector similarity search.
    This tool wraps the generic `QueryUploadedDocs` tool, fixing the section to "education".
    
    Args:
        query (str): The search query to find relevant educational documents (e.g., "summary of textbook chapter 5", "notes on quantum physics lecture").
        user_token (str): The unique identifier for the user. Defaults to "default".
        export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
        k (int): The number of top relevant documents to retrieve. Defaults to 5.
    
    Returns:
        str: A string containing the combined content of the relevant document chunks,
             or a message indicating no data/results found, or the export path if exported.
    """
    logger.info(f"Tool: education_query_uploaded_docs called with query: '{query}' for user: '{user_token}'")
    return QueryUploadedDocs(query=query, user_token=user_token, section="education", export=export, k=k)

@tool
def education_summarize_document_by_path(file_path_str: str) -> str:
    """
    Summarizes a document related to education or academic topics located at the given file path.
    The file path should be accessible by the system (e.g., in the 'uploads' directory).
    This tool wraps the generic `summarize_document` tool.
    
    Args:
        file_path_str (str): The full path to the document file to be summarized.
                              Example: "uploads/default/education/research_paper.pdf"
    
    Returns:
        str: A concise summary of the document content.
    """
    logger.info(f"Tool: education_summarize_document_by_path called for file: '{file_path_str}'")
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
            self.education_api_key = "MOCK_EDUCATION_API_KEY"
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
                    'education': 'education_api'
                }
            }
            self._api_providers_data = { # Mock api_providers_data for education
                "education": {
                    "education_api": {
                        "base_url": "https://api.example.com/education",
                        "api_key_name": "education_api_key",
                        "api_key_param_name": "api_key",
                        "functions": {
                            "search_courses": {
                                "endpoint": "/courses/search",
                                "required_params": ["query"],
                                "optional_params": ["level", "provider"],
                                "response_path": ["data"],
                                "data_map": {
                                    "course_id": "id",
                                    "title": "title",
                                    "description": "description",
                                    "provider": "provider",
                                    "level": "level",
                                    "duration": "duration",
                                    "cost": "cost",
                                    "url": "url"
                                }
                            },
                            "get_university_info": {
                                "endpoint": "/universities",
                                "required_params": ["name"],
                                "response_path": ["data", 0], # Assuming first result is most relevant
                                "data_map": {
                                    "name": "name",
                                    "location": "location",
                                    "ranking": "ranking",
                                    "programs": "programs",
                                    "website": "website"
                                }
                            },
                            "get_educational_resource": {
                                "endpoint": "/resources",
                                "required_params": ["name"],
                                "response_path": ["data", 0],
                                "data_map": {
                                    "title": "title",
                                    "type": "type",
                                    "description": "description",
                                    "url": "url"
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
                'education_tool_access': {
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
        # Simulate hypothetical Education API responses
        if "api.example.com/education" in url:
            if "/courses/search" in url:
                query = params.get("query", "").lower()
                level = params.get("level", "").lower()
                provider = params.get("provider", "").lower()
                
                mock_courses = [
                    {
                        "id": "CS101",
                        "title": "Introduction to Computer Science",
                        "description": "Fundamental concepts of computer science and programming.",
                        "provider": "University of Tech",
                        "level": "Beginner",
                        "duration": "12 weeks",
                        "cost": "Free",
                        "url": "http://example.edu/cs101"
                    },
                    {
                        "id": "MATH201",
                        "title": "Calculus I",
                        "description": "Introduction to differential and integral calculus.",
                        "provider": "Online Learning Platform",
                        "level": "Intermediate",
                        "duration": "10 weeks",
                        "cost": "$99",
                        "url": "http://example.edu/math201"
                    },
                    {
                        "id": "PY300",
                        "title": "Advanced Python Programming",
                        "description": "Deep dive into Python for data analysis and web development.",
                        "provider": "Code Academy",
                        "level": "Advanced",
                        "duration": "8 weeks",
                        "cost": "$199",
                        "url": "http://example.edu/py300"
                    }
                ]
                
                filtered_mock_courses = []
                for course in mock_courses:
                    match = True
                    if query and not (query in course["title"].lower() or query in course["description"].lower()):
                        match = False
                    if level and course["level"].lower() != level:
                        match = False
                    if provider and course["provider"].lower() != provider:
                        match = False
                    if match:
                        filtered_mock_courses.append(course)

                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"data": filtered_mock_courses}
                return mock_response

            elif "/universities" in url:
                name = params.get("name", "").lower()
                if "university of tech" in name:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "data": [{
                            "name": "University of Tech",
                            "location": "Tech City, CA",
                            "ranking": "Top 50 Global",
                            "programs": ["Computer Science", "Engineering"],
                            "website": "http://example.edu/tech"
                        }]
                    }
                    return mock_response
                else:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"data": []}
                    return mock_response
            
            elif "/resources" in url:
                name = params.get("name", "").lower()
                if "khan academy" in name:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "data": [{
                            "title": "Khan Academy - All Subjects",
                            "type": "Online Learning Platform",
                            "description": "Free online courses and practice exercises in various subjects.",
                            "url": "https://www.khanacademy.org"
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
            mock_response.text = f"<html><body><h1>Search results for {params.get('q', 'education')}</h1><p>Some educational content from web search.</p></body></html>"
            return mock_response

        return original_requests_get(url, params=params, headers=headers, timeout=timeout)

    requests.get = mock_requests_get_dynamic

    test_user_pro = "mock_pro_token"
    test_user_free = "mock_free_token"

    print("\n--- Testing education_tool functions ---")

    # Test search_courses
    print("\n--- Testing search_courses ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    result_courses = search_courses("computer science", level="Beginner", user_token=test_user_pro)
    print(f"Courses (Pro User, API):\n{result_courses[:500]}...")
    assert "Found Educational Courses:" in result_courses
    assert "Introduction to Computer Science" in result_courses
    print("Test 1 Passed.")

    # Test search_courses (fallback)
    print("\n--- Testing search_courses (Fallback) ---")
    with patch('domain_tools.education_tools.education_tool._make_dynamic_api_request', return_value=None):
        result_courses_fallback = search_courses("history", user_token=test_user_pro)
        print(f"Courses (Pro User, Fallback):\n{result_courses_fallback[:500]}...")
        assert "Found Educational Courses (Mock Data Fallback):" in result_courses_fallback
    print("Test 2 Passed.")

    # Test get_university_info
    print("\n--- Testing get_university_info ---")
    result_university = get_university_info("University of Tech", user_token=test_user_pro)
    print(f"University Info (Pro User, API):\n{result_university[:200]}...")
    assert "Information for University of Tech:" in result_university
    assert "Top 50 Global" in result_university
    print("Test 3 Passed.")

    # Test get_educational_resource
    print("\n--- Testing get_educational_resource ---")
    result_resource = get_educational_resource("Khan Academy", user_token=test_user_pro)
    print(f"Educational Resource (Pro User, API):\n{result_resource[:200]}...")
    assert "Information for Khan Academy - All Subjects:" in result_resource
    assert "Free online courses" in result_resource
    print("Test 4 Passed.")

    # Test RBAC for education_tool_access (e.g., search_courses for free user)
    print("\n--- Testing RBAC for education_tool_access (Free User) ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_free
    result_rbac_denied = search_courses("art history", user_token=test_user_free)
    print(f"Courses (Free User, RBAC Denied): {result_rbac_denied}")
    assert "Error: Access to education tools is not enabled for your current tier." in result_rbac_denied
    print("Test 5 Passed.")

    # Test education_search_web
    print("\n--- Testing education_search_web ---")
    sys.modules['utils.user_manager']._current_mock_user = test_user_pro
    search_web_query = "online degree programs in psychology"
    search_web_result = education_search_web(search_web_query, user_token=test_user_pro)
    print(f"Web Search Result for '{search_web_query}':\n{search_web_result[:500]}...")
    assert "Search results for online degree programs in psychology" in search_web_result
    print("Test 6 Passed.")

    # Test education_summarize_document_by_path (requires a dummy file)
    print("\n--- Testing education_summarize_document_by_path ---")
    dummy_upload_dir = Path("uploads") / test_user_pro / "education"
    dummy_upload_dir.mkdir(parents=True, exist_ok=True)
    dummy_file_path = dummy_upload_dir / "lecture_notes.txt"
    with open(dummy_file_path, "w") as f:
        f.write("These are notes from a lecture on quantum mechanics. It covered wave-particle duality and the Schrödinger equation.")
    
    result_summary = education_summarize_document_by_path(str(dummy_file_path))
    print(f"Lecture Notes Summary (Pro User): {result_summary}")
    assert "Mock summary of the provided text." in result_summary
    assert "quantum mechanics" in result_summary
    print("Test 7 Passed.")

    print("\nAll education_tool tests completed.")

    # Restore original requests.get
    requests.get = original_requests_get

    # Clean up dummy files and directories
    test_user_dirs = [Path("uploads") / test_user_pro, BASE_VECTOR_DIR / test_user_pro]
    for d in test_user_dirs:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            print(f"Cleaned up {d}")
