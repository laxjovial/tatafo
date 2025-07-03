Implementing domain_tools/ Directory - Part 1
The domain_tools/ directory will contain specialized tools categorized by domain (e.g., finance, medical, legal, news, sports, weather, entertainment). Each sub-directory will contain an _tool.py file with the specific tools for that domain.

Let's start by creating the main domain_tools/__init__.py file and then the finance_tools/ sub-directory with its finance_tool.py.

1. domain_tools/__init__.py
This file simply marks domain_tools as a Python package.
domain_tools/__init__.py
Jul 3, 1:11 AM

Open

2. domain_tools/finance_tools/__init__.py
This file marks domain_tools/finance_tools as a Python package.
domain_tools/finance_tools/__init__.py
Jul 3, 1:11 AM

Open

3. domain_tools/finance_tools/finance_tool.py
This file will contain tools specific to financial data, such as fetching stock prices or company information. We will integrate config_manager for API keys and ensure RBAC is considered for any rate limits or premium features.
domain_tools/finance_tools/finance_tool.py
Jul 3, 1:11 AM

Open

Key Features and Updates in domain_tools/finance_tools/finance_tool.py:

Multiple API Integration:

get_stock_price attempts to use Alpha Vantage first, then falls back to Finnhub if Alpha Vantage fails or its key is not configured. This provides robustness.

get_company_news uses Finnhub.

_get_finance_api_key uses config_manager.get_secret() to retrieve API keys, ensuring they are loaded securely from secrets.toml.

RBAC Integration:

Both get_stock_price and get_company_news perform an RBAC check using get_user_tier_capability(user_token, 'finance_tool_access', True). This assumes a finance_tool_access capability defined in rbac_capabilities.yaml to control access to all finance tools.

Date Handling: get_company_news handles from_date and to_date parameters, defaulting to the last 7 days if not provided, and includes validation for date format and order.

Robust Error Handling: Includes try-except blocks for network requests (requests.exceptions.RequestException), API-specific errors (e.g., "Error Message" from Alpha Vantage), and general exceptions.

Test Suite Enhancement: The if __name__ == "__main__": block has been significantly expanded to include:

Comprehensive mocks for st.secrets, config_manager, and user_manager.

Mocks for requests.get to simulate responses from Alpha Vantage and Finnhub for both stock prices and company news, including success, API-specific errors, and network errors.

Tests for different user tiers (Pro, Free, Admin) to verify RBAC access control.

Tests for valid and invalid inputs (e.g., date formats, date ranges).
