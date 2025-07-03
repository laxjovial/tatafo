Implementing domain_tools/ Directory - Part 3: crypto_tools/
We will now create the domain_tools/crypto_tools/ sub-directory and its crypto_tool.py to handle cryptocurrency-specific data, including current prices and historical data. This will follow a similar structure to our finance_tools.

1. domain_tools/crypto_tools/__init__.py
This file marks domain_tools/crypto_tools as a Python package.
domain_tools/crypto_tools/__init__.py
Jul 3, 2:49 AM

Open

2. domain_tools/crypto_tools/crypto_tool.py
This file will contain tools for fetching current and historical cryptocurrency prices. We will use CoinGecko API as an example.
domain_tools/crypto_tools/crypto_tool.py
Jul 3, 2:49 AM

Open

Key Features and Updates in domain_tools/crypto_tools/crypto_tool.py:

New Tool: get_crypto_price: Fetches the current price of a cryptocurrency using CoinGecko API, supporting different vs_currency options.

New Tool: get_historical_crypto_prices: Retrieves historical daily prices for a cryptocurrency for a given number of days using CoinGecko API.

Returns data as a JSON string, making it directly compatible with chart_generation_tool.

Includes a cap on days (365) to align with CoinGecko's free tier limitations.

New Tool: get_crypto_id_by_symbol: This is a crucial lookup tool that converts a common crypto symbol (e.g., "btc") into its CoinGecko ID (e.g., "bitcoin"). This directly addresses your request for handling symbols/abbreviations for crypto.

RBAC Integration:

All tools check for crypto_tool_access.

get_historical_crypto_prices also checks for historical_data_access, ensuring consistency with stock historical data.

_get_crypto_api_key: Helper function to retrieve API keys from secrets.

Robust Error Handling: Includes try-except blocks for network requests and API-specific errors.

Test Suite Enhancement: Comprehensive mocks for CoinGecko API responses (current price, historical data, coin list) and RBAC checks for various user tiers and scenarios.
