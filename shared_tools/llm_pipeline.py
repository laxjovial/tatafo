from shared_tools.ai_tool import get_ai_insight
# Import the classes, not the methods directly
from domain_tools.finance_tools.finance_tool import FinanceTools
from domain_tools.crypto_tools.crypto_tool import CryptoTools
from backend.models.user_models import UserProfile # Assuming UserProfile is needed for tools

def execute_pipeline(query: str, user_context: UserProfile) -> str:
    """
    Executes the LLM pipeline to answer a user's query.

    :param query: The user's query.
    :param user_context: The UserProfile object for the current user.
    :return: The response to the user's query.
    """
    # Initialize tool instances (you might want to pass dependencies here if they are real)
    # For simplicity in this pipeline, we'll instantiate with None for managers/loggers.
    # In a real application, these would be properly injected.
    finance_tools = FinanceTools(config_manager=None, firestore_manager=None, log_event=None, document_tools=None)
    crypto_tools = CryptoTools(config_manager=None, firestore_manager=None, log_event=None, document_tools=None)

    # Use the AI to determine which tool to use
    prompt = f"""
    Given the user's query, determine which tool to use and what parameters to pass to it.
    The available tools are:
    - `finance_get_historical_stock_prices(symbol: str, start_date: str, end_date: str)`: Gets historical stock prices for a given stock symbol within a date range.
    - `crypto_get_historical_crypto_price(symbol: str, start_date: str, end_date: str)`: Gets historical crypto prices for a given symbol within a date range.
    - `finance_get_stock_price(symbol: str)`: Retrieves the current stock price for a given stock symbol.
    - `finance_get_company_overview(symbol: str)`: Retrieves a company's overview.
    - `finance_get_forex_exchange_rate(from_currency: str, to_currency: str)`: Retrieves the current exchange rate.
    - `crypto_get_crypto_price(symbol: str)`: Retrieves the current price for a given cryptocurrency symbol.

    Query: "{query}"

    Respond with a JSON object containing the tool name and the parameters.
    For example: {{"tool": "finance_get_historical_stock_prices", "params": {{"symbol": "AAPL", "start_date": "2023-01-01", "end_date": "2023-01-31"}}}}
    """
    ai_response = get_ai_insight(data={}, prompt=prompt)

    try:
        import json
        tool_info = json.loads(ai_response)
        tool_name = tool_info.get("tool")
        params = tool_info.get("params", {})

        result = "I'm sorry, I don't know how to answer that question." # Default response

        # Pass user_context to all tool calls
        if tool_name == "finance_get_historical_stock_prices":
            result = finance_tools.finance_get_historical_stock_prices(user_context=user_context, **params)
        elif tool_name == "crypto_get_historical_crypto_price":
            result = crypto_tools.crypto_get_historical_crypto_price(user_context=user_context, **params)
        elif tool_name == "finance_get_stock_price":
            result = finance_tools.finance_get_stock_price(user_context=user_context, **params)
        elif tool_name == "finance_get_company_overview":
            result = finance_tools.finance_get_company_overview(user_context=user_context, **params)
        elif tool_name == "finance_get_forex_exchange_rate":
            result = finance_tools.finance_get_forex_exchange_rate(user_context=user_context, **params)
        elif tool_name == "crypto_get_crypto_price":
            result = crypto_tools.crypto_get_crypto_price(user_context=user_context, **params)
        
        # Await the result since the tool methods are async
        if hasattr(result, '__await__'):
            result = await result

        # Use the AI to generate a natural language response
        prompt = f"""
        Given the following data, provide a natural language response to the user's query: "{query}"

        Data:
        {result}
        """
        return get_ai_insight(data={}, prompt=prompt)

    except Exception as e:
        return f"An error occurred while executing the pipeline: {e}"

