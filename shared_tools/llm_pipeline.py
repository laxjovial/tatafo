from shared_tools.ai_tool import get_ai_insight
from domain_tools.finance_tools.finance_tool import finance_get_historical_stock_prices
from domain_tools.crypto_tools.crypto_tool import crypto_get_historical_crypto_price

def execute_pipeline(query: str) -> str:
    """
    Executes the LLM pipeline to answer a user's query.

    :param query: The user's query.
    :return: The response to the user's query.
    """
    # Use the AI to determine which tool to use
    prompt = f"""
    Given the user's query, determine which tool to use and what parameters to pass to it.
    The available tools are:
    - `finance_get_historical_stock_prices(ticker: str)`: Gets historical stock prices for a given ticker.
    - `crypto_get_historical_crypto_price(symbol: str)`: Gets historical crypto prices for a given symbol.

    Query: "{query}"

    Respond with a JSON object containing the tool name and the parameters.
    For example: {{"tool": "finance_get_historical_stock_prices", "params": {{"ticker": "AAPL"}}}}
    """
    ai_response = get_ai_insight(data={}, prompt=prompt)

    try:
        import json
        tool_info = json.loads(ai_response)
        tool_name = tool_info.get("tool")
        params = tool_info.get("params", {})

        if tool_name == "finance_get_historical_stock_prices":
            result = finance_get_historical_stock_prices(**params)
        elif tool_name == "crypto_get_historical_crypto_price":
            result = crypto_get_historical_crypto_price(**params)
        else:
            return "I'm sorry, I don't know how to answer that question."

        # Use the AI to generate a natural language response
        prompt = f"""
        Given the following data, provide a natural language response to the user's query: "{query}"

        Data:
        {result}
        """
        return get_ai_insight(data={}, prompt=prompt)

    except Exception as e:
        return f"An error occurred while executing the pipeline: {e}"
