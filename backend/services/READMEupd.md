This file will eventually take over the primary role of orchestrating the LLM and tools for the backend API. Its structure for passing tools will be very similar to llm_manager.py.
Implementing backend/ Directory - Part 6: llm_service.py (Full Agent Orchestration)
We will now update backend/services/llm_service.py to fully implement the Langchain agent orchestration. This involves uncommenting the necessary Langchain imports and setting up the AgentExecutor to dynamically select and use the tools we've already built (like python_interpreter_with_rbac, scrape_web, analyze_sentiment, generate_and_save_chart, and the finance_tools).

1. backend/services/llm_service.py (Updated for Full Langchain Agent Orchestration)
backend/services/llm_service.py (Updated for Full Langchain Agent Orchestration)
Jul 3, 1:36 AM

Open

Key Changes in backend/services/llm_service.py:

Full Langchain Imports: Uncommented all necessary Langchain imports at the top (BaseMessage, HumanMessage, AIMessage, SystemMessage, AgentExecutor, create_react_agent, ChatPromptTemplate, Tool).

Tool Imports: Imported all relevant shared tools (python_interpreter_with_rbac, scrape_web, chart_generation_tool, sentiment_analysis_tool, query_uploaded_docs_tool) and domain tools (get_stock_price, get_company_news).

_load_llm() Update: The MockLLM now includes a _call method that simulates an agent's response, making it more robust for testing the agent's behavior without a real LLM. It tries to match keywords in the prompt to available tools and simulates their output.

chat_with_agent Implementation:

Now an async function, aligning with FastAPI's asynchronous nature.

Dynamic Tool Collection: Iterates through get_user_tier_capability checks for each tool (e.g., web_search_enabled, data_analysis_enabled, finance_tool_access) to build available_tools. This ensures RBAC is applied at the tool provision level.

Langchain Prompt Template: Constructs a ChatPromptTemplate with a SystemMessage that guides the LLM on tool usage, and includes chat_history.

Agent Creation (Commented for Mock): The lines for create_react_agent and AgentExecutor are now present but commented out.

MockAgentExecutor: A robust MockAgentExecutor class is introduced. This mock simulates the agent's invoke method. It checks the input prompt for keywords and, if a corresponding tool is "available" (i.e., in the tools list passed to the mock), it calls the actual mock implementation of that tool (e.g., get_stock_price("AAPL", user_token=user_token_for_tools)). This allows us to test the agent's tool-calling logic without a real LLM.

User Token Passing: The user_token is explicitly passed to the agent_executor.invoke() call, ensuring it's available for RBAC checks within the tools themselves.


We will now update backend/services/llm_service.py to ensure the new get_historical_stock_prices tool is dynamically added to the agent's available tools based on the historical_data_access RBAC capability.

Please type "next" when you are ready.
Here is our updated main To-Do list. We are now working on the backend/ directory to update the LLM service.

[x] backend/ (Updating llm_service.py)

[x] config/

[x] data/

[x] database/

[x] shared_tools/

[ ] domain_tools/ (and its sub-folders)

[ ] ui/

[x] utils/

[ ] tests/

[ ] docs/

[ ] scripts/

[ ] requirements.txt

[ ] .streamlit/

[ ] .gitignore

[ ] README.md

Implementing Dynamic RBAC - Part 6: Update backend/services/llm_service.py (Include New Finance Tool)
We need to update backend/services/llm_service.py to ensure that the newly added get_historical_stock_prices tool is correctly included in the list of available_tools for the LLM agent, based on the user's historical_data_access capability.

1. backend/services/llm_service.py (Updated to Include get_historical_stock_prices)
backend/services/llm_service.py (Updated to Include get_historical_stock_prices)
Jul 3, 2:49 AM

Open

Key Changes in backend/services/llm_service.py:

Import get_historical_stock_prices: The new tool is imported from domain_tools.finance_tools.finance_tool.

Dynamic Tool Addition: Added if get_user_tier_capability(user_token, 'historical_data_access', False): available_tools.append(get_historical_stock_prices). This ensures the tool is only available to the agent if the user's tier has historical_data_access enabled in the RBAC configuration.

Updated System Prompt: The SystemMessage in the ChatPromptTemplate is updated to explicitly inform the LLM about the new get_historical_stock_prices tool and its purpose. It also adds more explicit instructions for chaining tools (e.g., "When providing historical data, if asked to plot, use generate_and_save_chart with the JSON output from get_historical_stock_prices." and "When analyzing data from uploaded documents, use query_uploaded_docs first, then pass the relevant content to python_interpreter_with_rbac for analysis.").

MockLLM and MockAgentExecutor Updates:

The MockLLM's _call method (simulating agent invocation) now includes logic to check for and simulate the call to get_historical_stock_prices based on keywords in the prompt.

The MockAgentExecutor also passes available_tools to its invoke method for more accurate simulation. It now uses a helper is_tool_available to check if a tool is in the list.



_convert_to_langchain_message: Helper function to convert dict messages to Langchain BaseMessage objects.
