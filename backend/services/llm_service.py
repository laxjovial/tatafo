# backend/services/llm_service.py

import logging
from typing import List, Dict, Any, Optional
from fastapi import HTTPException, status # Import HTTPException for error handling
from datetime import datetime, timedelta # Needed for mock agent's date calculations

# Langchain Imports - UNCOMMENT THESE FOR REAL SETUP
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI # Example LLM for OpenAI
from langchain_community.llms import GoogleGenerativeAI # Example LLM for Google
from langchain_core.tools import Tool # This is already imported by @tool decorator, but good to have explicitly

# Import config_manager
from config.config_manager import config_manager

# Import user_manager for RBAC checks within services
from utils.user_manager import get_user_tier_capability, get_current_user

# Import all shared tools
from shared_tools.python_interpreter_tool import python_interpreter_with_rbac
from shared_tools.scraper_tool import scrape_web
from shared_tools.doc_summarizer import summarize_document # Not directly a @tool, will be wrapped if needed
from shared_tools.chart_generation_tool import generate_and_save_chart
from shared_tools.sentiment_analysis_tool import analyze_sentiment
from shared_tools.query_uploaded_docs_tool import query_uploaded_docs

# Import domain-specific tools
from domain_tools.finance_tools.finance_tool import get_stock_price, get_company_news, get_historical_stock_prices
# from domain_tools.medical_tools.medical_tool import get_drug_info, get_symptom_info # Future medical tools

logger = logging.getLogger(__name__)

class LLMService:
    """
    Manages interactions with Large Language Models and orchestrates tool usage.
    This service will be called by FastAPI endpoints.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initializes LLM and related components."""
        self.llm = self._load_llm()
        # Agent executor will be created dynamically per request based on user tools
        logger.info("LLMService initialized.")

    def _load_llm(self):
        """Loads the appropriate LLM based on configuration."""
        llm_provider = config_manager.get("llm.provider", "openai")
        model_name = config_manager.get("llm.model_name", "gpt-3.5-turbo")
        temperature = config_manager.get("llm.temperature", 0.5)
        api_key = None

        if llm_provider == "openai":
            api_key = config_manager.get_secret("openai_api_key")
            if not api_key:
                logger.error("OpenAI API key not found in secrets.")
                # For a real app, this should be a critical error or a fallback to a free model
                raise ValueError("OpenAI API key is required for OpenAI LLM provider.")
            
            # UNCOMMENT THIS FOR REAL SETUP
            # return ChatOpenAI(model_name=model_name, temperature=temperature, api_key=api_key)
            
            logger.warning("Using mock LLM for backend. Uncomment Langchain LLM import and instantiation for real use.")
            class MockLLM:
                def invoke(self, messages: List[BaseMessage]) -> Any:
                    # Simulate LLM response for chat_completion
                    last_user_message = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else "No user message"
                    return AIMessage(content=f"Mock LLM response to: {last_user_message}")
                
                def stream(self, messages: List[BaseMessage]) -> Any:
                    yield AIMessage(content=f"Mock streaming part 1...")
                    yield AIMessage(content=f"Mock streaming part 2...")
                
                # Mock for agent's invoke method (simplified)
                def _call(self, inputs: Dict[str, Any], stop: Optional[List[str]] = None) -> Dict[str, Any]:
                    # This mock simulates an agent's response, potentially calling a tool.
                    # It's a very simplified agent loop for testing.
                    prompt = inputs.get('input', '').lower()
                    tools_available_names = [t.name for t in inputs.get('tools', [])] # Extract tool names from mock tools list
                    
                    # Simulate tool calls based on prompt keywords and available tools
                    if "stock price" in prompt and "get_stock_price" in tools_available_names:
                        symbol = "AAPL" # Hardcoded for mock
                        mock_tool_output = get_stock_price(symbol, user_token=inputs.get('user_token', 'default'))
                        return {"output": f"I used get_stock_price. Output:\n{mock_tool_output}"}
                    
                    if "historical stock prices" in prompt and "get_historical_stock_prices" in tools_available_names:
                        symbol = "MSFT" # Hardcoded for mock
                        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                        end_date = datetime.now().strftime("%Y-%m-%d")
                        mock_tool_output = get_historical_stock_prices(symbol, start_date, end_date, user_token=inputs.get('user_token', 'default'))
                        return {"output": f"I used get_historical_stock_prices. Output:\n{mock_tool_output}"}

                    if "company news" in prompt and "get_company_news" in tools_available_names:
                        symbol = "TSLA" # Hardcoded for mock
                        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                        to_date = datetime.now().strftime("%Y-%m-%d")
                        mock_tool_output = get_company_news(symbol, from_date, to_date, user_token=inputs.get('user_token', 'default'))
                        return {"output": f"I used get_company_news. Output:\n{mock_tool_output}"}
                    
                    if "analyze data" in prompt or "run python" in prompt and "python_interpreter_with_rbac" in tools_available_names:
                        code_to_run = "print('Mock Python analysis result.')"
                        mock_tool_output = python_interpreter_with_rbac(code_to_run, user_token=inputs.get('user_token', 'default'))
                        return {"output": f"I used python_interpreter_with_rbac. Output:\n{mock_tool_output}"}
                    
                    if "search web" in prompt and "scrape_web" in tools_available_names:
                        mock_tool_output = scrape_web("mock web search query", user_token=inputs.get('user_token', 'default'))
                        return {"output": f"I used scrape_web. Output:\n{mock_tool_output}"}

                    if "sentiment" in prompt and "analyze_sentiment" in tools_available_names:
                        text_for_sentiment = "This is a test sentence for sentiment analysis."
                        mock_tool_output = analyze_sentiment(text_for_sentiment)
                        return {"output": f"I used analyze_sentiment. Output:\n{mock_tool_output}"}

                    if "query document" in prompt and "query_uploaded_docs" in tools_available_names:
                        mock_tool_output = query_uploaded_docs("mock document query", user_token=inputs.get('user_token', 'default'), section="general")
                        return {"output": f"I used query_uploaded_docs. Output:\n{mock_tool_output}"}
                    
                    if "generate chart" in prompt and "generate_and_save_chart" in tools_available_names:
                        mock_data = json.dumps([{"x": 1, "y": 10}, {"x": 2, "y": 20}])
                        mock_tool_output = generate_and_save_chart(mock_data, "line", "x", "y", user_token=inputs.get('user_token', 'default'))
                        return {"output": f"I used generate_and_save_chart. Output:\n{mock_tool_output}"}

                    # Fallback if no specific tool action is simulated
                    return {"output": f"Mock LLM agent response to: '{prompt}'. I considered the available tools but didn't find a direct match for a tool call based on keywords. If you need a specific tool, please be explicit."}

            return MockLLM()

        elif llm_provider == "google":
            api_key = config_manager.get_secret("google_api_key")
            if not api_key:
                logger.error("Google API key not found in secrets.")
                raise ValueError("Google API key is required for Google LLM provider.")
            
            # UNCOMMENT THIS FOR REAL SETUP
            # return GoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key)
            
            logger.warning("Using mock LLM for backend. Uncomment Langchain LLM import and instantiation for real use.")
            class MockLLM:
                def invoke(self, messages: List[BaseMessage]) -> Any:
                    last_user_message = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else "No user message"
                    return AIMessage(content=f"Mock Google LLM response to: {last_user_message}")
                
                def stream(self, messages: List[BaseMessage]) -> Any:
                    yield AIMessage(content=f"Mock Google streaming part 1...")
                    yield AIMessage(content=f"Mock Google streaming part 2...")
                
                def _call(self, inputs: Dict[str, Any], stop: Optional[List[str]] = None) -> Dict[str, Any]:
                    prompt = inputs.get('input', '')
                    # Simplified agent mock for Google LLM
                    return {"output": f"Mock Google LLM agent response to: {prompt}. (Tool actions would be simulated here)"}
            return MockLLM()
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """
        Generates a basic chat completion using the configured LLM (without tools).
        
        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries
                                             (e.g., [{"role": "user", "content": "Hello"}]).
        
        Returns:
            str: The AI's response content.
        """
        try:
            # Convert dict messages to Langchain BaseMessage objects
            langchain_messages = [self._convert_to_langchain_message(msg) for msg in messages]
            response = self.llm.invoke(langchain_messages)
            
            return response.content
        except Exception as e:
            logger.error(f"Error during LLM chat completion: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"LLM chat completion failed: {e}")

    async def chat_with_agent(self, prompt: str, chat_history: List[Dict[str, str]], user_token: str) -> str:
        """
        Orchestrates a chat with an agent, dynamically providing tools based on user's capabilities.
        This method is now fully implemented to use Langchain's AgentExecutor.
        
        Args:
            prompt (str): The current user prompt.
            chat_history (List[Dict[str, str]]): The full chat history.
            user_token (str): The user's authentication token for RBAC checks within tools.
        
        Returns:
            str: The agent's response.
        """
        logger.info(f"Agent chat initiated for user: {user_token}, prompt: '{prompt[:100]}...'")

        # Dynamically collect tools based on user's capabilities
        available_tools = []

        # Shared Tools
        if get_user_tier_capability(user_token, 'web_search_enabled', False):
            available_tools.append(scrape_web)
            logger.debug(f"Tool 'scrape_web' added for user {user_token}")
        
        if get_user_tier_capability(user_token, 'data_analysis_enabled', False):
            available_tools.append(python_interpreter_with_rbac)
            logger.debug(f"Tool 'python_interpreter_with_rbac' added for user {user_token}")
        
        if get_user_tier_capability(user_token, 'chart_generation_enabled', False):
            available_tools.append(generate_and_save_chart)
            logger.debug(f"Tool 'generate_and_save_chart' added for user {user_token}")

        if get_user_tier_capability(user_token, 'sentiment_analysis_enabled', False):
            available_tools.append(analyze_sentiment)
            logger.debug(f"Tool 'analyze_sentiment' added for user {user_token}")
        
        if get_user_tier_capability(user_token, 'document_query_enabled', False):
            available_tools.append(query_uploaded_docs)
            logger.debug(f"Tool 'query_uploaded_docs' added for user {user_token}")

        # Domain-specific Tools
        if get_user_tier_capability(user_token, 'finance_tool_access', False):
            available_tools.extend([get_stock_price, get_company_news])
            logger.debug(f"Finance tools (current price, company news) added for user {user_token}")
        
        # NEW: Add historical data tool based on capability
        if get_user_tier_capability(user_token, 'historical_data_access', False):
            available_tools.append(get_historical_stock_prices)
            logger.debug(f"Tool 'get_historical_stock_prices' added for user {user_token}")

        # Future Medical Tools
        # if get_user_tier_capability(user_token, 'medical_tool_access', False):
        #     available_tools.extend([get_drug_info, get_symptom_info])
        #     logger.debug(f"Medical tools added for user {user_token}")

        # Future Crypto Tools
        # if get_user_tier_capability(user_token, 'crypto_tool_access', False):
        #     available_tools.extend([get_crypto_price, get_historical_crypto_prices])
        #     logger.debug(f"Crypto tools added for user {user_token}")

        # Future News Tools
        # if get_user_tier_capability(user_token, 'news_tool_access', False):
        #     available_tools.append(get_general_news)
        #     logger.debug(f"General news tool added for user {user_token}")


        if not available_tools:
            logger.info(f"No specialized tools available for user {user_token}. Falling back to chat completion.")
            return self.chat_completion(chat_history + [{"role": "user", "content": prompt}])

        # Convert chat history to Langchain BaseMessage format
        langchain_chat_history = [self._convert_to_langchain_message(msg) for msg in chat_history]

        # Define the prompt template for the agent
        # This prompt guides the LLM to use the tools effectively.
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(
                "You are a helpful AI assistant with access to various tools. "
                "Use the tools to answer questions and fulfill requests. "
                "For web search, use `scrape_web`. "
                "For sentiment analysis, use `analyze_sentiment`. "
                "For current stock prices, use `get_stock_price`. "
                "For historical stock prices, use `get_historical_stock_prices`. " # Added to prompt
                "For company news, use `get_company_news`. "
                "For querying uploaded documents, use `query_uploaded_docs`. "
                "For data analysis or complex calculations, use the `python_interpreter_with_rbac` tool. "
                "For generating charts from data, use `generate_and_save_chart`. "
                "Always provide comprehensive answers based on tool outputs. "
                "If a tool call fails, inform the user and try to explain why or suggest alternatives."
                "When providing historical data, if asked to plot, use `generate_and_save_chart` with the JSON output from `get_historical_stock_prices`."
                "When analyzing data from uploaded documents, use `query_uploaded_docs` first, then pass the relevant content to `python_interpreter_with_rbac` for analysis."
                "If asked for time series analysis, use `python_interpreter_with_rbac`."
                "Remember to pass the `user_token` to any tool that requires it." # Explicit reminder for LLM
            ),
            *langchain_chat_history, # Previous chat history
            HumanMessage(content="{input}"), # Current user input
            AIMessage(content="{agent_scratchpad}"), # Where agent's thoughts and tool calls go
        ])

        # Create the Langchain agent
        # UNCOMMENT THIS FOR REAL SETUP
        # agent = create_react_agent(self.llm, available_tools, prompt_template)
        # agent_executor = AgentExecutor(agent=agent, tools=available_tools, verbose=True, handle_parsing_errors=True)
        
        logger.warning("Using mock agent executor. Uncomment Langchain agent creation for real use.")
        class MockAgentExecutor:
            def __init__(self, llm, tools, prompt):
                self.llm = llm
                self.tools = tools
                self.prompt = prompt
                logger.info(f"MockAgentExecutor initialized with {len(tools)} tools.")

            async def invoke(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                # Simulate agent logic: check prompt for keywords and simulate tool calls
                prompt_text = inputs.get('input', '').lower()
                user_token_for_tools = inputs.get('user_token', 'default') # Pass user_token to mock tool calls
                
                # Helper to check if tool is available by name
                def is_tool_available(tool_name: str) -> bool:
                    return any(t.name == tool_name for t in self.tools)

                # Mock tool calls based on prompt keywords and available tools
                if "historical stock prices" in prompt_text and is_tool_available("get_historical_stock_prices"):
                    symbol = "MSFT" # Hardcoded for mock
                    start_date = "2023-01-01" # Hardcoded for mock
                    end_date = "2023-01-05" # Hardcoded for mock
                    mock_tool_output = get_historical_stock_prices(symbol, start_date, end_date, user_token=user_token_for_tools)
                    return {"output": f"I used get_historical_stock_prices. Output:\n{mock_tool_output}"}
                
                if "stock price" in prompt_text and is_tool_available("get_stock_price"):
                    symbol = "AAPL" # Hardcoded for mock
                    mock_tool_output = get_stock_price(symbol, user_token=user_token_for_tools)
                    return {"output": f"I used get_stock_price. Output:\n{mock_tool_output}"}
                
                if "company news" in prompt_text and is_tool_available("get_company_news"):
                    symbol = "TSLA" # Hardcoded for mock
                    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                    to_date = datetime.now().strftime("%Y-%m-%d")
                    mock_tool_output = get_company_news(symbol, from_date, to_date, user_token=user_token_for_tools)
                    return {"output": f"I used get_company_news. Output:\n{mock_tool_output}"}
                
                if ("analyze data" in prompt_text or "run python" in prompt_text or "time series analysis" in prompt_text) and is_tool_available("python_interpreter_with_rbac"):
                    code_to_run = "print('Mock Python analysis result for your data.')"
                    mock_tool_output = python_interpreter_with_rbac(code_to_run, user_token=user_token_for_tools)
                    return {"output": f"I used python_interpreter_with_rbac. Output:\n{mock_tool_output}"}
                
                if "search web" in prompt_text and is_tool_available("scrape_web"):
                    mock_tool_output = scrape_web("mock web search query", user_token=user_token_for_tools)
                    return {"output": f"I used scrape_web. Output:\n{mock_tool_output}"}

                if "sentiment" in prompt_text and is_tool_available("analyze_sentiment"):
                    text_for_sentiment = "This is a test sentence for sentiment analysis."
                    mock_tool_output = analyze_sentiment(text_for_sentiment)
                    return {"output": f"I used analyze_sentiment. Output:\n{mock_tool_output}"}

                if "query document" in prompt_text and is_tool_available("query_uploaded_docs"):
                    mock_tool_output = query_uploaded_docs("mock document query", user_token=user_token_for_tools, section="general")
                    return {"output": f"I used query_uploaded_docs. Output:\n{mock_tool_output}"}
                
                if "generate chart" in prompt_text and is_tool_available("generate_and_save_chart"):
                    mock_data = json.dumps([{"x": 1, "y": 10}, {"x": 2, "y": 20}])
                    mock_tool_output = generate_and_save_chart(mock_data, "line", "x", "y", user_token=user_token_for_tools)
                    return {"output": f"I used generate_and_save_chart. Output:\n{mock_tool_output}"}

                # Fallback if no specific tool action is simulated
                return {"output": f"Mock LLM agent response to: '{prompt_text}'. I considered the available tools but didn't find a direct match for a tool call based on keywords. If you need a specific tool, please be explicit."}

        agent_executor = MockAgentExecutor(self.llm, available_tools, prompt_template) # Use mock agent executor

        response = await agent_executor.invoke({"input": prompt, "chat_history": langchain_chat_history, "user_token": user_token, "tools": available_tools}) # Pass available_tools to mock agent

        return response["output"]

    def _convert_to_langchain_message(self, message: Dict[str, str]) -> BaseMessage:
        """Helper to convert dictionary messages to Langchain BaseMessage objects."""
        if message["role"] == "user":
            return HumanMessage(content=message["content"])
        elif message["role"] == "assistant":
            return AIMessage(content=message["content"])
        elif message["role"] == "system":
            return SystemMessage(content=message["content"])
        else:
            raise ValueError(f"Unknown message role: {message['role']}")

# Instantiate the LLMService as a singleton
llm_service = LLMService()

