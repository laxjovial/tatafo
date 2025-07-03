# backend/services/llm_service.py

import logging
import json
from typing import List, Dict, Any, Optional
from fastapi import HTTPException, status
from datetime import datetime, timedelta

# Langchain Imports - UNCOMMENT THESE FOR REAL SETUP
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI # Example LLM for OpenAI
from langchain_community.llms import GoogleGenerativeAI # Example LLM for Google
from langchain_core.tools import Tool

# Import config_manager
from config.config_manager import config_manager

# Import user_manager for RBAC checks within services
from utils.user_manager import get_user_tier_capability, get_current_user

# Import all shared tools
from shared_tools.python_interpreter_tool import python_interpreter_with_rbac
from shared_tools.scraper_tool import scrape_web
from shared_tools.doc_summarizer import summarize_document
from shared_tools.chart_generation_tool import generate_and_save_chart
from shared_tools.sentiment_analysis_tool import analyze_sentiment
from shared_tools.query_uploaded_docs_tool import query_uploaded_docs

# Import domain-specific tools
from domain_tools.finance_tools.finance_tool import get_stock_price, get_company_news, get_historical_stock_prices, lookup_stock_symbol
from domain_tools.crypto_tools.crypto_tool import get_crypto_price, get_historical_crypto_prices, get_crypto_id_by_symbol
# from domain_tools.medical_tools.medical_tool import get_drug_info, get_symptom_info # Future medical tools
# from domain_tools.news_tools.news_tool import get_general_news # Future news tools

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
        """Initializes LLM and related components. LLM is now loaded dynamically per request."""
        # LLM is now loaded within chat_with_agent to apply dynamic temperature
        logger.info("LLMService initialized. LLM will be loaded per request.")
        self.llm = None # Initialize as None, will be set in chat_with_agent

    def _load_llm(self, user_token: str, user_provided_temperature: Optional[float] = None):
        """
        Loads the appropriate LLM based on configuration and user's RBAC capabilities for temperature.
        """
        llm_provider = config_manager.get("llm.provider", "openai")
        model_name = config_manager.get("llm.model_name", "gpt-3.5-turbo")
        
        # Determine effective temperature based on RBAC
        can_control_temp = get_user_tier_capability(user_token, 'llm_temperature_control_enabled', False)
        tier_default_temp = get_user_tier_capability(user_token, 'llm_default_temperature', config_manager.get('llm.temperature', 0.7))
        max_allowed_temp = get_user_tier_capability(user_token, 'llm_max_temperature', 1.0)

        effective_temperature = tier_default_temp
        if can_control_temp and user_provided_temperature is not None:
            effective_temperature = min(user_provided_temperature, max_allowed_temp)
            logger.info(f"User {user_token} can control temperature. Using provided {user_provided_temperature}, capped at {max_allowed_temp}. Effective: {effective_temperature}")
        else:
            logger.info(f"User {user_token} cannot control temperature or none provided. Using tier default: {effective_temperature}")

        api_key = None

        if llm_provider == "openai":
            api_key = config_manager.get_secret("openai_api_key")
            if not api_key:
                logger.error("OpenAI API key not found in secrets.")
                raise ValueError("OpenAI API key is required for OpenAI LLM provider.")
            
            # UNCOMMENT THIS FOR REAL SETUP
            # return ChatOpenAI(model_name=model_name, temperature=effective_temperature, api_key=api_key)
            
            logger.warning("Using mock LLM for backend. Uncomment Langchain LLM import and instantiation for real use.")
            class MockLLM:
                def __init__(self, temp: float):
                    self.temperature = temp
                    logger.info(f"Mock LLM initialized with temperature: {self.temperature}")

                def invoke(self, messages: List[BaseMessage]) -> Any:
                    last_user_message = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else "No user message"
                    return AIMessage(content=f"Mock LLM response (temp={self.temperature}) to: {last_user_message}")
                
                def stream(self, messages: List[BaseMessage]) -> Any:
                    yield AIMessage(content=f"Mock streaming part 1 (temp={self.temperature})...")
                    yield AIMessage(content=f"Mock streaming part 2 (temp={self.temperature})...")
                
                # Mock for agent's invoke method (simplified)
                async def _call(self, inputs: Dict[str, Any], stop: Optional[List[str]] = None) -> Dict[str, Any]:
                    prompt = inputs.get('input', '').lower()
                    tools_available_names = [t.name for t in inputs.get('tools', [])]
                    user_token_for_tools = inputs.get('user_token', 'default')
                    
                    def is_tool_available(tool_name: str) -> bool:
                        return tool_name in tools_available_names

                    # Simulate tool calls based on prompt keywords and available tools
                    if ("price of apple" in prompt or "apple stock" in prompt) and is_tool_available("lookup_stock_symbol") and is_tool_available("get_stock_price"):
                        mock_symbol = lookup_stock_symbol("Apple", user_token=user_token_for_tools)
                        if "Error" not in mock_symbol:
                            mock_tool_output = get_stock_price(mock_symbol, user_token=user_token_for_tools)
                            return {"output": f"I used lookup_stock_symbol to get '{mock_symbol}' and then get_stock_price. Output:\n{mock_tool_output}"}
                    
                    if ("price of bitcoin" in prompt or "btc price" in prompt) and is_tool_available("get_crypto_id_by_symbol") and is_tool_available("get_crypto_price"):
                        mock_coin_id = get_crypto_id_by_symbol("btc", user_token=user_token_for_tools)
                        if "Error" not in mock_coin_id:
                            mock_tool_output = get_crypto_price(mock_coin_id, user_token=user_token_for_tools)
                            return {"output": f"I used get_crypto_id_by_symbol to get '{mock_coin_id}' and then get_crypto_price. Output:\n{mock_tool_output}"}

                    if "stock price" in prompt and is_tool_available("get_stock_price"):
                        symbol = "AAPL"
                        mock_tool_output = get_stock_price(symbol, user_token=user_token_for_tools)
                        return {"output": f"I used get_stock_price. Output:\n{mock_tool_output}"}
                    
                    if "historical stock prices" in prompt and is_tool_available("get_historical_stock_prices"):
                        symbol = "MSFT"
                        start_date = "2023-01-01"
                        end_date = "2023-01-05"
                        mock_tool_output = get_historical_stock_prices(symbol, start_date, end_date, user_token=user_token_for_tools)
                        return {"output": f"I used get_historical_stock_prices. Output:\n{mock_tool_output}"}

                    if "company news" in prompt and is_tool_available("get_company_news"):
                        symbol = "TSLA"
                        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                        to_date = datetime.now().strftime("%Y-%m-%d")
                        mock_tool_output = get_company_news(symbol, from_date, to_date, user_token=user_token_for_tools)
                        return {"output": f"I used get_company_news. Output:\n{mock_tool_output}"}
                    
                    if "lookup stock symbol" in prompt and is_tool_available("lookup_stock_symbol"):
                        company_name = "Google"
                        mock_tool_output = lookup_stock_symbol(company_name, user_token=user_token_for_tools)
                        return {"output": f"I used lookup_stock_symbol. Output:\n{mock_tool_output}"}

                    if "crypto price" in prompt and is_tool_available("get_crypto_price"):
                        coin_id = "ethereum"
                        mock_tool_output = get_crypto_price(coin_id, user_token=user_token_for_tools)
                        return {"output": f"I used get_crypto_price. Output:\n{mock_tool_output}"}

                    if "historical crypto prices" in prompt and is_tool_available("get_historical_crypto_prices"):
                        coin_id = "bitcoin"
                        mock_tool_output = get_historical_crypto_prices(coin_id, "usd", 7, user_token=user_token_for_tools)
                        return {"output": f"I used get_historical_crypto_prices. Output:\n{mock_tool_output}"}

                    if "lookup crypto id" in prompt and is_tool_available("get_crypto_id_by_symbol"):
                        symbol = "sol"
                        mock_tool_output = get_crypto_id_by_symbol(symbol, user_token=user_token_for_tools)
                        return {"output": f"I used get_crypto_id_by_symbol. Output:\n{mock_tool_output}"}
                    
                    if ("analyze data" in prompt or "run python" in prompt or "time series analysis" in prompt or "regression analysis" in prompt or "machine learning" in prompt or "ml model" in prompt) and is_tool_available("python_interpreter_with_rbac"):
                        code_to_run = "print('Mock Python analysis result for your data, potentially including ML/regression.')"
                        mock_tool_output = python_interpreter_with_rbac(code_to_run, user_token=user_token_for_tools)
                        return {"output": f"I used python_interpreter_with_rbac. Output:\n{mock_tool_output}"}
                    
                    if "search web" in prompt and is_tool_available("scrape_web"):
                        mock_tool_output = scrape_web("mock web search query", user_token=user_token_for_tools)
                        return {"output": f"I used scrape_web. Output:\n{mock_tool_output}"}

                    if "sentiment" in prompt and is_tool_available("analyze_sentiment"):
                        text_for_sentiment = "This is a test sentence for sentiment analysis."
                        mock_tool_output = analyze_sentiment(text_for_sentiment)
                        return {"output": f"I used analyze_sentiment. Output:\n{mock_tool_output}"}

                    if "query document" in prompt and is_tool_available("query_uploaded_docs"):
                        mock_tool_output = query_uploaded_docs("mock document query", user_token=user_token_for_tools, section="general")
                        return {"output": f"I used query_uploaded_docs. Output:\n{mock_tool_output}"}
                    
                    if "generate chart" in prompt and is_tool_available("generate_and_save_chart"):
                        mock_data = json.dumps([{"x": 1, "y": 10}, {"x": 2, "y": 20}])
                        mock_tool_output = generate_and_save_chart(mock_data, "line", "x", "y", user_token=user_token_for_tools)
                        return {"output": f"I used generate_and_save_chart. Output:\n{mock_tool_output}"}

                    # Fallback if no specific tool action is simulated
                    return {"output": f"Mock LLM agent response (temp={self.temperature}) to: '{prompt}'. I considered the available tools but didn't find a direct match for a tool call based on keywords. If you need a specific tool, please be explicit."}

            return MockLLM(effective_temperature)

        elif llm_provider == "google":
            api_key = config_manager.get_secret("google_api_key")
            if not api_key:
                logger.error("Google API key not found in secrets.")
                raise ValueError("Google API key is required for Google LLM provider.")
            
            # UNCOMMENT THIS FOR REAL SETUP
            # return GoogleGenerativeAI(model=model_name, temperature=effective_temperature, google_api_key=api_key)
            
            logger.warning("Using mock LLM for backend. Uncomment Langchain LLM import and instantiation for real use.")
            class MockLLM:
                def __init__(self, temp: float):
                    self.temperature = temp
                    logger.info(f"Mock Google LLM initialized with temperature: {self.temperature}")

                def invoke(self, messages: List[BaseMessage]) -> Any:
                    last_user_message = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else "No user message"
                    return AIMessage(content=f"Mock Google LLM response (temp={self.temperature}) to: {last_user_message}")
                
                def stream(self, messages: List[BaseMessage]) -> Any:
                    yield AIMessage(content=f"Mock Google streaming part 1 (temp={self.temperature})...")
                    yield AIMessage(content=f"Mock Google streaming part 2 (temp={self.temperature})...")
                
                async def _call(self, inputs: Dict[str, Any], stop: Optional[List[str]] = None) -> Dict[str, Any]:
                    prompt = inputs.get('input', '')
                    return {"output": f"Mock Google LLM agent response (temp={self.temperature}) to: {prompt}. (Tool actions would be simulated here)"}
            return MockLLM(effective_temperature)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    def chat_completion(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        """
        Generates a basic chat completion using the configured LLM (without tools).
        
        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries.
            temperature (float, optional): The LLM temperature to use for this completion.
                                           If None, the default from _load_llm will be used.
        Returns:
            str: The AI's response content.
        """
        try:
            # For chat_completion, we'll load a temporary LLM instance with the specified temperature
            # This is less efficient but ensures the temperature is applied.
            # In a real setup, if chat_completion is always used with an agent, this might be simplified.
            temp_llm = self._load_llm(user_token="default", user_provided_temperature=temperature) # Use default user for chat_completion if no agent context
            
            langchain_messages = [self._convert_to_langchain_message(msg) for msg in messages]
            response = temp_llm.invoke(langchain_messages) # Use temp_llm here
            
            return response.content
        except Exception as e:
            logger.error(f"Error during LLM chat completion: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"LLM chat completion failed: {e}")

    async def chat_with_agent(self, prompt: str, chat_history: List[Dict[str, str]], user_token: str, user_provided_temperature: Optional[float] = None) -> str:
        """
        Orchestrates a chat with an agent, dynamically providing tools based on user's capabilities.
        This method is now fully implemented to use Langchain's AgentExecutor.
        
        Args:
            prompt (str): The current user prompt.
            chat_history (List[Dict[str, str]]): The full chat history.
            user_token (str): The user's authentication token for RBAC checks within tools.
            user_provided_temperature (float, optional): The temperature provided by the user from the frontend.
                                                         Will be applied based on RBAC.
        Returns:
            str: The agent's response.
        """
        logger.info(f"Agent chat initiated for user: {user_token}, prompt: '{prompt[:100]}...', user_provided_temp: {user_provided_temperature}")

        # Load LLM for this request with the determined temperature
        self.llm = self._load_llm(user_token, user_provided_temperature)

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
            available_tools.extend([get_stock_price, get_company_news, lookup_stock_symbol])
            logger.debug(f"Finance tools (current price, company news, symbol lookup) added for user {user_token}")
        
        if get_user_tier_capability(user_token, 'historical_data_access', False):
            available_tools.append(get_historical_stock_prices)
            logger.debug(f"Tool 'get_historical_stock_prices' added for user {user_token}")

        if get_user_tier_capability(user_token, 'crypto_tool_access', False):
            available_tools.extend([get_crypto_price, get_historical_crypto_prices, get_crypto_id_by_symbol])
            logger.debug(f"Crypto tools added for user {user_token}")

        # Future Medical Tools
        # if get_user_tier_capability(user_token, 'medical_tool_access', False):
        #     available_tools.extend([get_drug_info, get_symptom_info])
        #     logger.debug(f"Medical tools added for user {user_token}")

        # Future News Tools
        # if get_user_tier_capability(user_token, 'news_tool_access', False):
        #     available_tools.append(get_general_news)
        #     logger.debug(f"General news tool added for user {user_token}")


        if not available_tools:
            logger.info(f"No specialized tools available for user {user_token}. Falling back to chat completion.")
            return self.chat_completion(chat_history + [{"role": "user", "content": prompt}], temperature=user_provided_temperature)

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
                "For historical stock prices, use `get_historical_stock_prices`. "
                "For company news, use `get_company_news`. "
                "To find a stock symbol from a company name, use `lookup_stock_symbol`. "
                "For current cryptocurrency prices, use `get_crypto_price`. "
                "For historical cryptocurrency prices, use `get_historical_crypto_prices`. "
                "To find a cryptocurrency ID from its symbol, use `get_crypto_id_by_symbol`. "
                "For querying uploaded documents, use `query_uploaded_docs`. "
                "For **data analysis**, complex calculations, time series analysis, regression analysis, " # Explicitly added "data analysis"
                "or any other machine learning tasks (supervised or unsupervised), use the `python_interpreter_with_rbac` tool. "
                "For generating charts from data, use `generate_and_save_chart`. "
                "Always provide comprehensive answers based on tool outputs. "
                "If a tool call fails, inform the user and try to explain why or suggest alternatives."
                "When providing historical data, if asked to plot, use `generate_and_save_chart` with the JSON output from `get_historical_stock_prices` or `get_historical_crypto_prices`."
                "When analyzing data from uploaded documents, use `query_uploaded_docs` first, then pass the relevant content to `python_interpreter_with_rbac` for analysis."
                "Remember to pass the `user_token` to any tool that requires it."
                "If a user asks for a stock by name (e.g., 'Apple'), first use `lookup_stock_symbol` to get the ticker, then use the appropriate stock tool."
                "If a user asks for crypto by symbol (e.g., 'btc'), first use `get_crypto_id_by_symbol` to get the ID, then use the appropriate crypto tool."
            ),
            *langchain_chat_history,
            HumanMessage(content="{input}"),
            AIMessage(content="{agent_scratchpad}"),
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
                logger.info(f"MockAgentExecutor initialized with {len(tools)} tools. LLM Temp: {self.llm.temperature}")

            async def invoke(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                prompt_text = inputs.get('input', '').lower()
                user_token_for_tools = inputs.get('user_token', 'default')
                
                def is_tool_available(tool_name: str) -> bool:
                    return any(t.name == tool_name for t in self.tools)

                if ("price of apple" in prompt_text or "apple stock" in prompt_text) and is_tool_available("lookup_stock_symbol") and is_tool_available("get_stock_price"):
                    mock_symbol = lookup_stock_symbol("Apple", user_token=user_token_for_tools)
                    if "Error" not in mock_symbol:
                        mock_tool_output = get_stock_price(mock_symbol, user_token=user_token_for_tools)
                        return {"output": f"I used lookup_stock_symbol to get '{mock_symbol}' and then get_stock_price. Output:\n{mock_tool_output}"}
                
                if ("price of bitcoin" in prompt_text or "btc price" in prompt_text) and is_tool_available("get_crypto_id_by_symbol") and is_tool_available("get_crypto_price"):
                    mock_coin_id = get_crypto_id_by_symbol("btc", user_token=user_token_for_tools)
                    if "Error" not in mock_coin_id:
                        mock_tool_output = get_crypto_price(mock_coin_id, user_token=user_token_for_tools)
                        return {"output": f"I used get_crypto_id_by_symbol to get '{mock_coin_id}' and then get_crypto_price. Output:\n{mock_tool_output}"}

                if "stock price" in prompt_text and is_tool_available("get_stock_price"):
                    symbol = "AAPL"
                    mock_tool_output = get_stock_price(symbol, user_token=user_token_for_tools)
                    return {"output": f"I used get_stock_price. Output:\n{mock_tool_output}"}
                
                if "historical stock prices" in prompt_text and is_tool_available("get_historical_stock_prices"):
                    symbol = "MSFT"
                    start_date = "2023-01-01"
                    end_date = "2023-01-05"
                    mock_tool_output = get_historical_stock_prices(symbol, start_date, end_date, user_token=user_token_for_tools)
                    return {"output": f"I used get_historical_stock_prices. Output:\n{mock_tool_output}"}

                if "company news" in prompt_text and is_tool_available("get_company_news"):
                    symbol = "TSLA"
                    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                    to_date = datetime.now().strftime("%Y-%m-%d")
                    mock_tool_output = get_company_news(symbol, from_date, to_date, user_token=user_token_for_tools)
                    return {"output": f"I used get_company_news. Output:\n{mock_tool_output}"}
                
                if "lookup stock symbol" in prompt_text and is_tool_available("lookup_stock_symbol"):
                    company_name = "Google"
                    mock_tool_output = lookup_stock_symbol(company_name, user_token=user_token_for_tools)
                    return {"output": f"I used lookup_stock_symbol. Output:\n{mock_tool_output}"}

                if "crypto price" in prompt_text and is_tool_available("get_crypto_price"):
                    coin_id = "ethereum"
                    mock_tool_output = get_crypto_price(coin_id, user_token=user_token_for_tools)
                    return {"output": f"I used get_crypto_price. Output:\n{mock_tool_output}"}

                if "historical crypto prices" in prompt_text and is_tool_available("get_historical_crypto_prices"):
                    coin_id = "bitcoin"
                    mock_tool_output = get_historical_crypto_prices(coin_id, "usd", 7, user_token=user_token_for_tools)
                    return {"output": f"I used get_historical_crypto_prices. Output:\n{mock_tool_output}"}

                if "lookup crypto id" in prompt_text and is_tool_available("get_crypto_id_by_symbol"):
                    symbol = "sol"
                    mock_tool_output = get_crypto_id_by_symbol(symbol, user_token=user_token_for_tools)
                    return {"output": f"I used get_crypto_id_by_symbol. Output:\n{mock_tool_output}"}
                
                if ("analyze data" in prompt_text or "run python" in prompt_text or "time series analysis" in prompt_text or "regression analysis" in prompt_text or "machine learning" in prompt_text or "ml model" in prompt) and is_tool_available("python_interpreter_with_rbac"):
                    code_to_run = "print('Mock Python analysis result for your data, potentially including ML/regression.')"
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
                return {"output": f"Mock LLM agent response (temp={self.llm.temperature}) to: '{prompt}'. I considered the available tools but didn't find a direct match for a tool call based on keywords. If you need a specific tool, please be explicit."}

        agent_executor = MockAgentExecutor(self.llm, available_tools, prompt_template)

        response = await agent_executor.invoke({"input": prompt, "chat_history": langchain_chat_history, "user_token": user_token, "tools": available_tools})

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

