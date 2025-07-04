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
from langchain_community.chat_models import ChatOllama # For Ollama
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
from domain_tools.medical_tools.medical_tool import get_drug_info, get_symptom_info
from domain_tools.news_tools.news_tool import get_general_news
from domain_tools.legal_tools.legal_tool import get_legal_definition, get_case_summary
from domain_tools.education_tools.education_tool import get_academic_definition, get_historical_event_summary
from domain_tools.entertainment_tools.entertainment_tool import get_movie_details, get_music_artist_info
from domain_tools.weather_tools.weather_tool import get_current_weather, get_weather_forecast
from domain_tools.travel_tools.travel_tool import find_flights, find_hotels # NEW: Import travel tools

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
        logger.info("LLMService initialized. LLM will be loaded per request based on user preferences and RBAC.")
        self.llm = None # Initialize as None, will be set in chat_with_agent

    def _load_llm(self, user_token: str, 
                  user_provided_temperature: Optional[float] = None,
                  user_provided_llm_provider: Optional[str] = None,
                  user_provided_model_name: Optional[str] = None):
        """
        Loads the appropriate LLM based on configuration, user's RBAC capabilities,
        and user-provided selections for temperature, provider, and model name.
        """
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

        # Determine effective LLM provider and model name based on RBAC
        can_select_model = get_user_tier_capability(user_token, 'llm_model_selection_enabled', False)
        
        effective_llm_provider = config_manager.get("llm.provider", "openai")
        effective_model_name = config_manager.get("llm.model_name", "gpt-3.5-turbo")

        if can_select_model:
            if user_provided_llm_provider:
                effective_llm_provider = user_provided_llm_provider
            if user_provided_model_name:
                effective_model_name = user_provided_model_name
            logger.info(f"User {user_token} can select model. Using provided provider '{user_provided_llm_provider}' and model '{user_provided_model_name}'. Effective: {effective_llm_provider}/{effective_model_name}")
        else:
            logger.info(f"User {user_token} cannot select model. Using config defaults: {effective_llm_provider}/{effective_model_name}")

        api_key = None # Will be set based on provider

        if effective_llm_provider == "openai":
            api_key = config_manager.get_secret("openai_api_key")
            if not api_key:
                logger.error("OpenAI API key not found in secrets.")
                raise ValueError("OpenAI API key is required for OpenAI LLM provider.")
            
            # UNCOMMENT THIS FOR REAL SETUP
            # return ChatOpenAI(model_name=effective_model_name, temperature=effective_temperature, api_key=api_key)
            
            logger.warning(f"Using mock LLM for backend (OpenAI). Temp: {effective_temperature}, Model: {effective_model_name}. Uncomment Langchain LLM import and instantiation for real use.")
            class MockLLM:
                def __init__(self, temp: float, model: str):
                    self.temperature = temp
                    self.model_name = model
                    logger.info(f"Mock LLM initialized with temperature: {self.temperature}, model: {self.model_name}")

                def invoke(self, messages: List[BaseMessage]) -> Any:
                    last_user_message = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else "No user message"
                    return AIMessage(content=f"Mock LLM response (provider=OpenAI, model={self.model_name}, temp={self.temperature}) to: {last_user_message}")
                
                def stream(self, messages: List[BaseMessage]) -> Any:
                    yield AIMessage(content=f"Mock streaming part 1 (OpenAI, {self.model_name}, {self.temperature})...")
                    yield AIMessage(content=f"Mock streaming part 2 (OpenAI, {self.model_name}, {self.temperature})...")
                
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
                    
                    if ("drug info" in prompt or "medication" in prompt) and is_tool_available("get_drug_info"):
                        drug_name = "Aspirin"
                        mock_tool_output = get_drug_info(drug_name, user_token=user_token_for_tools)
                        return {"output": f"I used get_drug_info. Output:\n{mock_tool_output}"}
                    
                    if ("symptom info" in prompt or "what causes" in prompt) and is_tool_available("get_symptom_info"):
                        symptom_name = "Headache"
                        mock_tool_output = get_symptom_info(symptom_name, user_token=user_token_for_tools)
                        return {"output": f"I used get_symptom_info. Output:\n{mock_tool_output}"}

                    if ("general news" in prompt or "latest news" in prompt) and is_tool_available("get_general_news"):
                        news_query = "technology"
                        mock_tool_output = get_general_news(news_query, user_token=user_token_for_tools)
                        return {"output": f"I used get_general_news. Output:\n{mock_tool_output}"}

                    if ("legal definition" in prompt or "define legal term" in prompt) and is_tool_available("get_legal_definition"):
                        term = "contract"
                        mock_tool_output = get_legal_definition(term, user_token=user_token_for_tools)
                        return {"output": f"I used get_legal_definition. Output:\n{mock_tool_output}"}

                    if ("case summary" in prompt or "summary of case" in prompt) and is_tool_available("get_case_summary"):
                        case_name = "smith v. jones"
                        mock_tool_output = get_case_summary(case_name, user_token=user_token_for_tools)
                        return {"output": f"I used get_case_summary. Output:\n{mock_tool_output}"}

                    if ("academic definition" in prompt or "define academic term" in prompt or "what is" in prompt) and is_tool_available("get_academic_definition"):
                        term = "photosynthesis"
                        mock_tool_output = get_academic_definition(term, user_token=user_token_for_tools)
                        return {"output": f"I used get_academic_definition. Output:\n{mock_tool_output}"}

                    if ("historical event" in prompt or "summary of history" in prompt) and is_tool_available("get_historical_event_summary"):
                        event_name = "moon landing"
                        mock_tool_output = get_historical_event_summary(event_name, user_token=user_token_for_tools)
                        return {"output": f"I used get_historical_event_summary. Output:\n{mock_tool_output}"}

                    if ("movie details" in prompt or "info about movie" in prompt) and is_tool_available("get_movie_details"):
                        movie_title = "Inception"
                        mock_tool_output = get_movie_details(movie_title, user_token=user_token_for_tools)
                        return {"output": f"I used get_movie_details. Output:\n{mock_tool_output}"}

                    if ("music artist" in prompt or "info about artist" in prompt) and is_tool_available("get_music_artist_info"):
                        artist_name = "Taylor Swift"
                        mock_tool_output = get_music_artist_info(artist_name, user_token=user_token_for_tools)
                        return {"output": f"I used get_music_artist_info. Output:\n{mock_tool_output}"}

                    if ("current weather" in prompt or "weather in" in prompt) and is_tool_available("get_current_weather"):
                        location = "London"
                        mock_tool_output = get_current_weather(location, user_token=user_token_for_tools)
                        return {"output": f"I used get_current_weather. Output:\n{mock_tool_output}"}

                    if ("weather forecast" in prompt or "forecast for" in prompt) and is_tool_available("get_weather_forecast"):
                        location = "New York"
                        days = 3
                        mock_tool_output = get_weather_forecast(location, days, user_token=user_token_for_tools)
                        return {"output": f"I used get_weather_forecast. Output:\n{mock_tool_output}"}

                    if ("find flights" in prompt or "flight info" in prompt) and is_tool_available("find_flights"): # NEW: Mock travel tool call
                        origin = "London"
                        destination = "New York"
                        date = "2025-07-15"
                        mock_tool_output = find_flights(origin, destination, date, user_token=user_token_for_tools)
                        return {"output": f"I used find_flights. Output:\n{mock_tool_output}"}

                    if ("find hotels" in prompt or "hotel availability" in prompt) and is_tool_available("find_hotels"): # NEW: Mock travel tool call
                        location = "Paris"
                        check_in_date = "2025-09-01"
                        check_out_date = "2025-09-05"
                        mock_tool_output = find_hotels(location, check_in_date, check_out_date, user_token=user_token_for_tools)
                        return {"output": f"I used find_hotels. Output:\n{mock_tool_output}"}

                    if ("analyze data" in prompt or "run python" in prompt or "time series analysis" in prompt or "regression analysis" in prompt or "machine learning" in prompt or "ml model" in prompt) and is_tool_available("python_interpreter_with_rbac"):
                        code_to_run = "print('Mock Python analysis result for your data, potentially including ML/regression.')"
                        mock_tool_output = python_interpreter_with_rbac(code_to_run, user_token=user_token_for_tools)
                        return {"output": f"I used python_interpreter_with_rbac. Output:\n{mock_tool_output}"}
                    
                    if "search web" in prompt and is_tool_available("scrape_web"):
                        mock_tool_output = scrape_web("mock web search query", user_token=user_token_for_tools)
                        return {"output": f"I used scrape_web. Output:\n{mock_tool_output}"}

                    if "sentiment" in prompt and is_tool_available("analyze_sentiment"):
                        text_for_sentiment = "This is a test sentence for sentiment analysis."
                        mock_tool_output = analyze_tool_output = analyze_sentiment(text_for_sentiment)
                        return {"output": f"I used analyze_sentiment. Output:\n{mock_tool_output}"}

                    if "query document" in prompt and is_tool_available("query_uploaded_docs"):
                        mock_tool_output = query_uploaded_docs("mock document query", user_token=user_token_for_tools, section="general")
                        return {"output": f"I used query_uploaded_docs. Output:\n{mock_tool_output}"}
                    
                    if "generate chart" in prompt and is_tool_available("generate_and_save_chart"):
                        mock_data = json.dumps([{"x": 1, "y": 10}, {"x": 2, "y": 20}])
                        mock_tool_output = generate_and_save_chart(mock_data, "line", "x", "y", user_token=user_token_for_tools)
                        return {"output": f"I used generate_and_save_chart. Output:\n{mock_tool_output}"}

                    # Fallback if no specific tool action is simulated
                    return {"output": f"Mock LLM agent response (provider={self.llm.model_name.split('-')[0]}, model={self.llm.model_name}, temp={self.llm.temperature}) to: '{prompt}'. I considered the available tools but didn't find a direct match for a tool call based on keywords. If you need a specific tool, please be explicit."}

            return MockLLM(effective_temperature, effective_model_name)

        elif effective_llm_provider == "google":
            api_key = config_manager.get_secret("google_api_key")
            if not api_key:
                logger.error("Google API key not found in secrets.")
                raise ValueError("Google API key is required for Google LLM provider.")
            
            # UNCOMMENT THIS FOR REAL SETUP
            # return GoogleGenerativeAI(model=effective_model_name, temperature=effective_temperature, google_api_key=api_key)
            
            logger.warning(f"Using mock LLM for backend (Google). Temp: {effective_temperature}, Model: {effective_model_name}. Uncomment Langchain LLM import and instantiation for real use.")
            class MockLLM:
                def __init__(self, temp: float, model: str):
                    self.temperature = temp
                    self.model_name = model
                    logger.info(f"Mock Google LLM initialized with temperature: {self.temperature}, model: {self.model_name}")

                def invoke(self, messages: List[BaseMessage]) -> Any:
                    last_user_message = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else "No user message"
                    return AIMessage(content=f"Mock Google LLM response (provider=Google, model={self.model_name}, temp={self.temperature}) to: {last_user_message}")
                
                def stream(self, messages: List[BaseMessage]) -> Any:
                    yield AIMessage(content=f"Mock Google streaming part 1 (Google, {self.model_name}, {self.temperature})...")
                    yield AIMessage(content=f"Mock Google streaming part 2 (Google, {self.model_name}, {self.temperature})...")
                
                async def _call(self, inputs: Dict[str, Any], stop: Optional[List[str]] = None) -> Dict[str, Any]:
                    prompt = inputs.get('input', '')
                    return {"output": f"Mock Google LLM agent response (provider={self.model_name}, temp={self.temperature}) to: {prompt}. (Tool actions would be simulated here)"}
            return MockLLM(effective_temperature, effective_model_name)

        elif effective_llm_provider == "ollama":
            # UNCOMMENT THIS FOR REAL SETUP
            # return ChatOllama(model=effective_model_name, temperature=effective_temperature)
            
            logger.warning(f"Using mock LLM for backend (Ollama). Temp: {effective_temperature}, Model: {effective_model_name}. Uncomment Langchain LLM import and instantiation for real use.")
            class MockLLM:
                def __init__(self, temp: float, model: str):
                    self.temperature = temp
                    self.model_name = model
                    logger.info(f"Mock Ollama LLM initialized with temperature: {self.temperature}, model: {self.model_name}")

                def invoke(self, messages: List[BaseMessage]) -> Any:
                    last_user_message = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else "No user message"
                    return AIMessage(content=f"Mock Ollama LLM response (provider=Ollama, model={self.model_name}, temp={self.temperature}) to: {last_user_message}")
                
                def stream(self, messages: List[BaseMessage]) -> Any:
                    yield AIMessage(content=f"Mock Ollama streaming part 1 (Ollama, {self.model_name}, {self.temperature})...")
                    yield AIMessage(content=f"Mock Ollama streaming part 2 (Ollama, {self.model_name}, {self.temperature})...")
                
                async def _call(self, inputs: Dict[str, Any], stop: Optional[List[str]] = None) -> Dict[str, Any]:
                    prompt = inputs.get('input', '')
                    return {"output": f"Mock Ollama LLM agent response (provider={self.model_name}, temp={self.temperature}) to: {prompt}. (Tool actions would be simulated here)"}
            return MockLLM(effective_temperature, effective_model_name)
        else:
            raise ValueError(f"Unsupported LLM provider: {effective_llm_provider}")

    def chat_completion(self, messages: List[Dict[str, str]], temperature: Optional[float] = None,
                        llm_provider: Optional[str] = None, model_name: Optional[str] = None) -> str:
        """
        Generates a basic chat completion using the configured LLM (without tools).
        
        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries.
            temperature (float, optional): The LLM temperature to use for this completion.
            llm_provider (str, optional): The LLM provider to use for this completion.
            model_name (str, optional): The LLM model name to use for this completion.
        Returns:
            str: The AI's response content.
        """
        try:
            temp_llm = self._load_llm(user_token="default",
                                      user_provided_temperature=temperature,
                                      user_provided_llm_provider=llm_provider,
                                      user_provided_model_name=model_name)
            
            langchain_messages = [self._convert_to_langchain_message(msg) for msg in messages]
            response = temp_llm.invoke(langchain_messages)
            
            return response.content
        except Exception as e:
            logger.error(f"Error during LLM chat completion: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"LLM chat completion failed: {e}")

    async def chat_with_agent(self, prompt: str, chat_history: List[Dict[str, str]], user_token: str, 
                              user_provided_temperature: Optional[float] = None,
                              user_provided_llm_provider: Optional[str] = None,
                              user_provided_model_name: Optional[str] = None) -> str:
        """
        Orchestrates a chat with an agent, dynamically providing tools based on user's capabilities.
        This method is now fully implemented to use Langchain's AgentExecutor.
        
        Args:
            prompt (str): The current user prompt.
            chat_history (List[Dict[str, str]]): The full chat history.
            user_token (str): The user's authentication token for RBAC checks within tools.
            user_provided_temperature (float, optional): The temperature provided by the user from the frontend.
            user_provided_llm_provider (str, optional): The LLM provider provided by the user from the frontend.
            user_provided_model_name (str, optional): The LLM model name provided by the user from the frontend.
        Returns:
            str: The agent's response.
        """
        logger.info(f"Agent chat initiated for user: {user_token}, prompt: '{prompt[:100]}...', user_provided_temp: {user_provided_temperature}, user_provided_provider: {user_provided_llm_provider}, user_provided_model: {user_provided_model_name}")

        self.llm = self._load_llm(user_token, user_provided_temperature, user_provided_llm_provider, user_provided_model_name)

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

        if get_user_tier_capability(user_token, 'medical_tool_access', False):
            available_tools.extend([get_drug_info, get_symptom_info])
            logger.debug(f"Medical tools added for user {user_token}")

        if get_user_tier_capability(user_token, 'news_tool_access', False):
            available_tools.append(get_general_news)
            logger.debug(f"General news tool added for user {user_token}")
        
        if get_user_tier_capability(user_token, 'legal_tool_access', False):
            available_tools.extend([get_legal_definition, get_case_summary])
            logger.debug(f"Legal tools (definition, case summary) added for user {user_token}")
        
        if get_user_tier_capability(user_token, 'education_tool_access', False):
            available_tools.extend([get_academic_definition, get_historical_event_summary])
            logger.debug(f"Education tools (academic definition, historical event summary) added for user {user_token}")
        
        if get_user_tier_capability(user_token, 'entertainment_tool_access', False):
            available_tools.extend([get_movie_details, get_music_artist_info])
            logger.debug(f"Entertainment tools (movie details, music artist info) added for user {user_token}")

        if get_user_tier_capability(user_token, 'weather_tool_access', False):
            available_tools.extend([get_current_weather, get_weather_forecast])
            logger.debug(f"Weather tools (current weather, forecast) added for user {user_token}")
        
        if get_user_tier_capability(user_token, 'travel_tool_access', False): # NEW: Add travel tools
            available_tools.extend([find_flights, find_hotels])
            logger.debug(f"Travel tools (find flights, find hotels) added for user {user_token}")


        if not available_tools:
            logger.info(f"No specialized tools available for user {user_token}. Falling back to chat completion.")
            return self.chat_completion(chat_history + [{"role": "user", "content": prompt}], 
                                        temperature=user_provided_temperature,
                                        llm_provider=user_provided_llm_provider,
                                        model_name=user_provided_model_name)

        # Convert chat history to Langchain BaseMessage format
        langchain_chat_history = [self._convert_to_langchain_message(msg) for msg in chat_history]

        # Define the prompt template for the agent
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
                "For drug information, use `get_drug_info`. "
                "For symptom information, use `get_symptom_info`. "
                "For general news, use `get_general_news`. "
                "For legal term definitions, use `get_legal_definition`. "
                "For legal case summaries, use `get_case_summary`. "
                "For academic term definitions, use `get_academic_definition`. "
                "For historical event summaries, use `get_historical_event_summary`. "
                "For movie details, use `get_movie_details`. "
                "For music artist information, use `get_music_artist_info`. "
                "For current weather, use `get_current_weather`. "
                "For weather forecasts, use `get_weather_forecast`. "
                "For finding flights, use `find_flights`. " # NEW: Added to prompt
                "For finding hotels, use `find_hotels`. " # NEW: Added to prompt
                "For **data analysis**, complex calculations, time series analysis, regression analysis, "
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
                logger.info(f"MockAgentExecutor initialized with {len(tools)} tools. LLM Provider: {self.llm.model_name.split('-')[0]}, Model: {self.llm.model_name}, Temp: {self.llm.temperature}")

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
                    
                if ("drug info" in prompt_text or "medication" in prompt_text) and is_tool_available("get_drug_info"):
                    drug_name = "Aspirin"
                    mock_tool_output = get_drug_info(drug_name, user_token=user_token_for_tools)
                    return {"output": f"I used get_drug_info. Output:\n{mock_tool_output}"}
                
                if ("symptom info" in prompt_text or "what causes" in prompt_text) and is_tool_available("get_symptom_info"):
                    symptom_name = "Headache"
                    mock_tool_output = get_symptom_info(symptom_name, user_token=user_token_for_tools)
                    return {"output": f"I used get_symptom_info. Output:\n{mock_tool_output}"}

                if ("general news" in prompt_text or "latest news" in prompt_text) and is_tool_available("get_general_news"):
                    news_query = "technology"
                    mock_tool_output = get_general_news(news_query, user_token=user_token_for_tools)
                    return {"output": f"I used get_general_news. Output:\n{mock_tool_output}"}

                if ("legal definition" in prompt_text or "define legal term" in prompt_text) and is_tool_available("get_legal_definition"):
                    term = "contract"
                    mock_tool_output = get_legal_definition(term, user_token=user_token_for_tools)
                    return {"output": f"I used get_legal_definition. Output:\n{mock_tool_output}"}

                if ("case summary" in prompt_text or "summary of case" in prompt_text) and is_tool_available("get_case_summary"):
                    case_name = "smith v. jones"
                    mock_tool_output = get_case_summary(case_name, user_token=user_token_for_tools)
                    return {"output": f"I used get_case_summary. Output:\n{mock_tool_output}"}

                if ("academic definition" in prompt_text or "define academic term" in prompt_text or "what is" in prompt_text) and is_tool_available("get_academic_definition"):
                    term = "photosynthesis"
                    mock_tool_output = get_academic_definition(term, user_token=user_token_for_tools)
                    return {"output": f"I used get_academic_definition. Output:\n{mock_tool_output}"}

                if ("historical event" in prompt_text or "summary of history" in prompt_text) and is_tool_available("get_historical_event_summary"):
                    event_name = "moon landing"
                    mock_tool_output = get_historical_event_summary(event_name, user_token=user_token_for_tools)
                    return {"output": f"I used get_historical_event_summary. Output:\n{mock_tool_output}"}

                if ("movie details" in prompt_text or "info about movie" in prompt_text) and is_tool_available("get_movie_details"):
                    movie_title = "Inception"
                    mock_tool_output = get_movie_details(movie_title, user_token=user_token_for_tools)
                    return {"output": f"I used get_movie_details. Output:\n{mock_tool_output}"}

                if ("music artist" in prompt_text or "info about artist" in prompt_text) and is_tool_available("get_music_artist_info"):
                    artist_name = "Taylor Swift"
                    mock_tool_output = get_music_artist_info(artist_name, user_token=user_token_for_tools)
                    return {"output": f"I used get_music_artist_info. Output:\n{mock_tool_output}"}

                if ("current weather" in prompt_text or "weather in" in prompt_text) and is_tool_available("get_current_weather"):
                    location = "London"
                    mock_tool_output = get_current_weather(location, user_token=user_token_for_tools)
                    return {"output": f"I used get_current_weather. Output:\n{mock_tool_output}"}

                if ("weather forecast" in prompt_text or "forecast for" in prompt_text) and is_tool_available("get_weather_forecast"):
                    location = "New York"
                    days = 3
                    mock_tool_output = get_weather_forecast(location, days, user_token=user_token_for_tools)
                    return {"output": f"I used get_weather_forecast. Output:\n{mock_tool_output}"}

                if ("find flights" in prompt_text or "flight info" in prompt_text) and is_tool_available("find_flights"):
                    origin = "London"
                    destination = "New York"
                    date = "2025-07-15"
                    mock_tool_output = find_flights(origin, destination, date, user_token=user_token_for_tools)
                    return {"output": f"I used find_flights. Output:\n{mock_tool_output}"}

                if ("find hotels" in prompt_text or "hotel availability" in prompt_text) and is_tool_available("find_hotels"):
                    location = "Paris"
                    check_in_date = "2025-09-01"
                    check_out_date = "2025-09-05"
                    mock_tool_output = find_hotels(location, check_in_date, check_out_date, user_token=user_token_for_tools)
                    return {"output": f"I used find_hotels. Output:\n{mock_tool_output}"}

                if ("analyze data" in prompt_text or "run python" in prompt_text or "time series analysis" in prompt_text or "regression analysis" in prompt_text or "machine learning" in prompt_text or "ml model" in prompt_text) and is_tool_available("python_interpreter_with_rbac"):
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
                return {"output": f"Mock LLM agent response (provider={self.llm.model_name.split('-')[0]}, model={self.llm.model_name}, temp={self.llm.temperature}) to: '{prompt}'. I considered the available tools but didn't find a direct match for a tool call based on keywords. If you need a specific tool, please be explicit."}

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
