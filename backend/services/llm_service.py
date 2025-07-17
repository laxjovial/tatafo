# backend/services/llm_service.py

import logging
import json
from typing import List, Dict, Any, Optional
from fastapi import HTTPException, status
from datetime import datetime, timedelta, timezone

# Langchain Imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI # Corrected Import
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import Tool

# Import config_manager
from config.config_manager import config_manager

# Import user_manager for RBAC checks within services
from utils.user_manager import UserManager
from backend.models.user_models import UserProfile

# NEW: Import ApiUsageService for API limit checks and usage tracking
from backend.services.api_usage_service import ApiUsageService

# Import all shared tools (these will be wrapped as Langchain Tools)
from shared_tools.python_interpreter_tool import python_interpreter_with_rbac
from shared_tools.scraper_tool import scrape_web
from shared_tools.doc_summarizer import summarize_document
from shared_tools.chart_generation_tool import ChartTools # Import the class
from shared_tools.sentiment_analysis_tool import analyze_sentiment
from shared_tools.query_uploaded_docs_tool import query_uploaded_docs

# Import the export function from its utility module
from utils.export_utils import export_dataframe_to_file

# Import domain-specific tools
from domain_tools.finance_tools.finance_tool import get_stock_price, get_company_news, get_historical_stock_prices, lookup_stock_symbol
from domain_tools.crypto_tools.crypto_tool import get_crypto_price, get_historical_crypto_prices, get_crypto_id_by_symbol
from domain_tools.medical_tools.medical_tool import get_drug_info, get_symptom_info
from domain_tools.news_tools.news_tool import get_general_news
from domain_tools.legal_tools.legal_tool import get_legal_definition, get_case_summary
from domain_tools.education_tools.education_tool import get_academic_definition, get_historical_event_summary
from domain_tools.entertainment_tools.entertainment_tool import get_movie_details, get_music_artist_info
from domain_tools.weather_tools.weather_tool import get_current_weather, get_weather_forecast
from domain_tools.travel_tools.travel_tool import find_flights, find_hotels
from domain_tools.sports_tools.sports_tool import get_player_stats, get_team_stats, get_league_info

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class LLMService:
    """
    Manages interactions with Large Language Models and orchestrates tool usage.
    This service will be called by FastAPI endpoints.
    """
    def __init__(self, user_manager: UserManager, api_usage_service: ApiUsageService):
        """
        Initializes LLMService with necessary dependencies.
        LLM will be loaded dynamically per request.
        """
        self.user_manager = user_manager
        self.api_usage_service = api_usage_service
        self.llm = None # Initialize as None, will be set in chat_with_agent or chat_completion

        # Instantiate ChartTools and store the export function reference
        self.chart_tools_instance = ChartTools()
        self.export_dataframe_to_file_func = export_dataframe_to_file

        logger.info("LLMService initialized with UserManager, ApiUsageService, ChartTools, and ExportUtil.")

    def _load_llm(self, user_profile: UserProfile, 
                  user_provided_temperature: Optional[float] = None,
                  user_provided_llm_provider: Optional[str] = None,
                  user_provided_model_name: Optional[str] = None):
        """
        Loads the appropriate LLM based on configuration, user's RBAC capabilities,
        and user-provided selections for temperature, provider, and model name.
        """
        user_id = user_profile.user_id
        
        # Determine effective temperature based on RBAC
        can_control_temp = self.user_manager.get_user_tier_capability(user_profile.tier, 'llm_temperature_control_enabled', False)
        tier_default_temp = self.user_manager.get_user_tier_capability(user_profile.tier, 'llm_default_temperature', config_manager.get('llm.temperature', 0.7))
        max_allowed_temp = self.user_manager.get_user_tier_capability(user_profile.tier, 'llm_max_temperature', 1.0)

        effective_temperature = tier_default_temp
        if can_control_temp and user_provided_temperature is not None:
            effective_temperature = min(user_provided_temperature, max_allowed_temp)
            logger.info(f"User {user_id} can control temperature. Using provided {user_provided_temperature}, capped at {max_allowed_temp}. Effective: {effective_temperature}")
        else:
            logger.info(f"User {user_id} cannot control temperature or none provided. Using tier default: {effective_temperature}")

        # Determine effective LLM provider and model name based on RBAC
        can_select_model = self.user_manager.get_user_tier_capability(user_profile.tier, 'llm_model_selection_enabled', False)
        
        effective_llm_provider = config_manager.get("llm.provider", "openai")
        effective_model_name = config_manager.get("llm.model_name", "gpt-3.5-turbo")

        if can_select_model:
            if user_provided_llm_provider:
                effective_llm_provider = user_provided_llm_provider
            if user_provided_model_name:
                effective_model_name = user_provided_model_name
            logger.info(f"User {user_id} can select model. Using provided provider '{user_provided_llm_provider}' and model '{user_provided_model_name}'. Effective: {effective_llm_provider}/{effective_model_name}")
        else:
            logger.info(f"User {user_id} cannot select model. Using config defaults: {effective_llm_provider}/{effective_model_name}")

        api_key = None

        if effective_llm_provider == "openai":
            api_key = config_manager.get_secret("openai_api_key")
            if not api_key:
                logger.error("OpenAI API key not found in secrets.")
                raise ValueError("OpenAI API key is required for OpenAI LLM provider.")
            
            return ChatOpenAI(model_name=effective_model_name, temperature=effective_temperature, api_key=api_key)
            
        elif effective_llm_provider == "google":
            api_key = config_manager.get_secret("google_api_key")
            if not api_key:
                logger.error("Google API key not found in secrets.")
                raise ValueError("Google API key is required for Google LLM provider.")
            
            # Use ChatGoogleGenerativeAI as imported
            return ChatGoogleGenerativeAI(model=effective_model_name, temperature=effective_temperature, api_key=api_key)
            
        elif effective_llm_provider == "ollama":
            ollama_base_url = config_manager.get("ollama.base_url", "http://localhost:11434")
            logger.info(f"Connecting to Ollama at: {ollama_base_url}")
            return ChatOllama(model=effective_model_name, temperature=effective_temperature, base_url=ollama_base_url)
            
        else:
            raise ValueError(f"Unsupported LLM provider: {effective_llm_provider}")


    def chat_completion(self, messages: List[Dict[str, str]], user_profile: UserProfile,
                        temperature: Optional[float] = None,
                        llm_provider: Optional[str] = None, model_name: Optional[str] = None) -> str:
        """
        Generates a basic chat completion using the configured LLM (without tools).
        """
        try:
            temp_llm = self._load_llm(user_profile=user_profile,
                                      user_provided_temperature=temperature,
                                      user_provided_llm_provider=llm_provider,
                                      user_provided_model_name=model_name)
            
            langchain_messages = [self._convert_to_langchain_message(msg) for msg in messages]
            response = temp_llm.invoke(langchain_messages)
            
            return response.content
        except Exception as e:
            logger.error(f"Error during LLM chat completion for user {user_profile.user_id}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"LLM chat completion failed: {e}")

    async def chat_with_agent(self, prompt: str, chat_history: List[Dict[str, str]], user_profile: UserProfile, 
                              user_provided_temperature: Optional[float] = None,
                              user_provided_llm_provider: Optional[str] = None,
                              user_provided_model_name: Optional[str] = None) -> str:
        """
        Orchestrates a chat with an agent, dynamically providing tools based on user's capabilities.
        """
        user_id = user_profile.user_id
        logger.info(f"Agent chat initiated for user: {user_id}, prompt: '{prompt[:100]}...', user_provided_temp: {user_provided_temperature}, user_provided_provider: {user_provided_llm_provider}, user_provided_model: {user_provided_model_name}")

        self.llm = self._load_llm(user_profile, user_provided_temperature, user_provided_llm_provider, user_provided_model_name)

        def get_tool_api_id(tool_func) -> str:
            tool_name = tool_func.__name__
            if "stock" in tool_name or "finance" in tool_name:
                return "finance-api-default"
            if "crypto" in tool_name:
                return "crypto-api-default"
            if "medical" in tool_name:
                return "medical-api-default"
            if "news" in tool_name:
                return "news-api-default"
            if "legal" in tool_name:
                return "legal-api-default"
            if "education" in tool_name:
                return "education-api-default"
            if "entertainment" in tool_name:
                return "entertainment-api-default"
            if "weather" in tool_name:
                return "weather-api-default"
            if "travel" in tool_name:
                return "travel-api-default"
            if "sports" in tool_name:
                return "sports-api-default"
            if "python_interpreter" in tool_name:
                return "python-interpreter-api"
            # Specific check for scrape_web as it's a function directly imported
            if tool_func == scrape_web:
                return "web-scraper-api"
            # Specific check for generate_and_save_chart as it's a method on an instance
            if hasattr(tool_func, '__self__') and isinstance(tool_func.__self__, ChartTools) and tool_func.__name__ == 'generate_and_save_chart':
                return "chart-gen-api"
            if tool_func == analyze_sentiment:
                return "sentiment-api"
            if tool_func == query_uploaded_docs:
                return "document-query-api"
            return "general-tool-api"

        async def wrapped_tool_executor(tool_func, *args, **kwargs):
            api_id = get_tool_api_id(tool_func)
            
            can_proceed = await self.api_usage_service.check_api_limit(user_profile, api_id)
            if not can_proceed:
                logger.warning(f"API limit exceeded for user {user_id}, API {api_id}.")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "message": f"API limit exceeded for {api_id} for your tier ({user_profile.tier}). Please upgrade your plan or try again later.",
                        "code": "API_LIMIT_EXCEEDED"
                    }
                )
            
            # Pass user_context to tools that need it for internal RBAC/logging
            if 'user_context' in tool_func.__code__.co_varnames:
                kwargs['user_context'] = user_profile
            
            # Special handling for python_interpreter_with_rbac and generate_and_save_chart
            # to pass their specific dependencies
            if tool_func == python_interpreter_with_rbac:
                kwargs['chart_tools'] = self.chart_tools_instance
                kwargs['export_dataframe_to_file_func'] = self.export_dataframe_to_file_func
            
            logger.debug(f"Executing tool {tool_func.__name__} for user {user_id} (API: {api_id})...")
            tool_output = await tool_func(*args, **kwargs)
            
            await self.api_usage_service.increment_api_usage(user_id, api_id)
            logger.debug(f"Tool {tool_func.__name__} executed successfully. Usage incremented.")
            return tool_output

        available_tools = []

        if self.user_manager.get_user_tier_capability(user_profile.tier, 'web_search_enabled', False):
            available_tools.append(Tool(
                name="scrape_web",
                func=lambda query: wrapped_tool_executor(scrape_web, query, user_context=user_profile), # Pass user_context
                description="A tool to perform web searches and scrape content from URLs. Input should be a search query string."
            ))
            logger.debug(f"Tool 'scrape_web' added for user {user_id}")
        
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'data_analysis_enabled', False):
            available_tools.append(Tool(
                name="python_interpreter_with_rbac",
                func=lambda code: wrapped_tool_executor(
                    python_interpreter_with_rbac,
                    code,
                    user_context=user_profile,
                    chart_tools=self.chart_tools_instance,
                    export_dataframe_to_file_func=self.export_dataframe_to_file_func
                ), # Pass dependencies
                description="A powerful Python interpreter for data analysis, complex calculations, time series analysis, regression analysis, or any machine learning tasks. Input should be valid Python code. This tool also provides access to `chart_tools` for charting and `export_data_to_file` for exporting dataframes."
            ))
            logger.debug(f"Tool 'python_interpreter_with_rbac' added for user {user_id}")
        
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'chart_generation_enabled', False):
            available_tools.append(Tool(
                name="generate_and_save_chart",
                func=lambda data_json, chart_type, x_column=None, y_column=None, color_column=None, title="Generated Chart", x_label=None, y_label=None, library="matplotlib", export_format="png": wrapped_tool_executor(
                    self.chart_tools_instance.generate_and_save_chart, # Call the method on the instance
                    data_json=data_json,
                    chart_type=chart_type,
                    x_column=x_column,
                    y_column=y_column,
                    color_column=color_column,
                    title=title,
                    x_label=x_label,
                    y_label=y_label,
                    user_context=user_profile, # Pass user_context
                    library=library,
                    export_format=export_format
                ),
                description="Generates and saves a chart (e.g., line, bar, scatter, pie, histogram, boxplot) from provided JSON data. Input should be a JSON string of data, chart type, and optional columns for x, y, color, title, and labels. Supported libraries are matplotlib, seaborn, plotly. Supported export formats are png, jpeg, svg, html (for plotly)."
            ))
            logger.debug(f"Tool 'generate_and_save_chart' added for user {user_id}")

        if self.user_manager.get_user_tier_capability(user_profile.tier, 'sentiment_analysis_enabled', False):
            available_tools.append(Tool(
                name="analyze_sentiment",
                func=lambda text: wrapped_tool_executor(analyze_sentiment, text, user_context=user_profile),
                description="Analyzes the sentiment of a given text. Input should be a string of text."
            ))
            logger.debug(f"Tool 'analyze_sentiment' added for user {user_id}")
        
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'document_query_enabled', False):
            available_tools.append(Tool(
                name="query_uploaded_docs",
                func=lambda query_text, section: wrapped_tool_executor(query_uploaded_docs, query_text, section=section, user_context=user_profile),
                description="Queries user-uploaded documents to find relevant information. Input should be a query string and an optional section (e.g., 'general', 'financial')."
            ))
            logger.debug(f"Tool 'query_uploaded_docs' added for user {user_id}")

        # Domain-specific Tools - Finance Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'finance_tool_access', False):
            available_tools.extend([
                Tool(
                    name="get_stock_price",
                    func=lambda symbol: wrapped_tool_executor(get_stock_price, symbol, user_context=user_profile),
                    description="Retrieves the current stock price for a given stock symbol. Input should be a stock symbol (e.g., 'AAPL')."
                ),
                Tool(
                    name="get_company_news",
                    func=lambda symbol, from_date, to_date: wrapped_tool_executor(get_company_news, symbol, from_date, to_date, user_context=user_profile),
                    description="Fetches recent news for a company by its stock symbol within a date range. Input: symbol (str), from_date (YYYY-MM-DD), to_date (YYYY-MM-DD)."
                ),
                Tool(
                    name="lookup_stock_symbol",
                    func=lambda company_name: wrapped_tool_executor(lookup_stock_symbol, company_name, user_context=user_profile),
                    description="Looks up the stock symbol for a given company name. Input: company_name (str)."
                )
            ])
            logger.debug(f"Finance tools (current price, company news, symbol lookup) added for user {user_id}")
        
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'historical_data_access', False):
            available_tools.append(Tool(
                name="get_historical_stock_prices",
                func=lambda symbol, start_date, end_date: wrapped_tool_executor(get_historical_stock_prices, symbol, start_date, end_date, user_context=user_profile),
                description="Retrieves historical stock prices for a given symbol and date range. Input: symbol (str), start_date (YYYY-MM-DD), end_date (YYYY-MM-DD)."
            ))
            logger.debug(f"Tool 'get_historical_stock_prices' added for user {user_id}")

        # Crypto Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'crypto_tool_access', False):
            available_tools.extend([
                Tool(name="get_crypto_price", func=lambda coin_id: wrapped_tool_executor(get_crypto_price, coin_id, user_context=user_profile), description="Retrieves the current price of a cryptocurrency by its ID."),
                Tool(name="get_historical_crypto_prices", func=lambda coin_id, vs_currency, days: wrapped_tool_executor(get_historical_crypto_prices, coin_id, vs_currency, days, user_context=user_profile), description="Retrieves historical prices for a cryptocurrency."),
                Tool(name="get_crypto_id_by_symbol", func=lambda symbol: wrapped_tool_executor(get_crypto_id_by_symbol, symbol, user_context=user_profile), description="Looks up the cryptocurrency ID by its symbol.")
            ])
            logger.debug(f"Crypto tools added for user {user_id}")

        # Medical Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'medical_tool_access', False):
            available_tools.extend([
                Tool(name="get_drug_info", func=lambda drug_name: wrapped_tool_executor(get_drug_info, drug_name, user_context=user_profile), description="Retrieves information about a specific drug."),
                Tool(name="get_symptom_info", func=lambda symptom_name: wrapped_tool_executor(get_symptom_info, symptom_name, user_context=user_profile), description="Retrieves information about a specific symptom.")
            ])
            logger.debug(f"Medical tools added for user {user_id}")

        # News Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'news_tool_access', False):
            available_tools.append(Tool(name="get_general_news", func=lambda query: wrapped_tool_executor(get_general_news, query, user_context=user_profile), description="Fetches general news articles based on a query."))
            logger.debug(f"General news tool added for user {user_id}")
        
        # Legal Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'legal_tool_access', False):
            available_tools.extend([
                Tool(name="get_legal_definition", func=lambda term: wrapped_tool_executor(get_legal_definition, term, user_context=user_profile), description="Retrieves the definition of a legal term."),
                Tool(name="get_case_summary", func=lambda case_name: wrapped_tool_executor(get_case_summary, case_name, user_context=user_profile), description="Retrieves a summary of a legal case.")
            ])
            logger.debug(f"Legal tools (definition, case summary) added for user {user_id}")
        
        # Education Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'education_tool_access', False):
            available_tools.extend([
                Tool(name="get_academic_definition", func=lambda term: wrapped_tool_executor(get_academic_definition, term, user_context=user_profile), description="Retrieves the definition of an academic term."),
                Tool(name="get_historical_event_summary", func=lambda event_name: wrapped_tool_executor(get_historical_event_summary, event_name, user_context=user_profile), description="Retrieves a summary of a historical event.")
            ])
            logger.debug(f"Education tools (academic definition, historical event summary) added for user {user_id}")
        
        # Entertainment Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'entertainment_tool_access', False):
            available_tools.extend([
                Tool(name="get_movie_details", func=lambda movie_title: wrapped_tool_executor(get_movie_details, movie_title, user_context=user_profile), description="Retrieves details about a movie."),
                Tool(name="get_music_artist_info", func=lambda artist_name: wrapped_tool_executor(get_music_artist_info, artist_name, user_context=user_profile), description="Retrieves information about a music artist.")
            ])
            logger.debug(f"Entertainment tools (movie details, music artist info) added for user {user_id}")

        # Weather Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'weather_tool_access', False):
            available_tools.extend([
                Tool(name="get_current_weather", func=lambda location: wrapped_tool_executor(get_current_weather, location, user_context=user_profile), description="Retrieves current weather conditions for a location."),
                Tool(name="get_weather_forecast", func=lambda location, days: wrapped_tool_executor(get_weather_forecast, location, days, user_context=user_profile), description="Retrieves weather forecast for a location for a number of days.")
            ])
            logger.debug(f"Weather tools (current weather, forecast) added for user {user_id}")
        
        # Travel Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'travel_tool_access', False):
            available_tools.extend([
                Tool(name="find_flights", func=lambda origin, destination, date: wrapped_tool_executor(find_flights, origin, destination, date, user_context=user_profile), description="Finds flights between an origin and destination on a specific date. Input: origin (str), destination (str), date (YYYY-MM-DD)."),
                Tool(name="find_hotels", func=lambda location, check_in_date, check_out_date: wrapped_tool_executor(find_hotels, location, check_in_date, check_out_date, user_context=user_profile), description="Finds available hotels in a location for specific check-in/out dates. Input: location (str), check_in_date (YYYY-MM-DD), check_out_date (YYYY-MM-DD).")
            ])
            logger.debug(f"Travel tools (find flights, find hotels) added for user {user_id}")
        
        # Sports Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'sports_tool_access', False):
            available_tools.extend([
                Tool(name="get_player_stats", func=lambda player_name, sport: wrapped_tool_executor(get_player_stats, player_name, sport=sport, user_context=user_profile), description="Retrieves statistics for a sports player in a given sport."),
                Tool(name="get_team_stats", func=lambda team_name, sport: wrapped_tool_executor(get_team_stats, team_name, sport=sport, user_context=user_profile), description="Retrieves statistics for a sports team in a given sport."),
                Tool(name="get_league_info", func=lambda league_name, sport: wrapped_tool_executor(get_league_info, league_name, sport=sport, user_context=user_profile), description="Retrieves information about a sports league in a given sport.")
            ])
            logger.debug(f"Sports tools added for user {user_id}")


        if not available_tools:
            logger.info(f"No specialized tools available for user {user_id}. Falling back to chat completion.")
            return await self.chat_completion(chat_history + [{"role": "user", "content": prompt}],
                                        user_profile=user_profile,
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
                "For finding flights, use `find_flights`. "
                "For finding hotels, use `find_hotels`. "
                "For player statistics (e.g., career stats, trophies, rings), use `get_player_stats`. "
                "For team or club statistics (e.g., season stats, major trophies, standings), use `get_team_stats`. "
                "For sports league information (e.g., champions, top scorers), use `get_league_info`. "
                "For **data analysis**, complex calculations, time series analysis, regression analysis, "
                "or any other machine learning tasks (supervised or unsupervised), use the `python_interpreter_with_rbac` tool. "
                "For generating charts from data, use `generate_and_save_chart`. "
                "Always provide comprehensive answers based on tool outputs. "
                "If a tool call fails, inform the user and try to explain why or suggest alternatives."
                "When providing historical data, if asked to plot, use `generate_and_save_chart` with the JSON output from `get_historical_stock_prices` or `get_historical_crypto_prices`."
                "When analyzing data from uploaded documents, use `query_uploaded_docs` first, then pass the relevant content to `python_interpreter_with_rbac` for analysis."
                "Remember to pass the `user_context` (which is the UserProfile object) to any tool that requires it for RBAC or logging."
                "If a user asks for a stock by name (e.g., 'Apple'), first use `lookup_stock_symbol` to get the ticker, then use the appropriate stock tool."
                "If a user asks for crypto by symbol (e.g., 'btc'), first use `get_crypto_id_by_symbol` to get the ID, then use the appropriate crypto tool."
                "If you need to export data, you can do so by calling the `export_data_to_file` function *from within* the `python_interpreter_with_rbac` tool."
            ),
            *langchain_chat_history,
            HumanMessage(content="{input}"),
            AIMessage(content="{agent_scratchpad}"),
        ])

        # Create the Langchain agent
        agent = create_react_agent(self.llm, available_tools, prompt_template)
        # AgentExecutor will handle parsing errors and verbose logging
        agent_executor = AgentExecutor(agent=agent, tools=available_tools, verbose=True, handle_parsing_errors=True)
        
        logger.info("Using real Langchain AgentExecutor.")

        try:
            # Invoke the agent with the current prompt and chat history
            response = await agent_executor.invoke({
                "input": prompt,
                "chat_history": langchain_chat_history,
                "user_profile": user_profile
            })
            return response["output"]
        except HTTPException as e:
            raise
        except Exception as e:
            logger.error(f"Error during Langchain agent invocation for user {user_id}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Agent execution failed: {str(e)}")

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
