# backend/services/llm_service.py

import logging
import json
from typing import List, Dict, Any, Optional
from fastapi import HTTPException, status, Depends
from datetime import datetime, timedelta, timezone

# Langchain Imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import Tool

# Import config_manager
from config.config_manager import config_manager

# Import user_manager for RBAC checks within services
from utils.user_manager import UserManager
from backend.models.user_models import UserProfile
from utils import analytics_tracker # Import analytics_tracker

# NEW: Import ApiUsageService for API limit checks and usage tracking
from backend.services.api_usage_service import ApiUsageService

# Import all shared tools (these will be wrapped as Langchain Tools)
from shared_tools.python_interpreter_tool import python_interpreter_with_rbac
from shared_tools.scrapper_tool import scrape_web
from shared_tools.doc_summarizer import summarize_document
from shared_tools.chart_generation_tool import ChartTools # Import the class
from shared_tools.sentiment_analysis_tool import analyze_sentiment
from shared_tools.query_uploaded_docs_tool import query_uploaded_docs

# Import the export function from shared_tools.export_tool if needed
try:
    from shared_tools.export_tool import export_data
except ImportError:
    logging.warning("export_data tool not available. Please ensure shared_tools.export_tool is properly configured.")
    export_data = None

# Import domain-specific tools
from domain_tools.finance_tools.finance_tool import FinanceTools # Only import the class
from domain_tools.crypto_tools import CryptoTools # Corrected: Import CryptoTools class from the package's __init__.py

# Import standalone crypto tool functions directly from the module as they are used directly in available_tools
from domain_tools.crypto_tools.crypto_tool import get_crypto_price, get_historical_crypto_prices, get_crypto_id_by_symbol

# Import the new DocumentTools class
from domain_tools.document_tools.document_tool import DocumentTools

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, user_manager: UserManager, api_usage_service: ApiUsageService):
        self.user_manager = user_manager
        self.api_usage_service = api_usage_service
        self.firestore_manager = config_manager.firestore_manager # Assuming firestore_manager is available via config_manager
        self.log_event = analytics_tracker.log_event_wrapper # Use the wrapped log_event for analytics
        self.document_tools = DocumentTools(config_manager, self.firestore_manager, self.log_event) # Initialize DocumentTools

        # Initialize domain-specific tool classes
        self.finance_tools = FinanceTools(
            config_manager=config_manager,
            firestore_manager=self.firestore_manager,
            log_event=self.log_event,
            document_tools=self.document_tools # Pass DocumentTools instance
        )
        self.crypto_tools = CryptoTools(
            config_manager=config_manager,
            log_event=self.log_event,
            document_tools=self.document_tools # Pass DocumentTools instance
        )
        
        logger.info("LLMService initialized with tool managers.")

    async def run_tool_by_name(self, tool_name: str, tool_args: Dict[str, Any], user_profile: UserProfile) -> Any:
        """
        Dynamically runs a tool by its name. This is used for direct tool invocation
        via the API endpoint, bypassing the LLM agent.
        """
        # Create a dictionary of all callable tools for direct lookup
        callable_tools = {
            "python_interpreter": python_interpreter_with_rbac,
            "scrape_web": scrape_web,
            "summarize_document": summarize_document,
            "analyze_sentiment": analyze_sentiment,
            "query_uploaded_docs": self.document_tools.query_uploaded_docs, # Use method from DocumentTools instance
            # Finance Tools
            "finance_get_stock_price": self.finance_tools.finance_get_stock_price,
            "finance_get_historical_stock_prices": self.finance_tools.finance_get_historical_stock_prices,
            "finance_get_company_overview": self.finance_tools.finance_get_company_overview,
            "finance_get_forex_exchange_rate": self.finance_tools.finance_get_forex_exchange_rate,
            "finance_query_uploaded_docs": self.finance_tools.finance_query_uploaded_docs,
            "finance_summarize_document_by_path": self.finance_tools.finance_summarize_document_by_path,
            "finance_search_web": self.finance_tools.finance_search_web,
            # Crypto Tools (using the standalone functions that are imported directly)
            "get_crypto_price": get_crypto_price,
            "get_historical_crypto_prices": get_historical_crypto_prices,
            "get_crypto_id_by_symbol": get_crypto_id_by_symbol,
            "crypto_search_web": self.crypto_tools.crypto_search_web,
            "crypto_query_uploaded_docs": self.crypto_tools.crypto_query_uploaded_docs,
            "crypto_summarize_document_by_path": self.crypto_tools.crypto_summarize_document_by_path,
        }

        # Add export_data if it was successfully imported
        if export_data:
            callable_tools["export_data"] = export_data

        tool_func = callable_tools.get(tool_name)

        if not tool_func:
            raise ValueError(f"Tool '{tool_name}' not found.")

        # Ensure user_context is passed for tools that require it
        # This check is more robust and less prone to issues than inspecting __code__.co_varnames directly
        if 'user_context' in tool_args or hasattr(tool_func, '__wrapped__') and 'user_context' in tool_func.__wrapped__.__code__.co_varnames:
            tool_args['user_context'] = user_profile


        # Log direct tool usage
        self.log_event(
            user_id=user_profile.user_id,
            event_type="direct_tool_invocation",
            details={"tool_name": tool_name, "tool_args": tool_args},
            success=True
        )

        try:
            result = tool_func(**tool_args)
            # If the tool is async, await it
            if hasattr(result, '__await__'):
                result = await result
            return result
        except Exception as e:
            logger.error(f"Error running tool '{tool_name}' directly for user {user_profile.user_id}: {e}", exc_info=True)
            self.log_event(
                user_id=user_profile.user_id,
                event_type="direct_tool_invocation",
                details={"tool_name": tool_name, "tool_args": tool_args, "error": str(e)},
                success=False
            )
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error executing tool '{tool_name}': {str(e)}")


    async def process_chat_message(self, prompt: str, chat_history: List[Dict[str, str]], user_profile: UserProfile) -> str:
        user_id = user_profile.user_id
        
        # Check API limits before processing the message
        if not self.api_usage_service.check_api_limit(user_id, user_profile.tier):
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="API limit exceeded for your tier.")

        # Convert chat history to Langchain format
        langchain_chat_history = [self._convert_to_langchain_message(msg) for msg in chat_history]

        # Define the prompt template
        # Adjusted system message to reflect current date and location
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", f"You are a helpful AI assistant. The current date is {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}. The user is in Ikorodu, Lagos, Nigeria."),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # Initialize the LLM based on user's tier
        model_name = config_manager.get_model_name_for_tier(user_profile.tier)
        llm = self._initialize_llm(model_name)
        
        # Dynamically create Langchain Tools based on user's access rights
        available_tools = []

        # Python Interpreter Tool
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'python_interpreter_access', False):
            available_tools.append(Tool(name="python_interpreter", func=lambda code: self.wrapped_tool_executor(python_interpreter_with_rbac, code, user_context=user_profile), description="Executes Python code safely within a sandboxed environment. Use this for mathematical calculations, data processing, or any task requiring code execution. Input should be the Python code string."))
            logger.debug(f"Python interpreter added for user {user_id}")

        # Web Scraper Tool
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'web_scrape_access', False):
            available_tools.append(Tool(name="scrape_web", func=lambda url: self.wrapped_tool_executor(scrape_web, url, user_context=user_profile), description="Accesses and extracts content from a given URL. Use this tool when you need to get information from a specific webpage. Input should be the URL string."))
            logger.debug(f"Web scraper added for user {user_id}")
        
        # Document Summarizer Tool
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'document_summarizer_access', False):
            available_tools.append(Tool(name="summarize_document", func=lambda file_path: self.wrapped_tool_executor(summarize_document, file_path, user_context=user_profile), description="Summarizes the content of a document located at a given file path."))
            logger.debug(f"Document summarizer added for user {user_id}")

        # Chart Generation Tools
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'chart_generation_access', False):
            available_tools.extend([
                Tool(name="create_chart", func=lambda **kwargs: self.wrapped_tool_executor(self.finance_tools.create_chart, user_context=user_profile, **kwargs), description="Generates a chart (e.g., line, bar, pie) from provided data. Use this when the user asks for a visualization of data. Input schema for this tool is complex and includes 'chart_type', 'data' (list of dicts), 'x_axis', 'y_axis', 'title', 'x_label', 'y_label', 'color', 'tooltip', 'interactive'."),
                Tool(name="save_chart", func=lambda file_path: self.wrapped_tool_executor(self.finance_tools.save_chart, file_path, user_context=user_profile), description="Saves the last generated chart to a specified file path. Use this when the user explicitly asks to save a chart. Input should be the 'file_path' string (e.g., 'chart.json').")
            ])
            logger.debug(f"Chart tools added for user {user_id}")

        # Sentiment Analysis Tool
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'sentiment_analysis_access', False):
            available_tools.append(Tool(name="analyze_sentiment", func=lambda text: self.wrapped_tool_executor(analyze_sentiment, text, user_context=user_profile), description="Analyzes the sentiment of a given text (e.g., 'positive', 'negative', 'neutral'). Input should be the text string."))
            logger.debug(f"Sentiment analysis tool added for user {user_id}")

        # Query Uploaded Documents Tool (General)
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'query_uploaded_docs_access', False):
            # This tool is implemented via DocumentTools instance
            available_tools.append(Tool(name="query_uploaded_docs", func=lambda query, export=False, k=5: self.wrapped_tool_executor(self.document_tools.query_uploaded_docs, query_text=query, user_context=user_profile, export=export, k=k), description="Queries previously uploaded and indexed documents for a user using vector similarity search. Returns relevant text snippets. Use this when the user asks questions about their uploaded documents. Input: 'query' (str), 'export' (bool, optional, default False), 'k' (int, optional, default 5 for number of results)."))
            logger.debug(f"Query uploaded docs tool added for user {user_id}")

        # Export Data Tool
        if export_data and self.user_manager.get_user_tier_capability(user_profile.tier, 'export_data_access', False):
            available_tools.append(Tool(name="export_data", func=lambda data, file_format, file_name: self.wrapped_tool_executor(export_data, data, file_format, file_name, user_context=user_profile), description="Exports given data to a specified file format (e.g., 'csv', 'json', 'xlsx') and saves it to a file. Use this when the user asks to export data. Input: 'data' (list of dicts), 'file_format' (str), 'file_name' (str)."))
            logger.debug(f"Export data tool added for user {user_id}")

        # Finance Tools (Methods from FinanceTools class)
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'finance_tool_access', False):
            available_tools.extend([
                Tool(name="finance_get_stock_price", func=lambda symbol: self.wrapped_tool_executor(self.finance_tools.finance_get_stock_price, symbol, user_context=user_profile), description="Retrieves the current stock price for a given stock symbol."),
                Tool(name="finance_get_historical_stock_prices", func=lambda symbol, range: self.wrapped_tool_executor(self.finance_tools.finance_get_historical_stock_prices, symbol, range, user_context=user_profile), description="Retrieves historical stock prices for a given stock symbol and date range. The range can be '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'."),
                Tool(name="finance_get_company_overview", func=lambda symbol: self.wrapped_tool_executor(self.finance_tools.finance_get_company_overview, symbol, user_context=user_profile), description="Retrieves a company overview for a given stock symbol."),
                Tool(name="finance_get_forex_exchange_rate", func=lambda from_currency, to_currency: self.wrapped_tool_executor(self.finance_tools.finance_get_forex_exchange_rate, from_currency, to_currency, user_context=user_profile), description="Retrieves the current exchange rate between two currencies."),
                Tool(name="finance_query_uploaded_docs", func=lambda query, export=False, k=5: self.wrapped_tool_executor(self.finance_tools.finance_query_uploaded_docs, query, user_context=user_profile, export=export, k=k), description="Queries previously uploaded and indexed finance-related documents for a user using vector similarity search. Returns relevant text snippets."),
                Tool(name="finance_summarize_document_by_path", func=lambda file_path: self.wrapped_tool_executor(self.finance_tools.finance_summarize_document_by_path, file_path, user_context=user_profile), description="Summarizes a finance-related document located at the given file path."),
                Tool(name="finance_search_web", func=lambda query: self.wrapped_tool_executor(self.finance_tools.finance_search_web, query, user_context=user_profile), description="Searches the web for finance-related information."),
            ])
            logger.debug(f"Finance tools added for user {user_id}")

        # Crypto Tools (using the standalone functions that are imported directly)
        if self.user_manager.get_user_tier_capability(user_profile.tier, 'crypto_tool_access', False):
            available_tools.extend([
                Tool(name="get_crypto_price", func=lambda coin_id: self.wrapped_tool_executor(get_crypto_price, coin_id, user_context=user_profile), description="Retrieves the current price of a cryptocurrency by its ID."),
                Tool(name="get_historical_crypto_prices", func=lambda coin_id, vs_currency, days: self.wrapped_tool_executor(get_historical_crypto_prices, coin_id, vs_currency, days, user_context=user_profile), description="Retrieves historical prices for a cryptocurrency."),
                Tool(name="get_crypto_id_by_symbol", func=lambda symbol: self.wrapped_tool_executor(get_crypto_id_by_symbol, symbol, user_context=user_profile), description="Looks up the cryptocurrency ID by its symbol."),
                Tool(name="crypto_search_web", func=lambda query: self.wrapped_tool_executor(self.crypto_tools.crypto_search_web, query, user_context=user_profile), description="Searches the web for cryptocurrency-related information."),
                Tool(name="crypto_query_uploaded_docs", func=lambda query, export=False, k=5: self.wrapped_tool_executor(self.crypto_tools.crypto_query_uploaded_docs, query, user_context=user_profile, export=export, k=k), description="Queries previously uploaded and indexed cryptocurrency documents for a user using vector similarity search. Returns relevant text snippets."),
                Tool(name="crypto_summarize_document_by_path", func=lambda file_path: self.wrapped_tool_executor(self.crypto_tools.crypto_summarize_document_by_path, file_path, user_context=user_profile), description="Summarizes a cryptocurrency-related document located at the given file path."),
            ])
            logger.debug(f"Crypto tools added for user {user_id}")

        # Corrected: Define wrapped_tool_executor as an async function
        async def wrapped_tool_executor(tool_func, *args, user_context: UserProfile, **kwargs):
            """
            A wrapper to execute tools, handling API usage tracking and potential errors.
            """
            tool_name = tool_func.__name__ # Get the original function's name
            
            # Log tool usage attempt
            self.log_event(
                user_id=user_context.user_id, # Changed from user_context to user_id directly
                event_type="tool_usage_attempt",
                details={"tool_name": tool_name, "args": args, "kwargs": kwargs},
                success=None # Indicates attempt, not final success/failure yet
            )

            try:
                # Deduct API usage before tool execution
                # For specific tools, you might want more granular checks
                self.api_usage_service.record_api_usage(user_context.user_id, user_context.tier, tool_name)
                
                result = tool_func(*args, user_context=user_context, **kwargs)
                # If the tool is async, await it
                if hasattr(result, '__await__'):
                    result = await result
                
                # Log successful tool usage
                self.log_event(
                    user_id=user_context.user_id,
                    event_type="tool_usage",
                    details={"tool_name": tool_name, "args": args, "kwargs": kwargs, "result_summary": str(result)[:200]},
                    success=True
                )
                return result
            except HTTPException as e:
                logger.error(f"HTTPException during tool execution for {tool_name}: {e.detail}")
                self.log_event(
                    user_id=user_context.user_id,
                    event_type="tool_usage",
                    details={"tool_name": tool_name, "args": args, "kwargs": kwargs, "error": e.detail},
                    success=False
                )
                raise # Re-raise HTTPExceptions directly to be caught by FastAPI
            except Exception as e:
                logger.error(f"Error during tool execution for {tool_name}: {e}", exc_info=True)
                # Log failed tool usage
                self.log_event(
                    user_id=user_context.user_id,
                    event_type="tool_usage",
                    details={"tool_name": tool_name, "args": args, "kwargs": kwargs, "error": str(e)},
                    success=False
                )
                # Depending on desired agent behavior, you might return a user-friendly error message
                # or re-raise the exception to terminate the agent's thought process.
                return f"An error occurred while using the {tool_name} tool: {str(e)}"

        # Create the Langchain agent
        agent = create_react_agent(llm, available_tools, prompt_template)
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

    def _initialize_llm(self, model_name: str):
        """Initializes the appropriate LLM based on the model name."""
        if model_name.startswith("gpt"):
            return ChatOpenAI(model=model_name, temperature=0.7)
        elif model_name.startswith("gemini"):
            return ChatGoogleGenerativeAI(model=model_name, temperature=0.7)
        elif model_name.startswith("ollama"):
            return ChatOllama(model=model_name, temperature=0.7)
        else:
            raise ValueError(f"Unsupported LLM model name: {model_name}")

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

# Dependency for FastAPI to inject LLMService
async def get_llm_service_dependency(
    user_manager: UserManager = Depends(UserManager),
    api_usage_service: ApiUsageService = Depends(ApiUsageService)
) -> LLMService:
    """
    FastAPI dependency that provides an LLMService instance.
    """
    return LLMService(user_manager=user_manager, api_usage_service=api_usage_service)
