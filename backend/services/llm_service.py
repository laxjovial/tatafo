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
from shared_tools.query_uploaded_docs_tool import query_uploaded_docs # Corrected: Changed to query_uploaded_docs

# Import domain-specific tools (e.g., CryptoTools)
from domain_tools.crypto_tools import CryptoTools # Corrected: Import CryptoTools class from the package's __init__.py
from domain_tools.document_tools.document_tool import DocumentTools # Import DocumentTools class

# Import individual functions from crypto_tool for direct use if needed,
# but prefer using CryptoTools class for consistency and dependency injection.
# If you are going to use them directly as Langchain Tools, they need to be imported.
# Corrected the import: 'get_historical_crypto_prices' -> 'get_historical_crypto_price'
from domain_tools.crypto_tools.crypto_tool import get_crypto_price, get_historical_crypto_price, get_crypto_id_by_symbol, get_crypto_info, crypto_search_web, crypto_summarize_document_by_path

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, user_manager: UserManager, api_usage_service: ApiUsageService):
        self.user_manager = user_manager
        self.api_usage_service = api_usage_service
        self.config_manager = config_manager # Access the global config_manager instance
        
        # Initialize ChartTools and DocumentTools once
        # Assuming config_manager and analytics_tracker are accessible globally or passed
        self.chart_tools = ChartTools(config_manager=self.config_manager, log_event=analytics_tracker.log_event)
        
        # Initialize DocumentTools with necessary dependencies
        self.document_tools = DocumentTools(config_manager=self.config_manager, log_event=analytics_tracker.log_event)

        # Initialize CryptoTools, passing the document_tools instance
        self.crypto_tools = CryptoTools(
            config_manager=self.config_manager,
            log_event=analytics_tracker.log_event,
            document_tools=self.document_tools # Pass the initialized document_tools
        )

        self.llm = self._initialize_llm(config_manager.get_llm_model())
        self.tools = self._get_llm_tools() # All available tools for the LLM
        self.agent_executor = self._create_agent_executor()

        logger.info("LLMService initialized with model: %s", config_manager.get_llm_model())

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

    def _get_llm_tools(self) -> List[Tool]:
        """
        Gathers and wraps all available tools for the LLM.
        Each tool must be wrapped as a Langchain Tool.
        """
        tools = [
            Tool(
                name="python_interpreter",
                func=python_interpreter_with_rbac,
                description="""
                A powerful Python interpreter. Use this to execute Python code for mathematical operations,
                data processing, or any programmatic task.
                Input should be a dictionary with a 'code' key containing the Python code string.
                Example: `{"code": "print(2 + 2)"}`.
                """,
                handle_tool_error=True # Allow the agent to handle tool errors
            ),
            Tool(
                name="scrape_web",
                func=scrape_web,
                description="""
                Useful for fetching the content of a given URL.
                Input should be a dictionary with a 'url' key containing the URL string.
                Example: `{"url": "https://www.example.com"}`.
                """,
                handle_tool_error=True
            ),
            Tool(
                name="summarize_document",
                func=summarize_document,
                description="""
                Summarizes the content of a document given its file path.
                Input should be a dictionary with a 'file_path_str' key containing the document's path string.
                Example: `{"file_path_str": "/path/to/document.pdf"}`.
                """,
                handle_tool_error=True
            ),
            # Integrate ChartTools methods
            Tool(
                name="create_line_chart",
                func=self.chart_tools.create_line_chart,
                description="""
                Generates a line chart from provided data.
                Input should be a dictionary with 'data' (list of dicts), 'x_col', 'y_col', 'title', 'x_label', 'y_label', 'output_path' (optional).
                Example: `{"data": [{"x": 1, "y": 2}, {"x": 2, "y": 4}], "x_col": "x", "y_col": "y", "title": "My Line Chart"}`.
                """,
                handle_tool_error=True
            ),
            Tool(
                name="create_bar_chart",
                func=self.chart_tools.create_bar_chart,
                description="""
                Generates a bar chart from provided data.
                Input should be a dictionary with 'data' (list of dicts), 'x_col', 'y_col', 'title', 'x_label', 'y_label', 'output_path' (optional).
                Example: `{"data": [{"category": "A", "value": 10}, {"category": "B", "value": 15}], "x_col": "category", "y_col": "value", "title": "My Bar Chart"}`.
                """,
                handle_tool_error=True
            ),
            Tool(
                name="analyze_sentiment",
                func=analyze_sentiment,
                description="""
                Analyzes the sentiment of a given text.
                Input should be a dictionary with a 'text' key containing the string to analyze.
                Example: `{"text": "This is a great movie!"}`.
                """,
                handle_tool_error=True
            ),
            Tool(
                name="query_uploaded_docs",
                func=query_uploaded_docs, # Changed func to query_uploaded_docs
                description="""
                Queries previously uploaded and indexed documents for a user using vector similarity search.
                This tool is useful for retrieving relevant information from a user's personal documents.
                Input should be a dictionary with 'query' (the question to search),
                'user_token' (the user's ID, e.g., 'user123'),
                'section' (e.g., 'general', 'finance', 'crypto'), 'export' (optional, bool), and 'k' (optional, int, number of results).
                Example: `{"query": "What is blockchain?", "user_token": "user123", "section": "crypto", "k": 3}`.
                """,
                handle_tool_error=True
            ),
            Tool(
                name="get_crypto_price",
                func=self.crypto_tools.get_crypto_price,
                description="""
                Retrieves the current price of a cryptocurrency.
                Input should be a dictionary with 'crypto_id' (e.g., "bitcoin"),
                'vs_currencies' (optional, comma-separated string, e.g., "usd,eur").
                Example: `{"crypto_id": "ethereum", "vs_currencies": "usd"}`.
                """,
                handle_tool_error=True
            ),
            Tool(
                name="get_crypto_info",
                func=self.crypto_tools.get_crypto_info,
                description="""
                Retrieves general information about a cryptocurrency.
                Input should be a dictionary with 'crypto_id' (e.g., "bitcoin").
                Example: `{"crypto_id": "solana"}`.
                """,
                handle_tool_error=True
            ),
            Tool(
                name="get_historical_crypto_price", # Corrected tool name to singular
                func=self.crypto_tools.get_historical_crypto_price, # Call the singular method
                description="""
                Retrieves the historical price of a cryptocurrency for a specific date.
                Input should be a dictionary with 'crypto_id' (e.g., "bitcoin"),
                'date' (string in 'DD-MM-YYYY' format, e.g., "01-01-2023"),
                'vs_currency' (optional, string, e.g., "usd").
                Example: `{"crypto_id": "bitcoin", "date": "15-06-2023", "vs_currency": "usd"}`.
                """,
                handle_tool_error=True
            ),
            Tool(
                name="crypto_search_web",
                func=self.crypto_tools.crypto_search_web,
                description="""
                Searches the web for cryptocurrency-related information.
                Input should be a dictionary with 'query' (the search query string).
                Example: `{"query": "latest news on ripple"}`.
                """,
                handle_tool_error=True
            ),
            Tool(
                name="crypto_query_uploaded_docs",
                func=self.crypto_tools.crypto_query_uploaded_docs,
                description="""
                Queries previously uploaded and indexed cryptocurrency documents for a user using vector similarity search.
                This tool is useful for retrieving relevant information from a user's personal crypto documents.
                Input should be a dictionary with 'query' (the question to search),
                'export' (optional, bool), and 'k' (optional, int, number of results).
                Example: `{"query": "What is the whitepaper about for Ethereum?", "k": 2}`.
                """,
                handle_tool_error=True
            ),
            Tool(
                name="crypto_summarize_document_by_path",
                func=self.crypto_tools.crypto_summarize_document_by_path,
                description="""
                Summarizes a document related to cryptocurrency or blockchain located at the given file path.
                Input should be a dictionary with 'file_path_str' (the document's path string).
                Example: `{"file_path_str": "/path/to/ethereum_whitepaper.pdf"}`.
                """,
                handle_tool_error=True
            ),
            Tool(
                name="get_crypto_id_by_symbol",
                func=self.crypto_tools.get_crypto_id_by_symbol,
                description="""
                Looks up the cryptocurrency ID by its symbol (e.g., "BTC", "ETH").
                Useful when only the symbol is known and the tool needs the full crypto ID (e.g., "bitcoin", "ethereum")
                for other cryptocurrency tools.
                Input should be a dictionary with 'symbol' (the cryptocurrency symbol string).
                Example: `{"symbol": "BTC"}`.
                """,
                handle_tool_error=True
            ),
            # Add other tools as needed
        ]
        return tools

    def _create_agent_executor(self) -> AgentExecutor:
        """
        Creates and returns the Langchain AgentExecutor.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant with access to various tools. Use the tools to answer questions and fulfill requests. If a user asks a question that can be answered by one of your tools, you must use the tool. If you do not have a tool that can answer the user's question, respond that you do not have a tool to answer the question. Only use the tools that are provided."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        # Create a ReAct agent
        agent = create_react_agent(self.llm, self.tools, prompt)

        # Create the AgentExecutor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True # Crucial for robustness
        )
        return agent_executor

    async def process_user_query(
        self,
        user_id: str,
        user_query: str,
        user_profile: UserProfile,
        chat_history: List[Dict[str, str]],
        model_name: Optional[str] = None
    ) -> str:
        """
        Processes a user query using the LLM agent,
        handling API usage checks and logging.
        """
        # 1. API Usage Check
        if not self.api_usage_service.check_and_track_usage(user_id, user_profile.tier):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="API usage limit exceeded for your tier."
            )

        # Update LLM model if specified and different from current
        if model_name and model_name != self.config_manager.get_llm_model():
            self.llm = self._initialize_llm(model_name)
            self.agent_executor = self._create_agent_executor() # Re-create agent with new LLM
            self.config_manager.set_llm_model(model_name) # Optionally update config or just use for this session

        # Convert chat history to Langchain format
        langchain_chat_history = [self._convert_to_langchain_message(msg) for msg in chat_history]

        try:
            # 2. Process Query with Agent
            logger.info(f"Processing query for user {user_id} with model {self.config_manager.get_llm_model()}")
            response = await self.agent_executor.ainvoke({
                "input": user_query,
                "chat_history": langchain_chat_history
            })
            result = response.get("output", "No response generated.")
            logger.info(f"Agent response for user {user_id}: {result}")

            # 3. Log API Usage (already done by check_and_track_usage, but can add more granular logging here)
            analytics_tracker.log_event(
                event_type="llm_query",
                user_id=user_id,
                details={
                    "model_name": self.config_manager.get_llm_model(),
                    "query": user_query,
                    "response_length": len(result),
                    "success": True
                }
            )
            return result
        except Exception as e:
            logger.error(f"Error processing user query for {user_id}: {e}", exc_info=True)
            analytics_tracker.log_event(
                event_type="llm_query",
                user_id=user_id,
                details={
                    "model_name": self.config_manager.get_llm_model(),
                    "query": user_query,
                    "success": False,
                    "error": str(e)
                }
            )
            # Re-raise or return a user-friendly error message
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An error occurred while processing your request: {e}"
            )

# Dependency for FastAPI to inject LLMService
async def get_llm_service_dependency(
    user_manager: UserManager = Depends(UserManager),
    api_usage_service: ApiUsageService = Depends(ApiUsageService)
) -> LLMService:
    """
    FastAPI dependency that provides an LLMService instance.
    """
    return LLMService(user_manager=user_manager, api_usage_service=api_usage_service)
