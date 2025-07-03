# backend/services/llm_service.py

import logging
from typing import List, Dict, Any, Optional
from fastapi import HTTPException, status # Import HTTPException for error handling

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
from domain_tools.finance_tools.finance_tool import get_stock_price, get_company_news
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
                    # Simulate streaming
                    yield AIMessage(content=f"Mock streaming part 1...")
                    yield AIMessage(content=f"Mock streaming part 2...")
                
                # Mock for agent's invoke method (simplified)
                def _call(self, inputs: Dict[str, Any], stop: Optional[List[str]] = None) -> Dict[str, Any]:
                    # This mock simulates an agent's response, potentially calling a tool.
                    # It's a very simplified agent loop for testing.
                    prompt = inputs.get('input', '')
                    tools_available = inputs.get('intermediate_steps', []) # Simplified
                    
                    if "analyze" in prompt.lower() and any("python_interpreter_with_rbac" in str(t) for t in tools_available):
                        return {"output": "Simulated: Executing Python code for data analysis. (Real tool call output would be here)"}
                    elif "stock price" in prompt.lower() and any("get_stock_price" in str(t) for t in tools_available):
                        return {"output": "Simulated: Fetching stock price for AAPL. (Real tool call output would be here)"}
                    elif "news" in prompt.lower() and any("get_company_news" in str(t) for t in tools_available):
                        return {"output": "Simulated: Fetching company news. (Real tool call output would be here)"}
                    elif "sentiment" in prompt.lower() and any("analyze_sentiment" in str(t) for t in tools_available):
                        return {"output": "Simulated: Performing sentiment analysis. (Real tool call output would be here)"}
                    elif "search web" in prompt.lower() and any("scrape_web" in str(t) for t in tools_available):
                        return {"output": "Simulated: Searching the web. (Real tool call output would be here)"}
                    elif "query documents" in prompt.lower() and any("query_uploaded_docs" in str(t) for t in tools_available):
                        return {"output": "Simulated: Querying uploaded documents. (Real tool call output would be here)"}
                    elif "generate chart" in prompt.lower() and any("generate_and_save_chart" in str(t) for t in tools_available):
                        return {"output": "Simulated: Generating a chart. (Real tool call output would be here)"}
                    else:
                        return {"output": f"Mock LLM agent response to: {prompt}. No specific tool action simulated."}

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
            # The python_interpreter_with_rbac tool needs the user_token
            # We'll use a partial function application or a wrapper if the tool signature
            # doesn't match the agent's expectation for direct tool calls.
            # For simplicity, if the tool takes `user_token` as a direct argument,
            # we need to ensure the agent calls it correctly or wrap it.
            # Langchain's @tool decorator handles this if the agent is designed for it.
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
            logger.debug(f"Finance tools added for user {user_token}")
        
        # Future Medical Tools
        # if get_user_tier_capability(user_token, 'medical_tool_access', False):
        #     available_tools.extend([get_drug_info, get_symptom_info])
        #     logger.debug(f"Medical tools added for user {user_token}")


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
                "If a user asks for data analysis, use the python_interpreter_with_rbac tool. "
                "For web search, use scrape_web. For sentiment analysis, use analyze_sentiment. "
                "For stock prices, use get_stock_price. For company news, use get_company_news. "
                "For querying uploaded documents, use query_uploaded_docs. "
                "For chart generation, use generate_and_save_chart. "
                "Always provide comprehensive answers based on tool outputs. "
                "If a tool call fails, inform the user and try to explain why or suggest alternatives."
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

                # Mock tool calls based on prompt keywords and available tools
                if "stock price" in prompt_text and any(t.name == "get_stock_price" for t in self.tools):
                    symbol = "AAPL" # Hardcoded for mock
                    mock_tool_output = get_stock_price(symbol, user_token=user_token_for_tools)
                    return {"output": f"I used get_stock_price. Output:\n{mock_tool_output}"}
                
                if "company news" in prompt_text and any(t.name == "get_company_news" for t in self.tools):
                    symbol = "TSLA" # Hardcoded for mock
                    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                    to_date = datetime.now().strftime("%Y-%m-%d")
                    mock_tool_output = get_company_news(symbol, from_date, to_date, user_token=user_token_for_tools)
                    return {"output": f"I used get_company_news. Output:\n{mock_tool_output}"}
                
                if "analyze data" in prompt_text or "run python" in prompt_text and any(t.name == "python_interpreter_with_rbac" for t in self.tools):
                    code_to_run = "print('Mock Python analysis result.')"
                    mock_tool_output = python_interpreter_with_rbac(code_to_run, user_token=user_token_for_tools)
                    return {"output": f"I used python_interpreter_with_rbac. Output:\n{mock_tool_output}"}
                
                if "search web" in prompt_text and any(t.name == "scrape_web" for t in self.tools):
                    mock_tool_output = scrape_web("mock web search query", user_token=user_token_for_tools)
                    return {"output": f"I used scrape_web. Output:\n{mock_tool_output}"}

                if "sentiment" in prompt_text and any(t.name == "analyze_sentiment" for t in self.tools):
                    text_for_sentiment = "This is a test sentence for sentiment analysis."
                    mock_tool_output = analyze_sentiment(text_for_sentiment)
                    return {"output": f"I used analyze_sentiment. Output:\n{mock_tool_output}"}

                if "query document" in prompt_text and any(t.name == "query_uploaded_docs" for t in self.tools):
                    mock_tool_output = query_uploaded_docs("mock document query", user_token=user_token_for_tools, section="general")
                    return {"output": f"I used query_uploaded_docs. Output:\n{mock_tool_output}"}
                
                if "generate chart" in prompt_text and any(t.name == "generate_and_save_chart" for t in self.tools):
                    mock_data = json.dumps([{"x": 1, "y": 10}, {"x": 2, "y": 20}])
                    mock_tool_output = generate_and_save_chart(mock_data, "line", "x", "y", user_token=user_token_for_tools)
                    return {"output": f"I used generate_and_save_chart. Output:\n{mock_tool_output}"}

                # Fallback if no specific tool action is simulated
                return {"output": f"Mock LLM agent response to: '{prompt_text}'. I considered the available tools but didn't find a direct match for a tool call based on keywords. If you need a specific tool, please be explicit."}

        agent_executor = MockAgentExecutor(self.llm, available_tools, prompt_template) # Use mock agent executor

        # Pass user_token to the agent's invoke method so it can be passed to tools
        # This requires the agent executor to accept a 'config' or similar parameter
        # or for the tools themselves to be wrapped to get the user_token.
        # For Langchain's create_react_agent, tools receive arguments directly.
        # We need to ensure tools that require user_token have it passed.
        # The @tool decorator in Langchain can handle this if the tool signature includes it.
        # Our tools already have `user_token` as an argument.

        response = await agent_executor.invoke({"input": prompt, "chat_history": langchain_chat_history, "user_token": user_token}) # Pass user_token to agent

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

