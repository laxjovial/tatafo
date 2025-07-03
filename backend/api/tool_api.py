# backend/api/tool_api.py

import logging
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Import the LLMService
from backend.services.llm_service import llm_service

# Import authentication middleware (for protected endpoints)
from backend.middleware.auth_middleware import get_current_active_user

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic model for incoming chat requests
class ChatRequest(BaseModel):
    prompt: str = Field(..., description="The user's current prompt or message.")
    chat_history: List[Dict[str, str]] = Field(default_factory=list, description="The full chat history, as a list of {'role': str, 'content': str} dictionaries.")
    user_token: str = Field(..., description="The user's authentication token (e.g., Firebase ID token or mock token).")

# Pydantic model for outgoing chat responses
class ChatResponse(BaseModel):
    response: str = Field(..., description="The AI's generated response.")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Optional: Details of any tool calls made by the AI.")
    # Add more fields as needed, e.g., token usage, sentiment analysis result, chart path

@router.post("/chat/agent", response_model=ChatResponse)
async def chat_with_ai_agent(request: ChatRequest):
    """
    Endpoint for interacting with the AI agent.
    The agent can use various tools based on the user's prompt and capabilities.
    """
    logger.info(f"Received chat request from user: {request.user_token}, prompt: '{request.prompt[:100]}...'")
    
    try:
        # Call the LLMService's chat_with_agent method
        # The llm_service will handle tool selection, execution, and RBAC checks
        agent_response_content = await llm_service.chat_with_agent(
            prompt=request.prompt,
            chat_history=request.chat_history,
            user_token=request.user_token
        )
        
        # In a more advanced setup, the agent_response_content might be a structured object
        # that includes tool call details, which you would parse here.
        
        return ChatResponse(response=agent_response_content)
    except ValueError as ve:
        logger.error(f"Validation error in chat_with_ai_agent: {ve}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except HTTPException as he:
        # Re-raise HTTPExceptions from underlying services (e.g., RBAC denial)
        raise he
    except Exception as e:
        logger.critical(f"Unexpected error in chat_with_ai_agent: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")

