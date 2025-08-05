# backend/api/chat_api.py

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Annotated
from pydantic import BaseModel

from backend.middleware.auth_middleware import get_current_user
from backend.models.user_models import UserProfile
from backend.services.chat_service import chat_service

logger = logging.getLogger(__name__)
router = APIRouter()

class ChatSession(BaseModel):
    title: str

class ChatMessage(BaseModel):
    role: str
    content: str

@router.get("/sessions", response_model=List[Dict[str, Any]])
async def get_chat_sessions(current_user: Annotated[UserProfile, Depends(get_current_user)]):
    """
    Retrieves a list of all chat sessions for the current user.
    """
    user_id = current_user.user_id
    sessions = await chat_service.get_sessions(user_id)
    return sessions

@router.get("/sessions/{session_id}", response_model=List[Dict[str, Any]])
async def get_chat_session(session_id: str, current_user: Annotated[UserProfile, Depends(get_current_user)]):
    """
    Retrieves the chat history for a specific session.
    """
    user_id = current_user.user_id
    # In a real app, you would also check if the user has access to this session
    messages = await chat_service.get_session_messages(session_id)
    return messages

@router.post("/sessions", status_code=status.HTTP_201_CREATED)
async def create_chat_session(session: ChatSession, current_user: Annotated[UserProfile, Depends(get_current_user)]):
    """
    Creates a new chat session.
    """
    user_id = current_user.user_id
    session_id = await chat_service.create_session(user_id, session.title)
    return {"session_id": session_id}

@router.post("/sessions/{session_id}/messages", status_code=status.HTTP_201_CREATED)
async def add_chat_message(session_id: str, message: ChatMessage, current_user: Annotated[UserProfile, Depends(get_current_user)]):
    """
    Adds a new message to a chat session.
    """
    user_id = current_user.user_id
    # In a real app, you would also check if the user has access to this session
    await chat_service.add_message(session_id, message.role, message.content)
    return {"message": "Message added successfully."}
