# backend/services/chat_service.py

import logging
from typing import List, Dict, Any
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud import firestore
from database.firestore_manager import FirestoreManager
from config.config_manager import config_manager

logger = logging.getLogger(__name__)

class ChatService:
    """
    A service for managing chat sessions in Firestore.
    """
    def __init__(self, firestore_manager: FirestoreManager):
        self.firestore_manager = firestore_manager
        logger.info("ChatService initialized.")

    async def get_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all chat sessions for a given user.
        """
        try:
            sessions_ref = self.firestore_manager.db.collection('chat_sessions').where(filter=FieldFilter("user_id", "==", user_id))
            sessions = []
            async for doc in sessions_ref.stream():
                session_data = doc.to_dict()
                session_data['id'] = doc.id
                sessions.append(session_data)
            return sessions
        except Exception as e:
            logger.error(f"Error getting chat sessions for user {user_id}: {e}", exc_info=True)
            return []

    async def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all messages for a given chat session.
        """
        try:
            messages_ref = self.firestore_manager.db.collection('chat_sessions').document(session_id).collection('messages').order_by('timestamp')
            messages = []
            async for doc in messages_ref.stream():
                messages.append(doc.to_dict())
            return messages
        except Exception as e:
            logger.error(f"Error getting messages for session {session_id}: {e}", exc_info=True)
            return []

    async def create_session(self, user_id: str, title: str) -> str:
        """
        Creates a new chat session.
        """
        try:
            new_session_ref = self.firestore_manager.db.collection('chat_sessions').document()
            await new_session_ref.set({
                "user_id": user_id,
                "title": title,
                "created_at": firestore.SERVER_TIMESTAMP
            })
            return new_session_ref.id
        except Exception as e:
            logger.error(f"Error creating chat session for user {user_id}: {e}", exc_info=True)
            return ""

    async def add_message(self, session_id: str, role: str, content: str):
        """
        Adds a new message to a chat session.
        """
        try:
            new_message_ref = self.firestore_manager.db.collection('chat_sessions').document(session_id).collection('messages').document()
            await new_message_ref.set({
                "role": role,
                "content": content,
                "timestamp": firestore.SERVER_TIMESTAMP
            })
        except Exception as e:
            logger.error(f"Error adding message to session {session_id}: {e}", exc_info=True)

chat_service = ChatService(FirestoreManager())
