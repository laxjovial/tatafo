# backend/services/context_manager.py

import logging
from typing import List, Dict, Any, Optional
from backend.models.user_models import UserProfile

logger = logging.getLogger(__name__)

class ContextManager:
    """
    Manages the context provided to the LLM.
    This includes chat history, system prompts, user information, and more.
    """
    def __init__(self, config_manager: Any):
        self.config_manager = config_manager
        logger.info("ContextManager initialized.")

    def build_context(
        self,
        prompt: str,
        chat_history: List[Dict[str, str]],
        user_profile: UserProfile
    ) -> List[Dict[str, str]]:
        """
        Builds the context to be sent to the LLM.

        This is the initial implementation. It will be enhanced in later steps.
        """
        # For now, just prepend the chat history with the new prompt
        max_messages = self.config_manager.get("context.max_messages", 20)

        # Trim the chat history
        if len(chat_history) > max_messages:
            chat_history = chat_history[-max_messages:]

        # Use the user's custom system prompt if available, otherwise use a default.
        system_prompt = user_profile.system_prompt or "You are a helpful assistant."

        # Add user-specific information to the system prompt
        system_prompt += f" The user's name is {user_profile.username} and their subscription tier is {user_profile.tier}."

        # Add the system prompt to the beginning of the context
        context = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": prompt}]

        # Placeholder for tool outputs and document snippets
        # This is where we would add logic to inject other types of context.

        return context

context_manager = ContextManager(config_manager=config_manager)
