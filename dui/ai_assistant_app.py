# ui/ai_assistant_app.py

import streamlit as st
import requests
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# --- Configuration for FastAPI Backend ---
FASTAPI_BASE_URL = "http://localhost:8000" # Assuming FastAPI runs on port 8000

# --- Helper Functions for API Calls ---
def get_rbac_capabilities_from_backend(user_token: str) -> Dict[str, Any]:
    """Fetches RBAC capabilities for the user from the backend."""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/rbac/capabilities/{user_token}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching RBAC capabilities for {user_token}: {e}")
        st.error(f"Could not load RBAC capabilities: {e}")
        return {} # Fallback

def chat_with_llm_agent_backend(prompt: str, chat_history: List[Dict[str, str]], user_token: str,
                                 temperature: Optional[float] = None,
                                 llm_provider: Optional[str] = None,
                                 model_name: Optional[str] = None) -> str:
    """Sends a chat message to the LLM agent and returns its response."""
    try:
        payload = {
            "prompt": prompt,
            "chat_history": chat_history,
            "user_token": user_token,
            "temperature": temperature,
            "llm_provider": llm_provider,
            "model_name": model_name
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{FASTAPI_BASE_URL}/chat/agent", json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get("response", "Error: No response from agent.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with LLM agent: {e}", exc_info=True)
        st.error(f"Error communicating with AI: {e}. Please ensure the backend is running.")
        return f"I'm sorry, I couldn't process your request due to a communication error: {e}"

def app():
    st.title("🤖 AI Assistant")

    # Initialize session state variables if not already present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_token" not in st.session_state:
        st.session_state.user_token = "default_user_token" # Default for unauthenticated access
    if "rbac_capabilities" not in st.session_state:
        st.session_state.rbac_capabilities = get_rbac_capabilities_from_backend(st.session_state.user_token)
    
    # Reload RBAC capabilities if user token changes or not loaded
    if st.session_state.get("last_rbac_user_token") != st.session_state.user_token:
        st.session_state.rbac_capabilities = get_rbac_capabilities_from_backend(st.session_state.user_token)
        st.session_state.last_rbac_user_token = st.session_state.user_token

    # --- LLM Settings (in main content area for simplicity, or could be in sidebar if desired) ---
    st.subheader("LLM Configuration")
    
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # LLM Temperature Control
        can_control_temp = st.session_state.rbac_capabilities.get('llm_temperature_control_enabled', False)
        tier_default_temp = st.session_state.rbac_capabilities.get('llm_default_temperature', 0.7)
        max_allowed_temp = st.session_state.rbac_capabilities.get('llm_max_temperature', 1.0)

        # Ensure value is within min/max range for slider
        current_temp_value = st.session_state.get('llm_temperature', tier_default_temp)
        if not (0.0 <= current_temp_value <= 1.0):
            current_temp_value = tier_default_temp # Reset if out of bounds

        st.session_state.llm_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=current_temp_value,
            step=0.01,
            disabled=not can_control_temp,
            help=f"Controls creativity. Lower for more focused, higher for more creative. Max allowed: {max_allowed_temp}. Current tier default: {tier_default_temp}."
        )
        if not can_control_temp:
            st.info("Upgrade to control temperature.")

    with col2:
        # LLM Provider Selection
        can_select_model = st.session_state.rbac_capabilities.get('llm_model_selection_enabled', False)
        
        available_providers = ["openai", "google", "ollama"]
        default_provider_index = available_providers.index(st.session_state.rbac_capabilities.get('llm_default_provider', 'openai')) if st.session_state.rbac_capabilities.get('llm_default_provider', 'openai') in available_providers else 0

        st.session_state.llm_provider = st.selectbox(
            "Provider",
            options=available_providers,
            index=default_provider_index,
            disabled=not can_select_model,
            help="Select the Large Language Model provider."
        )
        if not can_select_model:
            st.info("Upgrade to select provider.")

    with col3:
        # LLM Model Selection
        provider_models = {
            "openai": ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"],
            "google": ["gemini-pro", "gemini-1.5-flash", "gemini-1.5-pro"],
            "ollama": ["llama2", "mistral", "phi3"]
        }
        current_provider_models = provider_models.get(st.session_state.llm_provider, [])
        default_model_index = current_provider_models.index(st.session_state.rbac_capabilities.get('llm_default_model_name', 'gpt-3.5-turbo')) if st.session_state.rbac_capabilities.get('llm_default_model_name', 'gpt-3.5-turbo') in current_provider_models else 0

        st.session_state.llm_model_name = st.selectbox(
            "Model",
            options=current_provider_models,
            index=default_model_index,
            disabled=not can_select_model,
            help="Select the specific LLM model."
        )
        if not can_select_model:
            st.info("Upgrade to select model.")

    st.markdown("---")

    # Display chat messages from history
    chat_container = st.container(height=500, border=True)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Get AI response
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_with_llm_agent_backend(
                        prompt=prompt,
                        chat_history=st.session_state.chat_history,
                        user_token=st.session_state.user_token,
                        temperature=st.session_state.llm_temperature,
                        llm_provider=st.session_state.llm_provider,
                        model_name=st.session_state.llm_model_name
                    )
                st.markdown(response)
        # Add AI response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Optional: Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Example of how to run this app standalone for testing
if __name__ == "__main__":
    # Mock session state for standalone testing
    if "user_token" not in st.session_state:
        st.session_state.user_token = "mock_premium_token" # Or "mock_free_token", etc.
    
    # Mock requests.get and requests.post for backend calls if running standalone without FastAPI
    import unittest.mock as mock
    original_requests_get = requests.get
    original_requests_post = requests.post

    def mock_requests_get(url, *args, **kwargs):
        if f"{FASTAPI_BASE_URL}/rbac/capabilities/" in url:
            mock_rbac_data = {
                'llm_temperature_control_enabled': True,
                'llm_default_temperature': 0.7,
                'llm_max_temperature': 1.0,
                'llm_model_selection_enabled': True,
                'web_search_enabled': True,
                'data_analysis_enabled': True,
                'summarization_enabled': True,
                'chart_generation_enabled': True,
                'sentiment_analysis_enabled': True,
                'document_upload_enabled': True,
                'document_query_enabled': True,
                'chart_export_enabled': True,
                'finance_tool_access': True,
                'historical_data_access': True,
                'crypto_tool_access': True,
                'news_tool_access': True,
                'medical_tool_access': True,
                'legal_tool_access': True,
                'education_tool_access': True,
                'entertainment_tool_access': True,
                'weather_tool_access': True,
                'travel_tool_access': True,
                'sports_tool_access': True,
                'analytics_access': True,
                'analytics_charts_enabled': True,
                'analytics_user_specific_access': True,
            }
            mock_response = mock.Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_rbac_data
            mock_response.raise_for_status = lambda: None
            return mock_response
        return original_requests_get(url, *args, **kwargs)

    def mock_requests_post(url, json, *args, **kwargs):
        if f"{FASTAPI_BASE_URL}/chat/agent" in url:
            prompt = json.get("prompt", "")
            user_token = json.get("user_token", "default")
            temp = json.get("temperature", 0.7)
            model = json.get("model_name", "gpt-3.5-turbo")
            mock_response_content = f"Mock AI response from {model} (temp={temp}) for '{prompt}' by user {user_token}. (This is a mock, connect to backend for real AI)."
            
            mock_response = mock.Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": mock_response_content}
            mock_response.raise_for_status = lambda: None
            return mock_response
        return original_requests_post(url, json, *args, **kwargs)

    requests.get = mock_requests_get
    requests.post = mock_requests_post
    
    app()

    # Restore original requests.get and requests.post after testing
    requests.get = original_requests_get
    requests.post = original_requests_post
