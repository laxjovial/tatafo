# ui/app.py

import streamlit as st
import requests
import json
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration for FastAPI Backend ---
# In a real deployment, this would be an environment variable or a config file
FASTAPI_BASE_URL = "http://localhost:8000" # Assuming FastAPI runs on port 8000

# --- Helper Functions for API Calls ---
def get_user_profile(user_token: str) -> Dict[str, Any]:
    """Fetches user profile from the backend."""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/users/{user_token}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching user profile for {user_token}: {e}")
        st.error(f"Could not load user profile: {e}")
        return {"user_id": user_token, "username": "Guest", "tier": "free", "roles": ["user"]} # Fallback

def get_rbac_capabilities(user_token: str) -> Dict[str, Any]:
    """Fetches RBAC capabilities for the user from the backend."""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/rbac/capabilities/{user_token}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching RBAC capabilities for {user_token}: {e}")
        st.error(f"Could not load RBAC capabilities: {e}")
        return {} # Fallback

def chat_with_llm_agent(prompt: str, chat_history: List[Dict[str, str]], user_token: str,
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
        st.error(f"Error communicating with AI: {e}. Please try again.")
        return f"I'm sorry, I couldn't process your request due to a communication error: {e}"

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Advanced AI Assistant")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_token" not in st.session_state:
    # In a real app, this would come from an authentication system
    st.session_state.user_token = "default_user_token" # Placeholder for now
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}
if "rbac_capabilities" not in st.session_state:
    st.session_state.rbac_capabilities = {}

# Load user profile and RBAC capabilities on first run or if token changes
if not st.session_state.user_profile or st.session_state.user_profile.get("user_id") != st.session_state.user_token:
    st.session_state.user_profile = get_user_profile(st.session_state.user_token)
    st.session_state.rbac_capabilities = get_rbac_capabilities(st.session_state.user_token)
    logger.info(f"Loaded profile for user: {st.session_state.user_profile.get('username')}, Tier: {st.session_state.user_profile.get('tier')}")
    logger.info(f"Loaded RBAC capabilities: {st.session_state.rbac_capabilities}")

# --- Sidebar for User Profile and Settings ---
with st.sidebar:
    st.header("User Profile")
    user_profile = st.session_state.user_profile
    st.write(f"**Username:** {user_profile.get('username', 'N/A')}")
    st.write(f"**User ID:** `{user_profile.get('user_id', 'N/A')}`") # Display full user ID
    st.write(f"**Tier:** {user_profile.get('tier', 'N/A').capitalize()}")
    st.write(f"**Roles:** {', '.join(user_profile.get('roles', []))}")

    st.markdown("---")
    st.header("LLM Settings")

    # LLM Temperature Control
    can_control_temp = st.session_state.rbac_capabilities.get('llm_temperature_control_enabled', False)
    tier_default_temp = st.session_state.rbac_capabilities.get('llm_default_temperature', 0.7)
    max_allowed_temp = st.session_state.rbac_capabilities.get('llm_max_temperature', 1.0)

    st.session_state.llm_temperature = st.slider(
        "LLM Temperature",
        min_value=0.0,
        max_value=1.0,
        value=tier_default_temp,
        step=0.01,
        disabled=not can_control_temp,
        help=f"Controls creativity. Lower for more focused, higher for more creative. Max allowed: {max_allowed_temp}. Current tier default: {tier_default_temp}."
    )
    if not can_control_temp:
        st.info("Upgrade your tier to control LLM temperature.")

    # LLM Provider and Model Selection
    can_select_model = st.session_state.rbac_capabilities.get('llm_model_selection_enabled', False)
    
    # Mock available providers and models for UI display
    available_providers = ["openai", "google", "ollama"]
    provider_models = {
        "openai": ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"],
        "google": ["gemini-pro", "gemini-1.5-flash", "gemini-1.5-pro"],
        "ollama": ["llama2", "mistral", "phi3"]
    }

    st.session_state.llm_provider = st.selectbox(
        "LLM Provider",
        options=available_providers,
        index=available_providers.index(st.session_state.rbac_capabilities.get('llm_default_provider', 'openai')) if st.session_state.rbac_capabilities.get('llm_default_provider', 'openai') in available_providers else 0,
        disabled=not can_select_model,
        help="Select the Large Language Model provider."
    )
    if not can_select_model:
        st.info("Upgrade your tier to select LLM provider and model.")

    current_provider_models = provider_models.get(st.session_state.llm_provider, [])
    st.session_state.llm_model_name = st.selectbox(
        "LLM Model",
        options=current_provider_models,
        index=current_provider_models.index(st.session_state.rbac_capabilities.get('llm_default_model_name', 'gpt-3.5-turbo')) if st.session_state.rbac_capabilities.get('llm_default_model_name', 'gpt-3.5-turbo') in current_provider_models else 0,
        disabled=not can_select_model,
        help="Select the specific LLM model."
    )

    st.markdown("---")
    st.header("Analytics (Coming Soon!)")
    # Placeholder for analytics features - visible but disabled for lower tiers
    # This section will be populated with actual analytics data later
    can_access_analytics = st.session_state.rbac_capabilities.get('analytics_access_enabled', False)
    
    if can_access_analytics:
        st.success("You have access to advanced analytics features!")
        st.button("View Usage Dashboard", disabled=False, help="View your detailed usage statistics.")
        st.button("Generate Custom Reports", disabled=False, help="Generate custom reports on your interactions.")
    else:
        st.info("Upgrade to a higher tier to unlock advanced analytics features.")
        st.button("View Usage Dashboard", disabled=True, help="Upgrade your tier to view detailed usage statistics.")
        st.button("Generate Custom Reports", disabled=True, help="Upgrade your tier to generate custom reports on your interactions.")
        st.markdown("---")
        st.markdown("### Upgrade Your Plan!")
        st.write("Unlock more powerful features, including advanced analytics, custom LLM control, and more!")
        st.button("Learn More & Upgrade", help="Click to see subscription options.")


# --- Main Chat Interface ---
st.title("Advanced AI Assistant")

# Display chat messages from history
chat_container = st.container(height=600, border=True)
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
                response = chat_with_llm_agent(
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
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

