# shared_tools/doc_summarizer.py

import logging
from typing import Optional, Dict, Any, List
import asyncio
from pathlib import Path
import os
import re
import tiktoken
import httpx
import json

from langchain_core.tools import tool

# Import config_manager and user_manager for RBAC checks
from config.config_manager import config_manager
from utils.user_manager import get_user_tier_capability
from utils.analytics_tracker import log_event

logger = logging.getLogger(__name__)

# Global variable to hold the LLM instance
# This will be initialized once and reused.
_llm_instance = None

# --- Helper to get LLM configurations ---
def _get_llm_config(user_token: str) -> Dict[str, Any]:
    """
    Retrieves LLM configuration based on user's tier and available models.
    """
    # Get default LLM settings from config_manager
    default_provider = config_manager.get("llm.default_provider", "gemini")
    default_model_name = config_manager.get("llm.default_model_name", "gemini-1.5-flash")
    default_temperature = config_manager.get("llm.default_temperature", 0.7)

    # Get user-specific overrides from RBAC capabilities
    provider = get_user_tier_capability(user_token, 'llm_default_provider', default_provider)
    model_name = get_user_tier_capability(user_token, 'llm_default_model_name', default_model_name)
    temperature = get_user_tier_capability(user_token, 'llm_default_temperature', default_temperature)

    # Ensure temperature is within valid range [0.0, 1.0]
    temperature = max(0.0, min(1.0, temperature))

    return {
        "provider": provider,
        "model_name": model_name,
        "temperature": temperature
    }

# --- LLM Initialization (Lazy Loading) ---
async def _initialize_llm(user_token: str):
    """
    Initializes the LLM instance if it hasn't been initialized yet.
    This function is designed to be called only once.
    """
    global _llm_instance
    if _llm_instance is None:
        llm_config = _get_llm_config(user_token)
        provider = llm_config["provider"]
        model_name = llm_config["model_name"]
        temperature = llm_config["temperature"]

        logger.info(f"Initializing LLM: Provider={provider}, Model={model_name}, Temperature={temperature}")

        if provider == "gemini":
            # Gemini models are typically accessed via Google's Generative AI API
            # The API key is managed by the config_manager
            api_key = config_manager.get_secret("gemini_api_key")
            if not api_key:
                logger.error("Gemini API key not found in secrets.")
                raise ValueError("Gemini API key is not configured.")
            
            # For simplicity, we'll use a direct HTTP client for Gemini API.
            # In a more complex setup, you might use a dedicated SDK.
            class GeminiLLM:
                def __init__(self, model: str, api_key: str, temperature: float):
                    self.model = model
                    self.api_key = api_key
                    self.temperature = temperature
                    self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
                    self.headers = {"Content-Type": "application/json"}

                async def generate_content(self, prompt: str) -> str:
                    payload = {
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": self.temperature,
                            "topK": 40,
                            "topP": 0.95,
                            "maxOutputTokens": 8192 # Max output tokens for gemini-1.5-flash
                        }
                    }
                    async with httpx.AsyncClient(timeout=60.0) as client: # Increased timeout
                        try:
                            response = await client.post(self.base_url, params={"key": self.api_key}, json=payload)
                            response.raise_for_status()
                            response_json = response.json()
                            if response_json and response_json.get("candidates"):
                                return response_json["candidates"][0]["content"]["parts"][0]["text"]
                            else:
                                logger.error(f"Unexpected Gemini API response structure: {response_json}")
                                return "Error: Could not get a valid response from the LLM."
                        except httpx.HTTPStatusError as e:
                            logger.error(f"HTTP error with Gemini API: {e.response.status_code} - {e.response.text}")
                            return f"Error from LLM API: {e.response.text}"
                        except httpx.RequestError as e:
                            logger.error(f"Network error with Gemini API: {e}")
                            return f"Network error connecting to LLM: {e}"
                        except Exception as e:
                            logger.error(f"An unexpected error occurred during Gemini API call: {e}")
                            return f"An unexpected error occurred with LLM: {e}"

            _llm_instance = GeminiLLM(model=model_name, api_key=api_key, temperature=temperature)
            logger.info("Gemini LLM instance created.")
        elif provider == "openai":
            api_key = config_manager.get_secret("openai_api_key")
            if not api_key:
                logger.error("OpenAI API key not found in secrets.")
                raise ValueError("OpenAI API key is not configured.")
            
            # Mock OpenAI for now, or integrate a simple client if needed
            class OpenAILLM:
                def __init__(self, model: str, api_key: str, temperature: float):
                    self.model = model
                    self.api_key = api_key
                    self.temperature = temperature
                    self.base_url = "https://api.openai.com/v1/chat/completions"
                    self.headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }

                async def generate_content(self, prompt: str) -> str:
                    payload = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.temperature,
                        "max_tokens": 4000 # Example max tokens for OpenAI
                    }
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        try:
                            response = await client.post(self.base_url, headers=self.headers, json=payload)
                            response.raise_for_status()
                            response_json = response.json()
                            if response_json and response_json.get("choices"):
                                return response_json["choices"][0]["message"]["content"]
                            else:
                                logger.error(f"Unexpected OpenAI API response structure: {response_json}")
                                return "Error: Could not get a valid response from the LLM."
                        except httpx.HTTPStatusError as e:
                            logger.error(f"HTTP error with OpenAI API: {e.response.status_code} - {e.response.text}")
                            return f"Error from LLM API: {e.response.text}"
                        except httpx.RequestError as e:
                            logger.error(f"Network error with OpenAI API: {e}")
                            return f"Network error connecting to LLM: {e}"
                        except Exception as e:
                            logger.error(f"An unexpected error occurred during OpenAI API call: {e}")
                            return f"An unexpected error occurred with LLM: {e}"
            _llm_instance = OpenAILLM(model=model_name, api_key=api_key, temperature=temperature)
            logger.info("OpenAI LLM instance created.")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

# --- Text Chunking Helper ---
def _num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def _chunk_text(text: str, max_tokens: int) -> List[str]:
    """
    Splits text into chunks that do not exceed max_tokens.
    Attempts to split by paragraphs/sentences first, then words if necessary.
    """
    if not text:
        return []

    tokens = _num_tokens_from_string(text)
    if tokens <= max_tokens:
        return [text]

    chunks = []
    current_chunk_tokens = 0
    current_chunk_text = []

    # Try splitting by paragraphs
    paragraphs = text.split('\n\n')
    for para in paragraphs:
        para_tokens = _num_tokens_from_string(para)
        if current_chunk_tokens + para_tokens <= max_tokens:
            current_chunk_text.append(para)
            current_chunk_tokens += para_tokens
        else:
            if current_chunk_text:
                chunks.append("\n\n".join(current_chunk_text))
            current_chunk_text = [para]
            current_chunk_tokens = para_tokens
            # If a single paragraph is too large, split it further
            if para_tokens > max_tokens:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sentence_chunk_text = []
                sentence_chunk_tokens = 0
                for sent in sentences:
                    sent_tokens = _num_tokens_from_string(sent)
                    if sentence_chunk_tokens + sent_tokens <= max_tokens:
                        sentence_chunk_text.append(sent)
                        sentence_chunk_tokens += sent_tokens
                    else:
                        if sentence_chunk_text:
                            chunks.append(" ".join(sentence_chunk_text))
                        sentence_chunk_text = [sent]
                        sentence_chunk_tokens = sent_tokens
                        # If a single sentence is too large, split by words
                        if sent_tokens > max_tokens:
                            words = sent.split(' ')
                            word_chunk_text = []
                            word_chunk_tokens = 0
                            for word in words:
                                word_tokens = _num_tokens_from_string(word)
                                if word_chunk_tokens + word_tokens <= max_tokens:
                                    word_chunk_text.append(word)
                                    word_chunk_tokens += word_tokens
                                else:
                                    if word_chunk_text:
                                        chunks.append(" ".join(word_chunk_text))
                                    word_chunk_text = [word]
                                    word_chunk_tokens = word_tokens
                            if word_chunk_text:
                                chunks.append(" ".join(word_chunk_text))
                            sentence_chunk_text = [] # Clear after handling oversized sentence
                            sentence_chunk_tokens = 0
                if sentence_chunk_text:
                    chunks.append(" ".join(sentence_chunk_text))
                current_chunk_text = [] # Clear after handling oversized paragraph
                current_chunk_tokens = 0
    
    if current_chunk_text:
        chunks.append("\n\n".join(current_chunk_text))

    # Final check: if any chunk is still too large, it means the splitting logic
    # needs to be more aggressive or the max_tokens is too small for basic units.
    # For robust handling, one might add a final word-level split here if needed.
    final_chunks = []
    for chunk in chunks:
        if _num_tokens_from_string(chunk) > max_tokens:
            # Fallback to aggressive word split if previous methods failed
            words = chunk.split(' ')
            temp_chunk = []
            temp_tokens = 0
            for word in words:
                word_tokens = _num_tokens_from_string(word)
                if temp_tokens + word_tokens <= max_tokens:
                    temp_chunk.append(word)
                    temp_tokens += word_tokens
                else:
                    final_chunks.append(" ".join(temp_chunk))
                    temp_chunk = [word]
                    temp_tokens = word_tokens
            if temp_chunk:
                final_chunks.append(" ".join(temp_chunk))
        else:
            final_chunks.append(chunk)

    return final_chunks


# --- Document Summarizer Tool ---
@tool
async def summarize_document(text: str, user_token: str = "default") -> str:
    """
    Summarizes a given text document using an LLM.
    The text will be chunked if it exceeds the LLM's maximum input capacity.
    For very long documents, it performs an iterative summarization (map-reduce style).

    Args:
        text (str): The full text content of the document to summarize.
        user_token (str, optional): The unique identifier for the user. Defaults to "default".
                                    Used for RBAC capability checks.

    Returns:
        str: A concise summary of the document, or an error message if summarization fails.
    """
    logger.info(f"Tool: summarize_document called for user: '{user_token}' (text length: {len(text)})")

    # Check if summarization is enabled for the user's tier/roles
    if not get_user_tier_capability(user_token, 'summarization_enabled', False):
        await log_event('tool_access_denied', {
            'tool_name': 'summarize_document',
            'user_id': user_token,
            'reason': 'summarization_not_enabled'
        }, user_id=user_token, success=False, error_message="Summarization capability not enabled for this user tier.")
        return "Summarization capability is not enabled for your account tier."

    try:
        # Initialize LLM if not already done
        await _initialize_llm(user_token)
        global _llm_instance # Ensure we are using the global instance

        if _llm_instance is None:
            raise ValueError("LLM instance could not be initialized.")

        # Get max input tokens for the chosen LLM model
        # This should ideally come from a model registry or LLM config
        # For Gemini 1.5 Flash, context window is 1M tokens, but we'll use a practical limit
        # for summarization to avoid excessive costs/latency for typical use cases.
        # This value should be configurable in config.yml.
        max_llm_input_tokens = config_manager.get("llm.max_summary_input_tokens", 128000) # Default to 128k tokens

        num_tokens = _num_tokens_from_string(text)
        logger.info(f"Document has {num_tokens} tokens. Max LLM input tokens: {max_llm_input_tokens}")

        if num_tokens <= max_llm_input_tokens:
            # Single pass summarization
            prompt = f"Please provide a concise summary of the following document:\n\n{text}\n\nSummary:"
            summary = await _llm_instance.generate_content(prompt)
            await log_event('summarize_document', {
                'user_id': user_token,
                'text_length': len(text),
                'num_tokens': num_tokens,
                'summary_length': len(summary),
                'mode': 'single_pass',
                'status': 'success'
            }, user_id=user_token, success=True)
            return summary
        else:
            # Iterative summarization (Map-Reduce style)
            logger.info("Document too long for single pass. Performing iterative summarization.")
            chunks = _chunk_text(text, max_llm_input_tokens // 2) # Divide by 2 to leave space for prompt and intermediate summary
            
            if not chunks:
                await log_event('summarize_document', {
                    'user_id': user_token,
                    'text_length': len(text),
                    'num_tokens': num_tokens,
                    'status': 'failure',
                    'reason': 'failed_to_chunk_text'
                }, user_id=user_token, success=False, error_message="Failed to chunk document for summarization.")
                return "Error: Document is too large to process for summarization."

            intermediate_summaries = []
            for i, chunk in enumerate(chunks):
                chunk_tokens = _num_tokens_from_string(chunk)
                logger.debug(f"Summarizing chunk {i+1}/{len(chunks)} (tokens: {chunk_tokens})")
                chunk_prompt = f"Please summarize the following text. Focus on key information and main ideas:\n\n{chunk}\n\nSummary of chunk {i+1}:"
                chunk_summary = await _llm_instance.generate_content(chunk_prompt)
                intermediate_summaries.append(chunk_summary)
                await log_event('summarize_document_chunk', {
                    'user_id': user_token,
                    'chunk_index': i,
                    'chunk_length': len(chunk),
                    'chunk_tokens': chunk_tokens,
                    'summary_length': len(chunk_summary),
                    'status': 'success'
                }, user_id=user_token, success=True)

            combined_summaries = "\n\n".join(intermediate_summaries)
            final_summary_tokens = _num_tokens_from_string(combined_summaries)
            logger.info(f"Combined intermediate summaries have {final_summary_tokens} tokens.")

            if final_summary_tokens <= max_llm_input_tokens:
                final_prompt = f"The following are summaries of different sections of a document. Please combine them into one coherent and concise overall summary:\n\n{combined_summaries}\n\nOverall Summary:"
                final_summary = await _llm_instance.generate_content(final_prompt)
                await log_event('summarize_document', {
                    'user_id': user_token,
                    'text_length': len(text),
                    'num_tokens': num_tokens,
                    'summary_length': len(final_summary),
                    'mode': 'map_reduce',
                    'status': 'success'
                }, user_id=user_token, success=True)
                return final_summary
            else:
                # If combined summaries are still too long, recursively summarize them
                logger.info("Combined summaries still too long. Recursively summarizing.")
                recursive_summary = await summarize_document(combined_summaries, user_token) # Recursive call
                await log_event('summarize_document', {
                    'user_id': user_token,
                    'text_length': len(text),
                    'num_tokens': num_tokens,
                    'summary_length': len(recursive_summary),
                    'mode': 'recursive_map_reduce',
                    'status': 'success'
                }, user_id=user_token, success=True)
                return recursive_summary

    except ValueError as ve:
        logger.error(f"Configuration error during summarization for user {user_token}: {ve}")
        await log_event('summarize_document', {
            'user_id': user_token,
            'text_length': len(text),
            'status': 'failure',
            'reason': 'configuration_error',
            'error_message': str(ve)
        }, user_id=user_token, success=False, error_message=str(ve))
        return f"Error: {ve}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during summarization for user {user_token}: {e}", exc_info=True)
        await log_event('summarize_document', {
            'user_id': user_token,
            'text_length': len(text),
            'status': 'failure',
            'reason': 'unexpected_error',
            'error_message': str(e)
        }, user_id=user_token, success=False, error_message=str(e))
        return f"An unexpected error occurred during summarization: {e}"

# CLI Test (optional)
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock, AsyncMock

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.gemini_api_key = "MOCK_GEMINI_API_KEY_123"
            self.openai_api_key = "MOCK_OPENAI_API_KEY_456"
            self.firebase_config = "{}" # Mock empty config for Firebase if not set

        def get(self, key, default=None):
            return getattr(self, key, default)
    
    class MockConfigManager:
        _instance = None
        _is_loaded = False
        def __init__(self):
            if MockConfigManager._instance is not None:
                raise Exception("ConfigManager is a singleton. Use get_instance().")
            MockConfigManager._instance = self
            self._config_data = {
                'llm': {
                    'default_provider': 'gemini',
                    'default_model_name': 'gemini-1.5-flash',
                    'default_temperature': 0.7,
                    'max_summary_input_tokens': 128000 # Max tokens for summarization
                },
                'web_scraping': { # Needed for other tools, but not directly by summarizer
                    'user_agent': 'Mozilla/5.0 (Test; Python)',
                    'timeout_seconds': 5,
                    'max_search_results': 5
                },
                'tiers': {}, # This will be overridden by tiers.yaml
                'default_user_tier': 'free',
                'default_user_roles': ['user'],
                'api_configs': []
            }
            self._is_loaded = True
        
        def get(self, key, default=None):
            parts = key.split('.')
            val = self._config_data
            for part in parts:
                if isinstance(val, dict) and part in val:
                    val = val[part]
                else:
                    return default
            return val
        
        def get_secret(self, key, default=None):
            return st.secrets.get(key, default)

        def set_secret(self, key, value):
            setattr(st.secrets, key, value)

        def get_api_provider_config(self, domain: str, provider_name: str) -> Optional[Dict[str, Any]]:
            return None

        def get_domain_api_providers(self, domain: str) -> Dict[str, Any]:
            return {}

    # Mock user_manager.get_user_tier_capability for testing RBAC
    _mock_users_for_test = {
        "default": {"user_id": "default", "username": "DefaultUser", "email": "default@example.com", "tier": "free", "roles": ["user"]},
        "mock_free_token": {"user_id": "mock_free_token", "username": "FreeUser", "email": "free@example.com", "tier": "free", "roles": ["user"]},
        "mock_pro_token": {"user_id": "mock_pro_token", "username": "ProUser", "email": "pro@example.com", "tier": "pro", "roles": ["user"]},
        "mock_premium_token": {"user_id": "mock_premium_token", "username": "PremiumUser", "email": "premium@example.com", "tier": "premium", "roles": ["user"]},
        "mock_admin_token": {"user_id": "mock_admin_token", "username": "AdminUser", "email": "admin@example.com", "tier": "admin", "roles": ["user", "admin"]},
    }
    _rbac_capabilities_for_test = {
        'capabilities': {
            'summarization_enabled': {
                'default': False,
                'roles': {'pro': True, 'premium': True, 'admin': True}
            },
            'llm_default_provider': {
                'default': 'gemini',
                'tiers': {'pro': 'gemini', 'premium': 'openai', 'admin': 'gemini'}
            },
            'llm_default_model_name': {
                'default': 'gemini-1.5-flash',
                'tiers': {'pro': 'gemini-1.5-flash', 'premium': 'gpt-4o', 'admin': 'gemini-1.5-flash'}
            },
            'llm_default_temperature': {
                'default': 0.7,
                'tiers': {'pro': 0.5, 'premium': 0.3, 'admin': 0.7}
            },
            'llm_temperature_control_enabled': {
                'default': False,
                'roles': {'premium': True, 'admin': True}
            },
            'llm_max_temperature': {
                'default': 1.0,
                'roles': {'premium': 0.8, 'admin': 1.0}
            },
            'web_search_enabled': {'default': False, 'roles': {'pro': True, 'premium': True, 'admin': True}},
            'web_search_limit_chars': {'default': 500, 'roles': {'pro': 3000, 'premium': 5000, 'admin': 10000}},
            'web_search_max_results': {'default': 2, 'roles': {'pro': 7, 'premium': 10, 'admin': 15}},
        }
    }

    def get_user_tier_capability_mock(user_token: Optional[str], capability_key: str, default_value: Any = None) -> Any:
        user_info = _mock_users_for_test.get(user_token, _mock_users_for_test["default"])
        user_tier = user_info.get('tier', 'free')
        user_roles = user_info.get('roles', [])

        if "admin" in user_roles:
            if capability_key in _rbac_capabilities_for_test['capabilities']:
                cap_config = _rbac_capabilities_for_test['capabilities'][capability_key]
                if isinstance(cap_config.get('default'), bool): return True
                if isinstance(cap_config.get('default'), (int, float)): return float('inf')
            return default_value
        
        capability_config = _rbac_capabilities_for_test.get('capabilities', {}).get(capability_key)
        if not capability_config:
            return default_value

        for role in user_roles:
            if role in capability_config.get('roles', {}):
                return capability_config['roles'][role]
        
        if user_tier in capability_config.get('tiers', {}):
            return capability_config['tiers'][user_tier]

        return capability_config.get('default', default_value)


    # Patch the actual imports for testing
    import streamlit as st_mock
    if not hasattr(st_mock, 'secrets'):
        st_mock.secrets = MockSecrets()
    
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager
    sys.modules['utils.user_manager'].get_user_tier_capability = get_user_tier_capability_mock

    # Mock log_event from analytics_tracker
    sys.modules['utils.analytics_tracker'].log_event = AsyncMock()

    # Mock LLM instances for testing
    class MockGeminiLLM:
        def __init__(self, model, api_key, temperature):
            self.model = model
            self.api_key = api_key
            self.temperature = temperature
            print(f"Mock Gemini LLM initialized: {model}, temp={temperature}")

        async def generate_content(self, prompt: str) -> str:
            if "Error" in prompt:
                raise Exception("Simulated LLM error")
            if "Please provide a concise summary" in prompt:
                return f"This is a mock summary of the document: {prompt[:50]}..."
            elif "Please summarize the following text" in prompt:
                return f"Mock chunk summary: {prompt[:30]}..."
            elif "combine them into one coherent" in prompt:
                return f"Mock final summary of combined chunks: {prompt[:50]}..."
            return "Mock LLM response."

    class MockOpenAILLM:
        def __init__(self, model, api_key, temperature):
            self.model = model
            self.api_key = api_key
            self.temperature = temperature
            print(f"Mock OpenAI LLM initialized: {model}, temp={temperature}")

        async def generate_content(self, prompt: str) -> str:
            if "Error" in prompt:
                raise Exception("Simulated LLM error")
            if "Please provide a concise summary" in prompt:
                return f"This is an OpenAI mock summary of the document: {prompt[:50]}..."
            elif "Please summarize the following text" in prompt:
                return f"OpenAI mock chunk summary: {prompt[:30]}..."
            elif "combine them into one coherent" in prompt:
                return f"OpenAI mock final summary of combined chunks: {prompt[:50]}..."
            return "OpenAI mock LLM response."

    # Patch the _initialize_llm function to use our mocks
    original_initialize_llm = _initialize_llm
    async def mock_initialize_llm(user_token: str):
        global _llm_instance
        if _llm_instance is None:
            llm_config = _get_llm_config(user_token)
            if llm_config["provider"] == "gemini":
                _llm_instance = MockGeminiLLM(llm_config["model_name"], "mock_key", llm_config["temperature"])
            elif llm_config["provider"] == "openai":
                _llm_instance = MockOpenAILLM(llm_config["model_name"], "mock_key", llm_config["temperature"])
            else:
                raise ValueError(f"Unsupported mock LLM provider: {llm_config['provider']}")
    _initialize_llm = mock_initialize_llm

    # Test users
    test_user_free = _mock_users_for_test["mock_free_token"]['user_id']
    test_user_pro = _mock_users_for_test["mock_pro_token"]['user_id']
    test_user_premium = _mock_users_for_test["mock_premium_token"]['user_id']
    test_user_admin = _mock_users_for_test["mock_admin_token"]['user_id']

    async def run_tests():
        print("\n--- Testing summarize_document function ---")

        # Test 1: Free user (summarization not enabled)
        print("\n--- Test 1: Free user (summarization not enabled) ---")
        short_text = "This is a short document about a new technology."
        result1 = await summarize_document(short_text, user_token=test_user_free)
        print(f"Result for free user: {result1}")
        assert "Summarization capability is not enabled" in result1
        sys.modules['utils.analytics_tracker'].log_event.assert_called_with(
            'tool_access_denied',
            {'tool_name': 'summarize_document', 'user_id': test_user_free, 'reason': 'summarization_not_enabled'},
            user_id=test_user_free, success=False, error_message="Summarization capability not enabled for this user tier."
        )
        sys.modules['utils.analytics_tracker'].log_event.reset_mock()
        global _llm_instance # Reset LLM instance for next test
        _llm_instance = None


        # Test 2: Pro user, short text (single pass)
        print("\n--- Test 2: Pro user, short text (single pass) ---")
        short_text = "This is a short document about the latest advancements in artificial intelligence. It covers machine learning, neural networks, and their applications in various industries. The future of AI looks promising with continuous research and development."
        result2 = await summarize_document(short_text, user_token=test_user_pro)
        print(f"Result for Pro user (short text):\n{result2}")
        assert "This is a mock summary of the document" in result2
        assert "gemini-1.5-flash" in str(_llm_instance.model) # Check if Gemini was used
        assert _llm_instance.temperature == 0.5 # Check temperature for Pro tier
        sys.modules['utils.analytics_tracker'].log_event.assert_called_with(
            'summarize_document',
            {'user_id': test_user_pro, 'text_length': len(short_text), 'num_tokens': _num_tokens_from_string(short_text), 'summary_length': len(result2), 'mode': 'single_pass', 'status': 'success'},
            user_id=test_user_pro, success=True
        )
        sys.modules['utils.analytics_tracker'].log_event.reset_mock()
        _llm_instance = None # Reset LLM instance for next test

        # Test 3: Premium user, long text (map-reduce)
        print("\n--- Test 3: Premium user, long text (map-reduce) ---")
        long_text = "This is a very long document that needs summarization. " * 5000 # Create a long text
        # Ensure it's long enough to trigger chunking based on default max_summary_input_tokens (128000)
        # Assuming average token length, 5000 words * 5 chars/word = 25000 chars, which is small.
        # Let's make it much longer.
        long_text = "This is a very long document that needs summarization. It discusses various aspects of climate change, including its causes, impacts, and potential mitigation strategies. The document delves into scientific evidence, economic implications, and social challenges. It also explores policy recommendations and international agreements aimed at addressing the global climate crisis. The importance of renewable energy sources, carbon capture technologies, and sustainable land use practices are highlighted. Furthermore, the document examines the role of individual actions and community efforts in fostering environmental stewardship. The challenges of transitioning to a low-carbon economy are acknowledged, alongside the opportunities for innovation and green job creation. Adaptation measures to cope with unavoidable climate impacts are also detailed. The need for interdisciplinary research and collaboration across sectors is emphasized to achieve long-term sustainability goals. Public awareness and education are identified as crucial for driving behavioral change and supporting climate action. The document concludes with a call for urgent and collective action to safeguard the planet for future generations. " * 5000 # Make it truly long

        result3 = await summarize_document(long_text, user_token=test_user_premium)
        print(f"Result for Premium user (long text):\n{result3[:500]}...")
        assert "OpenAI mock final summary of combined chunks" in result3 # Premium uses OpenAI mock
        assert "gpt-4o" in str(_llm_instance.model) # Check if OpenAI was used
        assert _llm_instance.temperature == 0.3 # Check temperature for Premium tier
        # Check that chunking and map-reduce mode were logged
        sys.modules['utils.analytics_tracker'].log_event.assert_any_call(
            'summarize_document_chunk',
            user_id=test_user_premium, success=True
        )
        sys.modules['utils.analytics_tracker'].log_event.assert_called_with(
            'summarize_document',
            {'user_id': test_user_premium, 'text_length': len(long_text), 'num_tokens': _num_tokens_from_string(long_text), 'summary_length': len(result3), 'mode': 'map_reduce', 'status': 'success'},
            user_id=test_user_premium, success=True
        )
        sys.modules['utils.analytics_tracker'].log_event.reset_mock()
        _llm_instance = None # Reset LLM instance for next test

        # Test 4: Admin user, very long text (recursive map-reduce)
        print("\n--- Test 4: Admin user, very long text (recursive map-reduce) ---")
        very_long_text = long_text * 5 # Even longer text to trigger recursive summarization
        result4 = await summarize_document(very_long_text, user_token=test_user_admin)
        print(f"Result for Admin user (very long text):\n{result4[:500]}...")
        assert "Mock final summary of combined chunks" in result4 # Admin uses Gemini mock
        assert "gemini-1.5-flash" in str(_llm_instance.model) # Check if Gemini was used
        assert _llm_instance.temperature == 0.7 # Check temperature for Admin tier
        # Check that recursive map-reduce mode was logged
        sys.modules['utils.analytics_tracker'].log_event.assert_any_call(
            'summarize_document_chunk',
            user_id=test_user_admin, success=True
        )
        sys.modules['utils.analytics_tracker'].log_event.assert_called_with(
            'summarize_document',
            {'user_id': test_user_admin, 'text_length': len(very_long_text), 'num_tokens': _num_tokens_from_string(very_long_text), 'summary_length': len(result4), 'mode': 'recursive_map_reduce', 'status': 'success'},
            user_id=test_user_admin, success=True
        )
        sys.modules['utils.analytics_tracker'].log_event.reset_mock()
        _llm_instance = None # Reset LLM instance for next test

        # Test 5: LLM error handling
        print("\n--- Test 5: LLM error handling ---")
        error_text = "This document will cause an Error in the LLM."
        result5 = await summarize_document(error_text, user_token=test_user_pro)
        print(f"Result for LLM error: {result5}")
        assert "An unexpected error occurred with LLM: Simulated LLM error" in result5
        sys.modules['utils.analytics_tracker'].log_event.assert_called_with(
            'summarize_document',
            {'user_id': test_user_pro, 'text_length': len(error_text), 'num_tokens': _num_tokens_from_string(error_text), 'status': 'failure', 'reason': 'unexpected_error', 'error_message': 'Simulated LLM error'},
            user_id=test_user_pro, success=False, error_message='Simulated LLM error'
        )
        sys.modules['utils.analytics_tracker'].log_event.reset_mock()
        _llm_instance = None # Reset LLM instance for next test

        # Test 6: Configuration error (e.g., missing API key)
        print("\n--- Test 6: Configuration error (e.g., missing API key) ---")
        # Temporarily disable Gemini API key
        st_mock.secrets.set_secret('gemini_api_key', None)
        result6 = await summarize_document(short_text, user_token=test_user_pro)
        print(f"Result for config error: {result6}")
        assert "Gemini API key is not configured." in result6
        sys.modules['utils.analytics_tracker'].log_event.assert_called_with(
            'summarize_document',
            {'user_id': test_user_pro, 'text_length': len(short_text), 'status': 'failure', 'reason': 'configuration_error', 'error_message': 'Gemini API key is not configured.'},
            user_id=test_user_pro, success=False, error_message='Gemini API key is not configured.'
        )
        sys.modules['utils.analytics_tracker'].log_event.reset_mock()
        st_mock.secrets.set_secret('gemini_api_key', "MOCK_GEMINI_API_KEY_123") # Restore
        _llm_instance = None # Reset LLM instance for next test

        # Test 7: Empty text
        print("\n--- Test 7: Empty text ---")
        result7 = await summarize_document("", user_token=test_user_pro)
        print(f"Result for empty text: {result7}")
        assert result7 == "" or "Error" in result7 # Depending on LLM behavior, might return empty or error
        sys.modules['utils.analytics_tracker'].log_event.assert_called_with(
            'summarize_document',
            {'user_id': test_user_pro, 'text_length': 0, 'num_tokens': 0, 'summary_length': len(result7), 'mode': 'single_pass', 'status': 'success'},
            user_id=test_user_pro, success=True
        )
        sys.modules['utils.analytics_tracker'].log_event.reset_mock()
        _llm_instance = None # Reset LLM instance for next test


    # Run the async tests
    asyncio.run(run_tests())
