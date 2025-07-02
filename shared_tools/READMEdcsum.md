Implementing shared_tools/ Directory - Part 4: doc_summarizer.py
We will now review and update shared_tools/doc_summarizer.py to ensure it uses config_manager for LLM configuration and integrates RBAC checks for summarization capabilities, such as summarization_enabled and summarization_max_input_chars.

1. shared_tools/doc_summarizer.py (Updated for RBAC and ConfigManager)
shared_tools/doc_summarizer.py (Updated for RBAC and ConfigManager)
Jul 2, 7:52 AM

Open

Key Changes in shared_tools/doc_summarizer.py:

RBAC for Capabilities:

summarization_enabled: The function now performs an RBAC check using get_user_tier_capability(user_token, 'summarization_enabled', False). If summarization is not enabled for the user's tier/roles, it returns an error.

summarization_max_input_chars: The maximum number of characters from the document that will be sent to the LLM for summarization is now dynamically retrieved using get_user_tier_capability(user_token, 'summarization_max_input_chars', default_value). This prevents sending excessively large documents to the LLM, which can be costly or hit token limits, and allows tier-based differentiation.

config_manager for LLM: The _get_llm_for_summarization() helper function now uses config_manager.get() and config_manager.get_secret() to retrieve the LLM provider, model name, temperature, and API keys. This centralizes LLM configuration.

LLM Lazy Loading: The _llm_instance is initialized only once when _get_llm_for_summarization() is first called, improving efficiency.

Mock LLM: The actual Langchain LLM imports are commented out, and a MockLLM class is used for testing. This mock simulates the LLM's invoke method, making tests runnable without real API keys and providing predictable output.

Robust Text Extraction: Includes helper functions _extract_text_from_pdf, _extract_text_from_docx, and _extract_text_from_txt with improved error handling for each file type.

Error Handling: More specific ValueError and general Exception handling for file operations and LLM calls.

Test Suite Enhancement: The if __name__ == "__main__": block has been significantly expanded to include:

Comprehensive mocks for st.secrets, config_manager, and user_manager to ensure consistent and isolated testing.

Tests for different user tiers (Pro, Premium, Free, Admin) to verify RBAC summarization_enabled and summarization_max_input_chars limits.

Tests for non-existent files, empty documents, and unsupported file types.

A test for long documents to verify the truncation logic based on max_input_chars_allowed.

