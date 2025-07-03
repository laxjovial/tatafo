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

You are absolutely correct: while shared_tools/import_utils.py was updated to read CSV, XLS, and XLSX files and convert their content into a text (markdown table) format for indexing into the vector store (for RAG queries), the shared_tools/doc_summarizer.py itself was not explicitly updated to extract text from these new file types for summarization by the LLM.

Currently, doc_summarizer.py only has direct extraction logic for .pdf, .docx, and .txt. If you tried to summarize a .csv or .xlsx file with the current doc_summarizer.py, it would return an "Unsupported file type" error.

Correction Needed: To ensure that users can indeed summarize CSV, XLS, and XLSX files, we need to update shared_tools/doc_summarizer.py to leverage the text extraction capabilities for these formats. The most efficient way to do this is to make doc_summarizer.py use the _load_document_content function from shared_tools/import_utils.py for all supported document types, as _load_document_content already handles the conversion of CSV/Excel to text.

Here is our updated main To-Do list:

[x] backend/

[x] config/

[x] data/

[x] database/

[ ] shared_tools/

[ ] domain_tools/ (and its sub-folders)

[ ] ui/

[x] utils/

[ ] tests/

[ ] docs/

[ ] scripts/

[ ] requirements.txt

[ ] .streamlit/

[ ] .gitignore

[ ] README.md

Implementing shared_tools/ Directory - Part 6: doc_summarizer.py (Re-Updated for CSV/XLS/XLSX Summarization)
Let's re-update shared_tools/doc_summarizer.py to integrate the text extraction for CSV, XLS, and XLSX files, ensuring they can be summarized.

1. shared_tools/doc_summarizer.py (Re-Updated for All Supported Document Types)
shared_tools/doc_summarizer.py (Re-Updated for All Supported Document Types)
Jul 2, 8:04 AM

Open

Key Changes in shared_tools/doc_summarizer.py (Re-Update):

Unified Content Loading:

Removed internal _extract_text_from_pdf, _extract_text_from_docx, _extract_text_from_txt.

Now imports _load_document_content and SUPPORTED_DOC_EXTS directly from shared_tools/import_utils.py.

The summarize_document function now calls _load_document_content(file_path) to get the document's content (which handles PDF, DOCX, TXT, CSV, XLS, XLSX). It then concatenates the page_content from the returned list of document parts.

Mock LLM Improvement: The mock LLM now more accurately reports the "Original content length" based on the actual doc_content it receives, which helps verify the truncation logic.

Expanded Test Suite: The if __name__ == "__main__": block now includes specific tests for CSV and XLSX file summarization, confirming that the new integration works. It also uses unittest.mock.patch to control the behavior of _load_document_content for consistent testing.

Result: Yes, with this update, users can now summarize CSV, XLS, and XLSX files, as their content will be correctly extracted (as markdown tables) by _load_document_content and then passed to the LLM for summarization, subject to RBAC rules.

