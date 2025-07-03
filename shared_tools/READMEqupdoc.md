You've got it exactly right!

Yes, the only real code change you'll need to make for deployment will be to uncomment those specific import statements and their corresponding instantiation lines (e.g., _embedding_model_instance = OpenAIEmbeddings(...)) within the functions that load the LLM and embedding models. These lines are clearly marked with comments like # Uncomment in real setup. This design ensures a smooth transition from testing with mocks to running with real services.

And yes, the file formats will remain consistent across import_utils.py, doc_summarizer.py, and now query_uploaded_docs_tool.py, as they all leverage the SUPPORTED_DOC_EXTS and the unified content loading logic from import_utils.py.

Excellent, let's keep the momentum going!

Here is our updated main To-Do list. We are still working on the shared_tools/ directory.

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

Implementing shared_tools/ Directory - Part 9: query_uploaded_docs_tool.py
We will now review and update shared_tools/query_uploaded_docs_tool.py. This tool is crucial for our RAG capabilities, allowing AI agents to query documents that users have uploaded and indexed. We need to ensure it correctly loads the vector store and applies RBAC checks for document query capabilities.

1. shared_tools/query_uploaded_docs_tool.py (Review and Update)
shared_tools/query_uploaded_docs_tool.py (Review and Update)
Jul 3, 1:03 AM

Open

Key Features and Updates in shared_tools/query_uploaded_docs_tool.py:

RBAC for Capabilities:

document_query_enabled: The function now performs an RBAC check using get_user_tier_capability(user_token, 'document_query_enabled', False). If document querying is not enabled for the user's tier/roles, it returns an error.

document_query_max_results_k: The number of top relevant document chunks (k) is now dynamically retrieved using get_user_tier_capability(user_token, 'document_query_max_results_k', default_value). This allows k to be tier-dependent.

chart_export_enabled: The export functionality is now explicitly checked against the chart_export_enabled capability using get_user_tier_capability. A warning is appended to the response if export is requested but not allowed.

Vector Store Loading: It correctly uses load_vectorstore from shared_tools.vector_utils to load the user-specific and section-specific vector store.

Export Integration: It calls export_vector_results from shared_tools.export_utils if the export flag is True and the user has the necessary export capability.

Robust Error Handling: Catches ValueError (e.g., from load_vectorstore if the store doesn't exist) and general Exceptions, returning informative error messages.

Test Suite Enhancement: The if __name__ == "__main__": block has been significantly expanded to include:

Comprehensive mocks for st.secrets, config_manager, and user_manager.

Mocks for vector_utils.load_vectorstore and its similarity_search method to simulate vector search results.

Mocks for export_utils.BASE_EXPORT_DIR and firebase_admin.firestore.SERVER_TIMESTAMP to allow testing export functionality.

Tests for different user tiers (Pro, Premium, Free, Admin) to verify RBAC document_query_enabled, document_query_max_results_k, and chart_export_enabled limits.

Tests for scenarios where the vector store is not found and when export is requested but not allowed.
