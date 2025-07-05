# backend/document_query_app.py

import streamlit as st
import logging
from typing import Dict, Any, Optional, List
import asyncio # For async operations

# Import config_manager to access configurations
from config.config_manager import config_manager
# Import analytics_tracker for logging events
from utils.analytics_tracker import log_event, initialize_analytics
# Import vector utilities
from shared_tools.vector_utils import get_vector_store, query_vector_store
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability

# Import Firebase Admin SDK components for backend initialization (if needed for context)
import firebase_admin
from firebase_admin import credentials, auth, firestore
import json
import os

logger = logging.getLogger(__name__)

# --- Firebase Admin SDK Initialization (for backend context) ---
if not firebase_admin._apps:
    try:
        firebase_config_str = config_manager.get_secret("firebase_config")
        if not firebase_config_str:
            raise ValueError("Firebase configuration not found in secrets.")
        
        firebase_config = json.loads(firebase_config_str)
        
        if os.environ.get("FIREBASE_ADMIN_CERT"):
            cred = credentials.Certificate(json.loads(os.environ.get("FIREBASE_ADMIN_CERT")))
        else:
            logger.warning("FIREBASE_ADMIN_CERT environment variable not found. Firebase Admin SDK functionality may be limited.")
            cred = credentials.Certificate({
                "type": "service_account",
                "project_id": firebase_config.get("projectId", "mock-project-id"),
                "private_key_id": "mock-key-id",
                "private_key": "-----BEGIN PRIVATE KEY-----\nMOCK_PRIVATE_KEY\n-----END PRIVATE KEY-----\n",
                "client_email": "mock-client@mock-project-id.iam.gserviceaccount.com",
                "client_id": "mock-client-id",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/mock-client%40{firebase_config.get('projectId', 'mock-project-id')}.iam.gserviceaccount.com",
                "universe_domain": "googleapis.com"
            })

        firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully in document_query_app.")
    except Exception as e:
        logger.error(f"Error initializing Firebase Admin SDK in document_query_app: {e}")

# Initialize analytics_tracker for backend context
if 'analytics_initialized_backend' not in globals(): # Use globals() for module-level check
    if firebase_admin._apps:
        try:
            db_instance = firestore.client()
            auth_instance = auth
            app_id_for_analytics = config_manager.get("app_id", firebase_config.get("projectId", "default-streamlit-app-id"))
            initialize_analytics(db_instance, auth_instance, app_id_for_analytics, "backend_system_user")
            globals()['analytics_initialized_backend'] = True
            logger.info("Analytics tracker initialized for document_query_app with live Firebase.")
        except Exception as e:
            logger.error(f"Failed to initialize analytics with live Firebase Admin SDK in document_query_app: {e}")
            mock_db = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: None})()})()
            mock_auth = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': None})()})()
            app_id_for_analytics = config_manager.get("app_id", "default-streamlit-app-id")
            initialize_analytics(mock_db, mock_auth, app_id_for_analytics, "backend_system_user")
            globals()['analytics_initialized_backend'] = True
            logger.warning("Analytics tracker initialized with mock Firebase for document_query_app.")
    else:
        mock_db = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: None})()})()
        mock_auth = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': None})()})()
        app_id_for_analytics = config_manager.get("app_id", "default-streamlit-app-id")
        initialize_analytics(mock_db, mock_auth, app_id_for_analytics, "backend_system_user")
        globals()['analytics_initialized_backend'] = True
        logger.warning("Analytics tracker initialized with mock Firebase for document_query_app (Admin SDK not available).")


def app():
    st.title("📚 Query Your Uploaded Documents")
    st.info("Select a domain and enter a query to retrieve information from your indexed documents.")

    # Ensure user is logged in
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.warning("Please log in to query documents.")
        asyncio.run(log_event('page_view', {
            'page_name': 'DocumentQuery',
            'status': 'access_denied',
            'reason': 'not_logged_in'
        }, user_id='unauthenticated', success=False))
        return

    user_id = st.session_state.get('user_id_from_backend')
    user_token = st.session_state.get('user_token') # For RBAC checks

    if not user_id or not user_token:
        st.error("User authentication information missing. Please log in again.")
        asyncio.run(log_event('page_view', {
            'page_name': 'DocumentQuery',
            'status': 'access_denied',
            'reason': 'missing_auth_info'
        }, user_id='unknown_user', success=False))
        return

    # RBAC check for document query capability
    if not get_user_tier_capability(user_token, 'document_query_enabled', False):
        st.warning("Your current subscription tier does not allow querying uploaded documents. Please upgrade your plan.")
        asyncio.run(log_event('page_view', {
            'page_name': 'DocumentQuery',
            'status': 'access_denied',
            'reason': 'rbac_denied'
        }, user_id=user_id, success=False))
        return

    # Log successful page view
    asyncio.run(log_event('page_view', {
        'page_name': 'DocumentQuery',
        'status': 'accessed',
        'user_id': user_id
    }, user_id=user_id, success=True))

    # Get configurable RAG parameters
    embedding_provider = config_manager.get('rag.embedding_provider', 'openai')
    embedding_model_name = config_manager.get('rag.embedding_model_name', 'text-embedding-ada-002')
    max_query_results_k = get_user_tier_capability(user_token, 'document_query_max_results_k', config_manager.get('rag.max_query_results_k', 4))

    st.subheader("Query Options")
    
    # Allow user to select a domain/collection to query
    available_domains = config_manager.get('rag.available_domains', ['general', 'medical', 'legal', 'finance', 'education', 'sports', 'news'])
    selected_domain = st.selectbox(
        "Select Document Domain to Query",
        options=available_domains,
        key="query_domain_select",
        help="Query the knowledge base for this domain."
    )

    query_text = st.text_area("Enter your query here:", key="document_query_text")

    if st.button("Search Documents"):
        if not query_text:
            st.error("Please enter a query.")
            asyncio.run(log_event('ui_interaction', {
                'component': 'DocumentQueryForm',
                'action': 'Submit Query',
                'details': {'domain': selected_domain, 'reason': 'empty_query'},
                'user_id': user_id,
                'success': False,
                'error_message': 'Empty query text'
            }))
            return

        with st.spinner(f"Searching documents in '{selected_domain}' knowledge base..."):
            try:
                # Construct the collection name for the user and domain
                collection_name = f"{user_id}_{selected_domain}"

                # Get the vector store
                vector_store = await get_vector_store(
                    collection_name,
                    user_id=user_id,
                    embedding_provider=embedding_provider,
                    embedding_model_name=embedding_model_name
                )

                if vector_store is None:
                    st.warning(f"No knowledge base found for domain '{selected_domain}'. Please upload documents first for this domain.")
                    asyncio.run(log_event('document_query', {
                        'query_text': query_text,
                        'domain': selected_domain,
                        'status': 'failure',
                        'reason': 'vector_store_not_found'
                    }, user_id=user_id, success=False, error_message="Vector store not found for selected domain."))
                    return

                # Query the vector store
                results = await query_vector_store(vector_store, query_text, user_id=user_id, k=max_query_results_k)

                if results:
                    st.subheader("Relevant Documents Found:")
                    for i, doc in enumerate(results):
                        st.markdown(f"**Result {i+1}:**")
                        st.markdown(f"**Source:** `{doc.get('metadata', {}).get('source', 'N/A')}`")
                        st.markdown(f"**Page/Chunk:** `{doc.get('metadata', {}).get('page', 'N/A')}`")
                        st.code(doc.get('page_content', ''))
                        st.markdown("---")
                    
                    st.success(f"Found {len(results)} relevant document chunks.")
                    asyncio.run(log_event('document_query', {
                        'query_text': query_text,
                        'domain': selected_domain,
                        'num_results': len(results),
                        'status': 'success'
                    }, user_id=user_id, success=True))
                else:
                    st.info(f"No relevant documents found for your query in the '{selected_domain}' knowledge base.")
                    asyncio.run(log_event('document_query', {
                        'query_text': query_text,
                        'domain': selected_domain,
                        'num_results': 0,
                        'status': 'success' # Still a successful query, just no results
                    }, user_id=user_id, success=True))

            except Exception as e:
                st.error(f"Error querying documents: {e}")
                logger.error(f"Document query failed for user {user_id}, domain {selected_domain}, query '{query_text}': {e}", exc_info=True)
                asyncio.run(log_event('document_query', {
                    'query_text': query_text,
                    'domain': selected_domain,
                    'status': 'failure',
                    'error_message': str(e)
                }, user_id=user_id, success=False, error_message=str(e)))

    st.markdown("---")
    st.markdown("Need to upload documents? Go to the [Upload Documents](/upload_documents) page.")


# Example of how to run this app standalone for testing
if __name__ == "__main__":
    # Mock session state for standalone testing
    if "user_id_from_backend" not in st.session_state:
        st.session_state.user_id_from_backend = "mock_premium_uid_query"
        st.session_state.user_token = "mock_token_premium_query"
        st.session_state.logged_in = True
    
    # Mock RBAC capability for testing
    import unittest.mock as mock
    original_get_user_tier_capability = get_user_tier_capability
    
    def mock_get_user_tier_capability(user_token, capability_key, default_value):
        if capability_key == 'document_query_enabled':
            return True # Enable for testing
        if capability_key == 'document_query_max_results_k':
            return 5 # Mock max results
        return original_get_user_tier_capability(user_token, capability_key, default_value)
    
    # Patch the actual function
    import sys
    sys.modules['utils.user_manager'].get_user_tier_capability = mock_get_user_tier_capability

    # Mock vector_utils functions
    mock_vector_store_instance = mock.MagicMock()
    mock_vector_store_instance.similarity_search.return_value = [
        mock.MagicMock(page_content="This is a relevant chunk about medical procedures.", metadata={"source": "med_doc.pdf", "page": 1}),
        mock.MagicMock(page_content="Another chunk discussing legal precedents.", metadata={"source": "legal_doc.txt", "page": 0})
    ]

    mock_get_vector_store = AsyncMock(return_value=mock_vector_store_instance)
    mock_query_vector_store = AsyncMock(side_effect=lambda vs, q, uid, k: mock_vector_store_instance.similarity_search(q, k=k))

    with patch('shared_tools.vector_utils.get_vector_store', new=mock_get_vector_store), \
         patch('shared_tools.vector_utils.query_vector_store', new=mock_query_vector_store):
        
        # Initialize analytics for the test run if not already done
        if 'analytics_initialized_backend' not in st.session_state:
            mock_db_for_analytics = type('FirestoreMock', (object,), {'collection': lambda s, path: type('CollectionMock', (object,), {'add': lambda s, data: asyncio.sleep(0.01)})()})()
            mock_auth_for_analytics = type('AuthMock', (object,), {'currentUser': type('CurrentUserMock', (object,), {'uid': "test_cli_user"})()})()
            initialize_analytics(mock_db_for_analytics, mock_auth_for_analytics, "test-app-id-cli", "test_cli_user")
            st.session_state.analytics_initialized_backend = True

        st.write("Running standalone Document Query App test. Enter a query to test.")
        app()

        # Restore original functions after testing
        sys.modules['utils.user_manager'].get_user_tier_capability = original_get_user_tier_capability
